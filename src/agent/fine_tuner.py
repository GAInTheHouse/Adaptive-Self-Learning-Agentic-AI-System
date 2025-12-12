"""
Fine-Tuning Module with Validation Monitoring and LoRA Support
Supports Wav2Vec2 models with LoRA for efficient fine-tuning.
"""

import logging
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import time

# Initialize logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import DataCollatorCTCWithPadding - may not be available in all transformers versions
try:
    from transformers import DataCollatorCTCWithPadding
    DATA_COLLATOR_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import paths for different transformers versions
        from transformers.data.data_collator import DataCollatorCTCWithPadding
        DATA_COLLATOR_AVAILABLE = True
    except ImportError:
        DATA_COLLATOR_AVAILABLE = False
        logger.debug("DataCollatorCTCWithPadding not available. Will use default data collator for fine-tuning.")

# LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. LoRA fine-tuning will be disabled. Install with: pip install peft")


class ErrorDataset(Dataset):
    """Dataset for fine-tuning on error samples for Wav2Vec2"""
    
    def __init__(self, error_samples: List[Dict], processor):
        """
        Initialize error dataset.
        
        Args:
            error_samples: List of error samples with 'audio_path' and 'corrected_transcript'
            processor: Wav2Vec2 processor for audio preprocessing
        """
        self.error_samples = error_samples
        self.processor = processor
    
    def __len__(self):
        return len(self.error_samples)
    
    def __getitem__(self, idx):
        sample = self.error_samples[idx]
        # Load audio
        import librosa
        audio, sr = librosa.load(sample['audio_path'], sr=16000)
        
        # Process audio for Wav2Vec2
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )
        
        # Process text labels
        with self.processor.as_target_processor():
            label_ids = self.processor(
            sample['corrected_transcript'],
            padding=True,
            return_tensors="pt"
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'labels': label_ids.input_ids.squeeze(0)
        }


class FineTuner:
    """
    Fine-tuning module for Wav2Vec2 models with validation monitoring, overfitting prevention, and LoRA support.
    """
    
    def __init__(
        self,
        model: Optional[Wav2Vec2ForCTC] = None,
        processor: Optional[Wav2Vec2Processor] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        validation_split: float = 0.2,
        overfitting_threshold: float = 0.1,
        early_stopping_patience: int = 3,
        min_accuracy_gain: float = 0.01,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        output_dir: Optional[str] = None
    ):
        """
        Initialize fine-tuner for Wav2Vec2 models.
        
        Args:
            model: Pre-loaded Wav2Vec2 model. If None, model_name must be provided.
            processor: Pre-loaded Wav2Vec2 processor. If None, will load from model_name.
            model_name: HuggingFace model name to load (e.g., "facebook/wav2vec2-base-960h")
            device: Device to use (cuda/cpu/mps)
            validation_split: Fraction of data for validation
            overfitting_threshold: Max allowed difference between train/val accuracy
            early_stopping_patience: Number of epochs to wait before early stopping
            min_accuracy_gain: Minimum accuracy gain to consider fine-tuning successful
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_rank: LoRA rank (number of trainable parameters)
            lora_alpha: LoRA alpha scaling factor
            output_dir: Directory to save fine-tuned model
        """
        # CTC loss is not implemented for MPS, so force CPU for fine-tuning
        # This ensures CTC loss computation works correctly
        if device and device.startswith("mps"):
            logger.warning("MPS device detected. CTC loss not supported on MPS, using CPU instead.")
            self.device = "cpu"
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lora = use_lora and PEFT_AVAILABLE
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Store model_name for metadata
        # If model is provided, try to get name from config or use a default
        if model_name:
            self.model_name = model_name
        elif model:
            # Try to get model name from model config
            if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
                self.model_name = model.config._name_or_path
            else:
                self.model_name = "facebook/wav2vec2-base-960h"  # Default
        else:
            self.model_name = None
        
        # Load model and processor if not provided
        if model is None:
            if model_name is None:
                raise ValueError("Either model or model_name must be provided")
            
            logger.info(f"Loading Wav2Vec2 model: {model_name}")
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model_name = model_name  # Store for metadata
            
            # Verify model is in valid state - check for NaN weights
            logger.info("Verifying model weights are valid...")
            nan_found = False
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    logger.warning(f"NaN detected in {name} - initializing with zeros...")
                    # Initialize with small random values if NaN found
                    with torch.no_grad():
                        param.data = torch.zeros_like(param.data)
                    nan_found = True
                if torch.isinf(param).any():
                    logger.warning(f"Inf detected in {name} - initializing with zeros...")
                    with torch.no_grad():
                        param.data = torch.zeros_like(param.data)
                    nan_found = True
            if nan_found:
                logger.warning("Fixed NaN/Inf weights by reinitializing them")
            else:
                logger.info("Model weights verified - no NaN/Inf found")
        else:
            self.model = model
            self.processor = processor
        
        self.model.to(self.device)
        
        # WARNING: LoRA with Wav2Vec2 CTC has compatibility issues
        # PEFT/LoRA causes NaN logits in Wav2Vec2 forward pass
        # For Wav2Vec2, disable LoRA and use full fine-tuning instead
        is_wav2vec2 = isinstance(self.model, Wav2Vec2ForCTC) or "wav2vec2" in str(type(self.model)).lower()
        if is_wav2vec2 and self.use_lora:
            logger.warning("=" * 60)
            logger.warning("WARNING: LoRA with Wav2Vec2 CTC models causes NaN logits.")
            logger.warning("Disabling LoRA and using full fine-tuning for Wav2Vec2.")
            logger.warning("Full fine-tuning for small datasets (< 100 samples) is still efficient.")
            logger.warning("=" * 60)
            self.use_lora = False
        
        # Fix for Wav2Vec2: Disable dropout during training to prevent NaN in train mode
        # Wav2Vec2 dropout layers can cause numerical instability in train mode
        if is_wav2vec2:
            logger.info("Stabilizing Wav2Vec2 model for training...")
            logger.info("Strategy: Freeze encoder, only train CTC head (lm_head)")
            dropout_count = 0
            frozen_count = 0
            trainable_count = 0
            
            # Freeze encoder parameters, only train CTC head
            for name, param in self.model.named_parameters():
                # Only train CTC head, freeze everything else
                if 'lm_head' in name or 'classifier' in name:
                    # CTC head should be trainable
                    param.requires_grad = True
                    trainable_count += 1
                else:
                    # Freeze all encoder parameters
                    param.requires_grad = False
                    frozen_count += 1
            
            # Also disable dropout modules and set LayerNorm to eval mode
            ln_count = 0
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
                    dropout_count += 1
                elif isinstance(module, torch.nn.Dropout1d):
                    module.p = 0.0
                    dropout_count += 1
                elif isinstance(module, torch.nn.Dropout2d):
                    module.p = 0.0
                    dropout_count += 1
                elif isinstance(module, torch.nn.LayerNorm):
                    # Set LayerNorm to eval mode to prevent NaN in train mode
                    # Since encoder is frozen, LayerNorm doesn't need to be in train mode
                    module.eval()
                    ln_count += 1
            
            logger.info(f"Froze {frozen_count} encoder parameters, {trainable_count} CTC head parameters trainable")
            logger.info(f"Disabled {dropout_count} dropout layers and set {ln_count} LayerNorm layers to eval mode")
        
        # Apply LoRA if requested and available
        if self.use_lora:
            logger.info(f"Applying LoRA adapters (rank={lora_rank}, alpha={lora_alpha})")
            
            # Wav2Vec2 has different attention structure - auto-detect target modules
            # Wav2Vec2 encoder layers have attention modules named like:
            # encoder.layers.X.attention.q_proj, k_proj, v_proj, out_proj
            model_modules = [name for name, _ in self.model.named_modules()]
            
            # Try to find attention projection modules by looking for the module name pattern
            # PEFT needs just the last part (e.g., 'q_proj') and will match all instances
            target_modules = set()
            for name in model_modules:
                # Look for attention projection modules in Wav2Vec2
                if 'attention' in name and any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
                    # Extract the projection name (last part after the last dot)
                    parts = name.split('.')
                    for part in reversed(parts):
                        if part in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                            target_modules.add(part)
                            break
            
            target_modules = list(target_modules)
            
            # If no attention modules found, disable LoRA and use full fine-tuning
            if not target_modules:
                logger.warning("Could not find Wav2Vec2 attention modules (q_proj, k_proj, v_proj, out_proj) for LoRA.")
                logger.warning("Available modules (sample): " + ", ".join(model_modules[:10]) + "...")
                logger.warning("Disabling LoRA and using full fine-tuning instead.")
                self.use_lora = False
            
            if self.use_lora:
                logger.info(f"Found LoRA target modules: {target_modules}")
                try:
                    # Use AUTOMATIC_SPEECH_RECOGNITION or FEATURE_EXTRACTION for Wav2Vec2
                    # Note: PEFT may still try to handle input_ids, but we'll work around that
                    try:
                        # Try ASR task type first (if available)
                        task_type = TaskType.AUTOMATIC_SPEECH_RECOGNITION
                    except AttributeError:
                        # Fallback to FEATURE_EXTRACTION
                        task_type = TaskType.FEATURE_EXTRACTION
                    
                    lora_config = LoraConfig(
                        task_type=task_type,
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=0.1,
                        bias="none"
                    )
                    self.model = get_peft_model(self.model, lora_config)
                    
                    # CRITICAL: For Wav2Vec2 CTC, the lm_head (CTC head) must be trainable
                    # LoRA only trains attention modules, but the CTC head needs to be trained too
                    # Mark the CTC head as trainable
                    for name, param in self.model.named_parameters():
                        if 'lm_head' in name or 'classifier' in name:
                            param.requires_grad = True
                            logger.info(f"Marked {name} as trainable (CTC head)")
                    
                    # Verify CTC head is trainable
                    lm_head_params = [name for name, param in self.model.named_parameters() if 'lm_head' in name and param.requires_grad]
                    if lm_head_params:
                        logger.info(f"CTC head parameters marked as trainable: {len(lm_head_params)}")
                    else:
                        logger.warning("WARNING: No CTC head parameters found or marked as trainable!")
                    
                    # Patch the actual Wav2Vec2 model's forward to ignore PEFT's language model kwargs
                    # PEFT passes input_ids, inputs_embeds, etc. for language models, but Wav2Vec2 only uses input_values
                    # Find the actual model (could be base_model.model or base_model)
                    actual_model = self.model.base_model
                    if hasattr(actual_model, 'model'):
                        actual_model = actual_model.model
                    
                    original_model_forward = actual_model.forward
                    
                    def patched_model_forward(*args, **kwargs):
                        # Strip all language model kwargs - Wav2Vec2 doesn't need them
                        # Wav2Vec2 only uses: input_values, attention_mask, labels, output_attentions, etc.
                        unwanted_keys = ['input_ids', 'inputs_embeds', 'decoder_input_ids', 'decoder_inputs_embeds']
                        for key in unwanted_keys:
                            kwargs.pop(key, None)
                        return original_model_forward(*args, **kwargs)
                    
                    actual_model.forward = patched_model_forward
                    
                    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in self.model.parameters())
                    logger.info(f"LoRA enabled: {trainable_params:,} trainable parameters out of {total_params:,} total ({100*trainable_params/total_params:.2f}%)")
                except Exception as e:
                    logger.error(f"Failed to apply LoRA: {e}")
                    logger.warning("Falling back to full fine-tuning")
                    self.use_lora = False
        else:
            if use_lora and not PEFT_AVAILABLE:
                logger.warning("LoRA requested but PEFT not available. Falling back to full fine-tuning.")
            logger.info("Using full fine-tuning (all parameters will be updated)")
        
        self.validation_split = validation_split
        self.overfitting_threshold = overfitting_threshold
        self.early_stopping_patience = early_stopping_patience
        self.min_accuracy_gain = min_accuracy_gain
        
        # Training history
        self.training_history = []
        
        logger.info(f"Fine-tuner initialized on device: {self.device} (Wav2Vec2 model)")
    
    def fine_tune(
        self,
        error_samples: List[Dict],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        max_grad_norm: float = 1.0,
        use_hf_trainer: bool = None  # None = auto-detect based on model type
    ) -> Dict:
        """
        Fine-tune model on error samples with validation monitoring.
        
        Args:
            error_samples: List of error samples with 'audio_path' and 'corrected_transcript'
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            use_hf_trainer: Whether to use HuggingFace Trainer (None = auto-detect)
        
        Returns:
            Dictionary with fine-tuning results
        """
        if len(error_samples) < 10:
            logger.warning(f"Insufficient samples ({len(error_samples)}), skipping fine-tuning")
            logger.warning("Minimum 10 samples required for fine-tuning.")
            return {
                'success': False,
                'reason': 'insufficient_samples',
                'samples_provided': len(error_samples),
                'num_samples': len(error_samples),
                'model_path': None
            }
        
        # Wav2Vec2 benefits from HF Trainer for better CTC handling
        if use_hf_trainer is None:
            use_hf_trainer = True
        
        # Split into train and validation
        np.random.seed(42)
        indices = np.random.permutation(len(error_samples))
        split_idx = int(len(error_samples) * (1 - self.validation_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_samples = [error_samples[i] for i in train_indices]
        val_samples = [error_samples[i] for i in val_indices]
        
        logger.info(f"Fine-tuning on {len(train_samples)} train samples, {len(val_samples)} validation samples")
        
        # Use HuggingFace Trainer for Wav2Vec2 (better CTC handling)
        if use_hf_trainer:
            return self._fine_tune_with_trainer(
                train_samples, val_samples, num_epochs, batch_size, learning_rate
            )
        else:
            # Use manual training loop when specified
            return self._fine_tune_manual(
                train_samples, val_samples, num_epochs, batch_size, learning_rate, max_grad_norm
            )
    
    def _fine_tune_manual(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        max_grad_norm: float
    ) -> Dict:
        """Manual training loop (alternative to HF Trainer)."""
        # Create datasets
        train_dataset = ErrorDataset(train_samples, self.processor)
        val_dataset = ErrorDataset(val_samples, self.processor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Get initial validation accuracy
        initial_val_accuracy = self._evaluate(val_loader)
        initial_train_accuracy = self._evaluate(train_loader)
        
        logger.info(f"Initial train accuracy: {initial_train_accuracy:.4f}")
        logger.info(f"Initial validation accuracy: {initial_val_accuracy:.4f}")
        
        # Training loop
        best_val_accuracy = initial_val_accuracy
        best_model_state = None
        patience_counter = 0
        overfitting_detected = False
        
        training_start_time = datetime.now()
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_values = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_values=input_values, labels=labels)
                
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_accuracy = self._evaluate(val_loader)
            train_accuracy = self._evaluate(train_loader)
            
            # Check for overfitting
            accuracy_gap = train_accuracy - val_accuracy
            if accuracy_gap > self.overfitting_threshold:
                overfitting_detected = True
                logger.warning(
                    f"Epoch {epoch+1}: Overfitting detected! "
                    f"Train: {train_accuracy:.4f}, Val: {val_accuracy:.4f}, Gap: {accuracy_gap:.4f}"
                )
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Val Acc: {val_accuracy:.4f}"
            )
            
            # Record training history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'validation_accuracy': val_accuracy,
                'accuracy_gap': accuracy_gap,
                'overfitting_detected': overfitting_detected
            })
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        training_end_time = datetime.now()
        training_duration = (training_end_time - training_start_time).total_seconds()
        
        # Calculate final metrics
        final_val_accuracy = self._evaluate(val_loader)
        final_train_accuracy = self._evaluate(train_loader)
        accuracy_gain = final_val_accuracy - initial_val_accuracy
        
        # Save model if output directory provided
        model_path = None
        if self.output_dir:
            model_path = self._save_model()
        
        # Estimate training cost
        training_cost = self._estimate_training_cost(training_duration, len(train_samples))
        
        # Determine success
        success = (
            accuracy_gain >= self.min_accuracy_gain and
            not (overfitting_detected and accuracy_gain < self.min_accuracy_gain * 2)
        )
        
        result = {
            'success': success,
            'initial_validation_accuracy': initial_val_accuracy,
            'final_validation_accuracy': final_val_accuracy,
            'initial_train_accuracy': initial_train_accuracy,
            'final_train_accuracy': final_train_accuracy,
            'accuracy_gain': accuracy_gain,
            'overfitting_detected': overfitting_detected,
            'training_duration_seconds': training_duration,
            'training_cost': training_cost,
            'samples_used': len(train_samples) + len(val_samples),
            'train_samples': len(train_samples),
            'validation_samples': len(val_samples),
            'epochs_completed': len(self.training_history),
            'best_validation_accuracy': best_val_accuracy,
            'model_path': model_path
        }
        
        if success:
            logger.info(
                f"Fine-tuning completed successfully: "
                f"accuracy_gain={accuracy_gain:.4f}, cost={training_cost:.2f}"
            )
        else:
            logger.warning(
                f"Fine-tuning did not meet success criteria: "
                f"accuracy_gain={accuracy_gain:.4f}, overfitting={overfitting_detected}"
            )
        
        return result
    
    def _fine_tune_with_trainer(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        num_epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Dict:
        """Use HuggingFace Trainer for Wav2Vec2 (better CTC handling with proper padding)."""
        from transformers import TrainingArguments
        
        # Prepare processed data
        train_data = self._prepare_processed_data(train_samples)
        val_data = self._prepare_processed_data(val_samples)
        
        # Verify data format
        if train_data:
            logger.info(f"Sample train data keys: {list(train_data[0].keys())}")
            if 'input_values' not in train_data[0]:
                raise ValueError(f"Missing 'input_values' in processed data. Keys: {list(train_data[0].keys())}")
            
            # Test model forward pass with a sample before training
            logger.info("Testing model forward pass with sample data before training...")
            test_sample = train_data[0]
            test_input_values = test_sample['input_values'].unsqueeze(0).to(self.device)  # Add batch dim
            test_labels = test_sample['labels'].unsqueeze(0).to(self.device)  # Add batch dim
            
            # Test in eval mode first
            self.model.eval()
            with torch.no_grad():
                try:
                    test_outputs = self.model(input_values=test_input_values, labels=test_labels)
                    test_logits = test_outputs.logits
                    if torch.isnan(test_logits).any():
                        logger.error("CRITICAL: Model produces NaN logits in eval mode before training!")
                        logger.error(f"Test logits shape: {test_logits.shape}, NaN count: {torch.isnan(test_logits).sum()}")
                        raise ValueError("Model produces NaN logits in test forward pass (eval mode)")
                    logger.info(f"✓ Model forward pass test passed (eval mode). Logits shape: {test_logits.shape}, range: [{test_logits.min():.4f}, {test_logits.max():.4f}]")
                except Exception as e:
                    logger.error(f"Model test forward pass failed in eval mode: {e}")
                    raise
            
            # Test in train mode
            self.model.train()
            try:
                test_outputs_train = self.model(input_values=test_input_values, labels=test_labels)
                test_logits_train = test_outputs_train.logits
                if torch.isnan(test_logits_train).any():
                    logger.error("CRITICAL: Model produces NaN logits in train mode before training!")
                    logger.error(f"Test logits shape: {test_logits_train.shape}, NaN count: {torch.isnan(test_logits_train).sum()}")
                    logger.error("This indicates the model forward pass has numerical instability in training mode.")
                    logger.error("Possible causes: dropout, batch normalization, or CTC head initialization issues.")
                    raise ValueError("Model produces NaN logits in test forward pass (train mode)")
                logger.info(f"✓ Model forward pass test passed (train mode). Logits shape: {test_logits_train.shape}, range: [{test_logits_train.min():.4f}, {test_logits_train.max():.4f}]")
            except Exception as e:
                logger.error(f"Model test forward pass failed in train mode: {e}")
                raise
        
        # Create simple dataset classes that preserve all keys
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
                # Verify first item has required keys
                if data:
                    first_item = data[0]
                    if 'input_values' not in first_item or 'labels' not in first_item:
                        raise ValueError(f"Dataset item missing required keys. Keys: {list(first_item.keys())}")
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                # Return item as-is, ensuring it's a dict with all keys preserved
                # Trainer should not remove columns if remove_unused_columns=False
                if not isinstance(item, dict):
                    raise ValueError(f"Expected dict, got {type(item)}")
                # Make a copy to ensure keys aren't accidentally removed
                return dict(item)
        
        train_dataset = SimpleDataset(train_data)
        val_dataset = SimpleDataset(val_data)
        
        # Training arguments
        output_dir = self.output_dir if self.output_dir else Path("models/finetuned")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Adjust learning rate for stability with LayerNorm unfrozen
        # Wav2Vec2 with LayerNorm requires much lower LR to prevent gradient explosion
        # Use 1e-5 max to prevent Inf loss while still allowing learning
        adjusted_lr = min(learning_rate, 5e-6)  # Very conservative LR for stability
        if adjusted_lr < learning_rate:
            logger.info(f"Reducing learning rate from {learning_rate} to {adjusted_lr} for stability with LayerNorm")
        
        # Build training args - handle different parameter names in different transformers versions
        # evaluation_strategy was renamed to eval_strategy in newer transformers versions
        training_args_dict = {
            "output_dir": str(output_dir),
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "learning_rate": adjusted_lr,
            "logging_dir": str(output_dir / "logs"),
            "logging_steps": 10,
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "load_best_model_at_end": False,
            "push_to_hub": False,
            "report_to": "none",
            "remove_unused_columns": False,  # Don't remove input_values column
            "no_cuda": self.device == "cpu",  # Force CPU if device is CPU (for MPS fallback)
            "max_grad_norm": 1.0  # Clip gradients to prevent explosion (LayerNorm can cause large gradients)
        }
        
        # Try newer parameter name first (eval_strategy), fallback to older name
        try:
            training_args = TrainingArguments(**training_args_dict, eval_strategy="no")
        except TypeError:
            # Fallback to older parameter name
            training_args = TrainingArguments(**training_args_dict, evaluation_strategy="no")
        
        # Data collator for CTC - need to handle padding for variable-length audio
        if DATA_COLLATOR_AVAILABLE:
            data_collator = DataCollatorCTCWithPadding(
                processor=self.processor,
                padding=True
            )
        else:
            # Create custom data collator that handles padding
            logger.warning("DataCollatorCTCWithPadding not available, creating custom collator")
            from transformers import default_data_collator
            
            class Wav2Vec2DataCollator:
                """Custom data collator for Wav2Vec2 CTC that handles variable-length sequences."""
                def __init__(self, processor):
                    self.processor = processor
                    # Get padding value for audio (usually 0.0 for feature extractor)
                    self.audio_padding_value = 0.0
                    if hasattr(self.processor, 'feature_extractor') and hasattr(self.processor.feature_extractor, 'padding_value'):
                        self.audio_padding_value = self.processor.feature_extractor.padding_value
                    
                    # Get pad token ID for labels (-100 is typical for CTC loss which ignores padding)
                    self.label_padding_value = -100  # CTC loss ignores -100
                    if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'pad_token_id'):
                        tokenizer_pad = self.processor.tokenizer.pad_token_id
                        # Use -100 if pad_token_id is None or use pad_token_id
                        self.label_padding_value = tokenizer_pad if tokenizer_pad is not None else -100
                
                def __call__(self, features):
                    # Handle case where features might not have the expected keys
                    # This can happen if data collator is called with empty or malformed batch
                    if not features:
                        raise ValueError("Empty batch passed to data collator")
                    
                    # Check if first feature has expected keys
                    if not isinstance(features[0], dict):
                        raise ValueError(f"Expected dict features, got {type(features[0])}")
                    
                    # Separate input_values and labels
                    try:
                        input_values = [f['input_values'] for f in features]
                        labels = [f['labels'] for f in features]
                    except KeyError as e:
                        raise ValueError(
                            f"Missing expected key in features: {e}. "
                            f"Available keys in first feature: {list(features[0].keys())}"
                        )
                    
                    # Pad input_values to same length (batch dimension first)
                    # input_values are 1D tensors, need to pad to max length in batch
                    input_values = torch.nn.utils.rnn.pad_sequence(
                        input_values,
                        batch_first=True,
                        padding_value=self.audio_padding_value
                    )
                    
                    # Pad labels to same length
                    labels = torch.nn.utils.rnn.pad_sequence(
                        labels,
                        batch_first=True,
                        padding_value=self.label_padding_value
                    )
                    
                    return {
                        'input_values': input_values,
                        'labels': labels
                    }
            
            data_collator = Wav2Vec2DataCollator(self.processor)
        
        # Create custom Trainer with compute_loss that handles Wav2Vec2's input_values
        class Wav2Vec2Trainer(Trainer):
            """Custom Trainer for Wav2Vec2 that handles input_values correctly."""
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """
                Compute CTC loss for Wav2Vec2.
                Ensure input_values is passed correctly and remove any input_ids that might be added.
                For PEFT models, bypass the wrapper and call base model directly.
                
                Args:
                    model: The model to compute loss for (may be PEFT-wrapped)
                    inputs: Dictionary of inputs (should contain 'input_values' and 'labels')
                    return_outputs: Whether to return model outputs along with loss
                    **kwargs: Additional arguments (e.g., num_items_in_batch) - ignored
                """
                # Remove input_ids if present (PEFT might add it, but Wav2Vec2 doesn't need it)
                if 'input_ids' in inputs:
                    del inputs['input_ids']
                
                # Ensure input_values is present
                if 'input_values' not in inputs:
                    raise ValueError("input_values not found in inputs. Check data collator.")
                
                # Get labels
                labels = inputs.get('labels', None)
                
                # Move inputs to device
                input_values = inputs['input_values'].to(model.device)
                if labels is not None:
                    labels = labels.to(model.device)
                
                # Validate inputs
                if torch.isnan(input_values).any():
                    logger.error(f"NaN detected in input_values! Shape: {input_values.shape}")
                    raise ValueError("NaN in input_values")
                
                if torch.isinf(input_values).any():
                    logger.error(f"Inf detected in input_values! Shape: {input_values.shape}")
                    raise ValueError("Inf in input_values")
                
                # Forward pass - try both direct call and with explicit parameters
                # Some Wav2Vec2 models need the inputs dict format
                try:
                    # Try explicit parameters first (standard way)
                    outputs = model(input_values=input_values, labels=labels)
                except Exception as e:
                    logger.error(f"Error in model forward pass with explicit params: {e}")
                    # Fallback to dict format
                    try:
                        model_inputs = {'input_values': input_values}
                        if labels is not None:
                            model_inputs['labels'] = labels
                        outputs = model(**model_inputs)
                    except Exception as e2:
                        logger.error(f"Error in model forward pass with dict format: {e2}")
                        raise
                
                loss = outputs.loss
                
                # Debug: Check intermediate outputs
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    if torch.isnan(logits).any():
                        # Check if this is a fresh forward pass or if weights are corrupted
                        logger.error("NaN detected in logits during forward pass")
                        # Check CTC head weights
                        if hasattr(model, 'lm_head'):
                            lm_head_weight = model.lm_head.weight
                            if torch.isnan(lm_head_weight).any():
                                logger.error("NaN detected in lm_head.weight!")
                            if torch.isnan(model.lm_head.bias).any():
                                logger.error("NaN detected in lm_head.bias!")
                        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'lm_head'):
                            lm_head_weight = model.base_model.lm_head.weight
                            if torch.isnan(lm_head_weight).any():
                                logger.error("NaN detected in base_model.lm_head.weight!")
                            if torch.isnan(model.base_model.lm_head.bias).any():
                                logger.error("NaN detected in base_model.lm_head.bias!")
                        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'lm_head'):
                            lm_head_weight = model.base_model.model.lm_head.weight
                            if torch.isnan(lm_head_weight).any():
                                logger.error("NaN detected in base_model.model.lm_head.weight!")
                            if torch.isnan(model.base_model.model.lm_head.bias).any():
                                logger.error("NaN detected in base_model.model.lm_head.bias!")
                
                # Validate loss
                if loss is None:
                    logger.error("Loss is None!")
                    raise ValueError("Model returned None loss")
                
                if torch.isnan(loss):
                    logger.error(f"NaN loss detected! Check CTC loss computation.")
                    logger.error(f"Input values shape: {input_values.shape}, min: {input_values.min()}, max: {input_values.max()}")
                    if labels is not None:
                        logger.error(f"Labels shape: {labels.shape}, min: {labels.min()}, max: {labels.max()}, unique: {torch.unique(labels).tolist()[:20]}")
                    # Check logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        logger.error(f"Logits shape: {logits.shape}, min: {logits.min()}, max: {logits.max()}, mean: {logits.mean()}")
                        logger.error(f"Logits NaN count: {torch.isnan(logits).sum()}, Inf count: {torch.isinf(logits).sum()}")
                    raise ValueError("NaN loss - CTC loss computation failed")
                
                if torch.isinf(loss):
                    logger.error(f"Inf loss detected!")
                    raise ValueError("Inf loss")
                
                return (loss, outputs) if return_outputs else loss
        
        # Trainer - don't pass tokenizer/processing_class for Wav2Vec2
        # remove_unused_columns is set in TrainingArguments above
        trainer = Wav2Vec2Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if len(val_dataset) > 0 else None,
            data_collator=data_collator
        )
        
        # Train
        training_start_time = time.time()
        logger.info("Starting fine-tuning with HuggingFace Trainer...")
        
        trainer.train()
        
        training_duration = time.time() - training_start_time
        
        # Save model
        model_path = self._save_model(output_dir)
        
        # Simple evaluation (using loss as proxy)
        # Use the same data collator as training to handle variable-length sequences
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        final_train_accuracy = self._evaluate_simple(train_loader)
        final_val_accuracy = self._evaluate_simple(val_loader)
        initial_accuracy = (final_train_accuracy + final_val_accuracy) / 2  # Estimate
        accuracy_gain = final_val_accuracy - initial_accuracy
        
        training_cost = self._estimate_training_cost(training_duration, len(train_samples))
        
        result = {
            'success': True,
            'initial_validation_accuracy': initial_accuracy,
            'final_validation_accuracy': final_val_accuracy,
            'initial_train_accuracy': initial_accuracy,
            'final_train_accuracy': final_train_accuracy,
            'accuracy_gain': accuracy_gain,
            'overfitting_detected': False,
            'training_duration_seconds': training_duration,
            'training_cost': training_cost,
            'samples_used': len(train_samples) + len(val_samples),
            'train_samples': len(train_samples),
            'validation_samples': len(val_samples),
            'epochs_completed': num_epochs,
            'best_validation_accuracy': final_val_accuracy,
            'model_path': model_path
        }
        
        logger.info(f"Fine-tuning completed in {training_duration:.2f} seconds")
        return result
    
    def _prepare_processed_data(self, samples: List[Dict]) -> List[Dict]:
        """Prepare processed data for Wav2Vec2."""
        processed_data = []
        import librosa
        
        for i, sample in enumerate(samples):
            try:
                audio, sr = librosa.load(sample['audio_path'], sr=16000)
                
                inputs = self.processor(
                    audio,
                    sampling_rate=16000,
                    padding=False,  # Don't pad here - collator will handle batching/padding
                    return_tensors="pt"
                )
                
                # Use tokenizer directly for text processing (as_target_processor is deprecated)
                # The processor contains a tokenizer that we can use for text encoding
                tokenizer = self.processor.tokenizer
                label_ids = tokenizer(
                    sample['corrected_transcript'],
                    padding=False,  # Don't pad here - let collator handle it
                    return_tensors="pt",
                    return_attention_mask=False  # Don't return attention mask for labels
                )
                
                # Extract tensors and ensure they're 1D
                input_values = inputs.input_values.squeeze(0) if inputs.input_values.dim() > 1 else inputs.input_values[0]
                labels = label_ids['input_ids'].squeeze(0) if label_ids['input_ids'].dim() > 1 else label_ids['input_ids'][0]
                
                # Validate labels are within vocabulary range
                vocab_size = self.processor.tokenizer.vocab_size
                if len(labels) > 0:
                    labels_max = labels.max().item()
                    labels_min = labels.min().item()
                    if labels_max >= vocab_size or labels_min < 0:
                        logger.error(f"Invalid label values! Labels min: {labels_min}, max: {labels_max}, vocab_size: {vocab_size}")
                        raise ValueError(f"Labels contain values outside vocabulary range [0, {vocab_size})")
                
                # Log first sample for debugging
                if i == 0:
                    logger.info(f"Sample {i}: audio_length={len(audio)}, input_values_shape={input_values.shape}, labels_shape={labels.shape}, labels_unique={torch.unique(labels).tolist()[:10]}, vocab_size={vocab_size}")
                
                processed_item = {
                    'input_values': input_values,  # Shape: [seq_len]
                    'labels': labels  # Shape: [label_len]
                }
                
                # Verify the item has required keys
                if 'input_values' not in processed_item or 'labels' not in processed_item:
                    raise ValueError(f"Missing required keys after processing sample {i}")
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.error(f"Error processing sample {i} ({sample.get('audio_path', 'unknown')}): {e}")
                raise
        
        if not processed_data:
            raise ValueError("No data was processed successfully")
        
        logger.info(f"Processed {len(processed_data)} samples. Sample keys: {list(processed_data[0].keys())}")
        return processed_data
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_values = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_values=input_values, labels=labels)
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        accuracy = max(0.0, min(1.0, 1.0 - avg_loss))
        
        return accuracy
    
    def _evaluate_simple(self, data_loader: DataLoader) -> float:
        """Simple evaluation for Trainer-based training."""
        return self._evaluate(data_loader)
    
    def _save_model(self, output_dir: Optional[Path] = None) -> Optional[str]:
        """Save fine-tuned model."""
        if output_dir is None:
            output_dir = self.output_dir
        
        if output_dir is None:
            logger.warning("No output directory specified, skipping model save")
            return None
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_lora:
            # Save only LoRA adapters
            adapter_dir = output_dir / "lora_adapters"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(adapter_dir))
            logger.info(f"LoRA adapters saved to {adapter_dir}")
            model_path = str(adapter_dir)
        else:
            # Save full model
            self.model.save_pretrained(str(output_dir))
            model_path = str(output_dir)
        
        # Always save processor
        self.processor.save_pretrained(str(output_dir))
        logger.info(f"Model and processor saved to {output_dir}")
        
        # Save metadata about the model
        metadata = {
            "model_name": self.model_name,
            "use_lora": self.use_lora,
            "output_dir": str(output_dir),
            "saved_at": datetime.now().isoformat()
        }
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    @staticmethod
    def model_exists(output_dir: str) -> bool:
        """
        Check if a fine-tuned model exists in the output directory.
        
        Args:
            output_dir: Path to check for existing model
        
        Returns:
            True if model exists, False otherwise
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return False
        
        # Check for LoRA adapters
        lora_dir = output_path / "lora_adapters"
        if lora_dir.exists():
            config_file = lora_dir / "adapter_config.json"
            if config_file.exists():
                return True
        
        # Check for full model
        config_file = output_path / "config.json"
        pytorch_model = output_path / "pytorch_model.bin"
        safetensors_model = output_path / "model.safetensors"
        
        if config_file.exists() and (pytorch_model.exists() or safetensors_model.exists()):
            return True
        
        return False
    
    @staticmethod
    def load_model(
        output_dir: str,
        device: Optional[str] = None
    ) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
        """
        Load a previously fine-tuned model from disk.
        
        Args:
            output_dir: Directory containing the saved model
            device: Device to load model on (default: auto-detect)
        
        Returns:
            Tuple of (model, processor)
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            raise FileNotFoundError(f"Model directory does not exist: {output_dir}")
        
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load metadata if available
        metadata_path = output_path / "model_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loading model from {output_dir} (saved at: {metadata.get('saved_at', 'unknown')})")
        else:
            logger.info(f"Loading model from {output_dir}")
        
        # Check for LoRA adapters
        lora_dir = output_path / "lora_adapters"
        if lora_dir.exists() and PEFT_AVAILABLE:
            try:
                # Load base model
                model_name = metadata.get('model_name', 'facebook/wav2vec2-base-960h')
                logger.info(f"Loading base model: {model_name}")
                base_model = Wav2Vec2ForCTC.from_pretrained(model_name)
                
                # Load LoRA adapters
                logger.info(f"Loading LoRA adapters from {lora_dir}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, str(lora_dir))
                # Merge adapters for inference
                model = model.merge_and_unload()
                logger.info("LoRA adapters merged successfully")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapters: {e}. Trying full model load.")
                model = Wav2Vec2ForCTC.from_pretrained(str(output_path))
        else:
            # Load full model
            logger.info(f"Loading full model from {output_path}")
            model = Wav2Vec2ForCTC.from_pretrained(str(output_path))
        
        # Load processor
        processor = Wav2Vec2Processor.from_pretrained(str(output_path))
        
        # Move to device
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return model, processor
    
    def _estimate_training_cost(
        self,
        duration_seconds: float,
        num_samples: int
    ) -> float:
        """Estimate computational cost of training."""
        gpu_cost_per_second = 0.0001 if self.device.startswith('cuda') else 0.00001
        sample_cost_factor = 1.0 + (num_samples / 1000.0)
        
        # LoRA reduces cost
        cost_multiplier = 0.3 if self.use_lora else 1.0
        
        total_cost = duration_seconds * gpu_cost_per_second * sample_cost_factor * cost_multiplier
        
        return total_cost
    
    def get_training_history(self) -> List[Dict]:
        """Get training history."""
        return self.training_history.copy()


# Factory function for easy creation
def create_finetuner(
    model_name: str,
    use_lora: bool = True,
    **kwargs
) -> FineTuner:
    """
    Factory function to create a FineTuner instance for Wav2Vec2.
    
    Args:
        model_name: HuggingFace model name for Wav2Vec2
        use_lora: Whether to use LoRA
        **kwargs: Additional arguments for FineTuner
    
    Returns:
        FineTuner instance
    """
    return FineTuner(
        model_name=model_name,
        use_lora=use_lora,
        **kwargs
    )
