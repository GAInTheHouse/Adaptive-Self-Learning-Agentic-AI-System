"""
Fine-Tuning Module with Validation Monitoring - Week 3
Automated fine-tuning with overfitting prevention and validation monitoring.
"""

import logging
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import numpy as np
from datetime import datetime
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorDataset(Dataset):
    """Dataset for fine-tuning on error samples"""
    
    def __init__(self, error_samples: List[Dict], processor):
        """
        Initialize error dataset.
        
        Args:
            error_samples: List of error samples with 'audio_path' and 'corrected_transcript'
            processor: Whisper processor for audio preprocessing
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
        
        # Process audio
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        
        # Process text
        labels = self.processor.tokenizer(
            sample['corrected_transcript'],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        return {
            'input_features': inputs['input_features'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0)
        }


class FineTuner:
    """
    Fine-tuning module with validation monitoring and overfitting prevention.
    """
    
    def __init__(
        self,
        model,
        processor,
        device: Optional[str] = None,
        validation_split: float = 0.2,
        overfitting_threshold: float = 0.1,
        early_stopping_patience: int = 3,
        min_accuracy_gain: float = 0.01
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model: Whisper model to fine-tune
            processor: Whisper processor
            device: Device to use (cuda/cpu)
            validation_split: Fraction of data for validation
            overfitting_threshold: Max allowed difference between train/val accuracy
            early_stopping_patience: Number of epochs to wait before early stopping
            min_accuracy_gain: Minimum accuracy gain to consider fine-tuning successful
        """
        self.model = model
        self.processor = processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.validation_split = validation_split
        self.overfitting_threshold = overfitting_threshold
        self.early_stopping_patience = early_stopping_patience
        self.min_accuracy_gain = min_accuracy_gain
        
        # Training history
        self.training_history = []
        
        logger.info(f"Fine-tuner initialized on device: {self.device}")
    
    def fine_tune(
        self,
        error_samples: List[Dict],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        max_grad_norm: float = 1.0
    ) -> Dict:
        """
        Fine-tune model on error samples with validation monitoring.
        
        Args:
            error_samples: List of error samples with 'audio_path' and 'corrected_transcript'
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            max_grad_norm: Maximum gradient norm for clipping
        
        Returns:
            Dictionary with fine-tuning results
        """
        if len(error_samples) < 10:
            logger.warning(f"Insufficient samples ({len(error_samples)}), skipping fine-tuning")
            return {
                'success': False,
                'reason': 'insufficient_samples',
                'samples_provided': len(error_samples)
            }
        
        # Split into train and validation
        np.random.seed(42)
        indices = np.random.permutation(len(error_samples))
        split_idx = int(len(error_samples) * (1 - self.validation_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_samples = [error_samples[i] for i in train_indices]
        val_samples = [error_samples[i] for i in val_indices]
        
        logger.info(f"Fine-tuning on {len(train_samples)} train samples, {len(val_samples)} validation samples")
        
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
                
                input_features = batch['input_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_features=input_features, labels=labels)
                loss = outputs.loss
                
                # Backward pass
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
        
        # Estimate training cost (simplified: based on training time and GPU usage)
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
            'samples_used': len(error_samples),
            'train_samples': len(train_samples),
            'validation_samples': len(val_samples),
            'epochs_completed': len(self.training_history),
            'best_validation_accuracy': best_val_accuracy
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
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate model on a data loader.
        
        Args:
            data_loader: DataLoader for evaluation
        
        Returns:
            Accuracy score (simplified: based on loss)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_features = batch['input_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_features=input_features, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Convert loss to accuracy estimate (simplified: lower loss = higher accuracy)
        # This is a simplified metric; in practice, you'd compute actual WER/CER
        accuracy = max(0.0, min(1.0, 1.0 - avg_loss))
        
        return accuracy
    
    def _estimate_training_cost(
        self,
        duration_seconds: float,
        num_samples: int
    ) -> float:
        """
        Estimate computational cost of training.
        
        Args:
            duration_seconds: Training duration in seconds
            num_samples: Number of training samples
        
        Returns:
            Estimated cost (normalized units)
        """
        # Simplified cost model:
        # Base cost per second of GPU time
        gpu_cost_per_second = 0.0001 if self.device.startswith('cuda') else 0.00001
        
        # Cost scales with number of samples
        sample_cost_factor = 1.0 + (num_samples / 1000.0)
        
        total_cost = duration_seconds * gpu_cost_per_second * sample_cost_factor
        
        return total_cost
    
    def get_training_history(self) -> List[Dict]:
        """Get training history."""
        return self.training_history.copy()
