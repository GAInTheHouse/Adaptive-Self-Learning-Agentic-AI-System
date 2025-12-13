#!/usr/bin/env python3
"""
Fine-tuning script for Wav2Vec2 STT model.
Evaluates on 200 audio files (100 clean, 100 noisy), uses LLM corrections as gold standard,
and fine-tunes only on incorrect predictions.
"""

import sys
import os
from pathlib import Path
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# Initialize logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)
try:
    from datasets import Dataset
except ImportError:
    logger.warning("datasets library not available, using fallback")
    Dataset = None
import librosa
import numpy as np
from jiwer import wer, cer

from src.baseline_model import BaselineSTTModel
from src.agent.llm_corrector import LlamaLLMCorrector
from src.evaluation.metrics import STTEvaluator
from src.agent.fine_tuner import FineTuner, create_finetuner
from src.utils.model_versioning import get_next_model_version, get_model_version_name


class Wav2Vec2FineTuner:
    """
    Wrapper around unified FineTuner for Wav2Vec2 models.
    This class maintains backward compatibility with the script interface.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        output_dir: str = None,  # Will be auto-generated with versioned name if None
        device: str = None,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16
    ):
        """Initialize using the unified FineTuner."""
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        
        # Use the unified FineTuner from src/agent/fine_tuner.py
        self.fine_tuner = create_finetuner(
            model_name=model_name,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            output_dir=output_dir,
            device=device
        )
        
        # Store references for compatibility
        self.model = self.fine_tuner.model
        self.processor = self.fine_tuner.processor
        self.device = self.fine_tuner.device
    
    def fine_tune(
        self,
        train_audio_files: List[str],
        train_transcripts: List[str],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 3e-5,
        warmup_steps: int = 500
    ) -> Dict:
        """Fine-tune model on training data using unified FineTuner."""
        # Convert to error_samples format expected by FineTuner
        error_samples = [
            {
                'audio_path': audio_path,
                'corrected_transcript': transcript
            }
            for audio_path, transcript in zip(train_audio_files, train_transcripts)
        ]
        
        # Use unified fine_tune method
        result = self.fine_tuner.fine_tune(
            error_samples=error_samples,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Adjust return format for backward compatibility
        if result.get('model_path'):
            # Update model_path to use output_dir
            result['model_path'] = str(self.output_dir)
        
        return result
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using the fine-tuned model."""
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.decode(predicted_ids[0])
        
        return transcript


def collect_error_cases(
    audio_files: List[str],
    stt_model: BaselineSTTModel,
    llm_corrector: LlamaLLMCorrector,
    evaluator: STTEvaluator
) -> Tuple[List[Dict], Dict]:
    """Collect error cases and calculate metrics"""
    error_cases = []
    all_stt_transcripts = []
    all_llm_transcripts = []
    
    logger.info(f"Processing {len(audio_files)} audio files...")
    
    for i, audio_path in enumerate(audio_files):
        try:
            # Get STT transcript
            stt_result = stt_model.transcribe(audio_path)
            stt_transcript = stt_result.get("transcript", "").strip()
            
            # Debug: Log if transcript is empty and check the raw result
            if not stt_transcript:
                logger.warning(f"Empty STT transcript for {audio_path}")
                logger.warning(f"STT result keys: {list(stt_result.keys())}")
                logger.warning(f"STT result: {stt_result}")
            
            all_stt_transcripts.append(stt_transcript)
            
            # Get LLM gold standard
            if not stt_transcript:
                # Skip LLM correction if transcript is empty (LLM can't improve empty text)
                logger.warning(f"Skipping LLM correction for empty transcript: {audio_path}")
                llm_transcript = ""
            elif llm_corrector and llm_corrector.is_available():
                llm_result = llm_corrector.correct_transcript(
                    stt_transcript,
                    errors=[],
                    context={}  # General conversational transcripts
                )
                llm_transcript = llm_result.get("corrected_transcript", stt_transcript).strip()
            else:
                logger.warning("LLM not available, using STT transcript as gold standard")
                llm_transcript = stt_transcript
            
            # Clean up LLM transcript: remove quotes and normalize case
            llm_transcript = re.sub(r'^["\'](.*)["\']$', r'\1', llm_transcript.strip())
            llm_transcript = llm_transcript.strip()
            
            all_llm_transcripts.append(llm_transcript)
            
            # Normalize case for WER/CER calculation (use lowercase for comparison)
            stt_normalized = stt_transcript.lower().strip()
            llm_normalized = llm_transcript.lower().strip()
            
            # Log progress every 10 datapoints with STT vs LLM comparison
            if (i + 1) % 10 == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"Progress: {i + 1}/{len(audio_files)} files processed")
                logger.info(f"STT:  {stt_transcript}")
                logger.info(f"LLM:  {llm_transcript}")
                logger.info(f"{'='*60}\n")
            
            # Calculate WER/CER using normalized (lowercase) transcripts for accurate comparison
            sample_wer = wer(llm_normalized, stt_normalized)
            sample_cer = cer(llm_normalized, stt_normalized)
            
            # If error exists, add to error cases
            if sample_wer > 0.0 or sample_cer > 0.0:
                error_cases.append({
                    'audio_path': audio_path,
                    'stt_transcript': stt_transcript,
                    'gold_transcript': llm_transcript,
                    'wer': sample_wer,
                    'cer': sample_cer
                })
        
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            continue
    
    # Calculate overall metrics using normalized (lowercase) transcripts for accurate comparison
    all_stt_normalized = [t.lower().strip() for t in all_stt_transcripts]
    all_llm_normalized = [t.lower().strip() for t in all_llm_transcripts]
    overall_wer = wer(all_llm_normalized, all_stt_normalized)
    overall_cer = cer(all_llm_normalized, all_stt_normalized)
    
    metrics = {
        'wer': overall_wer,
        'cer': overall_cer,
        'total_samples': len(audio_files),
        'error_samples': len(error_cases),
        'error_rate': len(error_cases) / len(audio_files) if audio_files else 0.0
    }
    
    return error_cases, metrics


def main():
    """Main fine-tuning pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 model")
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files (should have 'clean' and 'noisy' subdirectories)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for fine-tuned model (auto-generated with versioned name if not specified)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning (default: True)"
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA and use full fine-tuning"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force re-training even if a fine-tuned model already exists"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8). Higher rank = more parameters but potentially better accuracy"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor (default: 16)"
    )
    
    args = parser.parse_args()
    
    # Handle --no_lora flag
    use_lora = args.use_lora and not args.no_lora
    
    audio_dir = Path(args.audio_dir)
    
    # Collect audio files
    clean_dir = audio_dir / "clean"
    noisy_dir = audio_dir / "noisy"
    
    clean_files = sorted(list(clean_dir.glob("*.wav")) + list(clean_dir.glob("*.mp3"))) if clean_dir.exists() else []
    noisy_files = sorted(list(noisy_dir.glob("*.wav")) + list(noisy_dir.glob("*.mp3"))) if noisy_dir.exists() else []
    
    # If no subdirectories, assume all files in root
    if not clean_files and not noisy_files:
        all_files = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")))
        clean_files = all_files[:len(all_files)//2]
        noisy_files = all_files[len(all_files)//2:]
    
    all_files = [str(f) for f in clean_files[:100]] + [str(f) for f in noisy_files[:100]]
    
    logger.info(f"Found {len(clean_files)} clean files, {len(noisy_files)} noisy files")
    logger.info(f"Using {len(all_files)} files for evaluation")
    
    if len(all_files) == 0:
        logger.error("No audio files found!")
        return
    
    # Auto-generate output directory with versioned name if not specified
    if args.output_dir is None:
        next_version = get_next_model_version()
        version_name = get_model_version_name(next_version)
        args.output_dir = f"models/{version_name}"
        logger.info(f"📦 Auto-generated output directory: {args.output_dir} (version {next_version})")
    
    # Initialize models
    logger.info("Initializing STT model...")
    stt_model = BaselineSTTModel(model_name="wav2vec2-base")
    
    logger.info("Initializing LLM corrector (Ollama with Llama)...")
    try:
        llm_corrector = LlamaLLMCorrector(
            model_name="llama3.2:3b",  # Use Ollama Llama 3.2 3B
            use_quantization=False,  # Not used for Ollama
            fast_mode=True
        )
        if not llm_corrector.is_available():
            logger.warning("LLM not available! Fine-tuning will proceed with STT transcripts as gold standard")
    except Exception as e:
        logger.error(f"Failed to initialize LLM corrector: {e}")
        logger.error("Make sure Ollama is installed and running:")
        logger.error("  1. Install Ollama: https://ollama.ai/download")
        logger.error("  2. Pull the model: ollama pull llama3.2:3b")
        logger.error("  3. Ensure Ollama server is running: ollama serve")
        logger.warning("Fine-tuning will proceed with STT transcripts as gold standard")
        llm_corrector = None
    
    evaluator = STTEvaluator()
    
    # Step 1: Evaluate baseline
    logger.info("=" * 60)
    logger.info("STEP 1: Evaluating Baseline Model")
    logger.info("=" * 60)
    
    error_cases, baseline_metrics = collect_error_cases(all_files, stt_model, llm_corrector, evaluator)
    
    logger.info(f"\nBaseline Metrics:")
    logger.info(f"  WER: {baseline_metrics['wer']:.4f} ({baseline_metrics['wer']*100:.2f}%)")
    logger.info(f"  CER: {baseline_metrics['cer']:.4f} ({baseline_metrics['cer']*100:.2f}%)")
    logger.info(f"  Error Samples: {baseline_metrics['error_samples']}/{baseline_metrics['total_samples']}")
    logger.info(f"  Error Rate: {baseline_metrics['error_rate']:.4f} ({baseline_metrics['error_rate']*100:.2f}%)")
    
    if len(error_cases) == 0:
        logger.info("No error cases found! Model is perfect. Exiting.")
        return
    
    # Step 2: Fine-tune on error cases
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Fine-tuning on Error Cases")
    logger.info("=" * 60)
    
    # Check if model already exists
    from src.agent.fine_tuner import FineTuner
    model_exists = FineTuner.model_exists(args.output_dir)
    
    if model_exists and not args.force_retrain:
        logger.info(f"✅ Fine-tuned model already exists at {args.output_dir}")
        logger.info("Skipping fine-tuning. Use --force-retrain to retrain anyway.")
        logger.info("Loading existing model for evaluation...")
        model, processor = FineTuner.load_model(args.output_dir)
        fine_tune_result = {
            'success': True,
            'model_path': args.output_dir,
            'num_samples': len(error_cases),
            'skipped': True,
            'reason': 'model_already_exists'
        }
        fine_tuned_wav2vec2 = model
        fine_tuned_processor = processor
    else:
        if model_exists:
            logger.info("--force-retrain specified, proceeding with fine-tuning...")
        
        train_audio_files = [case['audio_path'] for case in error_cases]
        train_transcripts = [case['gold_transcript'] for case in error_cases]
        
        # Estimate training time (rough estimate)
        if use_lora:
            # LoRA is 3-5x faster
            time_per_sample = 30 / 4  # ~7.5 seconds per sample per epoch with LoRA
        else:
            time_per_sample = 30  # 30 seconds per sample per epoch for full fine-tuning
        estimated_time = len(error_cases) * args.num_epochs * time_per_sample / 60  # in minutes
        logger.info(f"Estimated training time: ~{estimated_time:.1f} minutes ({'LoRA' if use_lora else 'Full'} fine-tuning)")
        
        fine_tuner = Wav2Vec2FineTuner(
            output_dir=args.output_dir,
            use_lora=use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha
        )
        
        start_time = time.time()
        fine_tune_result = fine_tuner.fine_tune(
            train_audio_files=train_audio_files,
            train_transcripts=train_transcripts,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        # Use training time from result, fallback to calculated time
        actual_time = fine_tune_result.get('training_duration_seconds', time.time() - start_time)
        
        # Check if fine-tuning was successful
        if not fine_tune_result.get('success', False):
            logger.warning(f"\nFine-tuning was skipped: {fine_tune_result.get('reason', 'unknown')}")
            logger.warning(f"  Samples provided: {fine_tune_result.get('samples_provided', len(error_cases))}")
            logger.warning("Cannot evaluate fine-tuned model as no training occurred.")
            logger.warning("Please provide at least 10 error cases for fine-tuning.")
            return
        
        logger.info(f"\nFine-tuning completed!")
        logger.info(f"  Actual training time: {actual_time:.2f} seconds ({actual_time/60:.1f} minutes)")
        logger.info(f"  Samples used: {fine_tune_result.get('num_samples', len(error_cases))}")
        
        # Check if model was saved before trying to load
        model_path = fine_tune_result.get('model_path', args.output_dir)
        if FineTuner.model_exists(model_path):
            logger.info(f"  Model saved to: {model_path}")
            # Load the newly trained model for evaluation
            model, processor = FineTuner.load_model(model_path)
            fine_tuned_wav2vec2 = model
            fine_tuned_processor = processor
        else:
            logger.error(f"Model was not saved to {model_path}. Cannot proceed with evaluation.")
            return
    
    # Step 3: Evaluate fine-tuned model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Evaluating Fine-tuned Model")
    logger.info("=" * 60)
    
    # Load test files from separate test directory
    test_dir = Path("data/recordings_for_test")
    if test_dir.exists():
        test_files = sorted(list(test_dir.glob("*.wav")) + list(test_dir.glob("*.mp3")))
        test_files = [str(f) for f in test_files]
        logger.info(f"Found {len(test_files)} test files in {test_dir}")
    else:
        logger.warning(f"Test directory {test_dir} not found. Using training files for evaluation.")
        test_files = all_files
    
    if len(test_files) == 0:
        logger.warning("No test files found. Using training files for evaluation.")
        test_files = all_files
    
    # Model should already be loaded above, but verify and load if needed
    if 'fine_tuned_wav2vec2' not in locals() or 'fine_tuned_processor' not in locals():
        # Try to load existing model if it exists
        if FineTuner.model_exists(args.output_dir):
            logger.info("Loading fine-tuned model...")
            fine_tuned_wav2vec2, fine_tuned_processor = FineTuner.load_model(args.output_dir)
            
            # Test the model on a single file to verify it works
            test_audio = test_files[0] if test_files else None
            if test_audio:
                logger.info(f"Testing fine-tuned model on {test_audio}...")
                import librosa
                import torch
                audio, sr = librosa.load(test_audio, sr=16000)
                test_inputs = fine_tuned_processor(audio, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    test_logits = fine_tuned_wav2vec2(test_inputs.input_values).logits
                    test_predicted = torch.argmax(test_logits, dim=-1)
                    test_transcript = fine_tuned_processor.batch_decode(test_predicted)[0]
                logger.info(f"Test transcript: '{test_transcript}'")
                logger.info(f"Test logits shape: {test_logits.shape}, vocab size: {test_logits.shape[-1]}")
                logger.info(f"Test predicted IDs unique values: {torch.unique(test_predicted).tolist()[:10]}")
        else:
            logger.error(f"No fine-tuned model found at {args.output_dir}")
            logger.error("Cannot proceed with evaluation. Please ensure fine-tuning completed successfully.")
            return
    
    # Create a wrapper class for fine-tuned model
    class FineTunedSTTModel:
        def __init__(self, model, processor):
            import torch  # Import torch at the method level
            
            # Model should already be merged (not PEFT-wrapped) after FineTuner.load_model
            self.model = model
            
            # Use the processor from the fine-tuned model
            # If it doesn't work, we'll fall back to base processor
            self.processor = processor
            
            # Also load base processor as fallback
            from transformers import Wav2Vec2Processor
            try:
                self.base_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            except:
                self.base_processor = None
            
            self.model_name = "wav2vec2-finetuned"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
        
        def transcribe(self, audio_path: str):
            import librosa
            import torch
            
            # Load audio and process (same as baseline model)
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            
            # Forward pass (same as baseline model)
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits
                
                # Debug: Check logits
                if logits.shape[-1] == 1:  # Only one class (padding)
                    logger.error(f"Model logits have only 1 class! Shape: {logits.shape}")
                    logger.error(f"Logits sample: {logits[0, :5, :].tolist()}")
                
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Debug: Check if all predictions are 0
                unique_ids = torch.unique(predicted_ids)
                if len(unique_ids) == 1 and unique_ids[0] == 0:
                    logger.warning(f"Model predicting only padding token (0). Logits shape: {logits.shape}")
                    logger.warning(f"Logits stats - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")
                    # Try using the baseline model's processor instead
                    logger.warning("Attempting to use baseline processor for decoding...")
                    # Fallback: try to decode with more lenient settings
                    # For now, just return empty and log the issue
            
            # Decode using batch_decode (same as baseline model)
            transcript = self.processor.batch_decode(predicted_ids)[0]
            
            # If transcript is empty and all IDs are 0, the model isn't working
            # This suggests the fine-tuning didn't work properly or the model is broken
            if not transcript.strip() and torch.all(predicted_ids == 0):
                logger.error(f"CRITICAL: Fine-tuned model is predicting only padding tokens (0s)")
                logger.error(f"This indicates the model did not learn properly during fine-tuning.")
                logger.error(f"Training loss was 0.0, which suggests the CTC loss computation may have failed.")
                logger.error(f"Logits shape: {logits.shape}, vocab size: {logits.shape[-1]}")
                logger.error(f"Logits stats - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")
                logger.error(f"All logits are likely the same, causing argmax to always return 0")
                
                # Check if logits are all the same (which would cause all 0 predictions)
                logits_std = logits.std().item()
                if logits_std < 0.001:
                    logger.error(f"Logits have very low std ({logits_std:.6f}), indicating model output is constant")
                    logger.error(f"This confirms the model is not working. The CTC head may not be properly initialized.")
                
                # Try using base processor as last resort
                if self.base_processor:
                    logger.warning("Attempting fallback to base processor...")
                    transcript = self.base_processor.batch_decode(predicted_ids)[0]
                    if not transcript.strip():
                        logger.error("Base processor also failed. Model is completely broken.")
            
            # Debug: Log if transcript is still empty
            if not transcript.strip():
                logger.warning(f"Empty transcript from fine-tuned model for {audio_path}")
                logger.warning(f"Predicted IDs shape: {predicted_ids.shape}")
                logger.warning(f"Predicted IDs sample (first 20): {predicted_ids[0][:20].tolist()}")
                logger.warning(f"Unique predicted IDs: {torch.unique(predicted_ids).tolist()}")
            
            return {
                "transcript": transcript,
                "model": self.model_name,
                "version": "finetuned-v1"
            }
    
    fine_tuned_model = FineTunedSTTModel(fine_tuned_wav2vec2, fine_tuned_processor)
    
    # Test if fine-tuned model works - if not, skip evaluation
    logger.info("Testing fine-tuned model on a sample file...")
    test_sample = test_files[0] if test_files else None
    fine_tuned_works = False
    if test_sample:
        try:
            test_result = fine_tuned_model.transcribe(test_sample)
            test_transcript = test_result.get("transcript", "").strip()
            if test_transcript:
                logger.info(f"✅ Fine-tuned model works! Sample transcript: '{test_transcript[:50]}...'")
                fine_tuned_works = True
            else:
                logger.error(f"❌ Fine-tuned model produces empty transcripts.")
                fine_tuned_works = False
        except Exception as e:
            logger.error(f"❌ Fine-tuned model failed with error: {e}")
            fine_tuned_works = False
    
    if not fine_tuned_works:
        logger.error("=" * 60)
        logger.error("❌ FINE-TUNING FAILED: Fine-tuned model is not working properly!")
        logger.error("This likely indicates a problem with the fine-tuning process.")
        logger.error("Possible causes:")
        logger.error("  1. CTC loss computation failed (loss was 0.0 during training)")
        logger.error("  2. Model's CTC head not properly initialized")
        logger.error("  3. LoRA adapters not properly merged")
        logger.error("  4. Labels/transcripts not properly processed during training")
        logger.error("=" * 60)
        logger.error("⚠️  Skipping evaluation. Please investigate the fine-tuning process.")
        logger.error("   Check training logs for loss values, gradient norms, and any warnings.")
        logger.error("   Consider re-running fine-tuning with --force-retrain flag.")
        return
    
    logger.info(f"Evaluating fine-tuned model on {len(test_files)} test files...")
    fine_error_cases, fine_metrics = collect_error_cases(test_files, fine_tuned_model, llm_corrector, evaluator)
    
    logger.info(f"\nFine-tuned Metrics:")
    logger.info(f"  WER: {fine_metrics['wer']:.4f} ({fine_metrics['wer']*100:.2f}%)")
    logger.info(f"  CER: {fine_metrics['cer']:.4f} ({fine_metrics['cer']*100:.2f}%)")
    logger.info(f"  Error Samples: {fine_metrics['error_samples']}/{fine_metrics['total_samples']}")
    
    # Step 4: Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    wer_improvement = baseline_metrics['wer'] - fine_metrics['wer']
    cer_improvement = baseline_metrics['cer'] - fine_metrics['cer']
    
    logger.info(f"\nBaseline WER: {baseline_metrics['wer']:.4f} ({baseline_metrics['wer']*100:.2f}%)")
    logger.info(f"Fine-tuned WER: {fine_metrics['wer']:.4f} ({fine_metrics['wer']*100:.2f}%)")
    logger.info(f"WER Improvement: {wer_improvement:.4f} ({wer_improvement*100:.2f} percentage points)")
    
    logger.info(f"\nBaseline CER: {baseline_metrics['cer']:.4f} ({baseline_metrics['cer']*100:.2f}%)")
    logger.info(f"Fine-tuned CER: {fine_metrics['cer']:.4f} ({fine_metrics['cer']*100:.2f}%)")
    logger.info(f"CER Improvement: {cer_improvement:.4f} ({cer_improvement*100:.2f} percentage points)")
    
    logger.info(f"\nTraining Details:")
    # Use training time from result, or 0 if not available
    training_time = fine_tune_result.get('training_duration_seconds', 0.0)
    if training_time == 0.0 and 'actual_time' in locals():
        training_time = actual_time
    logger.info(f"  Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    logger.info(f"  Samples used: {len(error_cases)}")
    logger.info(f"  Epochs: {args.num_epochs}")
    
    # Save results
    results = {
        'baseline_metrics': baseline_metrics,
        'fine_tuned_metrics': fine_metrics,
        'improvements': {
            'wer_improvement': wer_improvement,
            'cer_improvement': cer_improvement,
            'wer_improvement_pct': (wer_improvement / baseline_metrics['wer'] * 100) if baseline_metrics['wer'] > 0 else 0,
            'cer_improvement_pct': (cer_improvement / baseline_metrics['cer'] * 100) if baseline_metrics['cer'] > 0 else 0
        },
        'training': {
            'training_time_seconds': training_time,
            'num_samples': len(error_cases),
            'num_epochs': args.num_epochs,
            'model_path': fine_tune_result['model_path']
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = Path(args.output_dir) / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

