"""
STT Agent - Week 2 & Week 3
Main agent class that integrates error detection, self-learning, and adaptive fine-tuning
"""

import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import time
import os
import numpy as np

from .error_detector import ErrorDetector, ErrorSignal
from .self_learner import SelfLearner
from .llm_corrector import GemmaLLMCorrector
from .adaptive_scheduler import AdaptiveScheduler
from .fine_tuner import FineTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STTAgent:
    """
    Autonomous STT Agent that integrates error detection and self-learning.
    Wraps the baseline STT model with intelligent error detection and correction.
    """
    
    def __init__(
        self,
        baseline_model,
        error_threshold: float = 0.3,
        use_llm_correction: bool = True,
        llm_model_name: Optional[str] = None,
        use_quantization: bool = False,
        enable_adaptive_fine_tuning: bool = True,
        scheduler_history_path: Optional[str] = None
    ):
        """
        Initialize STT Agent.
        
        Args:
            baseline_model: Instance of BaselineSTTModel
            error_threshold: Threshold for error detection confidence
            use_llm_correction: Whether to use Gemma LLM for intelligent correction
            llm_model_name: Gemma model name (default: "google/gemma-2b-it")
            use_quantization: Whether to use 8-bit quantization for LLM (saves memory)
            enable_adaptive_fine_tuning: Whether to enable adaptive fine-tuning (Week 3)
            scheduler_history_path: Path to save/load scheduler history
        """
        self.baseline_model = baseline_model
        self.error_detector = ErrorDetector(min_confidence_threshold=error_threshold)
        self.self_learner = SelfLearner()  # In-memory tracking only
        
        # Initialize LLM corrector if requested
        self.llm_corrector = None
        if use_llm_correction:
            try:
                self.llm_corrector = GemmaLLMCorrector(
                    model_name=llm_model_name or "mistralai/Mistral-7B-Instruct-v0.3",
                    use_quantization=use_quantization
                )
                if self.llm_corrector.is_available():
                    logger.info("✅ LLM corrector initialized successfully")
                else:
                    logger.warning("⚠️  LLM not available, using rule-based correction only")
                    self.llm_corrector = None
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize LLM: {e}. Using rule-based correction only.")
                self.llm_corrector = None
        
        # Initialize adaptive scheduler and fine-tuner (Week 3)
        self.enable_adaptive_fine_tuning = enable_adaptive_fine_tuning
        self.adaptive_scheduler = None
        self.fine_tuner = None
        
        if enable_adaptive_fine_tuning:
            try:
                # Initialize adaptive scheduler
                scheduler_path = scheduler_history_path or "data/processed/scheduler_history.json"
                self.adaptive_scheduler = AdaptiveScheduler(history_path=scheduler_path)
                
                # Initialize fine-tuner (requires access to model and processor)
                if hasattr(baseline_model, 'model') and hasattr(baseline_model, 'processor'):
                    self.fine_tuner = FineTuner(
                        model=baseline_model.model,
                        processor=baseline_model.processor,
                        device=baseline_model.device
                    )
                    logger.info("✅ Adaptive fine-tuning system initialized")
                else:
                    logger.warning("⚠️  Baseline model doesn't expose model/processor, fine-tuning disabled")
                    self.enable_adaptive_fine_tuning = False
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize adaptive fine-tuning: {e}")
                self.enable_adaptive_fine_tuning = False
        
        logger.info("STT Agent initialized with error detection and self-learning")
    
    def transcribe_with_agent(
        self,
        audio_path: str,
        audio_length_seconds: Optional[float] = None,
        enable_auto_correction: bool = True
    ) -> Dict:
        """
        Transcribe audio with agent-based error detection and correction.
        
        Args:
            audio_path: Path to audio file
            audio_length_seconds: Length of audio in seconds (optional)
            enable_auto_correction: Whether to apply automatic corrections
        
        Returns:
            Dictionary with transcript, error detection results, and metadata
        """
        # Step 1: Get baseline transcription
        start_time = time.time()
        baseline_result = self.baseline_model.transcribe(audio_path)
        inference_time = time.time() - start_time
        
        transcript = baseline_result.get('transcript', '')
        baseline_result.setdefault("original_transcript", transcript)
        
        # Step 2: Detect errors
        errors = self.error_detector.detect_errors(
            transcript=transcript,
            audio_length_seconds=audio_length_seconds,
            model_confidence=baseline_result.get('confidence')
        )
        
        # Step 3: Calculate error score
        error_score = self.error_detector.calculate_error_score(errors)
        error_summary = self.error_detector.get_error_summary(errors)
        
        # Step 4: Apply corrections if enabled
        corrected_transcript = transcript
        corrections_applied = []
        correction_method = "none"
        
        if enable_auto_correction and errors:
            # Try LLM-based correction first if available
            if self.llm_corrector and self.llm_corrector.is_available():
                try:
                    error_dicts = [
                        {
                            'type': e.error_type,
                            'description': e.description,
                            'confidence': e.confidence,
                            'location': e.location
                        }
                        for e in errors
                    ]
                    
                    llm_result = self.llm_corrector.correct_transcript(
                        transcript=transcript,
                        errors=error_dicts,
                        context={
                            'audio_length': audio_length_seconds,
                            'confidence': baseline_result.get('confidence')
                        }
                    )
                    
                    if llm_result.get('llm_used', False):
                        corrected_transcript = llm_result['corrected_transcript']
                        correction_method = llm_result['correction_method']
                        corrections_applied.append({
                            'error_type': 'llm_correction',
                            'original': transcript,
                            'corrected': corrected_transcript,
                            'method': 'gemma_llm',
                            'confidence': 0.8  # LLM corrections have high confidence
                        })
                        logger.info("✅ Applied LLM-based correction using Gemma")
                    else:
                        # Fall back to rule-based correction
                        corrected_transcript, corrections_applied = self._apply_corrections(
                            transcript, errors
                        )
                        correction_method = "rule_based"
                except Exception as e:
                    logger.warning(f"LLM correction failed, falling back to rule-based: {e}")
                    corrected_transcript, corrections_applied = self._apply_corrections(
                        transcript, errors
                    )
                    correction_method = "rule_based_fallback"
            else:
                # Use rule-based correction
                corrected_transcript, corrections_applied = self._apply_corrections(
                    transcript, errors
                )
                correction_method = "rule_based"
            
            # Record corrections for learning
            for error in errors:
                if error.suggested_correction or correction_method.startswith("gemma"):
                    self.self_learner.record_error(
                        error_type=error.error_type,
                        transcript=transcript,
                        context={
                            'audio_length': audio_length_seconds,
                            'confidence': baseline_result.get('confidence'),
                            'correction_method': correction_method
                        },
                        correction=corrected_transcript if correction_method.startswith("gemma") else error.suggested_correction
                    )
        
        # Step 5: Record errors for learning (even if not corrected)
        error_count = len(errors)
        for error in errors:
            self.self_learner.record_error(
                error_type=error.error_type,
                transcript=transcript,
                context={
                    'audio_path': audio_path,  # Store path for fine-tuning
                    'audio_length': audio_length_seconds,
                    'confidence': baseline_result.get('confidence'),
                    'error_confidence': error.confidence
                }
            )
        
        # Step 6: Update adaptive scheduler and check for fine-tuning trigger (Week 3)
        fine_tuning_triggered = False
        if self.enable_adaptive_fine_tuning and self.adaptive_scheduler:
            # Calculate accuracy (simplified: 1 - error_rate)
            accuracy = 1.0 - (error_count / max(1, len(transcript.split())))
            
            # Record performance metrics
            self.adaptive_scheduler.record_performance(
                error_count=error_count,
                accuracy=accuracy,
                inference_time=inference_time
            )
            
            # Record error samples if errors were detected
            if error_count > 0:
                self.adaptive_scheduler.record_error_sample(count=error_count)
            
            # Check if fine-tuning should be triggered
            should_trigger, trigger_info = self.adaptive_scheduler.should_trigger_fine_tuning()
            if should_trigger:
                fine_tuning_triggered = self._trigger_adaptive_fine_tuning()
        
        # Step 7: Prepare result
        result = {
            'transcript': corrected_transcript if enable_auto_correction else transcript,
            'original_transcript': transcript,
            'model': baseline_result.get('model'),
            'inference_time_seconds': inference_time,
            'error_detection': {
                'has_errors': error_summary['has_errors'],
                'error_count': error_summary['error_count'],
                'error_score': error_score,
                'errors': error_summary.get('errors', []),
                'error_types': error_summary.get('error_types', {})
            },
            'corrections': {
                'applied': enable_auto_correction and len(corrections_applied) > 0,
                'count': len(corrections_applied),
                'details': corrections_applied
            },
            'agent_metadata': {
                'error_threshold': self.error_detector.min_confidence_threshold,
                'auto_correction_enabled': enable_auto_correction,
                'correction_method': correction_method,
                'llm_available': self.llm_corrector.is_available() if self.llm_corrector else False,
                'adaptive_fine_tuning_enabled': self.enable_adaptive_fine_tuning,
                'fine_tuning_triggered': fine_tuning_triggered
            }
        }
        
        return result
    
    def _apply_corrections(
        self,
        transcript: str,
        errors: List[ErrorSignal]
    ) -> Tuple[str, List[Dict]]:
        """
        Apply corrections to transcript based on detected errors.
        
        Args:
            transcript: Original transcript
            errors: List of detected errors
        
        Returns:
            Tuple of (corrected_transcript, list_of_corrections_applied)
        """
        corrected = transcript
        corrections_applied = []
        
        # Sort errors by location (if available) to apply in order
        errors_with_location = [e for e in errors if e.location is not None]
        errors_without_location = [e for e in errors if e.location is None]
        sorted_errors = sorted(errors_with_location, key=lambda x: x.location) + errors_without_location
        
        for error in sorted_errors:
            if error.suggested_correction:
                # Apply correction
                if error.location is not None:
                    # Location-based correction
                    # This is simplified - in practice, you'd need more sophisticated text replacement
                    pass  # Skip complex location-based corrections for now
                else:
                    # Full transcript replacement
                    if error.error_type == "all_caps":
                        corrected = error.suggested_correction
                        corrections_applied.append({
                            'error_type': error.error_type,
                            'original': transcript,
                            'corrected': corrected,
                            'confidence': error.confidence
                        })
        
        return corrected, corrections_applied
    
    def get_agent_stats(self) -> Dict:
        """
        Get agent statistics and learning insights.
        
        Returns:
            Dictionary with agent statistics
        """
        error_stats = self.self_learner.get_error_statistics()
        
        stats = {
            'error_detection': {
                'threshold': self.error_detector.min_confidence_threshold,
                'total_errors_detected': error_stats['total_errors'],
                'error_type_distribution': error_stats['error_type_distribution']
            },
            'learning': {
                'corrections_made': error_stats['corrections_made'],
                'feedback_count': error_stats['feedback_count'],
                'total_errors_learned': error_stats['total_errors']
            },
            'model_info': self.baseline_model.get_model_info()
        }
        
        # Add LLM info if available
        if self.llm_corrector:
            stats['llm_info'] = self.llm_corrector.get_model_info()
        
        return stats
    
    def submit_feedback(
        self,
        transcript_id: str,
        user_feedback: str,
        is_correct: bool,
        corrected_transcript: Optional[str] = None
    ):
        """
        Submit user feedback for learning.
        
        Args:
            transcript_id: Identifier for the transcript
            user_feedback: User's feedback text
            is_correct: Whether the transcript was correct
            corrected_transcript: Corrected version if not correct
        """
        self.self_learner.record_feedback(
            transcript_id=transcript_id,
            user_feedback=user_feedback,
            is_correct=is_correct,
            corrected_transcript=corrected_transcript
        )
        logger.info(f"Feedback recorded for transcript {transcript_id}")
    
    def get_learning_data(self) -> Dict:
        """
        Get in-memory learning data for external persistence.
        Note: Data persistence handled by data management layer.
        
        Returns:
            Dictionary with learning data
        """
        return self.self_learner.get_in_memory_data()
    
    def _trigger_adaptive_fine_tuning(self) -> bool:
        """
        Trigger adaptive fine-tuning when threshold n is reached.
        
        Returns:
            True if fine-tuning was triggered successfully
        """
        if not self.enable_adaptive_fine_tuning or not self.fine_tuner:
            return False
        
        logger.info("🚀 Triggering adaptive fine-tuning...")
        
        # Collect error samples with corrections
        learning_data = self.self_learner.get_in_memory_data()
        error_samples = []
        
        # Extract error samples that have corrections
        for error_type, patterns in learning_data.get('error_patterns', {}).items():
            for pattern in patterns:
                if pattern.get('correction') and pattern.get('context', {}).get('audio_path'):
                    error_samples.append({
                        'audio_path': pattern['context']['audio_path'],
                        'original_transcript': pattern['transcript'],
                        'corrected_transcript': pattern['correction'],
                        'error_type': error_type
                    })
        
        if len(error_samples) < 10:
            logger.warning(f"Insufficient error samples ({len(error_samples)}), skipping fine-tuning")
            return False
        
        # Get current model performance for comparison
        recent_metrics = list(self.adaptive_scheduler.performance_history)[-5:] if self.adaptive_scheduler.performance_history else []
        initial_val_accuracy = np.mean([m.accuracy for m in recent_metrics]) if recent_metrics else 0.5
        
        # Perform fine-tuning
        try:
            fine_tuning_result = self.fine_tuner.fine_tune(
                error_samples=error_samples[:self.adaptive_scheduler.current_threshold_n],
                num_epochs=3,
                batch_size=4
            )
            
            if fine_tuning_result.get('success', False):
                # Evaluate after fine-tuning
                final_val_accuracy = fine_tuning_result.get('final_validation_accuracy', initial_val_accuracy)
                
                # Check for overfitting
                train_acc = fine_tuning_result.get('final_train_accuracy', final_val_accuracy)
                val_acc = final_val_accuracy
                overfitting_detected, overfitting_info = self.adaptive_scheduler.check_overfitting(
                    train_acc, val_acc
                )
                
                # Record fine-tuning event
                self.adaptive_scheduler.record_fine_tuning_event(
                    samples_used=len(error_samples),
                    validation_accuracy_before=initial_val_accuracy,
                    validation_accuracy_after=final_val_accuracy,
                    training_cost=fine_tuning_result.get('training_cost', 0.0),
                    overfitting_detected=overfitting_detected
                )
                
                # Save scheduler history
                self.adaptive_scheduler.save_history()
                
                logger.info(
                    f"✅ Fine-tuning completed: "
                    f"accuracy_gain={fine_tuning_result.get('accuracy_gain', 0.0):.4f}, "
                    f"overfitting={overfitting_detected}"
                )
                
                return True
            else:
                logger.warning(f"Fine-tuning did not meet success criteria: {fine_tuning_result.get('reason', 'unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}", exc_info=True)
            return False
    
    def get_adaptive_scheduler_stats(self) -> Optional[Dict]:
        """
        Get adaptive scheduler statistics.
        
        Returns:
            Dictionary with scheduler stats or None if not enabled
        """
        if not self.enable_adaptive_fine_tuning or not self.adaptive_scheduler:
            return None
        
        return self.adaptive_scheduler.get_scheduler_stats()
    
    def manually_trigger_fine_tuning(self) -> Dict:
        """
        Manually trigger fine-tuning (for testing/debugging).
        
        Returns:
            Dictionary with fine-tuning results
        """
        if not self.enable_adaptive_fine_tuning:
            return {'success': False, 'reason': 'adaptive_fine_tuning_disabled'}
        
        success = self._trigger_adaptive_fine_tuning()
        return {
            'success': success,
            'scheduler_stats': self.get_adaptive_scheduler_stats()
        }

