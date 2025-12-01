"""
STT Agent - Week 2
Main agent class that integrates error detection and self-learning
"""

import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import time

from .error_detector import ErrorDetector, ErrorSignal
from .self_learner import SelfLearner
from .llm_corrector import GemmaLLMCorrector

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
        use_quantization: bool = False
    ):
        """
        Initialize STT Agent.
        
        Args:
            baseline_model: Instance of BaselineSTTModel
            error_threshold: Threshold for error detection confidence
            use_llm_correction: Whether to use Gemma LLM for intelligent correction
            llm_model_name: Gemma model name (default: "google/gemma-2b-it")
            use_quantization: Whether to use 8-bit quantization for LLM (saves memory)
        """
        self.baseline_model = baseline_model
        self.error_detector = ErrorDetector(min_confidence_threshold=error_threshold)
        self.self_learner = SelfLearner()  # In-memory tracking only
        
        # Initialize Gemma LLM corrector if requested
        self.llm_corrector = None
        if use_llm_correction:
            try:
                self.llm_corrector = GemmaLLMCorrector(
                    model_name=llm_model_name or "google/gemma-2b-it",
                    use_quantization=use_quantization
                )
                if self.llm_corrector.is_available():
                    logger.info("✅ Gemma LLM corrector initialized successfully")
                else:
                    logger.warning("⚠️  Gemma LLM not available, using rule-based correction only")
                    self.llm_corrector = None
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize Gemma LLM: {e}. Using rule-based correction only.")
                self.llm_corrector = None
        
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
        for error in errors:
            self.self_learner.record_error(
                error_type=error.error_type,
                transcript=transcript,
                context={
                    'audio_length': audio_length_seconds,
                    'confidence': baseline_result.get('confidence'),
                    'error_confidence': error.confidence
                }
            )
        
        # Step 6: Prepare result
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
                'llm_available': self.llm_corrector.is_available() if self.llm_corrector else False
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

