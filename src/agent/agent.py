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
        error_threshold: float = 0.3
    ):
        """
        Initialize STT Agent.
        
        Args:
            baseline_model: Instance of BaselineSTTModel
            error_threshold: Threshold for error detection confidence
        """
        self.baseline_model = baseline_model
        self.error_detector = ErrorDetector(min_confidence_threshold=error_threshold)
        self.self_learner = SelfLearner()  # In-memory tracking only
        
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
        
        if enable_auto_correction and errors:
            corrected_transcript, corrections_applied = self._apply_corrections(
                transcript, errors
            )
            
            # Record corrections for learning
            for error in errors:
                if error.suggested_correction:
                    self.self_learner.record_error(
                        error_type=error.error_type,
                        transcript=transcript,
                        context={
                            'audio_length': audio_length_seconds,
                            'confidence': baseline_result.get('confidence')
                        },
                        correction=error.suggested_correction
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
                'auto_correction_enabled': enable_auto_correction
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
        
        return {
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

