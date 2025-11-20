"""
Error Detection Module - Week 2
Detects errors in STT transcriptions using multiple heuristics
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ErrorSignal:
    """Represents an error signal detected in transcription"""
    error_type: str
    confidence: float  # 0.0 to 1.0, higher = more confident error
    location: Optional[int] = None  # Character position if applicable
    description: str = ""
    suggested_correction: Optional[str] = None


class ErrorDetector:
    """
    Multi-heuristic error detection system for STT transcriptions.
    Uses linguistic patterns, confidence thresholds, and anomaly detection.
    """
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.3,
        max_length_ratio: float = 3.0,
        min_length_ratio: float = 0.1
    ):
        """
        Initialize error detector.
        
        Args:
            min_confidence_threshold: Minimum confidence to flag as error
            max_length_ratio: Max ratio of transcript to expected length
            min_length_ratio: Min ratio of transcript to expected length
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.max_length_ratio = max_length_ratio
        self.min_length_ratio = min_length_ratio
        
        # Common error patterns
        self.repeated_chars_pattern = re.compile(r'(.)\1{4,}')  # 5+ repeated chars
        self.nonsense_words_pattern = re.compile(r'\b[a-z]{1,2}\b')  # Very short words
        self.special_chars_pattern = re.compile(r'[^\w\s\.,!?;:\-\'"]')
        
        # Linguistic heuristics
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any',
            'these', 'give', 'day', 'most', 'us'
        }
    
    def detect_errors(
        self,
        transcript: str,
        audio_length_seconds: Optional[float] = None,
        model_confidence: Optional[float] = None
    ) -> List[ErrorSignal]:
        """
        Detect errors in transcription using multiple heuristics.
        
        Args:
            transcript: The transcription to check
            audio_length_seconds: Length of audio in seconds (for length checks)
            model_confidence: Model's confidence score if available
        
        Returns:
            List of ErrorSignal objects
        """
        errors = []
        
        # 1. Empty or very short transcript
        if not transcript or len(transcript.strip()) < 3:
            errors.append(ErrorSignal(
                error_type="empty_transcript",
                confidence=0.9,
                description="Transcript is empty or too short",
                suggested_correction=None
            ))
        
        # 2. Length anomaly detection
        if audio_length_seconds:
            expected_length = audio_length_seconds * 2.5  # ~2.5 chars per second average
            actual_length = len(transcript)
            ratio = actual_length / expected_length if expected_length > 0 else 1.0
            
            if ratio > self.max_length_ratio:
                errors.append(ErrorSignal(
                    error_type="too_long",
                    confidence=0.7,
                    description=f"Transcript is {ratio:.1f}x longer than expected",
                    suggested_correction=None
                ))
            elif ratio < self.min_length_ratio:
                errors.append(ErrorSignal(
                    error_type="too_short",
                    confidence=0.7,
                    description=f"Transcript is {ratio:.1f}x shorter than expected",
                    suggested_correction=None
                ))
        
        # 3. Repeated characters (common STT error)
        repeated_matches = self.repeated_chars_pattern.finditer(transcript)
        for match in repeated_matches:
            errors.append(ErrorSignal(
                error_type="repeated_chars",
                confidence=0.8,
                location=match.start(),
                description=f"Repeated character '{match.group(1)}' detected",
                suggested_correction=transcript[:match.start()] + match.group(1) + transcript[match.end():]
            ))
        
        # 4. Special characters (likely transcription errors)
        special_matches = self.special_chars_pattern.finditer(transcript)
        for match in special_matches:
            errors.append(ErrorSignal(
                error_type="special_chars",
                confidence=0.6,
                location=match.start(),
                description=f"Unexpected special character '{match.group()}'",
                suggested_correction=None
            ))
        
        # 5. Low model confidence
        if model_confidence is not None and model_confidence < self.min_confidence_threshold:
            errors.append(ErrorSignal(
                error_type="low_confidence",
                confidence=0.9,
                description=f"Model confidence ({model_confidence:.2f}) below threshold",
                suggested_correction=None
            ))
        
        # 6. Unusual word patterns (too many very short words)
        words = transcript.lower().split()
        if len(words) > 0:
            short_words_ratio = sum(1 for w in words if len(w) <= 2) / len(words)
            if short_words_ratio > 0.5:
                errors.append(ErrorSignal(
                    error_type="unusual_word_pattern",
                    confidence=0.5,
                    description=f"High ratio ({short_words_ratio:.2f}) of very short words",
                    suggested_correction=None
                ))
        
        # 7. All caps (unusual for normal speech)
        if transcript.isupper() and len(transcript) > 10:
            errors.append(ErrorSignal(
                error_type="all_caps",
                confidence=0.4,
                description="Transcript is entirely uppercase",
                suggested_correction=transcript.capitalize()
            ))
        
        # 8. No punctuation or sentence structure
        if len(transcript) > 50 and not re.search(r'[.!?]', transcript):
            errors.append(ErrorSignal(
                error_type="no_punctuation",
                confidence=0.3,
                description="Long transcript without sentence punctuation",
                suggested_correction=None
            ))
        
        return errors
    
    def calculate_error_score(self, errors: List[ErrorSignal]) -> float:
        """
        Calculate overall error score from detected errors.
        
        Args:
            errors: List of ErrorSignal objects
        
        Returns:
            Error score between 0.0 and 1.0
        """
        if not errors:
            return 0.0
        
        # Weighted average of error confidences
        total_confidence = sum(e.confidence for e in errors)
        return min(1.0, total_confidence / len(errors))
    
    def get_error_summary(self, errors: List[ErrorSignal]) -> Dict:
        """
        Get summary of detected errors.
        
        Args:
            errors: List of ErrorSignal objects
        
        Returns:
            Dictionary with error summary
        """
        if not errors:
            return {
                "has_errors": False,
                "error_count": 0,
                "error_score": 0.0,
                "error_types": []
            }
        
        error_types = {}
        for error in errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        return {
            "has_errors": True,
            "error_count": len(errors),
            "error_score": self.calculate_error_score(errors),
            "error_types": error_types,
            "errors": [
                {
                    "type": e.error_type,
                    "confidence": e.confidence,
                    "description": e.description,
                    "location": e.location
                }
                for e in errors
            ]
        }

