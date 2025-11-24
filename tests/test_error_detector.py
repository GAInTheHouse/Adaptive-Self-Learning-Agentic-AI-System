"""
Unit tests for error detector
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.error_detector import ErrorDetector, ErrorSignal
import pytest


class TestErrorDetector:
    """Test cases for ErrorDetector class"""
    
    def setup_method(self):
        """Setup test instance"""
        self.detector = ErrorDetector(min_confidence_threshold=0.3)
    
    def test_empty_transcript_detection(self):
        """Test detection of empty transcripts"""
        errors = self.detector.detect_errors("")
        
        assert len(errors) > 0, "Should detect error in empty transcript"
        assert any(e.error_type == "empty_transcript" for e in errors)
    
    def test_all_caps_detection(self):
        """Test detection of all caps text"""
        errors = self.detector.detect_errors("HELLO WORLD THIS IS ALL CAPS")
        
        assert len(errors) > 0, "Should detect all caps text"
        assert any(e.error_type == "all_caps" for e in errors)
    
    def test_repeated_chars_detection(self):
        """Test detection of repeated characters"""
        test_cases = [
            "Hellllllo world",
            "Tesssst case",
            "aaaaaaa"
        ]
        
        for transcript in test_cases:
            errors = self.detector.detect_errors(transcript)
            assert any(e.error_type == "repeated_chars" for e in errors), \
                f"Should detect repeated chars in: {transcript}"
    
    def test_too_short_detection(self):
        """Test detection of too short transcripts"""
        errors = self.detector.detect_errors("a")
        
        assert any(e.error_type == "too_short" for e in errors), \
            "Should detect too short transcript"
    
    def test_missing_punctuation_detection(self):
        """Test detection of missing punctuation"""
        # Long text without punctuation
        transcript = "this is a long transcript without any punctuation " \
                    "which should trigger the missing punctuation error"
        
        errors = self.detector.detect_errors(transcript)
        
        # May or may not trigger depending on implementation
        # This tests the detector runs without error
        assert isinstance(errors, list)
    
    def test_unusual_word_pattern_detection(self):
        """Test detection of unusual word patterns"""
        errors = self.detector.detect_errors("a b c d e f g h")
        
        # Very short words might trigger unusual pattern
        assert isinstance(errors, list)
    
    def test_normal_transcript_no_errors(self):
        """Test that normal transcript has few or no errors"""
        errors = self.detector.detect_errors("This is a normal transcript with proper formatting.")
        
        # Normal transcript might have low error score
        summary = self.detector.get_error_summary(errors)
        
        # Check that error score is reasonable
        assert 0.0 <= summary['error_score'] <= 1.0
    
    def test_low_confidence_detection(self):
        """Test low confidence detection"""
        errors = self.detector.detect_errors(
            "transcript",
            model_confidence=0.2  # Low confidence
        )
        
        assert any(e.error_type == "low_confidence" for e in errors), \
            "Should detect low confidence"
    
    def test_high_confidence_no_error(self):
        """Test high confidence doesn't trigger error"""
        errors = self.detector.detect_errors(
            "This is a good transcript.",
            model_confidence=0.95
        )
        
        # High confidence shouldn't trigger low_confidence error
        assert not any(e.error_type == "low_confidence" for e in errors)
    
    def test_length_anomaly_too_short(self):
        """Test detection of transcript that's too short for audio"""
        errors = self.detector.detect_errors(
            "hi",
            audio_length_seconds=10.0  # 10 second audio
        )
        
        # Very short transcript for long audio should trigger error
        assert any(e.error_type == "length_anomaly_short" for e in errors)
    
    def test_length_anomaly_too_long(self):
        """Test detection of transcript that's too long for audio"""
        transcript = "word " * 200  # Very long transcript
        errors = self.detector.detect_errors(
            transcript,
            audio_length_seconds=1.0  # 1 second audio
        )
        
        # Very long transcript for short audio should trigger error
        assert any(e.error_type == "length_anomaly_long" for e in errors)
    
    def test_error_summary(self):
        """Test error summary generation"""
        errors = self.detector.detect_errors("HELLO WORLD")
        summary = self.detector.get_error_summary(errors)
        
        assert 'has_errors' in summary
        assert 'error_count' in summary
        assert 'error_score' in summary
        assert 'error_types' in summary
        
        if len(errors) > 0:
            assert summary['has_errors'] is True
            assert summary['error_count'] > 0
            assert summary['error_score'] > 0.0
    
    def test_error_signal_structure(self):
        """Test ErrorSignal dataclass structure"""
        signal = ErrorSignal(
            error_type="test_error",
            confidence=0.8,
            location=5,
            description="Test error description",
            suggested_correction="corrected text"
        )
        
        assert signal.error_type == "test_error"
        assert signal.confidence == 0.8
        assert signal.location == 5
        assert signal.description == "Test error description"
        assert signal.suggested_correction == "corrected text"
    
    def test_confidence_threshold_filtering(self):
        """Test that confidence threshold filters errors"""
        # Create detector with high threshold
        strict_detector = ErrorDetector(min_confidence_threshold=0.9)
        lenient_detector = ErrorDetector(min_confidence_threshold=0.1)
        
        transcript = "MAYBE ALL CAPS"
        
        strict_errors = strict_detector.detect_errors(transcript)
        lenient_errors = lenient_detector.detect_errors(transcript)
        
        # Lenient detector should find at least as many errors
        assert len(lenient_errors) >= len(strict_errors)
    
    def test_special_characters(self):
        """Test handling of special characters"""
        errors = self.detector.detect_errors("Hello @#$%^&* world !!!")
        
        # Should handle special characters without crashing
        assert isinstance(errors, list)
    
    def test_unicode_characters(self):
        """Test handling of unicode characters"""
        errors = self.detector.detect_errors("Hello 你好 مرحبا world")
        
        # Should handle unicode without crashing
        assert isinstance(errors, list)
    
    def test_numbers(self):
        """Test handling of transcripts with numbers"""
        errors = self.detector.detect_errors("The year is 2024 and the time is 10:30")
        
        # Should handle numbers without crashing
        assert isinstance(errors, list)


def test_multiple_errors():
    """Test detection of multiple simultaneous errors"""
    detector = ErrorDetector()
    
    # Transcript with multiple issues
    transcript = "HELLO WORLD TESSSST"  # All caps + repeated chars
    errors = detector.detect_errors(transcript)
    
    # Should detect multiple error types
    error_types = [e.error_type for e in errors]
    assert "all_caps" in error_types
    assert "repeated_chars" in error_types


def test_error_score_calculation():
    """Test error score increases with more/worse errors"""
    detector = ErrorDetector()
    
    good_transcript = "This is a good transcript."
    bad_transcript = "THISSSS IS BADDD"
    
    good_errors = detector.detect_errors(good_transcript)
    bad_errors = detector.detect_errors(bad_transcript)
    
    good_summary = detector.get_error_summary(good_errors)
    bad_summary = detector.get_error_summary(bad_errors)
    
    # Bad transcript should have higher error score
    assert bad_summary['error_score'] > good_summary['error_score']


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

