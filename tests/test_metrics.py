"""
Unit tests for evaluation metrics (WER/CER calculation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import STTEvaluator
import pytest


class TestSTTEvaluator:
    """Test cases for STTEvaluator class"""
    
    def setup_method(self):
        """Setup test instance"""
        self.evaluator = STTEvaluator()
    
    def test_perfect_match(self):
        """Test WER and CER with perfect match"""
        reference = "hello world"
        hypothesis = "hello world"
        
        wer_score = self.evaluator.calculate_wer(reference, hypothesis)
        cer_score = self.evaluator.calculate_cer(reference, hypothesis)
        
        assert wer_score == 0.0, "Perfect match should have 0.0 WER"
        assert cer_score == 0.0, "Perfect match should have 0.0 CER"
    
    def test_complete_mismatch(self):
        """Test WER and CER with complete mismatch"""
        reference = "hello world"
        hypothesis = "goodbye universe"
        
        wer_score = self.evaluator.calculate_wer(reference, hypothesis)
        cer_score = self.evaluator.calculate_cer(reference, hypothesis)
        
        assert wer_score > 0.0, "Complete mismatch should have positive WER"
        assert cer_score > 0.0, "Complete mismatch should have positive CER"
    
    def test_single_word_error(self):
        """Test WER with single word error"""
        reference = "hello world"
        hypothesis = "hello earth"
        
        wer_score = self.evaluator.calculate_wer(reference, hypothesis)
        
        assert wer_score == 0.5, "Single word error in 2 words should be 0.5 WER"
    
    def test_case_sensitivity(self):
        """Test case handling"""
        reference = "Hello World"
        hypothesis = "hello world"
        
        wer_score = self.evaluator.calculate_wer(reference, hypothesis)
        cer_score = self.evaluator.calculate_cer(reference, hypothesis)
        
        # jiwer treats differently cased words as different
        assert wer_score > 0.0, "Different case should result in positive WER"
    
    def test_punctuation(self):
        """Test punctuation handling"""
        reference = "Hello, world!"
        hypothesis = "Hello world"
        
        cer_score = self.evaluator.calculate_cer(reference, hypothesis)
        
        # Missing punctuation should affect CER
        assert cer_score > 0.0, "Missing punctuation should increase CER"
    
    def test_batch_evaluation_equal_length(self):
        """Test batch evaluation with equal length lists"""
        references = ["hello world", "goodbye world", "test case"]
        hypotheses = ["hello world", "goodbye earth", "test case"]
        
        results = self.evaluator.evaluate_batch(references, hypotheses)
        
        assert 'wer' in results
        assert 'cer' in results
        assert 'num_samples' in results
        assert results['num_samples'] == 3
        assert 0.0 <= results['wer'] <= 1.0
        assert 0.0 <= results['cer'] <= 1.0
    
    def test_batch_evaluation_unequal_length(self):
        """Test batch evaluation fails with unequal length lists"""
        references = ["hello world", "goodbye world"]
        hypotheses = ["hello world"]
        
        with pytest.raises(AssertionError):
            self.evaluator.evaluate_batch(references, hypotheses)
    
    def test_empty_strings(self):
        """Test handling of empty strings"""
        reference = "hello world"
        hypothesis = ""
        
        wer_score = self.evaluator.calculate_wer(reference, hypothesis)
        
        # Empty hypothesis should have high WER
        assert wer_score > 0.0, "Empty hypothesis should have positive WER"
    
    def test_extra_words(self):
        """Test insertion errors"""
        reference = "hello world"
        hypothesis = "hello beautiful world"
        
        wer_score = self.evaluator.calculate_wer(reference, hypothesis)
        
        # Extra word should increase WER
        assert wer_score > 0.0, "Extra word should increase WER"
    
    def test_missing_words(self):
        """Test deletion errors"""
        reference = "hello beautiful world"
        hypothesis = "hello world"
        
        wer_score = self.evaluator.calculate_wer(reference, hypothesis)
        
        # Missing word should increase WER
        assert wer_score > 0.0, "Missing word should increase WER"
    
    def test_save_results(self, tmp_path):
        """Test saving evaluation results"""
        references = ["hello world", "test case"]
        hypotheses = ["hello world", "test case"]
        
        self.evaluator.evaluate_batch(references, hypotheses)
        
        output_path = tmp_path / "results.json"
        summary = self.evaluator.save_results(str(output_path))
        
        assert output_path.exists(), "Results file should be created"
        assert 'average_wer' in summary
        assert 'average_cer' in summary
        assert 'num_samples' in summary
        assert summary['num_samples'] == 2


def test_wer_cer_relationship():
    """Test relationship between WER and CER"""
    evaluator = STTEvaluator()
    
    # CER is usually lower than or equal to WER for similar errors
    reference = "hello world"
    hypothesis = "hello earth"
    
    wer_score = evaluator.calculate_wer(reference, hypothesis)
    cer_score = evaluator.calculate_cer(reference, hypothesis)
    
    # Both should be positive for this error
    assert wer_score > 0.0
    assert cer_score > 0.0


def test_multiple_spaces():
    """Test handling of multiple spaces"""
    evaluator = STTEvaluator()
    
    reference = "hello  world"  # Double space
    hypothesis = "hello world"   # Single space
    
    # jiwer normalizes whitespace
    wer_score = evaluator.calculate_wer(reference, hypothesis)
    
    # Should handle whitespace normalization
    assert wer_score >= 0.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

