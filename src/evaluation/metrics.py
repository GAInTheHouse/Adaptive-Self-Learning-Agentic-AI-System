"""
Evaluation metrics for STT models: WER and CER.
"""

from jiwer import wer, cer
import json
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STTEvaluator:
    """Calculate WER and CER for STT predictions"""
    
    def __init__(self):
        self.results = []
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate.
        
        Args:
            reference: Ground truth transcription
            hypothesis: Model prediction
        
        Returns:
            WER score
        """
        return wer(reference, hypothesis)
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate.
        
        Args:
            reference: Ground truth transcription
            hypothesis: Model prediction
        
        Returns:
            CER score
        """
        return cer(reference, hypothesis)
    
    def evaluate_batch(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate batch of predictions.
        
        Args:
            references: List of ground truth transcriptions
            hypotheses: List of model predictions
        
        Returns:
            Dictionary with WER and CER scores
        """
        assert len(references) == len(hypotheses), \
            "References and hypotheses must have same length"
        
        # Calculate metrics
        wer_score = wer(references, hypotheses)
        cer_score = cer(references, hypotheses)
        
        # Store detailed results
        for ref, hyp in zip(references, hypotheses):
            self.results.append({
                'reference': ref,
                'hypothesis': hyp,
                'wer': self.calculate_wer(ref, hyp),
                'cer': self.calculate_cer(ref, hyp)
            })
        
        return {
            'wer': wer_score,
            'cer': cer_score,
            'num_samples': len(references)
        }
    
    def save_results(self, output_path: str):
        """
        Save detailed evaluation results.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        summary = {
            'average_wer': sum(r['wer'] for r in self.results) / len(self.results),
            'average_cer': sum(r['cer'] for r in self.results) / len(self.results),
            'num_samples': len(self.results),
            'detailed_results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Average WER: {summary['average_wer']:.4f}")
        logger.info(f"Average CER: {summary['average_cer']:.4f}")
        
        return summary
