"""
Agent Evaluator - Correction Accuracy and Consistency Metrics

Evaluates how well the agent corrects transcription errors.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from datetime import datetime  

import numpy as np
from src.evaluation.metrics import STTEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of a single correction evaluation"""
    audio_path: str
    original_transcript: str
    corrected_transcript: str
    reference_transcript: Optional[str]
    
    # Error metrics
    original_wer: float
    corrected_wer: float
    original_cer: float
    corrected_cer: float
    
    # Correction impact
    wer_improvement: float  # negative means worse
    cer_improvement: float
    was_improved: bool  # True if correction helped
    
    # Agent metadata
    errors_detected: int
    error_score: float
    agent_confidence: Optional[float]
    
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)


class AgentEvaluator:
    """
    Evaluates agent correction accuracy and consistency.
    
    Metrics calculated:
    - Correction accuracy: % of corrections that improve WER/CER
    - Correction consistency: Std dev of improvements
    - Average improvement: Mean WER/CER reduction
    - Worsening rate: % of corrections that hurt
    """
    
    def __init__(self, 
                 agent=None,
                 baseline_model=None,
                 output_dir: str = "experiments/evaluation_outputs"):
        """
        Initialize evaluator.
        
        Args:
            agent: STTAgent instance
            baseline_model: BaselineSTTModel instance
            output_dir: Directory to save evaluation results
        """
        self.agent = agent
        self.baseline_model = baseline_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[CorrectionResult] = []
        self.evaluator = STTEvaluator()
        
        logger.info("Agent Evaluator initialized")
    
    def evaluate_correction(self,
                           audio_path: str,
                           reference_transcript: Optional[str] = None,
                           enable_correction: bool = True) -> CorrectionResult:
        """
        Evaluate a single correction.
        
        Args:
            audio_path: Path to audio file
            reference_transcript: Ground truth transcription (optional)
            enable_correction: Whether to apply agent corrections
            
        Returns:
            CorrectionResult with metrics
        """
        from datetime import datetime
        
        # Get baseline transcription
        baseline_result = self.baseline_model.transcribe(audio_path)
        original_transcript = baseline_result['transcript']
        
        # Get agent transcription
        agent_result = self.agent.transcribe_with_agent(
            audio_path,
            enable_auto_correction=enable_correction
        )
        corrected_transcript = agent_result['transcript']
        
        # Calculate WER/CER if reference available
        original_wer = 0.0
        original_cer = 0.0
        corrected_wer = 0.0
        corrected_cer = 0.0
        
        if reference_transcript:
            original_wer = self.evaluator.calculate_wer(reference_transcript, original_transcript)
            original_cer = self.evaluator.calculate_cer(reference_transcript, original_transcript)
            corrected_wer = self.evaluator.calculate_wer(reference_transcript, corrected_transcript)
            corrected_cer = self.evaluator.calculate_cer(reference_transcript, corrected_transcript)
        
        # Calculate improvements (negative means correction made it worse)
        wer_improvement = original_wer - corrected_wer
        cer_improvement = original_cer - corrected_cer
        was_improved = (wer_improvement > 0.01)  # 0.01 threshold for significance
        
        result = CorrectionResult(
            audio_path=str(audio_path),
            original_transcript=original_transcript,
            corrected_transcript=corrected_transcript,
            reference_transcript=reference_transcript,
            original_wer=original_wer,
            corrected_wer=corrected_wer,
            original_cer=original_cer,
            corrected_cer=corrected_cer,
            wer_improvement=wer_improvement,
            cer_improvement=cer_improvement,
            was_improved=was_improved,
            errors_detected=agent_result['error_detection']['error_count'],
            error_score=agent_result['error_detection']['error_score'],
            agent_confidence=None,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def evaluate_batch(self,
                      audio_paths: List[str],
                      reference_transcripts: Optional[List[str]] = None,
                      enable_correction: bool = True) -> Dict:
        """
        Evaluate a batch of corrections.
        
        Args:
            audio_paths: List of audio file paths
            reference_transcripts: List of ground truth transcriptions (optional)
            enable_correction: Whether to apply corrections
            
        Returns:
            Dictionary with batch evaluation metrics
        """
        logger.info(f"Evaluating {len(audio_paths)} audio files...")
        
        if reference_transcripts:
            assert len(audio_paths) == len(reference_transcripts), \
                "Audio paths and reference transcripts must have same length"
        
        results = []
        for idx, audio_path in enumerate(audio_paths):
            ref = reference_transcripts[idx] if reference_transcripts else None
            try:
                result = self.evaluate_correction(
                    audio_path=audio_path,
                    reference_transcript=ref,
                    enable_correction=enable_correction
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {audio_path}: {e}")
                continue
        
        return self._calculate_batch_metrics(results)
    
    def _calculate_batch_metrics(self, results: List[CorrectionResult]) -> Dict:
        """Calculate aggregate metrics from batch results"""
        
        if not results:
            return {"error": "No valid results"}
        
        improved_count = sum(1 for r in results if r.was_improved)
        worsened_count = sum(1 for r in results if not r.was_improved and r.wer_improvement < -0.01)
        neutral_count = len(results) - improved_count - worsened_count
        
        wer_improvements = [r.wer_improvement for r in results]
        cer_improvements = [r.cer_improvement for r in results]
        
        return {
            "total_samples": len(results),
            "improvement_metrics": {
                "improved_count": improved_count,
                "improvement_rate": improved_count / len(results),
                "worsened_count": worsened_count,
                "worsening_rate": worsened_count / len(results),
                "neutral_count": neutral_count,
                "neutral_rate": neutral_count / len(results)
            },
            "wer_metrics": {
                "mean_improvement": np.mean(wer_improvements),
                "std_improvement": np.std(wer_improvements),
                "min_improvement": np.min(wer_improvements),
                "max_improvement": np.max(wer_improvements),
                "median_improvement": np.median(wer_improvements)
            },
            "cer_metrics": {
                "mean_improvement": np.mean(cer_improvements),
                "std_improvement": np.std(cer_improvements),
                "min_improvement": np.min(cer_improvements),
                "max_improvement": np.max(cer_improvements),
                "median_improvement": np.median(cer_improvements)
            },
            "consistency": {
                "wer_improvement_std": np.std(wer_improvements),
                "cer_improvement_std": np.std(cer_improvements)
            },
            "raw_results": [r.to_dict() for r in results]
        }
    
    def get_correction_accuracy(self) -> Dict:
        """Get overall correction accuracy statistics"""
        
        if not self.results:
            return {"error": "No evaluation results"}
        
        improved = sum(1 for r in self.results if r.was_improved)
        
        return {
            "total_corrections": len(self.results),
            "successful_corrections": improved,
            "accuracy_rate": improved / len(self.results),
            "mean_wer_improvement": np.mean([r.wer_improvement for r in self.results]),
            "mean_cer_improvement": np.mean([r.cer_improvement for r in self.results]),
            "consistency_wer_std": np.std([r.wer_improvement for r in self.results]),
            "consistency_cer_std": np.std([r.cer_improvement for r in self.results])
        }
    
    def save_results(self, filename: str = "agent_evaluation_results.json"):
        """Save evaluation results to JSON"""
        
        results_data = {
            "accuracy": self.get_correction_accuracy(),
            "detailed_results": [r.to_dict() for r in self.results],
            "timestamp": datetime.now().isoformat()
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"✅ Results saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print evaluation summary"""
        
        accuracy = self.get_correction_accuracy()
        
        print("\n" + "="*70)
        print("AGENT EVALUATOR - CORRECTION ACCURACY REPORT")
        print("="*70)
        print(f"\nTotal corrections evaluated: {accuracy['total_corrections']}")
        print(f"Successful corrections: {accuracy['successful_corrections']}")
        print(f"Accuracy rate: {accuracy['accuracy_rate']:.2%}")
        print(f"\nMean WER improvement: {accuracy['mean_wer_improvement']:.4f}")
        print(f"Mean CER improvement: {accuracy['mean_cer_improvement']:.4f}")
        print(f"\nWER improvement consistency (std): {accuracy['consistency_wer_std']:.4f}")
        print(f"CER improvement consistency (std): {accuracy['consistency_cer_std']:.4f}")
        print("="*70 + "\n")
