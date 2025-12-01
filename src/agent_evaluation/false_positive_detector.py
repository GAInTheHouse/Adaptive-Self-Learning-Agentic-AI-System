"""
False Positive Detector - Identify Harmful Corrections

Detects when agent corrections make transcriptions worse.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from jiwer import wer, cer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FalsePositive:
    """Represents a false positive correction"""
    audio_path: str
    original_transcript: str
    corrected_transcript: str
    reference_transcript: Optional[str]
    
    original_wer: float
    corrected_wer: float
    wer_degradation: float  # positive = worse
    
    original_cer: float
    corrected_cer: float
    cer_degradation: float
    
    error_type: str  # What kind of error was detected
    error_confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'audio_path': self.audio_path,
            'original_transcript': self.original_transcript,
            'corrected_transcript': self.corrected_transcript,
            'reference_transcript': self.reference_transcript,
            'original_wer': self.original_wer,
            'corrected_wer': self.corrected_wer,
            'wer_degradation': self.wer_degradation,
            'original_cer': self.original_cer,
            'corrected_cer': self.corrected_cer,
            'cer_degradation': self.cer_degradation,
            'error_type': self.error_type,
            'error_confidence': self.error_confidence
        }


class FalsePositiveDetector:
    """
    Detects false positive corrections (corrections that hurt accuracy).
    
    A false positive occurs when:
    - Agent detects an "error" and "corrects" it
    - But the correction actually makes WER/CER worse
    
    Key metrics:
    - False positive rate: % of corrections that hurt
    - Average degradation: How much worse corrections made things
    - Error type breakdown: Which error types cause false positives most
    """
    
    def __init__(self, output_dir: str = "experiments/evaluation_outputs"):
        """Initialize detector"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.false_positives: List[FalsePositive] = []
        
        logger.info("False Positive Detector initialized")
    
    def detect_false_positive(self,
                             original_transcript: str,
                             corrected_transcript: str,
                             reference_transcript: str,
                             error_type: str,
                             error_confidence: float,
                             audio_path: str = "") -> Optional[FalsePositive]:
        """
        Detect if a correction is a false positive.
        
        Args:
            original_transcript: Baseline output
            corrected_transcript: Agent-corrected output
            reference_transcript: Ground truth
            error_type: Type of error detected
            error_confidence: Confidence of error detection
            audio_path: Path to audio file
            
        Returns:
            FalsePositive object if correction hurt, None otherwise
        """
        
        original_wer = wer(reference_transcript, original_transcript)
        original_cer = cer(reference_transcript, original_transcript)
        
        corrected_wer = wer(reference_transcript, corrected_transcript)
        corrected_cer = cer(reference_transcript, corrected_transcript)
        
        wer_degradation = corrected_wer - original_wer
        cer_degradation = corrected_cer - original_cer
        
        # False positive if correction made things worse (threshold: 0.01)
        if wer_degradation > 0.01 or cer_degradation > 0.01:
            fp = FalsePositive(
                audio_path=audio_path,
                original_transcript=original_transcript,
                corrected_transcript=corrected_transcript,
                reference_transcript=reference_transcript,
                original_wer=original_wer,
                corrected_wer=corrected_wer,
                wer_degradation=wer_degradation,
                original_cer=original_cer,
                corrected_cer=corrected_cer,
                cer_degradation=cer_degradation,
                error_type=error_type,
                error_confidence=error_confidence
            )
            
            self.false_positives.append(fp)
            return fp
        
        return None
    
    def analyze_false_positives(self) -> Dict:
        """Analyze all detected false positives"""
        
        if not self.false_positives:
            return {
                "total_false_positives": 0,
                "false_positive_rate": 0.0
            }
        
        # Error type breakdown
        error_type_counts = {}
        for fp in self.false_positives:
            error_type_counts[fp.error_type] = \
                error_type_counts.get(fp.error_type, 0) + 1
        
        # WER degradation stats
        wer_degradations = [fp.wer_degradation for fp in self.false_positives]
        cer_degradations = [fp.cer_degradation for fp in self.false_positives]
        
        return {
            "total_false_positives": len(self.false_positives),
            "error_type_breakdown": error_type_counts,
            "wer_degradation": {
                "mean": np.mean(wer_degradations),
                "std": np.std(wer_degradations),
                "max": np.max(wer_degradations),
                "min": np.min(wer_degradations)
            },
            "cer_degradation": {
                "mean": np.mean(cer_degradations),
                "std": np.std(cer_degradations),
                "max": np.max(cer_degradations),
                "min": np.min(cer_degradations)
            },
            "error_type_analysis": {
                error_type: {
                    "count": count,
                    "average_degradation": np.mean([
                        fp.wer_degradation for fp in self.false_positives
                        if fp.error_type == error_type
                    ])
                }
                for error_type, count in error_type_counts.items()
            }
        }
    
    def save_analysis(self, filename: str = "false_positives_analysis.json"):
        """Save false positive analysis"""
        
        analysis = self.analyze_false_positives()
        analysis['detailed_false_positives'] = [
            fp.to_dict() for fp in self.false_positives
        ]
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"✅ Analysis saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print false positive summary"""
        
        analysis = self.analyze_false_positives()
        
        print("\n" + "="*70)
        print("FALSE POSITIVE DETECTION REPORT")
        print("="*70)
        print(f"\nTotal false positives: {analysis['total_false_positives']}")
        
        if analysis['total_false_positives'] > 0:
            print(f"\nWER degradation (mean): {analysis['wer_degradation']['mean']:.4f}")
            print(f"CER degradation (mean): {analysis['cer_degradation']['mean']:.4f}")
            
            print("\nError types causing false positives:")
            for error_type, stats in analysis['error_type_analysis'].items():
                print(f"  {error_type}: {stats['count']} cases " +
                      f"(avg degradation: {stats['average_degradation']:.4f})")
        
        print("="*70 + "\n")
