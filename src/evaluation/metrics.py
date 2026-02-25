"""
Unified evaluation module for STT models: WER and CER.
Supports streaming predictions (inference) and batch/offline test sets.
"""

from jiwer import wer, cer
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_pairs_from_json(path: Path, ref_key: str, hyp_key: str) -> tuple:
    """Load (references, hypotheses) from JSON array or {'samples': [...]}."""
    with open(path, "r") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("samples", data.get("data", []))
    if not items:
        return [], []
    refs = [item.get(ref_key, item.get("reference", "")) for item in items]
    hyps = [item.get(hyp_key, item.get("hypothesis", "")) for item in items]
    return refs, hyps


def _load_pairs_from_jsonl(path: Path, ref_key: str, hyp_key: str) -> tuple:
    """Load (references, hypotheses) from JSONL."""
    refs, hyps = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            refs.append(item.get(ref_key, item.get("reference", "")))
            hyps.append(item.get(hyp_key, item.get("hypothesis", "")))
    return refs, hyps


def _load_pairs_from_csv(
    path: Path, ref_key: str, hyp_key: str
) -> tuple:
    """Load (references, hypotheses) from CSV. ref_key/hyp_key are column names."""
    refs, hyps = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs.append(row.get(ref_key, row.get("reference", "")))
            hyps.append(row.get(hyp_key, row.get("hypothesis", "")))
    return refs, hyps


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


class EvaluationModule:
    """
    Unified evaluation: streaming (inference) and batch/offline test sets.
    Use add_prediction() for streaming; evaluate_batch() or evaluate_from_file() for batch.
    """

    def __init__(self):
        self._references: List[str] = []
        self._hypotheses: List[str] = []
        self._stt_evaluator = STTEvaluator()

    def add_prediction(self, reference: str, hypothesis: str) -> None:
        """
        Add a single reference/hypothesis pair (streaming inference).
        Call get_metrics() for current corpus-level WER/CER; per-sample results in .results.
        """
        self._references.append(reference)
        self._hypotheses.append(hypothesis)
        self._stt_evaluator.results.append({
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": self._stt_evaluator.calculate_wer(reference, hypothesis),
            "cer": self._stt_evaluator.calculate_cer(reference, hypothesis),
        })

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Single-pair WER (delegates to STTEvaluator)."""
        return self._stt_evaluator.calculate_wer(reference, hypothesis)

    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Single-pair CER (delegates to STTEvaluator)."""
        return self._stt_evaluator.calculate_cer(reference, hypothesis)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return current metrics over all pairs added so far (streaming).
        Returns WER, CER, num_samples; empty dict if no pairs.
        """
        if not self._references or not self._hypotheses:
            return {}
        n = min(len(self._references), len(self._hypotheses))
        refs = self._references[:n]
        hyps = self._hypotheses[:n]
        return {
            "wer": wer(refs, hyps),
            "cer": cer(refs, hyps),
            "num_samples": n,
        }

    def reset(self) -> None:
        """Clear accumulated streaming state."""
        self._references.clear()
        self._hypotheses.clear()
        self._stt_evaluator.results = []

    def evaluate_batch(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of reference/hypothesis pairs (offline test set).
        Returns WER, CER, num_samples, and populates detailed results on internal STTEvaluator.
        """
        result = self._stt_evaluator.evaluate_batch(references, hypotheses)
        return {
            "wer": result["wer"],
            "cer": result["cer"],
            "num_samples": result["num_samples"],
        }

    def evaluate_from_file(
        self,
        path: Union[str, Path],
        reference_key: str = "reference",
        hypothesis_key: str = "hypothesis",
    ) -> Dict[str, Any]:
        """
        Load reference/hypothesis pairs from a batch file and compute metrics.
        Supports .json (array or {'samples': [...]}), .jsonl, and .csv.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            refs, hyps = _load_pairs_from_json(path, reference_key, hypothesis_key)
        elif suffix == ".jsonl":
            refs, hyps = _load_pairs_from_jsonl(path, reference_key, hypothesis_key)
        elif suffix == ".csv":
            refs, hyps = _load_pairs_from_csv(path, reference_key, hypothesis_key)
        else:
            raise ValueError(
                f"Unsupported batch file format: {suffix}. Use .json, .jsonl, or .csv"
            )
        if len(refs) != len(hyps):
            logger.warning(
                f"Length mismatch: {len(refs)} references, {len(hyps)} hypotheses; truncating to min"
            )
            n = min(len(refs), len(hyps))
            refs, hyps = refs[:n], hyps[:n]
        if not refs:
            logger.warning("No pairs loaded from %s", path)
            return {}
        return self.evaluate_batch(refs, hyps)

    @property
    def results(self) -> List[Dict]:
        """Per-sample results from last evaluate_batch (or from streaming via get_metrics)."""
        return self._stt_evaluator.results

    def save_results(self, output_path: str) -> Optional[Dict]:
        """Save detailed results (same as STTEvaluator.save_results)."""
        if not self._stt_evaluator.results:
            logger.warning("No results to save")
            return None
        return self._stt_evaluator.save_results(output_path)
