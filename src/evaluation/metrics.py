"""
Evaluation metrics for STT models: WER, CER, DER, Verb Error Rate, Domain Error Rate.
"""

from jiwer import wer, cer
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
import re
import nltk
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

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
    """Calculate WER, CER, DER, Verb Error Rate, and Domain Error Rate for STT predictions"""
    
    def __init__(self):
        self.results = []
        # Domain keywords for domain error rate (can be customized)
        self.domain_keywords = {
            'medical': ['patient', 'diagnosis', 'treatment', 'symptom', 'medication', 'doctor', 'hospital'],
            'legal': ['court', 'judge', 'defendant', 'plaintiff', 'attorney', 'testimony', 'evidence'],
            'technical': ['algorithm', 'implementation', 'function', 'variable', 'parameter', 'system'],
            'business': ['revenue', 'profit', 'customer', 'market', 'strategy', 'investment']
        }
    
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
    
    def calculate_der(
        self,
        reference_segments: List[Dict[str, Any]],
        hypothesis_segments: List[Dict[str, Any]],
        tolerance: float = 0.25
    ) -> float:
        """
        Calculate Diarization Error Rate (DER).
        
        DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Reference Duration
        
        Args:
            reference_segments: List of dicts with 'start', 'end', 'speaker' keys
            hypothesis_segments: List of dicts with 'start', 'end', 'speaker' keys
            tolerance: Tolerance collar in seconds (default 0.25)
        
        Returns:
            DER score (0-1, lower is better)
        """
        if not reference_segments:
            return 1.0 if hypothesis_segments else 0.0
        
        # Calculate total reference duration
        total_duration = sum(seg['end'] - seg['start'] for seg in reference_segments)
        if total_duration == 0:
            return 1.0
        
        # Initialize error counters
        missed_speech = 0.0
        false_alarm = 0.0
        speaker_confusion = 0.0
        
        # Create time-aligned segments
        ref_timeline = sorted(reference_segments, key=lambda x: x['start'])
        hyp_timeline = sorted(hypothesis_segments, key=lambda x: x['start'])
        
        # Simple overlap-based DER calculation
        ref_idx = 0
        hyp_idx = 0
        
        while ref_idx < len(ref_timeline) and hyp_idx < len(hyp_timeline):
            ref_seg = ref_timeline[ref_idx]
            hyp_seg = hyp_timeline[hyp_idx]
            
            # Check overlap
            overlap_start = max(ref_seg['start'], hyp_seg['start'])
            overlap_end = min(ref_seg['end'], hyp_seg['end'])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > tolerance:
                # Check speaker match
                if ref_seg.get('speaker') != hyp_seg.get('speaker'):
                    speaker_confusion += overlap_duration
            else:
                # Missed speech or false alarm
                if ref_seg['end'] < hyp_seg['start']:
                    missed_speech += ref_seg['end'] - ref_seg['start']
                    ref_idx += 1
                else:
                    false_alarm += hyp_seg['end'] - hyp_seg['start']
                    hyp_idx += 1
        
        # Remaining segments
        while ref_idx < len(ref_timeline):
            missed_speech += ref_timeline[ref_idx]['end'] - ref_timeline[ref_idx]['start']
            ref_idx += 1
        
        while hyp_idx < len(hyp_timeline):
            false_alarm += hyp_timeline[hyp_idx]['end'] - hyp_timeline[hyp_idx]['start']
            hyp_idx += 1
        
        der = (missed_speech + false_alarm + speaker_confusion) / total_duration
        return min(1.0, max(0.0, der))
    
    def extract_verbs(self, text: str) -> List[str]:
        """Extract verbs from text using NLTK POS tagging."""
        if NLTK_AVAILABLE:
            try:
                tokens = nltk.word_tokenize(text.lower())
                pos_tags = nltk.pos_tag(tokens)
                verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
                return verbs
            except Exception as e:
                logger.warning(f"Error extracting verbs with NLTK: {e}")
                # Fallback: simple verb detection
                return self._fallback_verb_extraction(text)
        else:
            # Fallback: simple verb detection when NLTK not available
            return self._fallback_verb_extraction(text)
    
    def _fallback_verb_extraction(self, text: str) -> List[str]:
        """Fallback verb extraction without NLTK."""
        words = text.lower().split()
        # Simple heuristic: words ending in common verb suffixes
        verb_suffixes = ('ed', 'ing', 's', 'es', 'en')
        verbs = [word for word in words if any(word.endswith(suffix) for suffix in verb_suffixes)]
        return verbs
    
    def calculate_verb_error_rate(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Verb Error Rate - measures accuracy of verb transcription.
        
        Args:
            reference: Ground truth transcription
            hypothesis: Model prediction
        
        Returns:
            Verb Error Rate (0-1, lower is better)
        """
        ref_verbs = set(self.extract_verbs(reference))
        hyp_verbs = set(self.extract_verbs(hypothesis))
        
        if not ref_verbs:
            return 0.0 if not hyp_verbs else 1.0
        
        # Calculate errors
        substitutions = len(ref_verbs - hyp_verbs)  # Missing verbs
        insertions = len(hyp_verbs - ref_verbs)  # Extra verbs
        
        total_errors = substitutions + insertions
        ver = total_errors / len(ref_verbs) if ref_verbs else 0.0
        
        return min(1.0, max(0.0, ver))
    
    def detect_domain(self, text: str) -> Optional[str]:
        """Detect domain of text based on keywords."""
        text_lower = text.lower()
        domain_scores = defaultdict(int)
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    domain_scores[domain] += 1
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def calculate_domain_error_rate(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Calculate Domain Error Rate - measures accuracy within specific domains.
        
        Args:
            references: List of ground truth transcriptions
            hypotheses: List of model predictions
        
        Returns:
            Dictionary with domain-specific error rates
        """
        domain_errors = defaultdict(lambda: {'total': 0, 'errors': 0})
        
        for ref, hyp in zip(references, hypotheses):
            domain = self.detect_domain(ref)
            if domain:
                domain_errors[domain]['total'] += 1
                # Calculate WER for this domain
                ref_wer = self.calculate_wer(ref, hyp)
                if ref_wer > 0:
                    domain_errors[domain]['errors'] += 1
        
        domain_rates = {}
        for domain, stats in domain_errors.items():
            if stats['total'] > 0:
                domain_rates[domain] = stats['errors'] / stats['total']
            else:
                domain_rates[domain] = 0.0
        
        return domain_rates
    
    def evaluate_batch(
        self,
        references: List[str],
        hypotheses: List[str],
        include_der: bool = False,
        reference_segments: Optional[List[List[Dict]]] = None,
        hypothesis_segments: Optional[List[List[Dict]]] = None,
        include_verb_rate: bool = True,
        include_domain_rate: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate batch of predictions with all metrics.
        
        Args:
            references: List of ground truth transcriptions
            hypotheses: List of model predictions
            include_der: Whether to calculate DER (requires segments)
            reference_segments: Optional list of speaker segments for reference
            hypothesis_segments: Optional list of speaker segments for hypothesis
            include_verb_rate: Whether to calculate Verb Error Rate
            include_domain_rate: Whether to calculate Domain Error Rate
        
        Returns:
            Dictionary with all metric scores
        """
        assert len(references) == len(hypotheses), \
            "References and hypotheses must have same length"
        
        # Calculate basic metrics
        wer_score = wer(references, hypotheses)
        cer_score = cer(references, hypotheses)
        
        results = {
            'wer': wer_score,
            'cer': cer_score,
            'num_samples': len(references)
        }
        
        # Calculate DER if segments provided
        if include_der and reference_segments and hypothesis_segments:
            assert len(reference_segments) == len(hypothesis_segments), \
                "Reference and hypothesis segments must have same length"
            der_scores = [
                self.calculate_der(ref_segs, hyp_segs)
                for ref_segs, hyp_segs in zip(reference_segments, hypothesis_segments)
            ]
            results['der'] = sum(der_scores) / len(der_scores) if der_scores else 0.0
        
        # Calculate Verb Error Rate
        if include_verb_rate:
            verb_rates = [
                self.calculate_verb_error_rate(ref, hyp)
                for ref, hyp in zip(references, hypotheses)
            ]
            results['verb_error_rate'] = sum(verb_rates) / len(verb_rates) if verb_rates else 0.0
        
        # Calculate Domain Error Rate
        if include_domain_rate:
            domain_rates = self.calculate_domain_error_rate(references, hypotheses)
            results['domain_error_rates'] = domain_rates
            if domain_rates:
                results['avg_domain_error_rate'] = sum(domain_rates.values()) / len(domain_rates)
        
        # Store detailed results
        for ref, hyp in zip(references, hypotheses):
            result_entry = {
                'reference': ref,
                'hypothesis': hyp,
                'wer': self.calculate_wer(ref, hyp),
                'cer': self.calculate_cer(ref, hyp)
            }
            if include_verb_rate:
                result_entry['verb_error_rate'] = self.calculate_verb_error_rate(ref, hyp)
            self.results.append(result_entry)
        
        return results
    
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
            'average_wer': sum(r['wer'] for r in self.results) / len(self.results) if self.results else 0.0,
            'average_cer': sum(r['cer'] for r in self.results) / len(self.results) if self.results else 0.0,
            'num_samples': len(self.results),
            'detailed_results': self.results
        }
        
        # Add verb error rate if available
        if 'verb_error_rate' in self.results[0] if self.results else False:
            summary['average_verb_error_rate'] = sum(
                r.get('verb_error_rate', 0) for r in self.results
            ) / len(self.results) if self.results else 0.0
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Average WER: {summary['average_wer']:.4f}")
        logger.info(f"Average CER: {summary['average_cer']:.4f}")
        if 'average_verb_error_rate' in summary:
            logger.info(f"Average Verb Error Rate: {summary['average_verb_error_rate']:.4f}")
        
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
