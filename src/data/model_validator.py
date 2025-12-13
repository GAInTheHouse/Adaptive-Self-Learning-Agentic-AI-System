"""
Model Validation System
Validates fine-tuned models against baseline using standardized evaluation sets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

from jiwer import wer, cer
from ..evaluation.metrics import STTEvaluator
from .metadata_tracker import MetadataTracker
from ..utils.gcs_utils import GCSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""
    model_id: str
    baseline_id: str
    
    # Aggregate metrics
    model_wer: float
    model_cer: float
    baseline_wer: float
    baseline_cer: float
    
    # Improvements
    wer_improvement: float
    cer_improvement: float
    
    # Statistical significance
    is_significant: bool
    p_value: Optional[float] = None
    
    # Detailed results
    num_samples: int = 0
    per_sample_results: List[Dict] = None
    
    # Validation metadata
    validation_set: str = ""
    timestamp: str = ""
    passed: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        if self.per_sample_results is None:
            result['per_sample_results'] = []
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        return convert_numpy_types(result)


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    # Performance thresholds
    min_wer_improvement: float = 0.0  # Must not degrade
    min_cer_improvement: float = 0.0
    
    # Significance testing
    require_significance: bool = True
    significance_alpha: float = 0.05
    
    # Quality gates
    max_wer_degradation_rate: float = 0.1  # Max 10% of samples can degrade
    require_no_major_degradation: bool = True  # No sample should have >50% degradation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ModelValidator:
    """
    Validates fine-tuned models against baseline using standardized tests.
    
    Features:
    - Compare model performance against baseline
    - Statistical significance testing
    - Regression detection
    - Multi-metric evaluation (WER, CER)
    - Standardized evaluation sets
    """
    
    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        evaluation_data_path: Optional[str] = None,
        storage_dir: str = "data/model_validation",
        use_gcs: bool = True,
        project_id: str = "stt-agentic-ai-2025"
    ):
        """
        Initialize model validator.
        
        Args:
            config: Validation configuration
            evaluation_data_path: Path to standardized evaluation dataset
            storage_dir: Directory for validation results
            use_gcs: Whether to use Google Cloud Storage
            project_id: GCP project ID
        """
        self.config = config or ValidationConfig()
        self.evaluation_data_path = evaluation_data_path
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-systems
        self.metadata_tracker = MetadataTracker(
            use_gcs=use_gcs,
            project_id=project_id
        )
        
        # GCS integration
        self.use_gcs = use_gcs
        self.gcs_manager = None
        if use_gcs:
            try:
                self.gcs_manager = GCSManager(project_id, "stt-project-models")
                logger.info("GCS integration enabled for validation")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS: {e}")
                self.use_gcs = False
        
        # Results storage
        self.results_file = self.storage_dir / "validation_results.jsonl"
        self.results: Dict[str, ValidationResult] = {}
        self._load_results()
        
        logger.info("Model Validator initialized")
    
    def _load_results(self):
        """Load validation results history."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result_data = json.loads(line)
                        result = ValidationResult(**result_data)
                        result_id = f"{result.model_id}_{result.timestamp}"
                        self.results[result_id] = result
            logger.info(f"Loaded {len(self.results)} validation results")
    
    def _save_result(self, result: ValidationResult):
        """Save validation result."""
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            try:
                gcs_path = "validation/validation_results.jsonl"
                self.gcs_manager.upload_file(str(self.results_file), gcs_path)
            except Exception as e:
                logger.error(f"Failed to sync results to GCS: {e}")
    
    def load_evaluation_set(self, evaluation_set_path: str) -> List[Dict]:
        """
        Load standardized evaluation set.
        
        Args:
            evaluation_set_path: Path to evaluation dataset
            
        Returns:
            List of evaluation samples with audio_path and reference
        """
        eval_path = Path(evaluation_set_path)
        samples = []
        
        if eval_path.suffix == '.jsonl':
            with open(eval_path, 'r') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        elif eval_path.suffix == '.json':
            with open(eval_path, 'r') as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get('samples', [])
        else:
            raise ValueError(f"Unsupported file format: {eval_path.suffix}")
        
        logger.info(f"Loaded {len(samples)} evaluation samples from {eval_path.name}")
        return samples
    
    def validate_model(
        self,
        model_id: str,
        model_transcribe_fn: callable,
        baseline_id: str,
        baseline_transcribe_fn: callable,
        evaluation_set: Optional[List[Dict]] = None,
        evaluation_set_path: Optional[str] = None,
        validation_set_name: str = "default"
    ) -> ValidationResult:
        """
        Validate model against baseline.
        
        Args:
            model_id: Model identifier
            model_transcribe_fn: Function(audio_path) -> str that returns transcript
            baseline_id: Baseline model identifier
            baseline_transcribe_fn: Baseline transcription function
            evaluation_set: Pre-loaded evaluation samples
            evaluation_set_path: Path to evaluation dataset
            validation_set_name: Name of validation set
            
        Returns:
            ValidationResult with comparison metrics
        """
        logger.info(f"Validating model '{model_id}' against baseline '{baseline_id}'...")
        
        # Load evaluation set
        if evaluation_set is None:
            if evaluation_set_path is None:
                evaluation_set_path = self.evaluation_data_path
            if evaluation_set_path is None:
                raise ValueError("Must provide either evaluation_set or evaluation_set_path")
            evaluation_set = self.load_evaluation_set(evaluation_set_path)
        
        if not evaluation_set:
            raise ValueError("Evaluation set is empty")
        
        # Run evaluations
        logger.info(f"Running evaluation on {len(evaluation_set)} samples...")
        
        model_transcripts = []
        baseline_transcripts = []
        references = []
        per_sample_results = []
        
        for idx, sample in enumerate(evaluation_set):
            try:
                audio_path = sample.get('audio_path') or sample.get('audio')
                reference = sample.get('reference') or sample.get('text') or sample.get('target_text')
                
                if not audio_path or not reference:
                    logger.warning(f"Sample {idx} missing audio_path or reference, skipping")
                    continue
                
                # Get transcripts
                model_transcript = model_transcribe_fn(audio_path)
                baseline_transcript = baseline_transcribe_fn(audio_path)
                
                # Handle case where functions return dict with 'transcript' key
                if isinstance(model_transcript, dict):
                    model_transcript = model_transcript.get('transcript', '')
                if isinstance(baseline_transcript, dict):
                    baseline_transcript = baseline_transcript.get('transcript', '')
                
                model_transcripts.append(model_transcript)
                baseline_transcripts.append(baseline_transcript)
                references.append(reference)
                
                # Calculate per-sample metrics
                sample_model_wer = wer(reference, model_transcript)
                sample_baseline_wer = wer(reference, baseline_transcript)
                sample_model_cer = cer(reference, model_transcript)
                sample_baseline_cer = cer(reference, baseline_transcript)
                
                per_sample_results.append({
                    'audio_path': audio_path,
                    'reference': reference,
                    'model_transcript': model_transcript,
                    'baseline_transcript': baseline_transcript,
                    'model_wer': sample_model_wer,
                    'baseline_wer': sample_baseline_wer,
                    'model_cer': sample_model_cer,
                    'baseline_cer': sample_baseline_cer,
                    'wer_improvement': sample_baseline_wer - sample_model_wer,
                    'cer_improvement': sample_baseline_cer - sample_model_cer
                })
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"  Processed {idx + 1}/{len(evaluation_set)} samples")
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
        
        if not references:
            raise ValueError("No valid samples could be evaluated")
        
        # Calculate aggregate metrics
        model_wer_score = wer(references, model_transcripts)
        baseline_wer_score = wer(references, baseline_transcripts)
        model_cer_score = cer(references, model_transcripts)
        baseline_cer_score = cer(references, baseline_transcripts)
        
        wer_improvement = baseline_wer_score - model_wer_score
        cer_improvement = baseline_cer_score - model_cer_score
        
        # Statistical significance testing
        is_significant, p_value = self._test_significance(per_sample_results)
        
        # Validation checks
        passed = self._check_validation_criteria(
            wer_improvement=wer_improvement,
            cer_improvement=cer_improvement,
            is_significant=is_significant,
            per_sample_results=per_sample_results
        )
        
        # Create result
        result = ValidationResult(
            model_id=model_id,
            baseline_id=baseline_id,
            model_wer=model_wer_score,
            model_cer=model_cer_score,
            baseline_wer=baseline_wer_score,
            baseline_cer=baseline_cer_score,
            wer_improvement=wer_improvement,
            cer_improvement=cer_improvement,
            is_significant=is_significant,
            p_value=p_value,
            num_samples=len(references),
            per_sample_results=per_sample_results,
            validation_set=validation_set_name,
            timestamp=datetime.now().isoformat(),
            passed=passed
        )
        
        # Save result
        result_id = f"{model_id}_{result.timestamp}"
        self.results[result_id] = result
        self._save_result(result)
        
        # Record in metadata tracker
        self.metadata_tracker.record_performance(
            wer=model_wer_score,
            cer=model_cer_score,
            metadata={
                'model_id': model_id,
                'baseline_id': baseline_id,
                'validation_set': validation_set_name,
                'passed': passed,
                'wer_improvement': wer_improvement,
                'cer_improvement': cer_improvement
            }
        )
        
        # Print summary
        self._print_validation_summary(result)
        
        return result
    
    def _test_significance(
        self,
        per_sample_results: List[Dict]
    ) -> Tuple[bool, float]:
        """
        Test statistical significance of improvements.
        
        Uses paired t-test on per-sample WER improvements.
        """
        if len(per_sample_results) < 2:
            return False, 1.0
        
        try:
            from scipy import stats
            
            # Extract WER improvements
            improvements = [r['wer_improvement'] for r in per_sample_results]
            
            # Paired t-test (testing if mean improvement > 0)
            t_stat, p_value = stats.ttest_1samp(improvements, 0)
            
            # One-tailed test (we care if model is better, not just different)
            p_value = float(p_value / 2 if t_stat > 0 else 1.0)
            
            is_significant = bool(p_value < self.config.significance_alpha)
            
            return is_significant, p_value
            
        except ImportError:
            logger.warning("scipy not available, skipping significance test")
            return False, 1.0
        except Exception as e:
            logger.error(f"Error in significance test: {e}")
            return False, 1.0
    
    def _check_validation_criteria(
        self,
        wer_improvement: float,
        cer_improvement: float,
        is_significant: bool,
        per_sample_results: List[Dict]
    ) -> bool:
        """Check if model meets validation criteria."""
        
        # Check minimum improvement
        if wer_improvement < self.config.min_wer_improvement:
            logger.warning(f"❌ WER improvement ({wer_improvement:.4f}) below threshold ({self.config.min_wer_improvement})")
            return False
        
        if cer_improvement < self.config.min_cer_improvement:
            logger.warning(f"❌ CER improvement ({cer_improvement:.4f}) below threshold ({self.config.min_cer_improvement})")
            return False
        
        # Check significance
        if self.config.require_significance and not is_significant:
            logger.warning("❌ Improvement not statistically significant")
            return False
        
        # Check degradation rate
        degraded_samples = sum(1 for r in per_sample_results if r['wer_improvement'] < -0.01)
        degradation_rate = degraded_samples / len(per_sample_results)
        
        if degradation_rate > self.config.max_wer_degradation_rate:
            logger.warning(f"❌ Degradation rate ({degradation_rate:.2%}) exceeds threshold ({self.config.max_wer_degradation_rate:.2%})")
            return False
        
        # Check for major degradations
        if self.config.require_no_major_degradation:
            major_degradations = [
                r for r in per_sample_results
                if r['wer_improvement'] < -0.5  # 50% degradation
            ]
            if major_degradations:
                logger.warning(f"❌ Found {len(major_degradations)} samples with major degradation (>50%)")
                return False
        
        return True
    
    def _print_validation_summary(self, result: ValidationResult):
        """Print validation summary."""
        print("\n" + "="*80)
        print("MODEL VALIDATION REPORT")
        print("="*80)
        print(f"\nModel ID: {result.model_id}")
        print(f"Baseline ID: {result.baseline_id}")
        print(f"Validation Set: {result.validation_set}")
        print(f"Samples: {result.num_samples}")
        print(f"\n{'Metric':<20} {'Baseline':<15} {'Model':<15} {'Improvement':<15}")
        print("-"*65)
        print(f"{'WER':<20} {result.baseline_wer:<15.4f} {result.model_wer:<15.4f} {result.wer_improvement:<15.4f}")
        print(f"{'CER':<20} {result.baseline_cer:<15.4f} {result.model_cer:<15.4f} {result.cer_improvement:<15.4f}")
        print(f"\nStatistical Significance: {'Yes' if result.is_significant else 'No'}")
        if result.p_value is not None:
            print(f"P-value: {result.p_value:.4f}")
        print(f"\n{'VALIDATION PASSED ✅' if result.passed else 'VALIDATION FAILED ❌'}")
        print("="*80 + "\n")
    
    def compare_multiple_models(
        self,
        model_configs: List[Dict],
        baseline_config: Dict,
        evaluation_set: Optional[List[Dict]] = None
    ) -> List[ValidationResult]:
        """
        Compare multiple models against baseline.
        
        Args:
            model_configs: List of dicts with 'id' and 'transcribe_fn'
            baseline_config: Dict with 'id' and 'transcribe_fn'
            evaluation_set: Evaluation samples
            
        Returns:
            List of ValidationResults
        """
        results = []
        
        for model_config in model_configs:
            result = self.validate_model(
                model_id=model_config['id'],
                model_transcribe_fn=model_config['transcribe_fn'],
                baseline_id=baseline_config['id'],
                baseline_transcribe_fn=baseline_config['transcribe_fn'],
                evaluation_set=evaluation_set
            )
            results.append(result)
        
        # Print comparison summary
        self._print_comparison_summary(results)
        
        return results
    
    def _print_comparison_summary(self, results: List[ValidationResult]):
        """Print comparison summary for multiple models."""
        print("\n" + "="*80)
        print("MULTI-MODEL COMPARISON")
        print("="*80)
        print(f"\n{'Model ID':<30} {'WER':<12} {'CER':<12} {'WER Δ':<12} {'Passed':<10}")
        print("-"*76)
        
        for result in results:
            passed_str = "✅" if result.passed else "❌"
            print(f"{result.model_id:<30} {result.model_wer:<12.4f} {result.model_cer:<12.4f} "
                  f"{result.wer_improvement:+<12.4f} {passed_str:<10}")
        
        print("="*80 + "\n")
    
    def get_best_model(
        self,
        metric: str = 'wer',
        only_passed: bool = True
    ) -> Optional[ValidationResult]:
        """
        Get best performing model from validation results.
        
        Args:
            metric: Metric to optimize ('wer' or 'cer')
            only_passed: Only consider models that passed validation
            
        Returns:
            Best ValidationResult or None
        """
        results = list(self.results.values())
        
        if only_passed:
            results = [r for r in results if r.passed]
        
        if not results:
            return None
        
        if metric == 'wer':
            best = min(results, key=lambda r: r.model_wer)
        elif metric == 'cer':
            best = min(results, key=lambda r: r.model_cer)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best
    
    def generate_validation_report(
        self,
        output_path: Optional[str] = None
    ) -> Dict:
        """Generate comprehensive validation report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'total_validations': len(self.results),
            'passed_validations': sum(1 for r in self.results.values() if r.passed),
            'failed_validations': sum(1 for r in self.results.values() if not r.passed),
            'recent_validations': []
        }
        
        # Get recent validations
        recent = sorted(self.results.values(), key=lambda r: r.timestamp, reverse=True)[:10]
        for result in recent:
            report['recent_validations'].append({
                'model_id': result.model_id,
                'timestamp': result.timestamp,
                'passed': result.passed,
                'wer_improvement': result.wer_improvement,
                'cer_improvement': result.cer_improvement,
                'is_significant': result.is_significant
            })
        
        # Find best models
        best_wer = self.get_best_model('wer')
        best_cer = self.get_best_model('cer')
        
        if best_wer:
            report['best_wer_model'] = {
                'model_id': best_wer.model_id,
                'wer': best_wer.model_wer,
                'timestamp': best_wer.timestamp
            }
        
        if best_cer:
            report['best_cer_model'] = {
                'model_id': best_cer.model_id,
                'cer': best_cer.model_cer,
                'timestamp': best_cer.timestamp
            }
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Validation report saved to {output_path}")
        
        return report


