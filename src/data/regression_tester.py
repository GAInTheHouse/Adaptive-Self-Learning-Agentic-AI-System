"""
Regression Testing Framework
Prevents model degradation through continuous testing and monitoring.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

from jiwer import wer, cer
from .metadata_tracker import MetadataTracker
from ..utils.gcs_utils import GCSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegressionTest:
    """Defines a regression test."""
    test_id: str
    test_name: str
    test_type: str  # 'benchmark', 'critical_samples', 'edge_cases'
    test_data_path: str
    
    # Baseline performance
    baseline_wer: float
    baseline_cer: float
    baseline_version: str
    
    # Thresholds
    max_wer_degradation: float = 0.05  # 5% degradation allowed
    max_cer_degradation: float = 0.05
    
    # Metadata
    created_at: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RegressionTestResult:
    """Result of a regression test."""
    test_id: str
    model_version: str
    
    # Performance metrics
    model_wer: float
    model_cer: float
    baseline_wer: float
    baseline_cer: float
    
    # Degradation
    wer_degradation: float
    cer_degradation: float
    
    # Test result
    passed: bool
    failure_reason: Optional[str] = None
    
    # Per-sample details
    num_samples: int = 0
    failed_samples: List[Dict] = None
    
    # Metadata
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        if self.failed_samples is None:
            result['failed_samples'] = []
        return result


@dataclass
class RegressionConfig:
    """Configuration for regression testing."""
    # Test execution
    run_on_deploy: bool = True
    run_on_schedule: bool = False
    schedule_interval_hours: int = 24
    
    # Test criteria
    fail_on_any_degradation: bool = False
    fail_on_critical_degradation: bool = True
    critical_degradation_threshold: float = 0.1  # 10%
    
    # Sample-level checks
    max_failed_samples_rate: float = 0.05  # 5% of samples can fail
    sample_degradation_threshold: float = 0.2  # 20% degradation per sample
    
    # Alerting
    alert_on_failure: bool = True
    alert_on_warning: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class RegressionTester:
    """
    Regression testing framework to prevent model degradation.
    
    Features:
    - Define regression test suites
    - Track baseline performance
    - Detect degradation across model versions
    - Alert on performance regressions
    - Continuous monitoring
    """
    
    def __init__(
        self,
        config: Optional[RegressionConfig] = None,
        storage_dir: str = "data/regression_tests",
        use_gcs: bool = True,
        project_id: str = "stt-agentic-ai-2025"
    ):
        """
        Initialize regression tester.
        
        Args:
            config: Regression testing configuration
            storage_dir: Directory for test data and results
            use_gcs: Whether to use Google Cloud Storage
            project_id: GCP project ID
        """
        self.config = config or RegressionConfig()
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
                self.gcs_manager = GCSManager(project_id, "stt-project-datasets")
                logger.info("GCS integration enabled for regression testing")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS: {e}")
                self.use_gcs = False
        
        # Test registry
        self.tests_file = self.storage_dir / "regression_tests.json"
        self.tests: Dict[str, RegressionTest] = {}
        self._load_tests()
        
        # Results storage
        self.results_file = self.storage_dir / "regression_results.jsonl"
        self.results: List[RegressionTestResult] = []
        self._load_results()
        
        logger.info("Regression Tester initialized")
        logger.info(f"  Registered tests: {len(self.tests)}")
    
    def _load_tests(self):
        """Load test registry."""
        if self.tests_file.exists():
            try:
                with open(self.tests_file, 'r') as f:
                    tests_data = json.load(f)
                    for test_id, test_data in tests_data.items():
                        self.tests[test_id] = RegressionTest(**test_data)
                logger.info(f"Loaded {len(self.tests)} regression tests")
            except Exception as e:
                logger.error(f"Failed to load tests: {e}")
    
    def _save_tests(self):
        """Save test registry."""
        tests_data = {
            test_id: test.to_dict()
            for test_id, test in self.tests.items()
        }
        
        with open(self.tests_file, 'w') as f:
            json.dump(tests_data, f, indent=2)
    
    def _load_results(self):
        """Load test results history."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result_data = json.loads(line)
                        self.results.append(RegressionTestResult(**result_data))
            logger.info(f"Loaded {len(self.results)} test results")
    
    def _save_result(self, result: RegressionTestResult):
        """Save test result."""
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            try:
                gcs_path = "regression_tests/results.jsonl"
                self.gcs_manager.upload_file(str(self.results_file), gcs_path)
            except Exception as e:
                logger.error(f"Failed to sync results to GCS: {e}")
    
    def register_test(
        self,
        test_name: str,
        test_type: str,
        test_data_path: str,
        baseline_wer: float,
        baseline_cer: float,
        baseline_version: str,
        max_wer_degradation: float = 0.05,
        max_cer_degradation: float = 0.05,
        description: str = ""
    ) -> str:
        """
        Register a regression test.
        
        Args:
            test_name: Test name
            test_type: Type of test ('benchmark', 'critical_samples', 'edge_cases')
            test_data_path: Path to test data
            baseline_wer: Baseline WER
            baseline_cer: Baseline CER
            baseline_version: Baseline model version
            max_wer_degradation: Maximum allowed WER degradation
            max_cer_degradation: Maximum allowed CER degradation
            description: Test description
            
        Returns:
            Test ID
        """
        test_id = f"test_{len(self.tests) + 1:03d}_{test_type}"
        
        test = RegressionTest(
            test_id=test_id,
            test_name=test_name,
            test_type=test_type,
            test_data_path=test_data_path,
            baseline_wer=baseline_wer,
            baseline_cer=baseline_cer,
            baseline_version=baseline_version,
            max_wer_degradation=max_wer_degradation,
            max_cer_degradation=max_cer_degradation,
            created_at=datetime.now().isoformat(),
            description=description
        )
        
        self.tests[test_id] = test
        self._save_tests()
        
        logger.info(f"Registered regression test: {test_id} ({test_name})")
        return test_id
    
    def load_test_data(self, test_data_path: str) -> List[Dict]:
        """Load test data from file."""
        path = Path(test_data_path)
        samples = []
        
        if not path.exists():
            logger.error(f"Test data not found: {test_data_path}")
            return samples
        
        if path.suffix == '.jsonl':
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get('samples', [])
        
        return samples
    
    def run_test(
        self,
        test_id: str,
        model_version: str,
        model_transcribe_fn: Callable
    ) -> RegressionTestResult:
        """
        Run a regression test.
        
        Args:
            test_id: Test ID
            model_version: Model version being tested
            model_transcribe_fn: Function(audio_path) -> str
            
        Returns:
            RegressionTestResult
        """
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        logger.info(f"Running regression test: {test.test_name} on {model_version}")
        
        # Load test data
        test_samples = self.load_test_data(test.test_data_path)
        if not test_samples:
            raise ValueError(f"No test data found for test: {test_id}")
        
        # Run model on test data
        references = []
        hypotheses = []
        failed_samples = []
        
        for idx, sample in enumerate(test_samples):
            try:
                audio_path = sample.get('audio_path') or sample.get('audio')
                reference = sample.get('reference') or sample.get('text') or sample.get('target_text')
                
                if not audio_path or not reference:
                    logger.warning(f"Sample {idx} missing required fields, skipping")
                    continue
                
                # Get transcript
                transcript = model_transcribe_fn(audio_path)
                
                # Handle dict return
                if isinstance(transcript, dict):
                    transcript = transcript.get('transcript', '')
                
                references.append(reference)
                hypotheses.append(transcript)
                
                # Check per-sample degradation
                sample_wer = wer(reference, transcript)
                baseline_sample_wer = sample.get('baseline_wer', test.baseline_wer)
                sample_degradation = sample_wer - baseline_sample_wer
                
                if sample_degradation > self.config.sample_degradation_threshold:
                    failed_samples.append({
                        'audio_path': audio_path,
                        'reference': reference,
                        'transcript': transcript,
                        'wer': sample_wer,
                        'baseline_wer': baseline_sample_wer,
                        'degradation': sample_degradation
                    })
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
        
        if not references:
            raise ValueError("No valid samples could be evaluated")
        
        # Calculate aggregate metrics
        model_wer = wer(references, hypotheses)
        model_cer = cer(references, hypotheses)
        
        wer_degradation = model_wer - test.baseline_wer
        cer_degradation = model_cer - test.baseline_cer
        
        # Determine pass/fail
        passed = True
        failure_reason = None
        
        # Check WER degradation
        if wer_degradation > test.max_wer_degradation:
            passed = False
            failure_reason = f"WER degradation ({wer_degradation:.4f}) exceeds threshold ({test.max_wer_degradation})"
        
        # Check CER degradation
        if cer_degradation > test.max_cer_degradation:
            passed = False
            if failure_reason:
                failure_reason += f"; CER degradation ({cer_degradation:.4f}) exceeds threshold ({test.max_cer_degradation})"
            else:
                failure_reason = f"CER degradation ({cer_degradation:.4f}) exceeds threshold ({test.max_cer_degradation})"
        
        # Check critical degradation
        if self.config.fail_on_critical_degradation:
            if wer_degradation > self.config.critical_degradation_threshold:
                passed = False
                if not failure_reason:
                    failure_reason = f"Critical WER degradation detected: {wer_degradation:.4f}"
        
        # Check failed samples rate
        failed_rate = len(failed_samples) / len(references)
        if failed_rate > self.config.max_failed_samples_rate:
            passed = False
            if not failure_reason:
                failure_reason = f"Failed samples rate ({failed_rate:.2%}) exceeds threshold ({self.config.max_failed_samples_rate:.2%})"
        
        # Create result
        result = RegressionTestResult(
            test_id=test_id,
            model_version=model_version,
            model_wer=model_wer,
            model_cer=model_cer,
            baseline_wer=test.baseline_wer,
            baseline_cer=test.baseline_cer,
            wer_degradation=wer_degradation,
            cer_degradation=cer_degradation,
            passed=passed,
            failure_reason=failure_reason,
            num_samples=len(references),
            failed_samples=failed_samples,
            timestamp=datetime.now().isoformat()
        )
        
        # Save result
        self.results.append(result)
        self._save_result(result)
        
        # Record in metadata
        self.metadata_tracker.record_performance(
            wer=model_wer,
            cer=model_cer,
            metadata={
                'test_type': 'regression',
                'test_id': test_id,
                'model_version': model_version,
                'passed': passed,
                'wer_degradation': wer_degradation
            }
        )
        
        # Print result
        self._print_test_result(test, result)
        
        return result
    
    def run_test_suite(
        self,
        model_version: str,
        model_transcribe_fn: Callable,
        test_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Run multiple regression tests.
        
        Args:
            model_version: Model version being tested
            model_transcribe_fn: Transcription function
            test_types: Filter by test types (None for all)
            
        Returns:
            Summary of test suite results
        """
        logger.info(f"Running regression test suite for {model_version}")
        
        # Select tests
        tests_to_run = list(self.tests.values())
        if test_types:
            tests_to_run = [t for t in tests_to_run if t.test_type in test_types]
        
        if not tests_to_run:
            logger.warning("No tests to run")
            return {'error': 'No tests selected'}
        
        logger.info(f"Running {len(tests_to_run)} tests...")
        
        # Run tests
        results = []
        for test in tests_to_run:
            try:
                result = self.run_test(
                    test_id=test.test_id,
                    model_version=model_version,
                    model_transcribe_fn=model_transcribe_fn
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Test {test.test_id} failed with error: {e}")
                continue
        
        # Calculate summary
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        
        avg_wer_degradation = np.mean([r.wer_degradation for r in results])
        avg_cer_degradation = np.mean([r.cer_degradation for r in results])
        
        all_passed = failed_count == 0
        
        summary = {
            'model_version': model_version,
            'total_tests': len(results),
            'passed': passed_count,
            'failed': failed_count,
            'pass_rate': passed_count / len(results) if results else 0,
            'all_passed': all_passed,
            'avg_wer_degradation': avg_wer_degradation,
            'avg_cer_degradation': avg_cer_degradation,
            'results': [r.to_dict() for r in results],
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        self._print_suite_summary(summary)
        
        return summary
    
    def _print_test_result(self, test: RegressionTest, result: RegressionTestResult):
        """Print test result."""
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        
        print(f"\n{'='*70}")
        print(f"Regression Test: {test.test_name}")
        print(f"{'='*70}")
        print(f"Status: {status}")
        print(f"Model: {result.model_version}")
        print(f"\nMetrics:")
        print(f"  WER: {result.model_wer:.4f} (baseline: {result.baseline_wer:.4f}, Δ: {result.wer_degradation:+.4f})")
        print(f"  CER: {result.model_cer:.4f} (baseline: {result.baseline_cer:.4f}, Δ: {result.cer_degradation:+.4f})")
        print(f"\nSamples: {result.num_samples}")
        print(f"Failed samples: {len(result.failed_samples or [])}")
        
        if not result.passed:
            print(f"\nFailure reason: {result.failure_reason}")
        
        print(f"{'='*70}\n")
    
    def _print_suite_summary(self, summary: Dict):
        """Print test suite summary."""
        print("\n" + "="*80)
        print("REGRESSION TEST SUITE SUMMARY")
        print("="*80)
        print(f"\nModel: {summary['model_version']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"\nAverage WER Degradation: {summary['avg_wer_degradation']:+.4f}")
        print(f"Average CER Degradation: {summary['avg_cer_degradation']:+.4f}")
        
        if summary['all_passed']:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("\nFailed Tests:")
            for result_dict in summary['results']:
                if not result_dict['passed']:
                    print(f"  - {result_dict['test_id']}: {result_dict['failure_reason']}")
        
        print("="*80 + "\n")
    
    def get_test_history(
        self,
        test_id: Optional[str] = None,
        model_version: Optional[str] = None,
        time_window_days: Optional[int] = None
    ) -> List[RegressionTestResult]:
        """Get test history with optional filters."""
        results = self.results
        
        if test_id:
            results = [r for r in results if r.test_id == test_id]
        
        if model_version:
            results = [r for r in results if r.model_version == model_version]
        
        if time_window_days:
            cutoff = datetime.now() - timedelta(days=time_window_days)
            results = [
                r for r in results
                if datetime.fromisoformat(r.timestamp) > cutoff
            ]
        
        return sorted(results, key=lambda r: r.timestamp, reverse=True)
    
    def generate_regression_report(self) -> Dict:
        """Generate comprehensive regression testing report."""
        # Calculate statistics
        total_runs = len(self.results)
        if total_runs == 0:
            return {
                'error': 'No test results available',
                'total_tests': len(self.tests)
            }
        
        passed_runs = sum(1 for r in self.results if r.passed)
        failed_runs = total_runs - passed_runs
        
        # Get degradation trends
        wer_degradations = [r.wer_degradation for r in self.results]
        cer_degradations = [r.cer_degradation for r in self.results]
        
        # Recent results
        recent_results = self.get_test_history(time_window_days=7)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'registered_tests': len(self.tests),
            'total_test_runs': total_runs,
            'passed_runs': passed_runs,
            'failed_runs': failed_runs,
            'pass_rate': passed_runs / total_runs,
            'degradation_stats': {
                'wer': {
                    'mean': float(np.mean(wer_degradations)),
                    'std': float(np.std(wer_degradations)),
                    'min': float(np.min(wer_degradations)),
                    'max': float(np.max(wer_degradations))
                },
                'cer': {
                    'mean': float(np.mean(cer_degradations)),
                    'std': float(np.std(cer_degradations)),
                    'min': float(np.min(cer_degradations)),
                    'max': float(np.max(cer_degradations))
                }
            },
            'recent_results_7days': len(recent_results),
            'recent_pass_rate': sum(1 for r in recent_results if r.passed) / len(recent_results) if recent_results else 0
        }
        
        return report


