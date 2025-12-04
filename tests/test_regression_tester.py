"""
Unit tests for Regression Tester
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.regression_tester import (
    RegressionTester,
    RegressionConfig,
    RegressionTest,
    RegressionTestResult
)


@pytest.fixture
def tester(temp_directory):
    """Create tester instance."""
    config = RegressionConfig(
        fail_on_critical_degradation=True,
        critical_degradation_threshold=0.1
    )
    
    with patch('src.data.regression_tester.GCSManager'):
        tester = RegressionTester(
            config=config,
            storage_dir=str(Path(temp_directory) / "regression"),
            use_gcs=False
        )
    
    return tester


@pytest.fixture
def test_data_file(temp_directory):
    """Create test data file."""
    test_file = Path(temp_directory) / "regression_test.jsonl"
    
    # Create test data
    test_data = [
        {'audio_path': 'test1.wav', 'reference': 'this is test one'},
        {'audio_path': 'test2.wav', 'reference': 'this is test two'},
        {'audio_path': 'test3.wav', 'reference': 'this is test three'}
    ]
    
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    return str(test_file)


def mock_model_transcribe(audio_path):
    """Mock model transcription."""
    if 'test1' in audio_path:
        return 'this is test one'
    elif 'test2' in audio_path:
        return 'this is test two'
    else:
        return 'this is test three'


class TestRegressionConfig:
    """Test RegressionConfig."""
    
    def test_config_creation(self):
        """Test config creation."""
        config = RegressionConfig()
        assert config.run_on_deploy is True
        assert config.fail_on_critical_degradation is True
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = RegressionConfig(critical_degradation_threshold=0.15)
        config_dict = config.to_dict()
        assert config_dict['critical_degradation_threshold'] == 0.15


class TestRegressionTest:
    """Test RegressionTest."""
    
    def test_test_creation(self):
        """Test creation."""
        test = RegressionTest(
            test_id="test_001",
            test_name="Benchmark Test",
            test_type="benchmark",
            test_data_path="/path/to/data.jsonl",
            baseline_wer=0.15,
            baseline_cer=0.08,
            baseline_version="baseline_v1"
        )
        
        assert test.test_id == "test_001"
        assert test.test_type == "benchmark"
        assert test.baseline_wer == 0.15
    
    def test_test_to_dict(self):
        """Test serialization."""
        test = RegressionTest(
            test_id="test_001",
            test_name="Benchmark Test",
            test_type="benchmark",
            test_data_path="/path/to/data.jsonl",
            baseline_wer=0.15,
            baseline_cer=0.08,
            baseline_version="baseline_v1"
        )
        
        test_dict = test.to_dict()
        assert test_dict['test_id'] == "test_001"


class TestRegressionTestResult:
    """Test RegressionTestResult."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = RegressionTestResult(
            test_id="test_001",
            model_version="v1",
            model_wer=0.12,
            model_cer=0.06,
            baseline_wer=0.15,
            baseline_cer=0.08,
            wer_degradation=-0.03,
            cer_degradation=-0.02,
            passed=True,
            num_samples=100
        )
        
        assert result.test_id == "test_001"
        assert result.passed is True
        assert result.wer_degradation == -0.03
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result = RegressionTestResult(
            test_id="test_001",
            model_version="v1",
            model_wer=0.12,
            model_cer=0.06,
            baseline_wer=0.15,
            baseline_cer=0.08,
            wer_degradation=-0.03,
            cer_degradation=-0.02,
            passed=True,
            num_samples=100
        )
        
        result_dict = result.to_dict()
        assert result_dict['test_id'] == "test_001"


class TestRegressionTester:
    """Test RegressionTester."""
    
    def test_initialization(self, tester):
        """Test tester initialization."""
        assert tester is not None
        assert tester.config.fail_on_critical_degradation is True
    
    def test_register_test(self, tester, test_data_file):
        """Test registering a regression test."""
        test_id = tester.register_test(
            test_name="Benchmark Test",
            test_type="benchmark",
            test_data_path=test_data_file,
            baseline_wer=0.15,
            baseline_cer=0.08,
            baseline_version="baseline_v1",
            description="Core benchmark test"
        )
        
        assert test_id is not None
        assert test_id in tester.tests
        assert tester.tests[test_id].test_name == "Benchmark Test"
    
    def test_load_test_data(self, tester, test_data_file):
        """Test loading test data."""
        samples = tester.load_test_data(test_data_file)
        
        assert len(samples) == 3
        assert all('audio_path' in s for s in samples)
        assert all('reference' in s for s in samples)
    
    def test_run_test(self, tester, test_data_file):
        """Test running a regression test."""
        # Register test
        test_id = tester.register_test(
            test_name="Test Run",
            test_type="benchmark",
            test_data_path=test_data_file,
            baseline_wer=0.15,
            baseline_cer=0.08,
            baseline_version="baseline_v1"
        )
        
        # Run test
        result = tester.run_test(
            test_id=test_id,
            model_version="test_model_v1",
            model_transcribe_fn=mock_model_transcribe
        )
        
        assert result is not None
        assert result.test_id == test_id
        assert result.model_version == "test_model_v1"
        assert result.num_samples == 3
    
    def test_run_test_with_improvement(self, tester, test_data_file):
        """Test running test with model improvement."""
        # Register test with high baseline WER
        test_id = tester.register_test(
            test_name="Improvement Test",
            test_type="benchmark",
            test_data_path=test_data_file,
            baseline_wer=0.50,  # High baseline (worse)
            baseline_cer=0.30,
            baseline_version="baseline_v1"
        )
        
        # Run test with good model
        result = tester.run_test(
            test_id=test_id,
            model_version="good_model",
            model_transcribe_fn=mock_model_transcribe
        )
        
        # Should show improvement (negative degradation)
        assert result.wer_degradation < 0
        assert result.passed is True
    
    def test_run_test_with_degradation(self, tester, test_data_file):
        """Test running test with model degradation."""
        # Register test with low baseline WER
        test_id = tester.register_test(
            test_name="Degradation Test",
            test_type="benchmark",
            test_data_path=test_data_file,
            baseline_wer=0.01,  # Very low (good)
            baseline_cer=0.005,
            baseline_version="baseline_v1",
            max_wer_degradation=0.05
        )
        
        def bad_model_transcribe(audio_path):
            return "wrong transcription"
        
        # Run test with bad model
        result = tester.run_test(
            test_id=test_id,
            model_version="bad_model",
            model_transcribe_fn=bad_model_transcribe
        )
        
        # Should show degradation
        assert result.wer_degradation > 0
        assert result.passed is False
    
    def test_run_test_suite(self, tester, test_data_file):
        """Test running test suite."""
        # Register multiple tests
        for i in range(3):
            tester.register_test(
                test_name=f"Test {i}",
                test_type="benchmark",
                test_data_path=test_data_file,
                baseline_wer=0.15,
                baseline_cer=0.08,
                baseline_version="baseline_v1"
            )
        
        # Run test suite
        summary = tester.run_test_suite(
            model_version="suite_test_model",
            model_transcribe_fn=mock_model_transcribe
        )
        
        assert 'total_tests' in summary
        assert summary['total_tests'] == 3
        assert 'passed' in summary
        assert 'failed' in summary
    
    def test_run_test_suite_with_filter(self, tester, test_data_file):
        """Test running test suite with type filter."""
        # Register tests of different types
        tester.register_test(
            test_name="Benchmark 1",
            test_type="benchmark",
            test_data_path=test_data_file,
            baseline_wer=0.15,
            baseline_cer=0.08,
            baseline_version="baseline_v1"
        )
        tester.register_test(
            test_name="Critical 1",
            test_type="critical_samples",
            test_data_path=test_data_file,
            baseline_wer=0.15,
            baseline_cer=0.08,
            baseline_version="baseline_v1"
        )
        
        # Run only benchmark tests
        summary = tester.run_test_suite(
            model_version="filtered_model",
            model_transcribe_fn=mock_model_transcribe,
            test_types=["benchmark"]
        )
        
        assert summary['total_tests'] == 1
    
    def test_get_test_history(self, tester, test_data_file):
        """Test getting test history."""
        # Register and run test
        test_id = tester.register_test(
            test_name="History Test",
            test_type="benchmark",
            test_data_path=test_data_file,
            baseline_wer=0.15,
            baseline_cer=0.08,
            baseline_version="baseline_v1"
        )
        
        tester.run_test(
            test_id=test_id,
            model_version="v1",
            model_transcribe_fn=mock_model_transcribe
        )
        
        # Get history
        history = tester.get_test_history()
        
        assert len(history) > 0
        assert history[0].test_id == test_id
    
    def test_get_test_history_with_filters(self, tester, test_data_file):
        """Test getting test history with filters."""
        # Register and run tests
        test_id = tester.register_test(
            test_name="Filter Test",
            test_type="benchmark",
            test_data_path=test_data_file,
            baseline_wer=0.15,
            baseline_cer=0.08,
            baseline_version="baseline_v1"
        )
        
        tester.run_test(test_id, "v1", mock_model_transcribe)
        tester.run_test(test_id, "v2", mock_model_transcribe)
        
        # Filter by model version
        history = tester.get_test_history(model_version="v1")
        
        assert all(r.model_version == "v1" for r in history)
    
    def test_generate_regression_report(self, tester, test_data_file):
        """Test regression report generation."""
        # Register and run some tests
        test_id = tester.register_test(
            test_name="Report Test",
            test_type="benchmark",
            test_data_path=test_data_file,
            baseline_wer=0.15,
            baseline_cer=0.08,
            baseline_version="baseline_v1"
        )
        
        tester.run_test(test_id, "v1", mock_model_transcribe)
        
        # Generate report
        report = tester.generate_regression_report()
        
        assert 'generated_at' in report
        assert 'registered_tests' in report
        assert 'total_test_runs' in report
        assert report['total_test_runs'] > 0


@pytest.mark.unit
def test_regression_tester_integration(temp_directory):
    """Integration test for regression tester."""
    # Create test data
    test_file = Path(temp_directory) / "integration_test.jsonl"
    test_data = [
        {'audio_path': f'test{i}.wav', 'reference': f'test transcript {i}'}
        for i in range(10)
    ]
    
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    # Create tester
    config = RegressionConfig(
        fail_on_critical_degradation=True,
        critical_degradation_threshold=0.1,
        max_failed_samples_rate=0.1
    )
    
    with patch('src.data.regression_tester.GCSManager'):
        tester = RegressionTester(
            config=config,
            storage_dir=str(Path(temp_directory) / "regression"),
            use_gcs=False
        )
    
    # Register tests
    benchmark_id = tester.register_test(
        test_name="Integration Benchmark",
        test_type="benchmark",
        test_data_path=str(test_file),
        baseline_wer=0.20,
        baseline_cer=0.10,
        baseline_version="baseline_v1",
        description="Integration test benchmark"
    )
    
    critical_id = tester.register_test(
        test_name="Integration Critical",
        test_type="critical_samples",
        test_data_path=str(test_file),
        baseline_wer=0.15,
        baseline_cer=0.08,
        baseline_version="baseline_v1",
        description="Integration critical test"
    )
    
    # Define model transcribe function
    def model_fn(audio_path):
        idx = int(audio_path.replace('test', '').replace('.wav', ''))
        return f'test transcript {idx}'
    
    # Run test suite
    summary = tester.run_test_suite(
        model_version="integration_model_v1",
        model_transcribe_fn=model_fn
    )
    
    assert summary['total_tests'] == 2
    assert summary['all_passed'] is True
    assert summary['pass_rate'] == 1.0
    
    # Generate report
    report = tester.generate_regression_report()
    assert report['registered_tests'] == 2
    assert report['total_test_runs'] == 2

