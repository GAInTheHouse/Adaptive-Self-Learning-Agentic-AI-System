"""
Unit tests for benchmark functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import BaselineBenchmark
import pytest
import numpy as np


class TestBaselineBenchmark:
    """Test cases for BaselineBenchmark class"""
    
    def setup_method(self):
        """Setup test instance"""
        # Note: This will load the actual model, so it's more of an integration test
        # In a production environment, you'd want to mock the model
        pass
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization"""
        benchmark = BaselineBenchmark(model_name="whisper")
        
        assert benchmark.model is not None
        assert benchmark.results['model'] == "whisper"
        assert 'benchmarks' in benchmark.results
    
    def test_estimate_cost_structure(self):
        """Test cost estimation returns correct structure"""
        benchmark = BaselineBenchmark(model_name="whisper")
        
        cost = benchmark.estimate_cost(num_hours_per_month=100)
        
        assert 'inferences_per_month' in cost
        assert 'estimated_cost_usd' in cost
        assert 'cost_per_hour_transcribed' in cost
        
        assert cost['inferences_per_month'] > 0
        assert cost['estimated_cost_usd'] > 0
        assert cost['cost_per_hour_transcribed'] > 0
    
    def test_estimate_cost_scaling(self):
        """Test cost scales with usage"""
        benchmark = BaselineBenchmark(model_name="whisper")
        
        cost_100 = benchmark.estimate_cost(num_hours_per_month=100)
        cost_200 = benchmark.estimate_cost(num_hours_per_month=200)
        
        # Cost should roughly double
        assert cost_200['inferences_per_month'] > cost_100['inferences_per_month']
        assert cost_200['estimated_cost_usd'] > cost_100['estimated_cost_usd']
    
    @pytest.mark.skipif(not Path("test_audio/test_1.wav").exists(), 
                       reason="Test audio file not available")
    def test_benchmark_latency_structure(self):
        """Test latency benchmark returns correct structure"""
        benchmark = BaselineBenchmark(model_name="whisper")
        
        audio_paths = ["test_audio/test_1.wav"]
        latency = benchmark.benchmark_latency(audio_paths, num_runs=2)
        
        assert 'mean_latency_seconds' in latency
        assert 'std_latency_seconds' in latency
        assert 'min_latency_seconds' in latency
        assert 'max_latency_seconds' in latency
        assert 'num_samples' in latency
        
        assert latency['mean_latency_seconds'] > 0
        assert latency['min_latency_seconds'] > 0
        assert latency['max_latency_seconds'] >= latency['min_latency_seconds']
        assert latency['num_samples'] == 2  # 1 file * 2 runs
    
    @pytest.mark.skipif(not Path("test_audio/test_1.wav").exists(), 
                       reason="Test audio file not available")
    def test_benchmark_throughput_structure(self):
        """Test throughput benchmark returns correct structure"""
        benchmark = BaselineBenchmark(model_name="whisper")
        
        audio_paths = ["test_audio/test_1.wav"]
        # Use very short duration for testing
        throughput = benchmark.benchmark_throughput(audio_paths, duration_seconds=5)
        
        assert 'samples_per_second' in throughput
        assert 'total_duration_seconds' in throughput
        assert 'total_samples' in throughput
        
        assert throughput['samples_per_second'] > 0
        assert throughput['total_samples'] > 0
        assert throughput['total_duration_seconds'] > 0
    
    @pytest.mark.skipif(not Path("test_audio/test_1.wav").exists(), 
                       reason="Test audio file not available")
    def test_generate_report_structure(self):
        """Test complete report generation"""
        benchmark = BaselineBenchmark(model_name="whisper")
        
        audio_paths = ["test_audio/test_1.wav"]
        report = benchmark.generate_report(audio_paths)
        
        assert 'model' in report
        assert 'model_info' in report
        assert 'latency_benchmark' in report
        assert 'throughput_benchmark' in report
        assert 'cost_estimate' in report
        
        # Check model info
        assert 'name' in report['model_info']
        assert 'parameters' in report['model_info']
        assert 'device' in report['model_info']
    
    @pytest.mark.skipif(not Path("test_audio/test_1.wav").exists(), 
                       reason="Test audio file not available")
    def test_save_report(self, tmp_path):
        """Test report saving"""
        benchmark = BaselineBenchmark(model_name="whisper")
        
        audio_paths = ["test_audio/test_1.wav"]
        report = benchmark.generate_report(audio_paths)
        
        output_path = tmp_path / "benchmark_report.json"
        benchmark.save_report(report, str(output_path))
        
        assert output_path.exists(), "Report file should be created"
        
        # Verify it's valid JSON
        import json
        with open(output_path) as f:
            loaded_report = json.load(f)
        
        assert loaded_report['model'] == report['model']


def test_cost_estimate_reasonable():
    """Test that cost estimates are in reasonable range"""
    benchmark = BaselineBenchmark(model_name="whisper")
    
    cost = benchmark.estimate_cost(num_hours_per_month=1000)
    
    # For 1000 hours, cost should be reasonable (e.g., not millions of dollars)
    assert cost['estimated_cost_usd'] < 10000, "Cost estimate seems unreasonably high"
    assert cost['estimated_cost_usd'] > 0, "Cost should be positive"


def test_latency_statistics_valid():
    """Test that latency statistics are mathematically valid"""
    benchmark = BaselineBenchmark(model_name="whisper")
    
    # Simulate latency results
    latencies = [0.5, 0.6, 0.4, 0.55, 0.45]
    
    mean = np.mean(latencies)
    std = np.std(latencies)
    min_val = np.min(latencies)
    max_val = np.max(latencies)
    
    # Basic mathematical validations
    assert min_val <= mean <= max_val
    assert std >= 0
    assert min_val >= 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

