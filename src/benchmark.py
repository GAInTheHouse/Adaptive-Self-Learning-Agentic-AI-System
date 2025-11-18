# src/benchmark.py
"""
Task 4: Benchmark baseline model on WER, latency, and costs
Run after deploying baseline to establish performance benchmarks
"""

import torch
import time
import numpy as np
from src.baseline_model import BaselineSTTModel  # Fixed import
from typing import List, Dict
import json

class BaselineBenchmark:
    """Evaluate baseline STT model performance"""
    
    def __init__(self, model_name="whisper"):
        self.model = BaselineSTTModel(model_name)
        self.results = {
            "model": model_name,
            "benchmarks": {}
        }
    
    def benchmark_latency(self, audio_paths: List[str], num_runs=3) -> Dict:
        """Measure inference latency"""
        latencies = []
        
        for audio_path in audio_paths:
            for _ in range(num_runs):
                start = time.time()
                self.model.transcribe(audio_path)
                latencies.append(time.time() - start)
        
        return {
            "mean_latency_seconds": np.mean(latencies),
            "std_latency_seconds": np.std(latencies),
            "min_latency_seconds": np.min(latencies),
            "max_latency_seconds": np.max(latencies),
            "num_samples": len(latencies)
        }
    
    def benchmark_throughput(self, audio_paths: List[str], duration_seconds=60) -> Dict:
        """Measure samples per second"""
        count = 0
        start = time.time()
        
        idx = 0
        while time.time() - start < duration_seconds:
            audio_path = audio_paths[idx % len(audio_paths)]
            self.model.transcribe(audio_path)
            count += 1
            idx += 1
        
        elapsed = time.time() - start
        
        return {
            "samples_per_second": count / elapsed,
            "total_duration_seconds": elapsed,
            "total_samples": count
        }
    
    def estimate_cost(self, num_hours_per_month=100) -> Dict:
        """Estimate computational cost"""
        # Whisper-base: ~140M params, ~200ms per sample on GPU
        # Rough estimate: $0.0001 per inference on p3.2xlarge
        
        inferences_per_month = (num_hours_per_month * 3600) / 0.2  # 0.2s per sample
        estimated_cost_usd = inferences_per_month * 0.0001
        
        return {
            "inferences_per_month": inferences_per_month,
            "estimated_cost_usd": estimated_cost_usd,
            "cost_per_hour_transcribed": estimated_cost_usd / num_hours_per_month
        }
    
    def generate_report(self, audio_paths: List[str]) -> Dict:
        """Generate complete benchmark report"""
        print("⏱️  Benchmarking latency...")
        latency = self.benchmark_latency(audio_paths)
        
        print("📊 Benchmarking throughput...")
        throughput = self.benchmark_throughput(audio_paths)
        
        print("💰 Estimating cost...")
        cost = self.estimate_cost()
        
        report = {
            "model": self.model.model_name,
            "model_info": self.model.get_model_info(),
            "latency_benchmark": latency,
            "throughput_benchmark": throughput,
            "cost_estimate": cost
        }
        
        return report
    
    def save_report(self, report: Dict, output_path="baseline_benchmark.json"):
        """Save benchmark report to JSON"""
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved to {output_path}")
