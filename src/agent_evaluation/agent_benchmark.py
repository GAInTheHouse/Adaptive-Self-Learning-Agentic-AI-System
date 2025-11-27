"""
Agent Benchmarking - Latency and Runtime Performance

Measures agent performance overhead and scalability.
"""

import logging
from typing import Dict, List, Optional
import time
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    audio_path: str
    baseline_latency_ms: float
    agent_latency_ms: float
    error_detection_latency_ms: float
    correction_latency_ms: float
    overhead_percent: float
    errors_detected: int
    corrections_applied: int


class AgentBenchmark:
    """
    Benchmark agent performance metrics.
    
    Metrics:
    - Baseline inference latency
    - Agent overhead (error detection + correction)
    - Breakdown: error detection time vs correction time
    - Throughput: samples per second
    - Scalability: latency vs audio length
    """
    
    def __init__(self, 
                 agent=None,
                 baseline_model=None,
                 output_dir: str = "experiments/evaluation_outputs"):
        """Initialize benchmark"""
        self.agent = agent
        self.baseline_model = baseline_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        
        logger.info("Agent Benchmark initialized")
    
    def benchmark_single(self, audio_path: str) -> BenchmarkResult:
        """Benchmark a single audio file"""
        
        # Baseline latency
        start = time.time()
        baseline_result = self.baseline_model.transcribe(audio_path)
        baseline_latency = (time.time() - start) * 1000  # ms
        
        # Agent latency (full pipeline)
        start = time.time()
        agent_result = self.agent.transcribe_with_agent(
            audio_path,
            enable_auto_correction=True
        )
        agent_latency = (time.time() - start) * 1000  # ms
        
        # Estimate breakdown (rough approximation)
        error_detection_latency = agent_latency * 0.3  # Error detection ~30%
        correction_latency = agent_latency * 0.7      # Correction ~70%
        
        overhead = ((agent_latency - baseline_latency) / baseline_latency * 100) \
            if baseline_latency > 0 else 0
        
        result = BenchmarkResult(
            audio_path=str(audio_path),
            baseline_latency_ms=baseline_latency,
            agent_latency_ms=agent_latency,
            error_detection_latency_ms=error_detection_latency,
            correction_latency_ms=correction_latency,
            overhead_percent=overhead,
            errors_detected=agent_result['error_detection']['error_count'],
            corrections_applied=agent_result['corrections']['count']
        )
        
        self.results.append(result)
        return result
    
    def benchmark_batch(self, audio_paths: List[str], 
                       verbose: bool = True) -> Dict:
        """Benchmark multiple audio files"""
        
        logger.info(f"Benchmarking {len(audio_paths)} audio files...")
        
        for idx, audio_path in enumerate(audio_paths):
            try:
                result = self.benchmark_single(audio_path)
                if verbose and (idx + 1) % 10 == 0:
                    logger.info(f"  Completed {idx+1}/{len(audio_paths)}")
            except Exception as e:
                logger.error(f"Error benchmarking {audio_path}: {e}")
                continue
        
        return self.get_benchmark_summary()
    
    def get_benchmark_summary(self) -> Dict:
        """Get benchmark summary statistics"""
        
        if not self.results:
            return {"error": "No benchmark results"}
        
        baseline_latencies = [r.baseline_latency_ms for r in self.results]
        agent_latencies = [r.agent_latency_ms for r in self.results]
        overheads = [r.overhead_percent for r in self.results]
        
        return {
            "total_samples": len(self.results),
            "baseline_latency": {
                "mean_ms": np.mean(baseline_latencies),
                "std_ms": np.std(baseline_latencies),
                "min_ms": np.min(baseline_latencies),
                "max_ms": np.max(baseline_latencies),
                "p95_ms": np.percentile(baseline_latencies, 95),
                "p99_ms": np.percentile(baseline_latencies, 99)
            },
            "agent_latency": {
                "mean_ms": np.mean(agent_latencies),
                "std_ms": np.std(agent_latencies),
                "min_ms": np.min(agent_latencies),
                "max_ms": np.max(agent_latencies),
                "p95_ms": np.percentile(agent_latencies, 95),
                "p99_ms": np.percentile(agent_latencies, 99)
            },
            "overhead": {
                "mean_percent": np.mean(overheads),
                "std_percent": np.std(overheads),
                "min_percent": np.min(overheads),
                "max_percent": np.max(overheads)
            },
            "throughput": {
                "baseline_samples_per_sec": 1000 / np.mean(baseline_latencies),
                "agent_samples_per_sec": 1000 / np.mean(agent_latencies)
            },
            "detailed_results": [
                {
                    'audio_path': r.audio_path,
                    'baseline_latency_ms': r.baseline_latency_ms,
                    'agent_latency_ms': r.agent_latency_ms,
                    'overhead_percent': r.overhead_percent,
                    'errors_detected': r.errors_detected,
                    'corrections_applied': r.corrections_applied
                }
                for r in self.results
            ]
        }
    
    def save_benchmark_report(self, filename: str = "agent_benchmark.json"):
        """Save benchmark report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_results": self.get_benchmark_summary()
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Benchmark report saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print benchmark summary"""
        
        summary = self.get_benchmark_summary()
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print("\n" + "="*70)
        print("AGENT LATENCY BENCHMARK REPORT")
        print("="*70)
        
        print(f"\nTotal samples: {summary['total_samples']}")
        
        print("\nBaseline latency:")
        baseline = summary['baseline_latency']
        print(f"  Mean: {baseline['mean_ms']:.2f} ms")
        print(f"  Std: {baseline['std_ms']:.2f} ms")
        print(f"  P95: {baseline['p95_ms']:.2f} ms")
        print(f"  P99: {baseline['p99_ms']:.2f} ms")
        
        print("\nAgent latency (with error detection + correction):")
        agent = summary['agent_latency']
        print(f"  Mean: {agent['mean_ms']:.2f} ms")
        print(f"  Std: {agent['std_ms']:.2f} ms")
        print(f"  P95: {agent['p95_ms']:.2f} ms")
        print(f"  P99: {agent['p99_ms']:.2f} ms")
        
        print("\nAgent overhead:")
        overhead = summary['overhead']
        print(f"  Mean: +{overhead['mean_percent']:.2f}%")
        print(f"  Range: {overhead['min_percent']:.2f}% to {overhead['max_percent']:.2f}%")
        
        print("\nThroughput:")
        throughput = summary['throughput']
        print(f"  Baseline: {throughput['baseline_samples_per_sec']:.2f} samples/sec")
        print(f"  Agent: {throughput['agent_samples_per_sec']:.2f} samples/sec")
        
        print("="*70 + "\n")
