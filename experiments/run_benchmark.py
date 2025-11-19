"""
Task 4: Benchmark the baseline model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.benchmark import BaselineBenchmark
import os
import json

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 4: Baseline Benchmarking")
    print("=" * 50)
    
    # Use test audio
    test_audio_files = ["test_audio/test_1.wav"]
    
    if not all(os.path.exists(f) for f in test_audio_files):
        print("⚠️  Test audio files not found. Run Task 1 first.")
        exit(1)
    
    benchmark = BaselineBenchmark(model_name="whisper")
    
    print("\n📊 Running benchmarks (this may take a few minutes)...\n")
    report = benchmark.generate_report(test_audio_files)
    
    print("\n✅ BENCHMARK REPORT:")
    print("-" * 50)
    print(json.dumps(report, indent=2))
    
    # Save report
    benchmark.save_report(report, "baseline_benchmark.json")
    
    print("\n" + "=" * 50)
    print("✅ Benchmarking complete!")
    print("=" * 50)
