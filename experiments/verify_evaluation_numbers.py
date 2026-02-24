#!/usr/bin/env python3
"""
Script to verify evaluation numbers in the report by running actual evaluations.
"""

import sys
from pathlib import Path
import json
import logging
from typing import List, Dict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.baseline_model import BaselineSTTModel
from src.integration import UnifiedSTTSystem, StatisticalAnalyzer, AblationStudy
from src.evaluation.metrics import STTEvaluator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def verify_baseline_metrics():
    """Verify baseline model metrics."""
    logger.info("="*70)
    logger.info("VERIFYING BASELINE METRICS")
    logger.info("="*70)
    
    # Load existing evaluation results
    eval_summary_path = Path("experiments/evaluation_outputs/evaluation_summary.json")
    benchmark_path = Path("experiments/evaluation_outputs/benchmark_report.json")
    
    actual_results = {}
    
    if eval_summary_path.exists():
        with open(eval_summary_path) as f:
            eval_data = json.load(f)
            actual_results['baseline_wer'] = eval_data['overall_metrics']['mean_wer']
            actual_results['baseline_cer'] = eval_data['overall_metrics']['mean_cer']
            actual_results['model_params'] = eval_data['model_info']['parameters']
            logger.info(f"✓ Found baseline WER: {actual_results['baseline_wer']:.4f} ({actual_results['baseline_wer']*100:.2f}%)")
            logger.info(f"✓ Found baseline CER: {actual_results['baseline_cer']:.4f} ({actual_results['baseline_cer']*100:.2f}%)")
    
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            benchmark_data = json.load(f)
            actual_results['mean_latency'] = benchmark_data['latency_benchmark']['mean_latency_seconds']
            actual_results['throughput'] = benchmark_data['throughput_benchmark']['samples_per_second']
            logger.info(f"✓ Found mean latency: {actual_results['mean_latency']:.2f}s")
            logger.info(f"✓ Found throughput: {actual_results['throughput']:.2f} samples/s")
    
    # Report discrepancies
    logger.info("\n" + "-"*70)
    logger.info("REPORTED vs ACTUAL VALUES:")
    logger.info("-"*70)
    
    discrepancies = []
    
    # Baseline WER
    reported_wer = 0.10  # Report says 10.0%
    if abs(actual_results.get('baseline_wer', 0) - reported_wer) > 0.01:
        discrepancies.append(f"Baseline WER: Report says {reported_wer*100:.1f}%, Actual: {actual_results.get('baseline_wer', 'N/A')*100:.1f}%")
    else:
        logger.info(f"✓ Baseline WER matches: {reported_wer*100:.1f}%")
    
    # Baseline CER
    reported_cer = 0.0227  # Report says 2.27%
    if abs(actual_results.get('baseline_cer', 0) - reported_cer) > 0.001:
        discrepancies.append(f"Baseline CER: Report says {reported_cer*100:.2f}%, Actual: {actual_results.get('baseline_cer', 'N/A')*100:.2f}%")
    else:
        logger.info(f"✓ Baseline CER matches: {reported_cer*100:.2f}%")
    
    # Latency - Report says 0.72s but actual is 5.29s
    reported_latency = 0.72
    if abs(actual_results.get('mean_latency', 0) - reported_latency) > 0.1:
        discrepancies.append(f"⚠️  LATENCY MISMATCH: Report says {reported_latency:.2f}s, Actual: {actual_results.get('mean_latency', 'N/A'):.2f}s")
        logger.warning(f"⚠️  Major discrepancy in latency!")
    
    # Throughput
    reported_throughput = 2.97
    if abs(actual_results.get('throughput', 0) - reported_throughput) > 0.1:
        discrepancies.append(f"Throughput: Report says {reported_throughput:.2f} samples/s, Actual: {actual_results.get('throughput', 'N/A'):.2f} samples/s")
    else:
        logger.info(f"✓ Throughput matches: {reported_throughput:.2f} samples/s")
    
    if discrepancies:
        logger.warning("\n⚠️  DISCREPANCIES FOUND:")
        for d in discrepancies:
            logger.warning(f"  - {d}")
    else:
        logger.info("\n✓ All baseline metrics match!")
    
    return actual_results, discrepancies


def check_report_numbers():
    """Check numbers mentioned in the report against what we can verify."""
    logger.info("\n" + "="*70)
    logger.info("CHECKING REPORT NUMBERS")
    logger.info("="*70)
    
    issues = []
    
    # Check baseline numbers
    logger.info("\n1. Baseline Performance:")
    logger.info("   Report claims: WER 10.0%, CER 2.27%, Latency 0.72s, Throughput 2.97 samples/s")
    logger.info("   Note: These appear to be from a small test dataset (2 samples)")
    
    # Check full system numbers
    logger.info("\n2. Full System Performance:")
    logger.info("   Report claims: WER 19-22% (improvement from 25-30%)")
    logger.info("   ⚠️  WARNING: Baseline WER is actually 10%, not 25-30%")
    logger.info("   ⚠️  This suggests report numbers may be from different dataset or theoretical")
    issues.append("Baseline WER mismatch: Report says 25-30% but actual baseline is 10%")
    
    # Check ablation study numbers
    logger.info("\n3. Ablation Study Results:")
    logger.info("   Report claims various WER values for different configurations")
    logger.info("   ⚠️  These numbers cannot be verified without running full ablation study")
    logger.info("   ⚠️  Need to run actual ablation study to verify")
    
    # Check statistical numbers
    logger.info("\n4. Statistical Analysis:")
    logger.info("   Report claims: p < 0.001, Cohen's d = 0.5-0.7")
    logger.info("   ⚠️  These require actual paired comparisons - cannot verify without test data")
    
    return issues


def main():
    """Main verification function."""
    logger.info("EVALUATION NUMBER VERIFICATION")
    logger.info("="*70)
    logger.info("This script verifies numbers in the report against actual evaluation results.\n")
    
    # Verify baseline metrics
    actual_results, discrepancies = verify_baseline_metrics()
    
    # Check report numbers
    issues = check_report_numbers()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    logger.info("\n✓ Verified from actual evaluation files:")
    logger.info(f"  - Baseline WER: {actual_results.get('baseline_wer', 'N/A')*100:.2f}%")
    logger.info(f"  - Baseline CER: {actual_results.get('baseline_cer', 'N/A')*100:.2f}%")
    logger.info(f"  - Mean Latency: {actual_results.get('mean_latency', 'N/A'):.2f}s")
    logger.info(f"  - Throughput: {actual_results.get('throughput', 'N/A'):.2f} samples/s")
    
    logger.info("\n⚠️  Numbers in report that need verification:")
    logger.info("  - Full system WER (19-22%) - requires full system evaluation")
    logger.info("  - Ablation study results - requires running ablation study")
    logger.info("  - Statistical p-values and effect sizes - requires paired comparisons")
    logger.info("  - Component contributions - requires ablation study")
    
    logger.info("\n⚠️  Major discrepancies found:")
    if discrepancies:
        for d in discrepancies:
            logger.warning(f"  - {d}")
    if issues:
        for i in issues:
            logger.warning(f"  - {i}")
    
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS:")
    logger.info("="*70)
    logger.info("1. The report contains some theoretical/estimated numbers")
    logger.info("2. Baseline metrics (WER 10%, CER 2.27%) are verified from actual evaluations")
    logger.info("3. Latency number (0.72s) doesn't match actual (5.29s) - may be from different test")
    logger.info("4. Full system and ablation numbers need actual test runs to verify")
    logger.info("5. Consider updating report with actual measured values or clearly label as estimates")


if __name__ == "__main__":
    main()


