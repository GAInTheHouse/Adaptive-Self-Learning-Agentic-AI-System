#!/usr/bin/env python3
"""
Demo: Weights & Biases Integration for Fine-Tuning

Demonstrates how W&B tracks metrics, creates visualizations, and logs experiments.

Usage:
    # Make sure wandb is installed and logged in
    pip install wandb
    wandb login
    
    # Run demo
    python experiments/demo_wandb_tracking.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime

from src.data.data_manager import DataManager
from src.data.finetuning_orchestrator import FinetuningOrchestrator, FinetuningConfig
from src.data.wandb_tracker import WandbTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_basic_tracking():
    """Demo 1: Basic W&B tracking."""
    print_section("DEMO 1: Basic W&B Tracking")
    
    # Initialize tracker
    tracker = WandbTracker(
        project_name="stt-finetuning-demo",
        enabled=True
    )
    
    # Start a run
    tracker.start_run(
        run_name="demo_basic_tracking",
        tags=["demo", "basic"]
    )
    
    print("✅ W&B run started")
    print(f"   View at: {tracker.run.url if tracker.run else 'N/A'}")
    
    # Log some training metrics
    print("\n📊 Logging training metrics...")
    for epoch in range(1, 6):
        tracker.log_training_metrics(
            epoch=epoch,
            train_loss=0.5 / epoch,  # Decreasing loss
            val_loss=0.6 / epoch,
            learning_rate=0.001 * (0.9 ** epoch)
        )
        print(f"   Epoch {epoch}: train_loss={0.5/epoch:.4f}, val_loss={0.6/epoch:.4f}")
    
    # Finish run
    tracker.finish_run()
    print("\n✅ Basic tracking demo completed")


def demo_validation_tracking():
    """Demo 2: Validation results tracking."""
    print_section("DEMO 2: Validation Results Tracking")
    
    tracker = WandbTracker(
        project_name="stt-finetuning-demo",
        enabled=True
    )
    
    tracker.start_run(
        run_name="demo_validation",
        tags=["demo", "validation"]
    )
    
    print("📊 Logging validation results...")
    
    # Mock validation result
    validation_result = {
        'model_wer': 0.12,
        'model_cer': 0.06,
        'baseline_wer': 0.20,
        'baseline_cer': 0.10,
        'wer_improvement': 0.08,
        'cer_improvement': 0.04,
        'passed': True,
        'num_samples': 100
    }
    
    tracker.log_validation_results(
        validation_result=validation_result,
        model_id="demo_model_v1"
    )
    
    print(f"   Model WER: {validation_result['model_wer']:.4f}")
    print(f"   Baseline WER: {validation_result['baseline_wer']:.4f}")
    print(f"   Improvement: {validation_result['wer_improvement']:.4f}")
    print("   ✅ W&B will generate comparison charts automatically!")
    
    tracker.finish_run()
    print("\n✅ Validation tracking demo completed")


def demo_regression_tracking():
    """Demo 3: Regression test tracking."""
    print_section("DEMO 3: Regression Test Tracking")
    
    tracker = WandbTracker(
        project_name="stt-finetuning-demo",
        enabled=True
    )
    
    tracker.start_run(
        run_name="demo_regression",
        tags=["demo", "regression"]
    )
    
    print("📊 Logging regression test results...")
    
    # Mock regression results
    regression_results = {
        'total_tests': 5,
        'passed': 4,
        'failed': 1,
        'pass_rate': 0.8,
        'avg_wer_degradation': -0.02,  # Negative means improvement
        'avg_cer_degradation': -0.01,
        'results': [
            {'test_id': 'test_001', 'wer_degradation': -0.05, 'passed': True},
            {'test_id': 'test_002', 'wer_degradation': -0.03, 'passed': True},
            {'test_id': 'test_003', 'wer_degradation': 0.02, 'passed': False},
            {'test_id': 'test_004', 'wer_degradation': -0.01, 'passed': True},
            {'test_id': 'test_005', 'wer_degradation': -0.04, 'passed': True}
        ]
    }
    
    tracker.log_regression_results(
        test_results=regression_results,
        model_version="demo_model_v1"
    )
    
    print(f"   Total tests: {regression_results['total_tests']}")
    print(f"   Pass rate: {regression_results['pass_rate']:.1%}")
    print(f"   Average WER change: {regression_results['avg_wer_degradation']:+.4f}")
    print("   ✅ W&B will generate test result charts!")
    
    tracker.finish_run()
    print("\n✅ Regression tracking demo completed")


def demo_dataset_tracking():
    """Demo 4: Dataset information tracking."""
    print_section("DEMO 4: Dataset Information Tracking")
    
    tracker = WandbTracker(
        project_name="stt-finetuning-demo",
        enabled=True
    )
    
    tracker.start_run(
        run_name="demo_dataset",
        tags=["demo", "dataset"]
    )
    
    print("📊 Logging dataset information...")
    
    split_sizes = {
        'train': 800,
        'val': 100,
        'test': 100
    }
    
    error_distribution = {
        'word_substitution': 450,
        'missing_word': 300,
        'extra_word': 150,
        'pronunciation_error': 100
    }
    
    tracker.log_dataset_info(
        dataset_id="demo_dataset_001",
        split_sizes=split_sizes,
        error_type_distribution=error_distribution
    )
    
    print(f"   Total samples: {sum(split_sizes.values())}")
    print(f"   Train: {split_sizes['train']}, Val: {split_sizes['val']}, Test: {split_sizes['test']}")
    print(f"   Error types: {len(error_distribution)}")
    print("   ✅ W&B will visualize error distribution!")
    
    tracker.finish_run()
    print("\n✅ Dataset tracking demo completed")


def demo_system_metrics():
    """Demo 5: System-level metrics tracking."""
    print_section("DEMO 5: System Metrics Tracking")
    
    tracker = WandbTracker(
        project_name="stt-finetuning-demo",
        enabled=True
    )
    
    tracker.start_run(
        run_name="demo_system_metrics",
        tags=["demo", "system"]
    )
    
    print("📊 Logging system metrics over time...")
    
    # Simulate system evolution
    for iteration in range(1, 6):
        error_cases = 50 * iteration
        corrected_cases = int(error_cases * 0.6)
        correction_rate = corrected_cases / error_cases
        
        tracker.log_system_metrics(
            error_cases=error_cases,
            corrected_cases=corrected_cases,
            correction_rate=correction_rate,
            models_deployed=iteration
        )
        
        print(f"   Iteration {iteration}: {error_cases} errors, {corrected_cases} corrected")
    
    print("   ✅ W&B tracks system growth over time!")
    
    tracker.finish_run()
    print("\n✅ System metrics demo completed")


def demo_orchestrator_integration():
    """Demo 6: Full orchestrator integration with W&B."""
    print_section("DEMO 6: Orchestrator Integration")
    
    print("🔧 Initializing fine-tuning orchestrator with W&B...")
    
    # Create data manager
    data_manager = DataManager(
        local_storage_dir="data/wandb_demo",
        use_gcs=False
    )
    
    # Add some test error cases
    for i in range(15):
        data_manager.store_failed_case(
            audio_path=f"test_{i}.wav",
            original_transcript=f"original transcript {i}",
            corrected_transcript=f"corrected transcript {i}",
            error_types=["word_substitution"],
            error_score=0.8
        )
    
    # Create orchestrator with W&B enabled
    config = FinetuningConfig(
        min_error_cases=10,
        auto_approve_finetuning=True,
        use_wandb=True,  # Enable W&B
        wandb_project="stt-finetuning-demo"
    )
    
    orchestrator = FinetuningOrchestrator(
        data_manager=data_manager,
        config=config,
        storage_dir="data/wandb_orchestration",
        use_gcs=False
    )
    
    print(f"   W&B tracking enabled: {orchestrator.wandb_tracker is not None}")
    
    # Check trigger conditions
    trigger_result = orchestrator.check_trigger_conditions()
    print(f"   Should trigger: {trigger_result['should_trigger']}")
    print(f"   Error cases: {trigger_result['metrics']['total_error_cases']}")
    
    # Trigger fine-tuning (this will start a W&B run automatically)
    print("\n🚀 Triggering fine-tuning...")
    job = orchestrator.trigger_finetuning(force=True)
    
    if job and orchestrator.wandb_tracker and orchestrator.wandb_tracker.run:
        print(f"   ✅ Job created: {job.job_id}")
        print(f"   📊 W&B run: {orchestrator.wandb_tracker.run.url}")
        print("   All metrics will be automatically logged to W&B!")
        
        # Finish W&B run
        orchestrator.wandb_tracker.finish_run()
    
    print("\n✅ Orchestrator integration demo completed")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("  WEIGHTS & BIASES INTEGRATION - COMPREHENSIVE DEMO")
    print("="*80)
    print("\nThis demo shows how W&B tracks fine-tuning experiments:")
    print("  1. Basic Training Metrics")
    print("  2. Validation Results")
    print("  3. Regression Tests")
    print("  4. Dataset Information")
    print("  5. System Metrics")
    print("  6. Full Orchestrator Integration")
    
    try:
        # Run demos
        demo_basic_tracking()
        demo_validation_tracking()
        demo_regression_tracking()
        demo_dataset_tracking()
        demo_system_metrics()
        demo_orchestrator_integration()
        
        # Final summary
        print_section("DEMO COMPLETE - Summary")
        
        print("✅ Successfully demonstrated:")
        print("  ✓ Training metrics tracking")
        print("  ✓ Validation results visualization")
        print("  ✓ Regression test tracking")
        print("  ✓ Dataset information logging")
        print("  ✓ System metrics over time")
        print("  ✓ Full orchestrator integration")
        
        print("\n📊 W&B Features Used:")
        print("  • Automatic chart generation")
        print("  • Metric comparison across runs")
        print("  • Dataset visualization")
        print("  • Model artifact logging")
        print("  • Custom plots and tables")
        
        print("\n🚀 Next Steps:")
        print("  1. Check W&B dashboard for visualizations")
        print("  2. Compare multiple training runs")
        print("  3. Analyze performance trends")
        print("  4. Share results with team")
        
        print("\n💡 View your W&B dashboard at:")
        print("   https://wandb.ai/")
        
        print("\n" + "="*80)
        print("  Demo completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed with error: {e}")
        print("   Make sure wandb is installed: pip install wandb")
        print("   And you're logged in: wandb login")
        sys.exit(1)


if __name__ == "__main__":
    main()

