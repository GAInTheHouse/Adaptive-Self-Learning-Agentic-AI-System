#!/usr/bin/env python3
"""
Comprehensive Demo: Fine-Tuning Orchestration System

Demonstrates the complete fine-tuning lifecycle:
1. Automated fine-tuning trigger based on error accumulation
2. Model validation against baseline
3. Model versioning and deployment
4. Regression testing to prevent degradation

Usage:
    python experiments/demo_finetuning_orchestration.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime

from src.data.data_manager import DataManager
from src.data.finetuning_coordinator import FinetuningCoordinator
from src.data.finetuning_orchestrator import FinetuningConfig
from src.data.model_validator import ValidationConfig
from src.data.model_deployer import DeploymentConfig
from src.data.regression_tester import RegressionConfig
from src.baseline_model import BaselineSTTModel

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


def demo_data_manager():
    """Demo 1: Data Manager - Track error cases."""
    print_section("DEMO 1: Data Manager - Tracking Error Cases")
    
    # Initialize data manager
    data_manager = DataManager(
        local_storage_dir="data/test_demo",
        use_gcs=False  # Set to True for GCS integration
    )
    
    print("📊 Simulating error case accumulation...")
    
    # Simulate storing failed cases
    for i in range(5):
        case_id = data_manager.store_failed_case(
            audio_path=f"data/test_audio/test_{i}.wav",
            original_transcript=f"This is test transcription number {i}",
            corrected_transcript=f"This is test transcription number {i} (corrected)",
            error_types=["word_substitution", "missing_word"],
            error_score=0.75,
            metadata={"model": "baseline", "timestamp": datetime.now().isoformat()}
        )
        print(f"  Stored case {i+1}: {case_id}")
    
    # Get statistics
    stats = data_manager.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"   Total error cases: {stats['total_failed_cases']}")
    print(f"   Corrected cases: {stats['corrected_cases']}")
    print(f"   Correction rate: {stats['correction_rate']:.1%}")
    
    return data_manager


def demo_orchestrator(data_manager):
    """Demo 2: Fine-Tuning Orchestrator - Automated triggering."""
    print_section("DEMO 2: Fine-Tuning Orchestrator - Automated Triggering")
    
    # Configure orchestrator
    config = FinetuningConfig(
        min_error_cases=3,  # Low threshold for demo
        min_corrected_cases=2,
        auto_approve_finetuning=True
    )
    
    from src.data.finetuning_orchestrator import FinetuningOrchestrator
    
    orchestrator = FinetuningOrchestrator(
        data_manager=data_manager,
        config=config,
        use_gcs=False
    )
    
    print("🔍 Checking trigger conditions...")
    trigger_result = orchestrator.check_trigger_conditions()
    
    if trigger_result['should_trigger']:
        print("✅ Trigger conditions met!")
        print(f"   Reasons: {', '.join(trigger_result['reasons'])}")
        
        # Trigger fine-tuning
        print("\n🚀 Triggering fine-tuning job...")
        job = orchestrator.trigger_finetuning(force=True)
        
        if job:
            print(f"✅ Job created: {job.job_id}")
            print(f"   Status: {job.status}")
            print(f"   Dataset: {job.dataset_id}")
            
            # Get job info
            job_info = orchestrator.get_job_info(job.job_id)
            print(f"\n📋 Job Details:")
            print(f"   Training samples: {job_info.get('dataset_info', {}).get('split_sizes', {}).get('train', 'N/A')}")
            print(f"   Validation samples: {job_info.get('dataset_info', {}).get('split_sizes', {}).get('val', 'N/A')}")
            
            return job
    else:
        print("⏸️  Trigger conditions not met")
        print(f"   Need {config.min_error_cases - trigger_result['metrics']['total_error_cases']} more cases")
    
    return None


def demo_validator():
    """Demo 3: Model Validator - Validate against baseline."""
    print_section("DEMO 3: Model Validator - Validation Against Baseline")
    
    config = ValidationConfig(
        min_wer_improvement=0.0,  # Allow any improvement
        require_significance=False  # Skip significance test for demo
    )
    
    from src.data.model_validator import ModelValidator
    
    validator = ModelValidator(
        config=config,
        use_gcs=False
    )
    
    print("📝 Creating mock evaluation data...")
    
    # Create simple mock data
    eval_data = [
        {
            'audio_path': f'data/test_audio/test_{i}.wav',
            'reference': f'This is test audio number {i}'
        }
        for i in range(3)
    ]
    
    print(f"   Evaluation samples: {len(eval_data)}")
    
    # Mock transcription functions
    def baseline_transcribe(audio_path):
        """Mock baseline transcription."""
        return "This is test audio number 0"  # Simplified
    
    def model_transcribe(audio_path):
        """Mock fine-tuned model transcription."""
        return "This is test audio number 0"  # Same for demo
    
    print("\n⚖️  Running validation...")
    print("   (Using mock transcription functions for demo)")
    
    try:
        result = validator.validate_model(
            model_id="demo_finetuned_v1",
            model_transcribe_fn=model_transcribe,
            baseline_id="baseline_v1",
            baseline_transcribe_fn=baseline_transcribe,
            evaluation_set=eval_data
        )
        
        print(f"\n📊 Validation Result:")
        print(f"   Passed: {'Yes ✅' if result.passed else 'No ❌'}")
        print(f"   Model WER: {result.model_wer:.4f}")
        print(f"   Baseline WER: {result.baseline_wer:.4f}")
        print(f"   Improvement: {result.wer_improvement:+.4f}")
        
        return result
    except Exception as e:
        print(f"⚠️  Validation demo skipped: {e}")
        return None


def demo_deployer():
    """Demo 4: Model Deployer - Version management and deployment."""
    print_section("DEMO 4: Model Deployer - Version Management")
    
    config = DeploymentConfig(
        keep_previous_versions=3,
        auto_backup_before_deploy=True
    )
    
    from src.data.model_deployer import ModelDeployer
    
    deployer = ModelDeployer(
        config=config,
        storage_dir="data/test_deployed_models",
        use_gcs=False
    )
    
    print("📦 Registering mock model versions...")
    
    # Register some mock versions
    for i in range(3):
        version_id = deployer.register_model(
            model_name=f"test-model-v{i+1}",
            model_path=f"/tmp/mock_model_v{i+1}",
            validation_result={
                'passed': True,
                'model_wer': 0.15 - (i * 0.02),  # Improving
                'model_cer': 0.08 - (i * 0.01)
            }
        )
        print(f"  ✓ Registered: {version_id}")
    
    # List versions
    print("\n📋 Registered Versions:")
    for version in deployer.list_versions(limit=5):
        print(f"  - {version.version_id}: WER={version.wer:.4f}" if version.wer else f"  - {version.version_id}")
    
    # Show deployment status
    deployer.print_status()
    
    return deployer


def demo_regression_tester():
    """Demo 5: Regression Tester - Prevent degradation."""
    print_section("DEMO 5: Regression Tester - Preventing Degradation")
    
    config = RegressionConfig(
        fail_on_any_degradation=False,
        max_failed_samples_rate=0.1
    )
    
    from src.data.regression_tester import RegressionTester
    
    tester = RegressionTester(
        config=config,
        storage_dir="data/test_regression",
        use_gcs=False
    )
    
    print("📝 Registering regression tests...")
    
    # Register a test
    test_id = tester.register_test(
        test_name="Core Benchmark Test",
        test_type="benchmark",
        test_data_path="data/test_regression_samples.jsonl",
        baseline_wer=0.15,
        baseline_cer=0.08,
        baseline_version="baseline_v1",
        max_wer_degradation=0.05,
        description="Critical benchmark that should not degrade"
    )
    
    print(f"  ✓ Registered test: {test_id}")
    
    print(f"\n📊 Registered Tests: {len(tester.tests)}")
    for tid, test in tester.tests.items():
        print(f"  - {test.test_name} ({test.test_type})")
        print(f"    Baseline WER: {test.baseline_wer:.4f}")
        print(f"    Max degradation: {test.max_wer_degradation:.4f}")
    
    return tester


def demo_coordinator(data_manager):
    """Demo 6: Full Coordinator - Complete workflow."""
    print_section("DEMO 6: Fine-Tuning Coordinator - Complete Workflow")
    
    # Initialize coordinator with all components
    coordinator = FinetuningCoordinator(
        data_manager=data_manager,
        finetuning_config=FinetuningConfig(
            min_error_cases=3,
            auto_approve_finetuning=True
        ),
        validation_config=ValidationConfig(
            min_wer_improvement=0.0,
            require_significance=False
        ),
        deployment_config=DeploymentConfig(
            keep_previous_versions=3
        ),
        regression_config=RegressionConfig(
            fail_on_any_degradation=False
        ),
        use_gcs=False,
        storage_dir="data/test_orchestration"
    )
    
    # Show system status
    coordinator.print_status()
    
    print("\n📋 System Configuration:")
    print(f"   Auto-trigger: {coordinator.orchestrator.config.auto_approve_finetuning}")
    print(f"   Min error cases: {coordinator.orchestrator.config.min_error_cases}")
    print(f"   Validation required: Yes")
    print(f"   Regression testing: Yes")
    
    # Check if ready to trigger
    trigger_result = coordinator.orchestrator.check_trigger_conditions()
    
    if trigger_result['should_trigger']:
        print("\n✅ System is ready to trigger fine-tuning!")
        print("\n💡 To run complete workflow:")
        print("   coordinator.run_complete_workflow(force_trigger=True, auto_deploy=True)")
    else:
        print("\n⏸️  Waiting for more error cases...")
    
    return coordinator


def demo_complete_workflow():
    """Demo 7: Complete End-to-End Workflow."""
    print_section("DEMO 7: Complete End-to-End Workflow (Simulated)")
    
    print("This would demonstrate the complete workflow:")
    print("\n1️⃣  Monitor error cases")
    print("    └─ Accumulate failed transcriptions")
    print("    └─ Track corrections")
    
    print("\n2️⃣  Trigger fine-tuning (when threshold met)")
    print("    └─ Prepare training dataset")
    print("    └─ Create data version")
    print("    └─ Launch training job")
    
    print("\n3️⃣  Validate trained model")
    print("    └─ Run on standardized evaluation set")
    print("    └─ Compare against baseline")
    print("    └─ Check statistical significance")
    
    print("\n4️⃣  Run regression tests")
    print("    └─ Test on critical samples")
    print("    └─ Check for degradation")
    print("    └─ Verify edge cases")
    
    print("\n5️⃣  Deploy model (if validation passes)")
    print("    └─ Register model version")
    print("    └─ Backup current model")
    print("    └─ Deploy new version")
    print("    └─ Update active model pointer")
    
    print("\n6️⃣  Continuous monitoring")
    print("    └─ Track performance metrics")
    print("    └─ Alert on degradation")
    print("    └─ Enable rollback if needed")
    
    print("\n💡 For production use:")
    print("   - Set use_gcs=True for cloud storage")
    print("   - Configure actual training callbacks")
    print("   - Set up monitoring and alerting")
    print("   - Use real evaluation datasets")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("  FINE-TUNING ORCHESTRATION SYSTEM - COMPREHENSIVE DEMO")
    print("="*80)
    print("\nThis demo showcases the complete fine-tuning orchestration system:")
    print("  1. Data Management")
    print("  2. Automated Triggering")
    print("  3. Model Validation")
    print("  4. Model Deployment")
    print("  5. Regression Testing")
    print("  6. Complete Workflow Coordination")
    
    try:
        # Run demos
        data_manager = demo_data_manager()
        job = demo_orchestrator(data_manager)
        validation_result = demo_validator()
        deployer = demo_deployer()
        tester = demo_regression_tester()
        coordinator = demo_coordinator(data_manager)
        demo_complete_workflow()
        
        # Final summary
        print_section("DEMO COMPLETE - Summary")
        
        print("✅ Successfully demonstrated:")
        print("  ✓ Data Manager: Error case tracking")
        print("  ✓ Orchestrator: Automated fine-tuning triggers")
        print("  ✓ Validator: Model validation against baseline")
        print("  ✓ Deployer: Version management and deployment")
        print("  ✓ Regression Tester: Degradation prevention")
        print("  ✓ Coordinator: Complete workflow orchestration")
        
        print("\n📚 Next Steps:")
        print("  1. Review the generated data in data/test_* directories")
        print("  2. Check src/data/ for implementation details")
        print("  3. See docs/ for comprehensive documentation")
        print("  4. Configure for production with GCS integration")
        print("  5. Set up training callbacks for actual model training")
        
        print("\n🚀 Ready for Production:")
        print("  - Enable GCS: use_gcs=True")
        print("  - Configure training: set_training_callback()")
        print("  - Set up validation: set_baseline_transcribe_function()")
        print("  - Deploy to GCP: python scripts/deploy_finetuning_to_gcp.py")
        
        print("\n" + "="*80)
        print("  Demo completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed with error: {e}")
        print("   Check logs for details")
        sys.exit(1)


if __name__ == "__main__":
    main()


