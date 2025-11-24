"""
Example Usage of Data Management System
Demonstrates practical usage scenarios.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.integration import IntegratedDataManagementSystem
from src.agent.agent import STTAgent
from src.baseline_model import BaselineSTTModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_1_record_failures():
    """Example 1: Record failed transcriptions during agent operation."""
    logger.info("=" * 80)
    logger.info("Example 1: Recording Failed Transcriptions")
    logger.info("=" * 80)
    
    # Initialize data management system
    data_system = IntegratedDataManagementSystem(
        base_dir="data/production",
        use_gcs=True  # Enable GCS for production
    )
    
    # Initialize STT agent (using mock for example)
    # In production, you would use actual model
    # baseline_model = BaselineSTTModel()
    # agent = STTAgent(baseline_model)
    
    # Simulate processing audio files
    audio_files = [
        "audio/user_recording_1.wav",
        "audio/user_recording_2.wav",
        "audio/user_recording_3.wav"
    ]
    
    for audio_path in audio_files:
        # In production, you would call: result = agent.transcribe_with_agent(audio_path)
        # For this example, we'll simulate the result
        
        result = {
            'transcript': 'THIS IS ALL CAPS',
            'original_transcript': 'THIS IS ALL CAPS',
            'inference_time_seconds': 0.5,
            'error_detection': {
                'has_errors': True,
                'error_score': 0.7,
                'error_types': {'all_caps': 1},
                'errors': [{'type': 'all_caps', 'confidence': 0.7}]
            },
            'confidence': 0.85
        }
        
        # If errors detected, record the case
        if result['error_detection']['has_errors']:
            case_id = data_system.record_failed_transcription(
                audio_path=audio_path,
                original_transcript=result['original_transcript'],
                corrected_transcript=None,  # Will be added later when user provides feedback
                error_types=list(result['error_detection']['error_types'].keys()),
                error_score=result['error_detection']['error_score'],
                inference_time=result['inference_time_seconds'],
                model_confidence=result.get('confidence'),
                additional_metadata={
                    'error_details': result['error_detection']['errors']
                }
            )
            logger.info(f"Recorded failed case {case_id} for {audio_path}")
    
    # Get statistics
    stats = data_system.data_manager.get_statistics()
    logger.info(f"\nCurrent statistics:")
    logger.info(f"  Total failed cases: {stats['total_failed_cases']}")
    logger.info(f"  Uncorrected cases: {stats['uncorrected_cases']}")


def example_2_add_corrections():
    """Example 2: Add user corrections to failed cases."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: Adding User Corrections")
    logger.info("=" * 80)
    
    # Initialize system
    data_system = IntegratedDataManagementSystem(
        base_dir="data/production",
        use_gcs=True
    )
    
    # Get uncorrected cases
    uncorrected = data_system.data_manager.get_uncorrected_cases()
    logger.info(f"Found {len(uncorrected)} uncorrected cases")
    
    # Simulate user providing corrections
    for case in uncorrected[:3]:  # Correct first 3 cases
        # In production, this would come from user feedback
        corrected_text = case.original_transcript.capitalize()
        
        success = data_system.add_correction(
            case_id=case.case_id,
            corrected_transcript=corrected_text,
            correction_method='user_feedback'
        )
        
        if success:
            logger.info(f"Added correction for case {case.case_id}")
    
    # Updated statistics
    stats = data_system.data_manager.get_statistics()
    logger.info(f"\nUpdated statistics:")
    logger.info(f"  Corrected cases: {stats['corrected_cases']}")
    logger.info(f"  Correction rate: {stats['correction_rate']:.2%}")


def example_3_prepare_finetuning():
    """Example 3: Prepare fine-tuning dataset."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: Preparing Fine-tuning Dataset")
    logger.info("=" * 80)
    
    # Initialize system
    data_system = IntegratedDataManagementSystem(
        base_dir="data/production",
        use_gcs=True
    )
    
    # Prepare dataset with versioning
    dataset_info = data_system.prepare_finetuning_dataset(
        min_error_score=0.5,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        max_samples=1000,
        balance_error_types=True,
        create_version=True
    )
    
    if 'error' not in dataset_info:
        logger.info(f"Dataset prepared successfully!")
        logger.info(f"  Dataset ID: {dataset_info['dataset_id']}")
        logger.info(f"  Version ID: {dataset_info.get('version_id', 'N/A')}")
        logger.info(f"  Total samples: {dataset_info['total_samples']}")
        logger.info(f"  Train: {dataset_info['split_sizes']['train']}")
        logger.info(f"  Val: {dataset_info['split_sizes']['val']}")
        logger.info(f"  Test: {dataset_info['split_sizes']['test']}")
        
        # Prepare in HuggingFace format for training
        hf_path = data_system.finetuning_pipeline.prepare_huggingface_dataset(
            dataset_id=dataset_info['dataset_id'],
            output_format='json'
        )
        logger.info(f"\nHuggingFace format dataset: {hf_path}")
        
        return dataset_info['dataset_id']
    else:
        logger.error(f"Dataset preparation failed: {dataset_info['error']}")
        return None


def example_4_track_training():
    """Example 4: Track model training performance."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 4: Tracking Training Performance")
    logger.info("=" * 80)
    
    # Initialize system
    data_system = IntegratedDataManagementSystem(
        base_dir="data/production",
        use_gcs=True
    )
    
    # Simulate training iterations
    model_versions = [
        ("whisper_base_v1", 0.15, 0.08),
        ("whisper_base_v2", 0.13, 0.07),
        ("whisper_base_v3", 0.11, 0.06)
    ]
    
    for version, wer, cer in model_versions:
        data_system.record_training_performance(
            model_version=version,
            wer=wer,
            cer=cer,
            training_metadata={
                'model_name': 'whisper-base',
                'training_data_size': 1000,
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 1e-5
            }
        )
        logger.info(f"Recorded performance for {version}: WER={wer:.4f}, CER={cer:.4f}")
    
    # Get performance trends
    wer_trend = data_system.metadata_tracker.get_performance_trend('wer')
    logger.info(f"\nWER Improvement:")
    logger.info(f"  Initial: {wer_trend['values'][0]:.4f}")
    logger.info(f"  Latest: {wer_trend['latest']:.4f}")
    logger.info(f"  Improvement: {wer_trend['improvement']:.4f} ({wer_trend['improvement_percent']:.1f}%)")


def example_5_quality_control():
    """Example 5: Quality control and validation."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 5: Quality Control")
    logger.info("=" * 80)
    
    # Initialize system
    data_system = IntegratedDataManagementSystem(
        base_dir="data/production",
        use_gcs=True
    )
    
    # List all datasets
    datasets = data_system.finetuning_pipeline.list_datasets()
    logger.info(f"Found {len(datasets)} datasets")
    
    # Validate each dataset
    for dataset in datasets:
        dataset_id = dataset['dataset_id']
        validation = data_system.finetuning_pipeline.validate_dataset(dataset_id)
        
        logger.info(f"\nDataset: {dataset_id}")
        logger.info(f"  Valid: {validation['is_valid']}")
        logger.info(f"  Issues: {len(validation['issues'])}")
        logger.info(f"  Warnings: {len(validation['warnings'])}")
        
        if validation['statistics']:
            logger.info(f"  Train samples: {validation['statistics'].get('train', {}).get('num_samples', 0)}")


def example_6_comprehensive_report():
    """Example 6: Generate comprehensive system report."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 6: Comprehensive System Report")
    logger.info("=" * 80)
    
    # Initialize system
    data_system = IntegratedDataManagementSystem(
        base_dir="data/production",
        use_gcs=True
    )
    
    # Generate report
    report = data_system.generate_comprehensive_report(
        output_path="data/production/reports/system_report.json"
    )
    
    logger.info("System Report Generated:")
    logger.info(f"  Data Quality: {report['data_quality']['quality_status']}")
    logger.info(f"  Total Failed Cases: {report['data_quality']['total_failed_cases']}")
    logger.info(f"  Correction Rate: {report['data_quality']['correction_rate']:.2%}")
    
    logger.info("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        logger.info(f"  {i}. {rec}")


def example_7_version_management():
    """Example 7: Dataset version management."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 7: Version Management")
    logger.info("=" * 80)
    
    # Initialize system
    data_system = IntegratedDataManagementSystem(
        base_dir="data/production",
        use_gcs=True
    )
    
    # List all versions
    versions = data_system.version_control.list_versions()
    logger.info(f"Found {len(versions)} dataset versions")
    
    # Show version details
    for version in versions[:3]:  # Show first 3
        logger.info(f"\nVersion: {version.version_id}")
        logger.info(f"  Created: {version.created_at}")
        logger.info(f"  Checksum: {version.checksum}")
        
        quality_report = version.metadata.get('quality_report', {})
        if quality_report:
            logger.info(f"  Quality Score: {quality_report.get('quality_metrics', {}).get('overall_score', 0):.2f}")
    
    # Compare versions if we have at least 2
    if len(versions) >= 2:
        comparison = data_system.version_control.compare_versions(
            versions[0].version_id,
            versions[1].version_id
        )
        logger.info(f"\nVersion Comparison:")
        logger.info(f"  Checksum match: {comparison['checksum_match']}")
        logger.info(f"  Quality improvement: {comparison['quality_comparison']['improvement']:.2f}")


def main():
    """Run all examples."""
    logger.info("Data Management System - Example Usage")
    logger.info("=" * 80)
    
    try:
        # Note: These examples assume you have some data already
        # For a fresh start, run test_data_management.py first
        
        # Example 1: Record failures during operation
        example_1_record_failures()
        
        # Example 2: Add user corrections
        example_2_add_corrections()
        
        # Example 3: Prepare fine-tuning dataset
        example_3_prepare_finetuning()
        
        # Example 4: Track training performance
        example_4_track_training()
        
        # Example 5: Quality control
        example_5_quality_control()
        
        # Example 6: Generate comprehensive report
        example_6_comprehensive_report()
        
        # Example 7: Version management
        example_7_version_management()
        
        logger.info("\n" + "=" * 80)
        logger.info("All examples completed!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

