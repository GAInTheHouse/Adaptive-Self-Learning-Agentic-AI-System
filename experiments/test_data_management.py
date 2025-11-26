"""
Test script for Data Management System
Demonstrates all features of the integrated data management system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.integration import IntegratedDataManagementSystem
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_manager():
    """Test DataManager functionality."""
    logger.info("=" * 80)
    logger.info("Testing DataManager")
    logger.info("=" * 80)
    
    # Initialize system (with GCS disabled for local testing)
    system = IntegratedDataManagementSystem(
        base_dir="data/test_data_management",
        use_gcs=False
    )
    
    # Test 1: Record failed cases
    logger.info("\n--- Test 1: Recording Failed Cases ---")
    
    case_ids = []
    for i in range(5):
        case_id = system.record_failed_transcription(
            audio_path=f"test_audio/sample_{i}.wav",
            original_transcript=f"THIS IS ALL CAPS TEXT {i}",
            corrected_transcript=f"This is proper text {i}",
            error_types=["all_caps", "low_confidence"],
            error_score=0.8,
            inference_time=0.5 + i * 0.1,
            model_confidence=0.6 + i * 0.05,
            additional_metadata={"test_id": i}
        )
        case_ids.append(case_id)
        logger.info(f"Recorded case: {case_id}")
    
    # Test 2: Get statistics
    logger.info("\n--- Test 2: Getting Statistics ---")
    stats = system.data_manager.get_statistics()
    logger.info(f"Total failed cases: {stats['total_failed_cases']}")
    logger.info(f"Corrected cases: {stats['corrected_cases']}")
    logger.info(f"Correction rate: {stats['correction_rate']:.2%}")
    logger.info(f"Error type distribution: {stats['error_type_distribution']}")
    
    # Test 3: Add corrections
    logger.info("\n--- Test 3: Adding Corrections ---")
    for case_id in case_ids[:3]:
        success = system.add_correction(
            case_id=case_id,
            corrected_transcript="Manually corrected text",
            correction_method="manual"
        )
        logger.info(f"Added correction for {case_id}: {success}")
    
    # Test 4: Export to DataFrame
    logger.info("\n--- Test 4: Exporting to DataFrame ---")
    df = system.data_manager.export_to_dataframe()
    logger.info(f"Exported {len(df)} cases to DataFrame")
    logger.info(f"Columns: {list(df.columns)}")
    
    return system


def test_metadata_tracker(system):
    """Test MetadataTracker functionality."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing MetadataTracker")
    logger.info("=" * 80)
    
    # Test 1: Record performance metrics
    logger.info("\n--- Test 1: Recording Performance Metrics ---")
    for i in range(5):
        system.metadata_tracker.record_performance(
            wer=0.15 - i * 0.01,  # Improving WER
            cer=0.08 - i * 0.005,  # Improving CER
            error_rate=0.20 - i * 0.02,
            correction_rate=0.60 + i * 0.05,
            inference_time=0.5 + i * 0.02
        )
    logger.info("Recorded 5 performance metrics")
    
    # Test 2: Get performance trends
    logger.info("\n--- Test 2: Performance Trends ---")
    wer_trend = system.metadata_tracker.get_performance_trend('wer')
    logger.info(f"WER Trend:")
    logger.info(f"  Latest: {wer_trend['latest']:.4f}")
    logger.info(f"  Mean: {wer_trend['mean']:.4f}")
    logger.info(f"  Improvement: {wer_trend['improvement']:.4f}")
    logger.info(f"  Improvement %: {wer_trend['improvement_percent']:.2f}%")
    
    # Test 3: Record model versions
    logger.info("\n--- Test 3: Recording Model Versions ---")
    system.metadata_tracker.record_model_version(
        version_id="whisper_base_v1",
        model_name="whisper-base",
        training_data_size=1000,
        training_metadata={
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 1e-5
        },
        performance_metrics={"wer": 0.12, "cer": 0.06}
    )
    logger.info("Recorded model version")
    
    # Test 4: Generate performance report
    logger.info("\n--- Test 4: Performance Report ---")
    report = system.metadata_tracker.generate_performance_report()
    logger.info(f"Generated report with {len(report['performance_trends'])} trend metrics")
    logger.info(f"Model versions: {report['model_versions']}")


def test_finetuning_pipeline(system):
    """Test FinetuningDatasetPipeline functionality."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing FinetuningDatasetPipeline")
    logger.info("=" * 80)
    
    # Add more test cases first
    logger.info("\n--- Adding More Test Cases ---")
    error_types_list = [
        ["all_caps"],
        ["low_confidence"],
        ["repeated_chars"],
        ["too_short"],
        ["unusual_word_pattern"]
    ]
    
    for i in range(20):
        system.record_failed_transcription(
            audio_path=f"test_audio/sample_{i+5}.wav",
            original_transcript=f"Test transcript with errors {i}",
            corrected_transcript=f"Test transcript corrected {i}",
            error_types=[error_types_list[i % len(error_types_list)]],
            error_score=0.5 + (i % 5) * 0.1,
            inference_time=0.4 + i * 0.01,
            model_confidence=0.5 + i * 0.02
        )
    
    logger.info("Added 20 more test cases")
    
    # Test 1: Prepare dataset
    logger.info("\n--- Test 1: Preparing Fine-tuning Dataset ---")
    dataset_info = system.prepare_finetuning_dataset(
        min_error_score=0.5,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        max_samples=20,
        balance_error_types=True,
        create_version=True
    )
    
    if 'error' not in dataset_info:
        logger.info(f"Dataset prepared: {dataset_info['dataset_id']}")
        logger.info(f"Train samples: {dataset_info['split_sizes']['train']}")
        logger.info(f"Val samples: {dataset_info['split_sizes']['val']}")
        logger.info(f"Test samples: {dataset_info['split_sizes']['test']}")
        logger.info(f"Version ID: {dataset_info.get('version_id', 'N/A')}")
        
        # Test 2: List datasets
        logger.info("\n--- Test 2: Listing Datasets ---")
        datasets = system.finetuning_pipeline.list_datasets()
        logger.info(f"Found {len(datasets)} datasets")
        for ds in datasets:
            logger.info(f"  - {ds['dataset_id']} (created: {ds['created_at']})")
        
        # Test 3: Validate dataset
        logger.info("\n--- Test 3: Validating Dataset ---")
        validation = system.finetuning_pipeline.validate_dataset(dataset_info['dataset_id'])
        logger.info(f"Validation result: {'PASSED' if validation['is_valid'] else 'FAILED'}")
        logger.info(f"Issues: {len(validation['issues'])}")
        logger.info(f"Warnings: {len(validation['warnings'])}")
        
        return dataset_info
    else:
        logger.error(f"Dataset preparation failed: {dataset_info['error']}")
        return None


def test_version_control(system, dataset_info):
    """Test DataVersionControl functionality."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing DataVersionControl")
    logger.info("=" * 80)
    
    if not dataset_info:
        logger.warning("No dataset info available, skipping version control tests")
        return
    
    # Test 1: List versions
    logger.info("\n--- Test 1: Listing Versions ---")
    versions = system.version_control.list_versions()
    logger.info(f"Found {len(versions)} versions")
    for version in versions:
        logger.info(f"  - {version.version_id}")
        logger.info(f"    Checksum: {version.checksum}")
        logger.info(f"    Created: {version.created_at}")
    
    # Test 2: Get version info
    if versions:
        logger.info("\n--- Test 2: Version Details ---")
        version = versions[0]
        logger.info(f"Version: {version.version_id}")
        logger.info(f"Dataset path: {version.dataset_path}")
        quality_report = version.metadata.get('quality_report', {})
        if quality_report:
            logger.info(f"Quality passed: {quality_report.get('passed', False)}")
            metrics = quality_report.get('quality_metrics', {})
            logger.info(f"Quality score: {metrics.get('overall_score', 0):.2f}")
    
    # Test 3: Generate version report
    logger.info("\n--- Test 3: Version Control Report ---")
    report = system.version_control.generate_version_report()
    logger.info(f"Total versions: {report['total_versions']}")
    logger.info(f"Versions by quality: {dict(report['versions_by_quality'])}")


def test_integrated_system():
    """Test integrated system functionality."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Integrated System")
    logger.info("=" * 80)
    
    system = IntegratedDataManagementSystem(
        base_dir="data/test_data_management",
        use_gcs=False
    )
    
    # Test 1: Get system statistics
    logger.info("\n--- Test 1: System Statistics ---")
    stats = system.get_system_statistics()
    logger.info(f"Data management stats:")
    logger.info(f"  Total failed cases: {stats['data_management']['total_failed_cases']}")
    logger.info(f"  Correction rate: {stats['data_management']['correction_rate']:.2%}")
    logger.info(f"Available datasets: {stats['available_datasets']}")
    
    # Test 2: Generate comprehensive report
    logger.info("\n--- Test 2: Comprehensive Report ---")
    report_path = "data/test_data_management/comprehensive_report.json"
    report = system.generate_comprehensive_report(output_path=report_path)
    logger.info(f"Report generated at: {report_path}")
    logger.info(f"Data quality status: {report['data_quality']['quality_status']}")
    logger.info(f"Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        logger.info(f"  {i}. {rec}")
    
    return system


def main():
    """Run all tests."""
    logger.info("Starting Data Management System Tests")
    logger.info("=" * 80)
    
    try:
        # Test individual components
        system = test_data_manager()
        test_metadata_tracker(system)
        dataset_info = test_finetuning_pipeline(system)
        test_version_control(system, dataset_info)
        
        # Test integrated system
        test_integrated_system()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

