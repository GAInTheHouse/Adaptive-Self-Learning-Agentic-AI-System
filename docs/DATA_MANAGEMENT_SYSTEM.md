# Data Management System Documentation

## Overview

The Data Management System is a comprehensive solution for managing failed transcription cases, tracking performance metrics, preparing fine-tuning datasets, and maintaining data quality with version control. It is designed to integrate seamlessly with Google Cloud Storage for scalable, production-ready deployments.

## Architecture

The system consists of four main components:

```
┌─────────────────────────────────────────────────────────────┐
│           Integrated Data Management System                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐  ┌──────────────────┐                    │
│  │ Data Manager  │  │ Metadata Tracker │                    │
│  │               │  │                  │                    │
│  │ - Store       │  │ - Performance    │                    │
│  │   failures    │  │   metrics        │                    │
│  │ - Corrections │  │ - Model versions │                    │
│  │ - Statistics  │  │ - Learning       │                    │
│  └───────────────┘  │   progress       │                    │
│                     └──────────────────┘                    │
│                                                               │
│  ┌───────────────┐  ┌──────────────────┐                    │
│  │ Fine-tuning   │  │ Version Control  │                    │
│  │ Pipeline      │  │                  │                    │
│  │               │  │ - Versioning     │                    │
│  │ - Dataset     │  │ - Quality checks │                    │
│  │   preparation │  │ - Validation     │                    │
│  │ - Validation  │  │ - Rollback       │                    │
│  └───────────────┘  └──────────────────┘                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Google Cloud Storage  │
                │                       │
                │ - Persistent storage  │
                │ - Sync & backup       │
                │ - Collaboration       │
                └───────────────────────┘
```

## Components

### 1. DataManager

Manages storage and retrieval of failed transcription cases and corrections.

**Key Features:**
- Store failed transcription cases with metadata
- Add corrections to cases
- Query cases by error type
- Export data to pandas DataFrame
- Sync with Google Cloud Storage

**Usage:**

```python
from src.data import DataManager

# Initialize
data_manager = DataManager(
    local_storage_dir="data/failed_cases",
    use_gcs=True,
    gcs_bucket_name="stt-project-datasets"
)

# Store a failed case
case_id = data_manager.store_failed_case(
    audio_path="audio/sample.wav",
    original_transcript="THIS IS ALL CAPS",
    corrected_transcript="This is proper text",
    error_types=["all_caps"],
    error_score=0.8,
    metadata={"confidence": 0.85}
)

# Get statistics
stats = data_manager.get_statistics()
print(f"Total cases: {stats['total_failed_cases']}")
print(f"Correction rate: {stats['correction_rate']:.2%}")
```

### 2. MetadataTracker

Tracks performance metrics, model versions, and learning progress over time.

**Key Features:**
- Record performance metrics (WER, CER, etc.)
- Track model versions
- Monitor learning progress
- Analyze performance trends
- Generate performance reports

**Usage:**

```python
from src.data import MetadataTracker

# Initialize
tracker = MetadataTracker(
    local_storage_dir="data/metadata",
    use_gcs=True
)

# Record performance
tracker.record_performance(
    wer=0.12,
    cer=0.06,
    error_rate=0.15,
    correction_rate=0.85,
    inference_time=0.5
)

# Get performance trend
wer_trend = tracker.get_performance_trend('wer')
print(f"WER improvement: {wer_trend['improvement_percent']:.2f}%")

# Record model version
tracker.record_model_version(
    version_id="whisper_base_v1",
    model_name="whisper-base",
    training_data_size=1000,
    training_metadata={"epochs": 10, "batch_size": 16},
    performance_metrics={"wer": 0.12, "cer": 0.06}
)
```

### 3. FinetuningDatasetPipeline

Prepares datasets for model fine-tuning from failed cases and corrections.

**Key Features:**
- Prepare train/val/test splits
- Balance error types
- Validate dataset quality
- Export to HuggingFace format
- Data augmentation

**Usage:**

```python
from src.data import FinetuningDatasetPipeline, DataManager

# Initialize
data_manager = DataManager()
pipeline = FinetuningDatasetPipeline(
    data_manager=data_manager,
    output_dir="data/finetuning"
)

# Prepare dataset
dataset_info = pipeline.prepare_dataset(
    min_error_score=0.5,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    max_samples=1000,
    balance_error_types=True
)

print(f"Dataset: {dataset_info['dataset_id']}")
print(f"Train samples: {dataset_info['split_sizes']['train']}")

# Validate dataset
validation = pipeline.validate_dataset(dataset_info['dataset_id'])
print(f"Valid: {validation['is_valid']}")

# Prepare for HuggingFace
hf_path = pipeline.prepare_huggingface_dataset(
    dataset_id=dataset_info['dataset_id'],
    output_format='json'
)
```

### 4. DataVersionControl

Manages dataset versions with quality control and validation.

**Key Features:**
- Create versioned datasets
- Quality validation
- Version comparison
- Lineage tracking
- Rollback capability

**Usage:**

```python
from src.data import DataVersionControl

# Initialize
version_control = DataVersionControl(
    local_storage_dir="data/versions",
    use_gcs=True
)

# Create version
version_id = version_control.create_version(
    dataset_path="data/finetuning/dataset_20250101",
    version_name="initial",
    metadata={"description": "Initial dataset"},
    run_quality_check=True
)

# List versions
versions = version_control.list_versions(min_quality_score=0.9)
for v in versions:
    print(f"{v.version_id}: {v.checksum}")

# Compare versions
comparison = version_control.compare_versions(version_id1, version_id2)
print(f"Quality improvement: {comparison['quality_comparison']['improvement']:.2f}")

# Check quality
quality_report = version_control.check_quality("data/finetuning/dataset_20250101")
print(f"Passed: {quality_report['passed']}")
print(f"Score: {quality_report['quality_metrics']['overall_score']:.2f}")
```

### 5. IntegratedDataManagementSystem

Unified interface for all components.

**Key Features:**
- Single entry point for all functionality
- Automated workflows
- Comprehensive reporting
- Recommendations generation

**Usage:**

```python
from src.data.integration import IntegratedDataManagementSystem

# Initialize
system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=True,
    gcs_bucket_name="stt-project-datasets"
)

# Record failed transcription
case_id = system.record_failed_transcription(
    audio_path="audio/sample.wav",
    original_transcript="original text",
    corrected_transcript="corrected text",
    error_types=["all_caps"],
    error_score=0.8,
    inference_time=0.5,
    model_confidence=0.85
)

# Prepare fine-tuning dataset with versioning
dataset_info = system.prepare_finetuning_dataset(
    min_error_score=0.5,
    max_samples=1000,
    create_version=True
)

# Record training performance
system.record_training_performance(
    model_version="whisper_base_v1",
    wer=0.12,
    cer=0.06,
    training_metadata={"epochs": 10}
)

# Get system statistics
stats = system.get_system_statistics()
print(stats)

# Generate comprehensive report
report = system.generate_comprehensive_report(
    output_path="reports/system_report.json"
)
```

## Quality Control Mechanisms

The system implements comprehensive quality control:

### 1. Completeness Check
- Verifies all required fields are present
- Checks for non-empty values
- Validates field types

### 2. Consistency Check
- Ensures logical consistency between fields
- Validates relationships (e.g., input vs. target text)
- Checks file path formats

### 3. Validity Check
- Validates text content (length, characters)
- Checks score ranges (0.0 to 1.0)
- Verifies data types

### 4. Uniqueness Check
- Detects duplicate samples
- Prevents data leakage between splits
- Ensures unique identifiers

### Quality Thresholds

Default thresholds (configurable):
- Minimum completeness: 95%
- Minimum consistency: 90%
- Minimum validity: 95%
- Minimum overall score: 90%

## Data Versioning

### Version Tracking

Each dataset version includes:
- **Version ID**: Unique identifier
- **Checksum**: Data integrity verification
- **Metadata**: Configuration and statistics
- **Parent Version**: For lineage tracking
- **Quality Report**: Validation results

### Version Lineage

Track dataset evolution:
```
v_initial_20250101
    └── v_augmented_20250115
            └── v_refined_20250130
```

### Rollback

Restore previous versions:
```python
version_control.rollback_to_version(
    version_id="v_initial_20250101",
    target_path="data/restored"
)
```

## Google Cloud Storage Integration

### Setup

1. Install Google Cloud SDK:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

2. Authenticate:
```bash
gcloud auth login
gcloud config set project stt-agentic-ai-2025
```

3. Create buckets:
```bash
gsutil mb gs://stt-project-datasets
```

### Sync Operations

**Automatic Sync:**
Data is automatically synced to GCS after each operation when `use_gcs=True`.

**Manual Sync:**
```python
# Sync to GCS
system.sync_all_to_gcs()

# Sync from GCS
system.sync_all_from_gcs()
```

### Bucket Structure

```
gs://stt-project-datasets/
├── learning_data/
│   ├── failed_cases.jsonl
│   ├── corrections.jsonl
│   └── metadata.json
├── metadata/
│   ├── performance_history.jsonl
│   ├── model_versions.json
│   ├── learning_progress.jsonl
│   └── inference_stats.jsonl
├── finetuning_datasets/
│   └── dataset_20250101/
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       └── metadata.json
└── data_versions/
    └── version_registry.json
```

## Best Practices

### 1. Regular Syncing

```python
# Sync every N operations
if operation_count % 100 == 0:
    system.sync_all_to_gcs()
```

### 2. Quality Gates

```python
# Only create versions that pass quality checks
quality_report = version_control.check_quality(dataset_path)
if quality_report['passed']:
    version_control.create_version(dataset_path)
```

### 3. Error Handling

```python
try:
    case_id = system.record_failed_transcription(...)
except Exception as e:
    logger.error(f"Failed to record case: {e}")
    # Fallback to local storage only
```

### 4. Monitoring

```python
# Regular reporting
report = system.generate_comprehensive_report()
for recommendation in report['recommendations']:
    logger.info(f"Recommendation: {recommendation}")
```

### 5. Dataset Validation

```python
# Always validate before training
validation = pipeline.validate_dataset(dataset_id)
if not validation['is_valid']:
    raise ValueError(f"Dataset validation failed: {validation['issues']}")
```

## Testing

Run comprehensive tests:

```bash
# Test all components
python experiments/test_data_management.py

# Run example usage
python experiments/example_usage.py
```

## Troubleshooting

### GCS Connection Issues

```python
# Disable GCS temporarily
system = IntegratedDataManagementSystem(use_gcs=False)
```

### Quality Check Failures

```python
# Adjust quality thresholds
version_control.quality_thresholds = {
    'min_completeness': 0.90,
    'min_consistency': 0.85,
    'min_validity': 0.90,
    'min_overall_score': 0.85
}
```

### Large Dataset Performance

```python
# Use sampling for large datasets
dataset_info = pipeline.prepare_dataset(
    max_samples=5000,
    balance_error_types=True
)
```

## API Reference

### DataManager

| Method | Description |
|--------|-------------|
| `store_failed_case()` | Store a failed transcription case |
| `store_correction()` | Add correction to a case |
| `get_failed_case()` | Retrieve case by ID |
| `get_failed_cases_by_error_type()` | Get cases by error type |
| `get_corrected_cases()` | Get all corrected cases |
| `get_statistics()` | Get data statistics |
| `export_to_dataframe()` | Export to pandas DataFrame |

### MetadataTracker

| Method | Description |
|--------|-------------|
| `record_performance()` | Record performance metrics |
| `record_model_version()` | Record model version |
| `record_learning_progress()` | Record learning stage |
| `record_inference_stats()` | Record inference statistics |
| `get_performance_trend()` | Get trend for metric |
| `generate_performance_report()` | Generate report |

### FinetuningDatasetPipeline

| Method | Description |
|--------|-------------|
| `prepare_dataset()` | Prepare fine-tuning dataset |
| `validate_dataset()` | Validate dataset quality |
| `prepare_huggingface_dataset()` | Convert to HF format |
| `augment_dataset()` | Apply data augmentation |
| `list_datasets()` | List all datasets |

### DataVersionControl

| Method | Description |
|--------|-------------|
| `create_version()` | Create new version |
| `get_version()` | Get version by ID |
| `list_versions()` | List all versions |
| `compare_versions()` | Compare two versions |
| `check_quality()` | Run quality checks |
| `rollback_to_version()` | Restore version |

## Performance Considerations

### Storage

- **Local**: ~1KB per failed case
- **GCS**: Negligible cost for metadata
- **Datasets**: ~100MB per 10,000 samples

### Speed

- **Record case**: ~10ms (local) + ~100ms (GCS sync)
- **Prepare dataset**: ~1-5 seconds per 1000 samples
- **Quality check**: ~2-10 seconds per dataset
- **Version creation**: ~1-3 seconds

### Scalability

- Designed for 100,000+ failed cases
- Handles datasets up to 1M samples
- Supports concurrent access (via GCS)

## Roadmap

- [ ] Advanced data augmentation techniques
- [ ] Real-time quality monitoring dashboard
- [ ] Automated dataset preparation triggers
- [ ] Multi-cloud support (AWS, Azure)
- [ ] Enhanced anomaly detection
- [ ] Collaborative correction workflows

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review example usage scripts
3. Run test suite for diagnostics
4. Consult GCP documentation for cloud issues

