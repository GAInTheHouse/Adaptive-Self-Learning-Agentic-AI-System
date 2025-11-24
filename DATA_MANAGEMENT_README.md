# Data Management System for Self-Learning STT Agent

## 🎯 Overview

A comprehensive, production-ready data management system for storing failed transcription cases, tracking performance metrics, preparing fine-tuning datasets, and maintaining data quality with version control. Fully integrated with Google Cloud Storage.

## ✨ Key Features

### 1. **Failed Case Management** 
- Store and organize failed transcriptions
- Track error types and patterns
- Add corrections (manual or automated)
- Export to pandas DataFrame for analysis

### 2. **Performance Tracking**
- Record WER, CER, and custom metrics
- Track model versions and training progress
- Analyze performance trends over time
- Generate comprehensive reports

### 3. **Fine-tuning Pipeline**
- Prepare datasets from failed cases
- Automatic train/val/test splitting
- Balance error types
- Export to HuggingFace format
- Data augmentation support

### 4. **Version Control & Quality**
- Version all datasets with checksums
- Comprehensive quality validation
- Track dataset lineage
- Rollback capability
- Quality gates and thresholds

### 5. **Cloud Integration**
- Seamless Google Cloud Storage sync
- Automatic backup and persistence
- Multi-environment support
- Collaborative workflows

## 🚀 Quick Start

### Installation

```bash
cd /path/to/Adaptive-Self-Learning-Agentic-AI-System
pip install -r requirements.txt
```

### Basic Usage

```python
from src.data.integration import IntegratedDataManagementSystem

# Initialize system
system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=True  # Enable Google Cloud Storage
)

# Record a failed transcription
case_id = system.record_failed_transcription(
    audio_path="audio/sample.wav",
    original_transcript="THIS IS ALL CAPS",
    corrected_transcript="This is proper text",
    error_types=["all_caps"],
    error_score=0.8,
    inference_time=0.5,
    model_confidence=0.85
)

# Prepare fine-tuning dataset
dataset_info = system.prepare_finetuning_dataset(
    min_error_score=0.5,
    max_samples=1000,
    create_version=True
)

# Track training performance
system.record_training_performance(
    model_version="whisper_base_v1",
    wer=0.12,
    cer=0.06,
    training_metadata={"epochs": 10, "batch_size": 16}
)

# Generate report
report = system.generate_comprehensive_report(
    output_path="reports/system_report.json"
)
```

### Run Tests

```bash
python experiments/test_data_management.py
```

## 📁 Project Structure

```
src/data/
├── __init__.py                 # Package exports
├── data_manager.py             # Failed case storage
├── metadata_tracker.py         # Performance tracking
├── finetuning_pipeline.py      # Dataset preparation
├── version_control.py          # Versioning & quality
└── integration.py              # Unified interface

experiments/
├── test_data_management.py     # Comprehensive tests
└── example_usage.py            # Usage examples

docs/
├── DATA_MANAGEMENT_SYSTEM.md   # Full documentation
└── QUICK_START_DATA_MANAGEMENT.md  # Quick start guide

data/  (created at runtime)
├── failed_cases/               # Stored failed cases
├── metadata/                   # Performance metrics
├── finetuning/                 # Prepared datasets
└── versions/                   # Dataset versions
```

## 🔧 Components

### DataManager
Manages storage of failed cases and corrections.

```python
from src.data import DataManager

dm = DataManager(local_storage_dir="data/failed_cases")
case_id = dm.store_failed_case(...)
stats = dm.get_statistics()
```

### MetadataTracker
Tracks performance metrics and model versions.

```python
from src.data import MetadataTracker

tracker = MetadataTracker()
tracker.record_performance(wer=0.12, cer=0.06)
trend = tracker.get_performance_trend('wer')
```

### FinetuningDatasetPipeline
Prepares datasets for model fine-tuning.

```python
from src.data import FinetuningDatasetPipeline

pipeline = FinetuningDatasetPipeline(data_manager=dm)
dataset_info = pipeline.prepare_dataset(max_samples=1000)
```

### DataVersionControl
Manages dataset versions with quality control.

```python
from src.data import DataVersionControl

vc = DataVersionControl()
version_id = vc.create_version(dataset_path, run_quality_check=True)
versions = vc.list_versions(min_quality_score=0.9)
```

### IntegratedDataManagementSystem
Unified interface for all components.

```python
from src.data.integration import IntegratedDataManagementSystem

system = IntegratedDataManagementSystem()
# Use all features through single interface
```

## 📊 Data Flow

```
┌─────────────────┐
│   STT Agent     │
│  (Production)   │
└────────┬────────┘
         │ Detects errors
         ▼
┌─────────────────┐
│  Data Manager   │  ◄─── Store failed cases
│                 │  ◄─── Add corrections
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fine-tuning     │  ◄─── Prepare dataset
│   Pipeline      │  ◄─── Validate quality
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Version Control │  ◄─── Create version
│                 │  ◄─── Quality check
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   GCS Storage   │  ◄─── Sync & backup
│                 │  ◄─── Collaborate
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │  ◄─── Use dataset
│                 │  ◄─── Fine-tune model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Metadata Tracker│  ◄─── Record performance
│                 │  ◄─── Track improvement
└─────────────────┘
```

## 🌟 Key Workflows

### Workflow 1: Production Monitoring
1. Agent detects errors in transcriptions
2. System records failed cases with metadata
3. User provides corrections via feedback
4. System tracks correction rate

### Workflow 2: Dataset Preparation
1. Collect sufficient corrected cases (500+)
2. Prepare fine-tuning dataset with balancing
3. Validate dataset quality
4. Create versioned snapshot
5. Export to HuggingFace format

### Workflow 3: Model Training
1. Load versioned dataset
2. Train model with fine-tuning
3. Record training performance
4. Track WER/CER improvements
5. Compare model versions

### Workflow 4: Continuous Improvement
1. Monitor performance trends
2. Generate regular reports
3. Identify areas for improvement
4. Iterate on dataset preparation
5. Track long-term progress

## 📈 Quality Control

The system implements comprehensive quality checks:

- **Completeness**: All required fields present (95%+ threshold)
- **Consistency**: Logical relationships valid (90%+ threshold)
- **Validity**: Data types and ranges correct (95%+ threshold)
- **Uniqueness**: No duplicates or data leakage (100% required)

Quality reports include:
```json
{
  "passed": true,
  "quality_metrics": {
    "overall_score": 0.95,
    "completeness": 0.98,
    "consistency": 0.93,
    "validity": 0.97,
    "uniqueness": 1.0
  },
  "issues": [],
  "warnings": []
}
```

## ☁️ Google Cloud Integration

### Setup

```bash
# Install gcloud SDK
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login
gcloud config set project stt-agentic-ai-2025

# Create bucket
gsutil mb gs://stt-project-datasets
```

### Automatic Sync

Data is automatically synced to GCS when `use_gcs=True`:
- Failed cases → `gs://bucket/learning_data/`
- Metadata → `gs://bucket/metadata/`
- Datasets → `gs://bucket/finetuning_datasets/`
- Versions → `gs://bucket/data_versions/`

### Manual Sync

```python
# Push to GCS
system.sync_all_to_gcs()

# Pull from GCS
system.sync_all_from_gcs()
```

## 📚 Documentation

- **[Full Documentation](docs/DATA_MANAGEMENT_SYSTEM.md)**: Complete API reference and guides
- **[Quick Start](docs/QUICK_START_DATA_MANAGEMENT.md)**: Get started in 5 minutes
- **[Examples](experiments/example_usage.py)**: Practical usage examples
- **[Tests](experiments/test_data_management.py)**: Comprehensive test suite

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test all components
python experiments/test_data_management.py

# Output:
# ✓ Testing DataManager
# ✓ Testing MetadataTracker  
# ✓ Testing FinetuningDatasetPipeline
# ✓ Testing DataVersionControl
# ✓ Testing IntegratedSystem
# ALL TESTS COMPLETED SUCCESSFULLY!
```

Run example usage scenarios:

```bash
python experiments/example_usage.py
```

## 📊 Statistics & Reporting

Get comprehensive system statistics:

```python
stats = system.get_system_statistics()

# Returns:
{
  'data_management': {
    'total_failed_cases': 1250,
    'corrected_cases': 850,
    'correction_rate': 0.68,
    'error_type_distribution': {...}
  },
  'performance_trends': {
    'wer': {'improvement_percent': 15.3, ...},
    'cer': {'improvement_percent': 12.7, ...}
  },
  'learning_summary': {...},
  'inference_stats': {...},
  'version_control': {...},
  'available_datasets': 5
}
```

Generate detailed reports:

```python
report = system.generate_comprehensive_report(
    output_path="reports/weekly_report.json"
)

# Includes:
# - System statistics
# - Performance trends
# - Data quality assessment
# - Actionable recommendations
```

## 🎓 Usage Examples

### Example 1: Record Failures

```python
# During agent operation
result = agent.transcribe_with_agent("audio.wav")

if result['error_detection']['has_errors']:
    system.record_failed_transcription(
        audio_path="audio.wav",
        original_transcript=result['transcript'],
        corrected_transcript=None,
        error_types=list(result['error_detection']['error_types'].keys()),
        error_score=result['error_detection']['error_score'],
        inference_time=result['inference_time_seconds']
    )
```

### Example 2: Add Corrections

```python
# Get uncorrected cases
uncorrected = system.data_manager.get_uncorrected_cases()

# Add user corrections
for case in uncorrected:
    corrected = get_user_correction(case.original_transcript)
    system.add_correction(case.case_id, corrected)
```

### Example 3: Prepare Dataset

```python
# Prepare when ready
dataset_info = system.prepare_finetuning_dataset(
    min_error_score=0.5,
    max_samples=1000,
    balance_error_types=True,
    create_version=True
)

# Use for training
train_data = load_dataset(dataset_info['local_path'])
```

## 🔍 Monitoring & Maintenance

### Daily
```python
stats = system.get_system_statistics()
print(f"New cases: {stats['data_management']['total_failed_cases']}")
```

### Weekly
```python
report = system.generate_comprehensive_report()
if stats['data_management']['corrected_cases'] >= 500:
    system.prepare_finetuning_dataset(max_samples=1000)
```

### Monthly
```python
system.sync_all_to_gcs()
version_report = system.version_control.generate_version_report()
```

## 🛠️ Troubleshooting

### Common Issues

**No cases being recorded**
```python
# Lower error threshold
agent = STTAgent(baseline_model, error_threshold=0.2)
```

**Dataset preparation fails**
```python
# Check if you have corrected cases
stats = system.data_manager.get_statistics()
if stats['corrected_cases'] < 10:
    print("Need more corrected cases!")
```

**GCS sync errors**
```python
# Test connection
try:
    system.sync_all_to_gcs()
except Exception as e:
    print(f"GCS error: {e}")
    # Use local-only mode
    system = IntegratedDataManagementSystem(use_gcs=False)
```

## 🚀 Performance

- **Storage**: ~1KB per failed case
- **Speed**: Record case in ~10ms (local) + ~100ms (GCS)
- **Scalability**: Handles 100,000+ cases
- **Dataset Prep**: ~1-5 seconds per 1000 samples

## 📦 Dependencies

Core requirements:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `google-cloud-storage`: GCS integration

All dependencies are in `requirements.txt`.

## 🤝 Integration with STT Agent

The data management system integrates seamlessly with the STT agent:

```python
from src.agent.agent import STTAgent
from src.baseline_model import BaselineSTTModel
from src.data.integration import IntegratedDataManagementSystem

# Initialize
baseline_model = BaselineSTTModel()
agent = STTAgent(baseline_model)
data_system = IntegratedDataManagementSystem()

# Process audio
result = agent.transcribe_with_agent("audio.wav")

# Automatically record failures
if result['error_detection']['has_errors']:
    data_system.record_failed_transcription(
        audio_path="audio.wav",
        original_transcript=result['original_transcript'],
        corrected_transcript=None,
        error_types=list(result['error_detection']['error_types'].keys()),
        error_score=result['error_detection']['error_score'],
        inference_time=result['inference_time_seconds']
    )
```

## 🎯 Next Steps

1. ✅ **Run Tests**: `python experiments/test_data_management.py`
2. 📖 **Read Docs**: Check `docs/DATA_MANAGEMENT_SYSTEM.md`
3. 🔧 **Set Up GCS**: Configure Google Cloud Storage
4. 🚀 **Integrate**: Add to your STT agent workflow
5. 📊 **Monitor**: Track performance and generate reports

## 📄 License

Part of the Adaptive Self-Learning Agentic AI System project.

## 👥 Contributors

- Team Member 2: Data Management & Infrastructure

---

**Need Help?** Check the documentation in `docs/` or run the example scripts in `experiments/`.

