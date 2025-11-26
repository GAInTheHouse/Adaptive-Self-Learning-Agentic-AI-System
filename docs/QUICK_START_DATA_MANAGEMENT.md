# Quick Start Guide: Data Management System

## 5-Minute Setup

### 1. Installation

Ensure you have the required dependencies:

```bash
cd /path/to/Adaptive-Self-Learning-Agentic-AI-System
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from src.data.integration import IntegratedDataManagementSystem

# Initialize (local-only for quick start)
system = IntegratedDataManagementSystem(
    base_dir="data/quickstart",
    use_gcs=False  # Set to True for GCS integration
)

# Record a failed transcription
case_id = system.record_failed_transcription(
    audio_path="audio/sample.wav",
    original_transcript="THIS IS ALL CAPS",
    corrected_transcript="This is proper text",
    error_types=["all_caps"],
    error_score=0.8,
    inference_time=0.5
)

print(f"Recorded case: {case_id}")

# Get statistics
stats = system.get_system_statistics()
print(f"Total cases: {stats['data_management']['total_failed_cases']}")
```

### 3. Run Tests

```bash
python experiments/test_data_management.py
```

## Common Workflows

### Workflow 1: Collect Failed Cases During Production

```python
from src.data.integration import IntegratedDataManagementSystem
from src.agent.agent import STTAgent
from src.baseline_model import BaselineSTTModel

# Initialize
system = IntegratedDataManagementSystem(base_dir="data/production")
baseline_model = BaselineSTTModel()
agent = STTAgent(baseline_model)

# Process audio
result = agent.transcribe_with_agent("audio/sample.wav")

# If errors detected, record the case
if result['error_detection']['has_errors']:
    case_id = system.record_failed_transcription(
        audio_path="audio/sample.wav",
        original_transcript=result['original_transcript'],
        corrected_transcript=None,  # Will add later
        error_types=list(result['error_detection']['error_types'].keys()),
        error_score=result['error_detection']['error_score'],
        inference_time=result['inference_time_seconds']
    )
```

### Workflow 2: Add User Corrections

```python
# Get uncorrected cases
uncorrected = system.data_manager.get_uncorrected_cases()

# Add corrections (from user feedback)
for case in uncorrected:
    # User provides corrected text
    corrected_text = get_user_correction(case.original_transcript)
    
    system.add_correction(
        case_id=case.case_id,
        corrected_transcript=corrected_text,
        correction_method='user_feedback'
    )
```

### Workflow 3: Prepare Fine-tuning Dataset

```python
# Prepare dataset when you have enough corrected cases
dataset_info = system.prepare_finetuning_dataset(
    min_error_score=0.5,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    max_samples=1000,
    balance_error_types=True,
    create_version=True  # Creates a versioned snapshot
)

print(f"Dataset ready: {dataset_info['dataset_id']}")
print(f"Train: {dataset_info['split_sizes']['train']} samples")
print(f"Val: {dataset_info['split_sizes']['val']} samples")
print(f"Test: {dataset_info['split_sizes']['test']} samples")

# Get path for training
dataset_path = dataset_info['local_path']
```

### Workflow 4: Track Training Progress

```python
# After training each model version
system.record_training_performance(
    model_version="whisper_base_v1",
    wer=0.12,
    cer=0.06,
    training_metadata={
        'model_name': 'whisper-base',
        'training_data_size': 1000,
        'epochs': 10,
        'batch_size': 16,
        'learning_rate': 1e-5
    }
)

# View performance trends
wer_trend = system.metadata_tracker.get_performance_trend('wer')
print(f"WER improved by {wer_trend['improvement_percent']:.1f}%")
```

### Workflow 5: Generate Reports

```python
# Generate comprehensive report
report = system.generate_comprehensive_report(
    output_path="reports/weekly_report.json"
)

print(f"Data Quality: {report['data_quality']['quality_status']}")
print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"  - {rec}")
```

## Google Cloud Setup (Optional)

### 1. Install Google Cloud SDK

```bash
# Download and install
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud config set project stt-agentic-ai-2025
```

### 2. Create Buckets

```bash
gsutil mb gs://stt-project-datasets
```

### 3. Enable GCS in Code

```python
system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=True,  # Enable GCS
    gcs_bucket_name="stt-project-datasets",
    project_id="stt-agentic-ai-2025"
)
```

## Monitoring & Maintenance

### Daily Checks

```python
# Check system health
stats = system.get_system_statistics()
print(f"New cases today: {stats['data_management']['total_failed_cases']}")
print(f"Correction rate: {stats['data_management']['correction_rate']:.2%}")
```

### Weekly Tasks

```python
# Generate weekly report
report = system.generate_comprehensive_report(
    output_path=f"reports/weekly_{datetime.now().strftime('%Y%m%d')}.json"
)

# Prepare new dataset if needed
if stats['data_management']['corrected_cases'] >= 500:
    dataset_info = system.prepare_finetuning_dataset(
        max_samples=1000,
        create_version=True
    )
```

### Monthly Tasks

```python
# Sync to GCS
system.sync_all_to_gcs()

# Review version history
versions = system.version_control.list_versions()
print(f"Total versions: {len(versions)}")

# Generate version report
version_report = system.version_control.generate_version_report()
```

## Troubleshooting

### Issue: No failed cases being recorded

**Solution:**
```python
# Check if errors are being detected
result = agent.transcribe_with_agent("audio/sample.wav")
print(result['error_detection'])

# Lower error threshold if needed
agent = STTAgent(baseline_model, error_threshold=0.2)
```

### Issue: Dataset preparation fails

**Solution:**
```python
# Check statistics
stats = system.data_manager.get_statistics()
print(f"Total cases: {stats['total_failed_cases']}")
print(f"Corrected: {stats['corrected_cases']}")

# Need at least some corrected cases
if stats['corrected_cases'] < 10:
    print("Need more corrected cases!")
```

### Issue: GCS sync errors

**Solution:**
```python
# Test GCS connection
try:
    system.sync_all_to_gcs()
except Exception as e:
    print(f"GCS error: {e}")
    # Fall back to local-only
    system = IntegratedDataManagementSystem(use_gcs=False)
```

## Next Steps

1. **Explore Examples**: Run `python experiments/example_usage.py`
2. **Read Full Documentation**: See `docs/DATA_MANAGEMENT_SYSTEM.md`
3. **Integrate with Agent**: Add data management to your STT agent workflow
4. **Set Up GCS**: Enable cloud storage for production use
5. **Monitor Performance**: Track metrics and generate regular reports

## Cheat Sheet

```python
# Initialize
system = IntegratedDataManagementSystem(base_dir="data", use_gcs=False)

# Record failure
case_id = system.record_failed_transcription(...)

# Add correction
system.add_correction(case_id, corrected_text)

# Prepare dataset
dataset_info = system.prepare_finetuning_dataset(max_samples=1000, create_version=True)

# Track training
system.record_training_performance(model_version, wer, cer, metadata)

# Get statistics
stats = system.get_system_statistics()

# Generate report
report = system.generate_comprehensive_report(output_path="report.json")

# Sync GCS
system.sync_all_to_gcs()
```

## Help & Support

- **Documentation**: `docs/DATA_MANAGEMENT_SYSTEM.md`
- **Examples**: `experiments/example_usage.py`
- **Tests**: `experiments/test_data_management.py`
- **Issues**: Check logs in `data/*/` directories

