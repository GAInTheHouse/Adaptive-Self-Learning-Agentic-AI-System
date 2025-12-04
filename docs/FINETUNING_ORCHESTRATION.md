# Fine-Tuning Orchestration System

**Complete Guide to Automated Model Fine-Tuning, Validation, and Deployment**

## Overview

The Fine-Tuning Orchestration System provides a comprehensive, automated pipeline for improving speech-to-text models through continuous learning from error cases. The system handles the complete lifecycle:

1. **Automated Triggering** - Monitors error accumulation and triggers fine-tuning
2. **Model Validation** - Validates models against baseline with statistical testing
3. **Version Management** - Manages model versions with deployment and rollback
4. **Regression Testing** - Prevents performance degradation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Fine-Tuning Coordinator                        │
│                  (Central Orchestration)                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬─────────────────┐
        │               │               │                 │
        ▼               ▼               ▼                 ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│  Fine-Tuning │ │    Model    │ │    Model    │ │  Regression  │
│ Orchestrator │ │  Validator  │ │  Deployer   │ │   Tester     │
└──────────────┘ └─────────────┘ └─────────────┘ └──────────────┘
        │               │               │                 │
        ▼               ▼               ▼                 ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│     Data     │ │ Evaluation  │ │   Version   │ │ Test Suites  │
│   Manager    │ │   Metrics   │ │   Control   │ │              │
└──────────────┘ └─────────────┘ └─────────────┘ └──────────────┘
        │               │               │                 │
        └───────────────┴───────────────┴─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Google Cloud   │
              │    Storage      │
              └─────────────────┘
```

---

## Components

### 1. Fine-Tuning Orchestrator

**Location:** `src/data/finetuning_orchestrator.py`

**Purpose:** Monitors error cases and automatically triggers fine-tuning when thresholds are met.

**Key Features:**
- Automatic trigger based on configurable thresholds
- Dataset preparation from failed cases
- Job management and tracking
- Integration with version control
- Monitoring loop for continuous operation

**Configuration:**
```python
from src.data.finetuning_orchestrator import FinetuningConfig

config = FinetuningConfig(
    min_error_cases=100,           # Minimum cases before triggering
    min_corrected_cases=50,         # Minimum corrected cases
    trigger_on_error_rate=True,     # Trigger on error rate
    error_rate_threshold=0.15,      # 15% error rate threshold
    auto_approve_finetuning=False   # Require manual approval
)
```

**Usage:**
```python
from src.data.finetuning_orchestrator import FinetuningOrchestrator
from src.data.data_manager import DataManager

# Initialize
data_manager = DataManager(use_gcs=True)
orchestrator = FinetuningOrchestrator(
    data_manager=data_manager,
    config=config
)

# Check trigger conditions
trigger_result = orchestrator.check_trigger_conditions()

if trigger_result['should_trigger']:
    # Trigger fine-tuning
    job = orchestrator.trigger_finetuning(force=True)
    print(f"Job created: {job.job_id}")
```

---

### 2. Model Validator

**Location:** `src/data/model_validator.py`

**Purpose:** Validates fine-tuned models against baseline using standardized evaluation sets.

**Key Features:**
- Compare model performance vs baseline
- Statistical significance testing
- Multi-metric evaluation (WER, CER)
- Per-sample analysis
- Degradation detection

**Configuration:**
```python
from src.data.model_validator import ValidationConfig

config = ValidationConfig(
    min_wer_improvement=0.0,                # Minimum improvement required
    min_cer_improvement=0.0,
    require_significance=True,               # Require statistical significance
    significance_alpha=0.05,                 # P-value threshold
    max_wer_degradation_rate=0.1,           # Max 10% samples can degrade
    require_no_major_degradation=True        # No >50% degradation per sample
)
```

**Usage:**
```python
from src.data.model_validator import ModelValidator

validator = ModelValidator(config=config, use_gcs=True)

# Validate model
result = validator.validate_model(
    model_id="finetuned_v1",
    model_transcribe_fn=fine_tuned_transcribe,
    baseline_id="baseline_v1",
    baseline_transcribe_fn=baseline_transcribe,
    evaluation_set_path="data/evaluation/test_set.jsonl"
)

print(f"Validation {'PASSED' if result.passed else 'FAILED'}")
print(f"WER Improvement: {result.wer_improvement:+.4f}")
```

---

### 3. Model Deployer

**Location:** `src/data/model_deployer.py`

**Purpose:** Manages model versioning, deployment, and rollback.

**Key Features:**
- Model version registry
- Deployment with backup
- Rollback capability
- Version history tracking
- GCS synchronization

**Configuration:**
```python
from src.data.model_deployer import DeploymentConfig

config = DeploymentConfig(
    deployment_strategy="replace",           # 'replace', 'canary', 'blue_green'
    keep_previous_versions=5,                # Keep 5 previous versions
    auto_backup_before_deploy=True,
    enable_auto_rollback=True,
    rollback_on_error_threshold=0.5
)
```

**Usage:**
```python
from src.data.model_deployer import ModelDeployer

deployer = ModelDeployer(config=config, use_gcs=True)

# Register model
version_id = deployer.register_model(
    model_name="fine-tuned-stt",
    model_path="/path/to/model",
    validation_result=validation_result.to_dict()
)

# Deploy model
deployer.deploy_model(version_id)

# Rollback if needed
deployer.rollback()  # Rollback to previous version
```

---

### 4. Regression Tester

**Location:** `src/data/regression_tester.py`

**Purpose:** Prevents model degradation through continuous testing.

**Key Features:**
- Register regression test suites
- Track baseline performance
- Detect degradation
- Per-test and aggregate metrics
- Test history tracking

**Configuration:**
```python
from src.data.regression_tester import RegressionConfig

config = RegressionConfig(
    run_on_deploy=True,
    fail_on_critical_degradation=True,
    critical_degradation_threshold=0.1,     # 10% degradation is critical
    max_failed_samples_rate=0.05,           # 5% samples can fail
    sample_degradation_threshold=0.2        # 20% per-sample threshold
)
```

**Usage:**
```python
from src.data.regression_tester import RegressionTester

tester = RegressionTester(config=config, use_gcs=True)

# Register test
test_id = tester.register_test(
    test_name="Critical Benchmark",
    test_type="benchmark",
    test_data_path="data/evaluation/benchmark.jsonl",
    baseline_wer=0.15,
    baseline_cer=0.08,
    baseline_version="baseline_v1",
    max_wer_degradation=0.05
)

# Run test
result = tester.run_test(
    test_id=test_id,
    model_version="finetuned_v1",
    model_transcribe_fn=model_transcribe
)

# Run full test suite
suite_results = tester.run_test_suite(
    model_version="finetuned_v1",
    model_transcribe_fn=model_transcribe
)
```

---

### 5. Fine-Tuning Coordinator

**Location:** `src/data/finetuning_coordinator.py`

**Purpose:** Central coordinator that orchestrates the complete workflow.

**Key Features:**
- Complete workflow automation
- Component integration
- Callback management
- Status monitoring
- Workflow tracking

**Usage:**
```python
from src.data.finetuning_coordinator import FinetuningCoordinator
from src.data.data_manager import DataManager

# Initialize
data_manager = DataManager(use_gcs=True)
coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    use_gcs=True
)

# Set callbacks
coordinator.set_training_callback(custom_training_function)
coordinator.set_baseline_transcribe_function(baseline_transcribe)
coordinator.set_model_transcribe_function_factory(model_factory)

# Run complete workflow
workflow = coordinator.run_complete_workflow(
    force_trigger=True,
    auto_deploy=True
)

# Check status
coordinator.print_status()
```

---

## Complete Workflow Example

### Step 1: Setup

```python
from src.data.data_manager import DataManager
from src.data.finetuning_coordinator import FinetuningCoordinator
from src.baseline_model import BaselineSTTModel

# Initialize components
data_manager = DataManager(use_gcs=True)
baseline_model = BaselineSTTModel()

# Initialize coordinator
coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    use_gcs=True,
    project_id="your-gcp-project"
)
```

### Step 2: Accumulate Error Cases

```python
# Store failed cases as they occur
case_id = data_manager.store_failed_case(
    audio_path="audio/sample.wav",
    original_transcript="incorrect transcription",
    corrected_transcript="correct transcription",
    error_types=["word_substitution"],
    error_score=0.85,
    metadata={"source": "production"}
)
```

### Step 3: Monitor and Trigger

```python
# Check if ready to trigger
trigger_result = coordinator.orchestrator.check_trigger_conditions()

if trigger_result['should_trigger']:
    print("Ready to trigger fine-tuning!")
    
    # Trigger workflow
    workflow = coordinator.run_complete_workflow(
        force_trigger=True,
        auto_deploy=False  # Manual deployment for safety
    )
```

### Step 4: Validate and Deploy

```python
# After training completes, deploy the model
coordinator.deploy_job_model(
    job_id=workflow['stages']['trigger']['job_id'],
    model_path="/path/to/trained/model",
    run_validation=True,
    run_regression=True
)
```

---

## Google Cloud Platform Integration

### Prerequisites

1. **GCP Setup:**
   ```bash
   # Install Google Cloud SDK
   curl https://sdk.cloud.google.com | bash
   
   # Authenticate
   gcloud auth login
   gcloud config set project your-project-id
   ```

2. **Create Storage Buckets:**
   ```bash
   gsutil mb gs://your-project-datasets
   gsutil mb gs://your-project-models
   ```

3. **Set Permissions:**
   ```bash
   gcloud projects add-iam-policy-binding your-project-id \
       --member="serviceAccount:your-service-account@your-project.iam.gserviceaccount.com" \
       --role="roles/storage.objectAdmin"
   ```

### Deploy Fine-Tuning to GCP

```bash
# Create GPU VM
python scripts/deploy_finetuning_to_gcp.py \
    --create-vm \
    --machine-type n1-standard-8

# Upload code and prepare
python scripts/deploy_finetuning_to_gcp.py \
    --prepare-dataset

# Run training
python scripts/deploy_finetuning_to_gcp.py \
    --run-training \
    --dataset-id finetuning_dataset_20231201_120000 \
    --epochs 5

# Download trained model
python scripts/deploy_finetuning_to_gcp.py \
    --download-model ~/stt-project/models/finetuned_model \
    --local-dest ./models

# Clean up
python scripts/deploy_finetuning_to_gcp.py --stop-vm
```

---

## Configuration Best Practices

### Development Environment

```python
# Use local storage, low thresholds
config = FinetuningConfig(
    min_error_cases=10,
    auto_approve_finetuning=True
)

coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    use_gcs=False  # Local storage
)
```

### Production Environment

```python
# Use GCS, higher thresholds, manual approval
config = FinetuningConfig(
    min_error_cases=100,
    min_corrected_cases=50,
    trigger_on_error_rate=True,
    error_rate_threshold=0.15,
    auto_approve_finetuning=False  # Require approval
)

coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    use_gcs=True,
    project_id="production-project-id"
)
```

---

## Monitoring and Alerting

### Get System Status

```python
# Get comprehensive status
status = coordinator.get_system_status()

print(f"Active model: {status['deployer']['active_version']}")
print(f"Error cases: {status['orchestrator']['trigger_conditions']['metrics']['total_error_cases']}")
print(f"Validation pass rate: {status['validation']['pass_rate']}")
```

### Track Metrics

```python
from src.data.metadata_tracker import MetadataTracker

tracker = MetadataTracker(use_gcs=True)

# Get performance trends
wer_trend = tracker.get_performance_trend('wer', time_window_days=30)
print(f"WER improvement: {wer_trend['improvement']:.4f}")

# Get inference statistics
stats = tracker.get_inference_statistics(time_window_hours=24)
print(f"Error detection rate: {stats['error_detection_rate']:.2%}")
```

---

## Troubleshooting

### Common Issues

1. **Trigger Not Working:**
   ```python
   # Check trigger conditions
   result = orchestrator.check_trigger_conditions()
   print(result['reasons'])
   print(result['metrics'])
   ```

2. **Validation Failing:**
   ```python
   # Check validation configuration
   print(validator.config.to_dict())
   
   # Review validation results
   result = validator.validate_model(...)
   print(result.failure_reason)
   ```

3. **Deployment Issues:**
   ```python
   # Check deployment status
   deployer.print_status()
   
   # Review deployment history
   history = deployer.get_deployment_history()
   for deployment in history:
       print(f"{deployment['version_id']}: {deployment['status']}")
   ```

### Rollback Procedure

```python
# If model performs poorly in production
deployer = ModelDeployer(use_gcs=True)

# Rollback to previous version
success = deployer.rollback()

if success:
    print("Rolled back to previous version")
else:
    # Manual rollback to specific version
    deployer.rollback(target_version_id="model_v123")
```

---

## Testing

### Run Demo

```bash
# Run comprehensive demo
python experiments/demo_finetuning_orchestration.py
```

### Unit Tests

```bash
# Run specific component tests
pytest tests/test_finetuning_orchestrator.py
pytest tests/test_model_validator.py
pytest tests/test_model_deployer.py
pytest tests/test_regression_tester.py
```

---

## API Reference

### Quick Reference

```python
# Data Manager
data_manager.store_failed_case(...)
data_manager.get_statistics()

# Fine-Tuning Orchestrator
orchestrator.check_trigger_conditions()
orchestrator.trigger_finetuning(force=True)
orchestrator.get_job_info(job_id)

# Model Validator
validator.validate_model(model_id, ...)
validator.get_best_model(metric='wer')

# Model Deployer
deployer.register_model(...)
deployer.deploy_model(version_id)
deployer.rollback()

# Regression Tester
tester.register_test(...)
tester.run_test_suite(...)

# Coordinator
coordinator.run_complete_workflow(...)
coordinator.get_system_status()
coordinator.print_status()
```

---

## Performance Considerations

### Scalability

- **Dataset Size:** System handles 1000s of error cases efficiently
- **Concurrent Jobs:** Supports multiple fine-tuning jobs
- **GCS Integration:** Offloads storage to cloud for scalability

### Optimization Tips

1. **Batch Operations:** Accumulate cases before triggering
2. **Parallel Processing:** Use GCP VMs for parallel training
3. **Caching:** Version control caches metadata locally
4. **Incremental Updates:** Only sync changed data to GCS

---

## Security

### Best Practices

1. **Authentication:** Use service accounts for GCS access
2. **Permissions:** Follow principle of least privilege
3. **Data Privacy:** Encrypt sensitive audio/transcript data
4. **Audit Logging:** All operations are logged with timestamps

### Configuration

```python
# Use service account key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/key.json'

coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    use_gcs=True,
    project_id="your-project"
)
```

---

## Support and Contributing

### Documentation
- Full API docs: `docs/API_REFERENCE.md`
- Setup guide: `docs/SETUP_INSTRUCTIONS.md`
- GCP guide: `docs/GCP_SETUP_GUIDE.md`

### Examples
- Demo script: `experiments/demo_finetuning_orchestration.py`
- Test cases: `tests/`

### Questions
For questions or issues, see the project repository.

---

## License

This system is part of the Adaptive Self-Learning Agentic AI System project.


