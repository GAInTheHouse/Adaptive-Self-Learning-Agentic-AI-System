# Fine-Tuning Orchestration - Quick Start Guide

Get started with the automated fine-tuning orchestration system in 5 minutes.

## 🚀 Quick Start

### 1. Run the Demo

```bash
# Run comprehensive demo
python experiments/demo_finetuning_orchestration.py
```

This demonstrates all components:
- ✅ Data Manager (error tracking)
- ✅ Fine-Tuning Orchestrator (automated triggering)
- ✅ Model Validator (baseline comparison)
- ✅ Model Deployer (version management)
- ✅ Regression Tester (degradation prevention)
- ✅ Complete Workflow Coordinator

### 2. Basic Usage

```python
from src.data.data_manager import DataManager
from src.data.finetuning_coordinator import FinetuningCoordinator

# Initialize
data_manager = DataManager(use_gcs=False)  # Local for testing
coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    use_gcs=False
)

# Check status
coordinator.print_status()

# Check if ready to trigger
trigger_result = coordinator.orchestrator.check_trigger_conditions()
print(f"Should trigger: {trigger_result['should_trigger']}")

# Trigger fine-tuning (when ready)
if trigger_result['should_trigger']:
    workflow = coordinator.run_complete_workflow(
        force_trigger=True,
        auto_deploy=False  # Manual approval for safety
    )
```

---

## 📋 4 Core Components

### 1. Automated Fine-Tuning Orchestrator

**What it does:** Monitors error cases and triggers fine-tuning automatically

```python
from src.data.finetuning_orchestrator import FinetuningOrchestrator, FinetuningConfig

config = FinetuningConfig(
    min_error_cases=100,        # Trigger after 100 errors
    auto_approve_finetuning=False  # Require approval
)

orchestrator = FinetuningOrchestrator(
    data_manager=data_manager,
    config=config
)

# Trigger fine-tuning
job = orchestrator.trigger_finetuning(force=True)
```

**Key Features:**
- ✅ Automatic trigger based on error thresholds
- ✅ Dataset preparation from failed cases
- ✅ Job management and tracking
- ✅ Integration with GCS

### 2. Model Validation System

**What it does:** Validates fine-tuned models against baseline

```python
from src.data.model_validator import ModelValidator, ValidationConfig

config = ValidationConfig(
    min_wer_improvement=0.0,
    require_significance=True
)

validator = ModelValidator(config=config)

# Validate model
result = validator.validate_model(
    model_id="finetuned_v1",
    model_transcribe_fn=model_transcribe,
    baseline_id="baseline_v1",
    baseline_transcribe_fn=baseline_transcribe
)

print(f"Passed: {result.passed}")
print(f"WER Improvement: {result.wer_improvement:+.4f}")
```

**Key Features:**
- ✅ Baseline comparison
- ✅ Statistical significance testing
- ✅ Multi-metric evaluation (WER, CER)
- ✅ Per-sample analysis

### 3. Model Versioning & Deployment

**What it does:** Manages model versions and deployment

```python
from src.data.model_deployer import ModelDeployer, DeploymentConfig

config = DeploymentConfig(
    keep_previous_versions=5,
    auto_backup_before_deploy=True
)

deployer = ModelDeployer(config=config)

# Register and deploy
version_id = deployer.register_model(
    model_name="fine-tuned-stt",
    model_path="/path/to/model",
    validation_result=validation_result.to_dict()
)

deployer.deploy_model(version_id)

# Rollback if needed
deployer.rollback()
```

**Key Features:**
- ✅ Version registry and history
- ✅ Automated backup before deployment
- ✅ One-click rollback
- ✅ GCS synchronization

### 4. Regression Testing

**What it does:** Prevents model degradation

```python
from src.data.regression_tester import RegressionTester, RegressionConfig

config = RegressionConfig(
    fail_on_critical_degradation=True,
    critical_degradation_threshold=0.1
)

tester = RegressionTester(config=config)

# Register test
test_id = tester.register_test(
    test_name="Critical Benchmark",
    test_type="benchmark",
    test_data_path="data/evaluation/test.jsonl",
    baseline_wer=0.15,
    baseline_cer=0.08,
    baseline_version="baseline_v1"
)

# Run tests
results = tester.run_test_suite(
    model_version="finetuned_v1",
    model_transcribe_fn=model_transcribe
)
```

**Key Features:**
- ✅ Multiple test suites (benchmark, critical, edge cases)
- ✅ Baseline tracking
- ✅ Automated degradation detection
- ✅ Test history and trends

---

## 🔄 Complete Workflow

The system automates the entire fine-tuning lifecycle:

```
1. Monitor Error Cases
   └─> Accumulate failed transcriptions
   └─> Track corrections

2. Trigger Fine-Tuning (automatic when threshold met)
   └─> Prepare dataset from error cases
   └─> Create data version
   └─> Launch training job

3. Validate Model
   └─> Compare against baseline
   └─> Statistical significance test
   └─> Check quality metrics

4. Run Regression Tests
   └─> Test critical samples
   └─> Check for degradation
   └─> Verify edge cases

5. Deploy Model (if validation passes)
   └─> Register new version
   └─> Backup current model
   └─> Deploy new version
   └─> Update active pointer

6. Continuous Monitoring
   └─> Track performance metrics
   └─> Alert on degradation
   └─> Enable rollback
```

---

## ☁️ Google Cloud Integration

### Setup GCP

```bash
# Authenticate
gcloud auth login
gcloud config set project your-project-id

# Create buckets
gsutil mb gs://your-project-datasets
gsutil mb gs://your-project-models
```

### Enable GCS in Code

```python
# Enable GCS for all components
coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    use_gcs=True,  # ← Enable GCS
    project_id="your-project-id"
)
```

### Deploy to GCP for Training

```bash
# Create GPU VM and run fine-tuning
python scripts/deploy_finetuning_to_gcp.py \
    --create-vm \
    --prepare-dataset \
    --run-training \
    --dataset-id your_dataset_id
```

---

## 📊 Monitoring

### Check System Status

```python
# Print comprehensive status
coordinator.print_status()

# Get detailed status
status = coordinator.get_system_status()

# Check specific components
trigger_result = coordinator.orchestrator.check_trigger_conditions()
validation_report = coordinator.validator.generate_validation_report()
deployment_report = coordinator.deployer.generate_deployment_report()
regression_report = coordinator.regression_tester.generate_regression_report()
```

### Track Metrics

```python
from src.data.metadata_tracker import MetadataTracker

tracker = MetadataTracker(use_gcs=True)

# Performance trends
wer_trend = tracker.get_performance_trend('wer', time_window_days=30)
print(f"WER improvement over 30 days: {wer_trend['improvement']:.4f}")

# Recent inference stats
stats = tracker.get_inference_statistics(time_window_hours=24)
print(f"Error detection rate: {stats['error_detection_rate']:.2%}")
```

---

## 🛠️ Configuration

### Development (Local Testing)

```python
from src.data.finetuning_orchestrator import FinetuningConfig

config = FinetuningConfig(
    min_error_cases=10,           # Low threshold
    auto_approve_finetuning=True  # Auto-approve
)

coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    finetuning_config=config,
    use_gcs=False  # Local storage
)
```

### Production

```python
config = FinetuningConfig(
    min_error_cases=100,              # Higher threshold
    min_corrected_cases=50,
    trigger_on_error_rate=True,
    error_rate_threshold=0.15,
    auto_approve_finetuning=False     # Manual approval
)

coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    finetuning_config=config,
    use_gcs=True,                     # Cloud storage
    project_id="production-project"
)
```

---

## 🚨 Troubleshooting

### Issue: Fine-tuning not triggering

```python
# Check trigger conditions
result = orchestrator.check_trigger_conditions()
print(f"Should trigger: {result['should_trigger']}")
print(f"Reasons: {result['reasons']}")
print(f"Current error cases: {result['metrics']['total_error_cases']}")
print(f"Required: {orchestrator.config.min_error_cases}")
```

### Issue: Validation failing

```python
# Check validation configuration
print(validator.config.to_dict())

# Review detailed results
result = validator.validate_model(...)
if not result.passed:
    print(f"Failure reason: {result.failure_reason}")
    print(f"Failed samples: {len(result.failed_samples)}")
```

### Issue: Deployment problems

```python
# Check deployment status
deployer.print_status()

# Review version history
for version in deployer.list_versions(limit=5):
    print(f"{version.version_id}: {version.status}")

# Rollback if needed
deployer.rollback()
```

---

## 📚 Further Reading

- **Full Documentation:** `docs/FINETUNING_ORCHESTRATION.md`
- **API Reference:** `docs/API_REFERENCE.md` (if exists)
- **GCP Setup:** `docs/GCP_SETUP_GUIDE.md`
- **Testing Guide:** `docs/TESTING_GUIDE.md`

---

## 💡 Tips

1. **Start Local:** Test with `use_gcs=False` first
2. **Low Thresholds:** Use low `min_error_cases` for testing
3. **Manual Approval:** Keep `auto_approve_finetuning=False` in production
4. **Monitor Metrics:** Regularly check system status
5. **Test Rollback:** Practice rollback procedure before production

---

## 🎯 Next Steps

1. ✅ Run demo: `python experiments/demo_finetuning_orchestration.py`
2. ✅ Configure for your use case
3. ✅ Set up GCP (optional but recommended)
4. ✅ Integrate with your training pipeline
5. ✅ Set up monitoring and alerts
6. ✅ Test rollback procedure

---

## 🤝 Support

- Check `docs/` for detailed documentation
- Review `examples/` for more use cases
- See `tests/` for test examples

**Ready to automate your fine-tuning? Start with the demo!**

```bash
python experiments/demo_finetuning_orchestration.py
```


