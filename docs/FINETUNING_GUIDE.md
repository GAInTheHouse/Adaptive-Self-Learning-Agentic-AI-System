# Fine-Tuning Guide

Complete guide to model fine-tuning and automated fine-tuning orchestration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Wav2Vec2 Model Fine-Tuning](#wav2vec2-model-fine-tuning)
3. [Fine-Tuning Orchestration System](#fine-tuning-orchestration-system)
4. [System Architecture](#system-architecture)
5. [Configuration](#configuration)
6. [GCP Integration](#gcp-integration)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Run the Demo

```bash
# Run comprehensive demo
python experiments/demo_finetuning_orchestration.py
```

### Basic Usage

```python
from src.data.data_manager import DataManager
from src.data.finetuning_coordinator import FinetuningCoordinator

# Initialize
data_manager = DataManager(
    local_storage_dir="data/failed_cases",
    use_gcs=False  # Local for testing
)
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

## Wav2Vec2 Model Fine-Tuning

This section covers fine-tuning the Wav2Vec2 STT model using LLM-generated gold standard transcripts.

### Overview

The fine-tuning process:
1. **Evaluation Phase**: Processes 200 audio files (100 clean, 100 noisy), gets STT transcripts, uses LLM to generate gold standard transcripts, and calculates baseline WER/CER
2. **Fine-tuning Phase**: Fine-tunes the model only on samples where STT made errors
3. **Re-evaluation Phase**: Evaluates the fine-tuned model and shows improvements

### Prerequisites

- Python 3.8+
- Audio files (200 total: 100 clean, 100 noisy)
- Ollama installed and running locally
- Llama model downloaded via Ollama

### Setup

1. **Install Python dependencies** (if not already installed):
```bash
pip install torch transformers librosa jiwer datasets peft bitsandbytes ollama
```

Optional (for faster LLM inference):
```bash
pip install flash-attn  # Requires CUDA and proper compilation
```

2. **Install and Setup Ollama**:

The fine-tuning process uses Ollama with Llama models for generating gold standard transcripts. Follow these steps:

#### Install Ollama

**macOS:**
```bash
# Download and install from https://ollama.ai/download
# Or use Homebrew:
brew install ollama
```

**Linux:**
```bash
# Download and install from https://ollama.ai/download
# Or use the install script:
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
- Download the installer from https://ollama.ai/download
- Run the installer and follow the setup wizard

#### Start Ollama Server

Ollama runs as a local server. Start it in a separate terminal:

```bash
# Start Ollama server (runs on http://localhost:11434 by default)
ollama serve
```

**Note**: Keep this terminal open while using the system. The server must be running for LLM correction to work.

#### Download Required Model

The system uses `llama3.2:3b` by default. Download it with:

```bash
# Pull the Llama 3.2 3B model (recommended - fast and efficient)
ollama pull llama3.2:3b
```

**Alternative models** (if you need better quality but slower):
```bash
# Llama 3.1 8B (better quality, slower)
ollama pull llama3.1:8b

# Llama 2 7B (alternative option)
ollama pull llama2:7b
```

**Note**: The 3B model is recommended for speed. The 8B model provides better quality but is slower.

#### Verify Ollama Setup

Test that Ollama is working correctly:

```bash
# Check if Ollama is running
ollama list

# Test the model
ollama run llama3.2:3b "Hello, how are you?"
```

Or use the provided test script:

```bash
python scripts/fine_tune_scripts/test_llm_connection.py
```

This will verify:
- Ollama server is running
- Model is downloaded and available
- LLM connection is working

2. **Organize your audio files**:
```
data/finetuning_audio/
├── clean/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ... (100 files)
└── noisy/
    ├── audio_101.wav
    ├── audio_102.wav
    └── ... (100 files)
```

Alternatively, if you put all files in one directory, the script will automatically split them in half.

### Test LLM Connection

Before fine-tuning, test that Ollama and the LLM are working:

```bash
# Use the provided test script
python scripts/fine_tune_scripts/test_llm_connection.py

# Or check available models
python scripts/fine_tune_scripts/check_ollama_models.py
```

The test script will verify:
- ✅ Ollama server is running
- ✅ Model is downloaded
- ✅ LLM can generate responses
- ✅ Connection is working properly

### LLM Configuration

The system uses Ollama with Llama models for LLM-based error correction and gold standard generation. The default model is `llama3.2:3b`, which provides a good balance of speed and quality.

**Model Options**:
- **llama3.2:3b** (default, recommended): Fast, efficient, good quality (~2GB download)
- **llama3.1:8b**: Better quality, slower (~4.7GB download)
- **llama2:7b**: Alternative option (~3.8GB download)

**Change the model** (if needed):
```python
# In your fine-tuning script or code
from src.agent.fine_tuner.llm_corrector import LlamaLLMCorrector

corrector = LlamaLLMCorrector(
    model_name="llama3.1:8b",  # Use 8B model instead
    ollama_base_url="http://localhost:11434"
)
```

**Note**: Make sure to pull the model with `ollama pull <model_name>` before using it. The 3B model is recommended for speed during fine-tuning, while the 8B model provides better quality for gold standard generation.

### Run Fine-tuning

#### Basic Usage

```bash
python scripts/finetune_wav2vec2.py --audio_dir data/finetuning_audio
```

By default, the script uses **LoRA** (Low-Rank Adaptation) for efficient fine-tuning, which is 3-5x faster and uses 3-5x less memory than full fine-tuning while maintaining comparable accuracy (within 0.3-0.5%).

#### Advanced Options

```bash
python scripts/finetune_wav2vec2.py \
    --audio_dir data/finetuning_audio \
    --output_dir models/finetuned_wav2vec2 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --lora_rank 8 \
    --lora_alpha 16
```

#### Arguments

- `--audio_dir`: Directory containing audio files (required)
- `--output_dir`: Output directory for fine-tuned model (default: `models/finetuned_wav2vec2`)
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 3e-5)
- `--use_lora`: Enable LoRA fine-tuning (default: True)
- `--no_lora`: Disable LoRA and use full fine-tuning
- `--lora_rank`: LoRA rank - controls number of trainable parameters (default: 8)
- `--lora_alpha`: LoRA alpha scaling factor (default: 16)

### LoRA vs Full Fine-Tuning

#### LoRA (Low-Rank Adaptation) - Default

**Benefits:**
- **3-5x faster** training time
- **3-5x less GPU memory** usage
- Only ~0.8% of parameters are trainable
- Comparable accuracy (typically within 0.3-0.5% of full fine-tuning)
- Smaller saved models (only adapters, not full model)

**When to use:**
- Limited computational resources
- Fast iteration and experimentation
- When slight accuracy trade-off is acceptable

#### Full Fine-Tuning

**Benefits:**
- Maximum accuracy potential
- All model parameters updated
- Better for complex domain-specific tasks

**When to use:**
- When maximum accuracy is critical
- When you have abundant computational resources
- For complex tasks requiring comprehensive model updates

**To use full fine-tuning:**
```bash
python scripts/finetune_wav2vec2.py --audio_dir data/finetuning_audio --no_lora
```

### Training Time Estimation

**LoRA**: ~7.5 seconds per sample per epoch (3-5x faster)
**Full Fine-tuning**: ~30 seconds per sample per epoch

**Examples**:
- **LoRA**: 150 error samples × 3 epochs × 7.5 seconds = ~56 minutes
- **Full**: 150 error samples × 3 epochs × 30 seconds = ~3.75 hours

### Using the Fine-tuned Model

After fine-tuning, the model will be saved to the output directory. To use it in the system:

1. Update `src/baseline_model.py` to load from the fine-tuned path for "wav2vec2-finetuned"
2. Or load directly:
```python
from src.baseline_model import BaselineSTTModel

model = BaselineSTTModel(model_name="path/to/finetuned/model")
result = model.transcribe("audio_file.wav")
```

---

## Fine-Tuning Orchestration System

The Fine-Tuning Orchestration System provides a comprehensive, automated pipeline for improving speech-to-text models through continuous learning from error cases. The system handles the complete lifecycle:

1. **Automated Triggering** - Monitors error accumulation and triggers fine-tuning
2. **Model Validation** - Validates models against baseline with statistical testing
3. **Version Management** - Manages model versions with deployment and rollback
4. **Regression Testing** - Prevents performance degradation

### System Architecture

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
```

### Core Components

#### 1. Automated Fine-Tuning Orchestrator

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

#### 2. Model Validation System

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

#### 3. Model Versioning & Deployment

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

#### 4. Regression Testing

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

### Complete Workflow

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

## Configuration

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

## GCP Integration

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

## Monitoring

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

## Troubleshooting

### Issue: Ollama LLM Connection Failed

**Problem**: Fine-tuning script fails with "Ollama server is not running" or "Model not available"

**Solutions**:

1. **Check if Ollama server is running**:
```bash
# Test Ollama connection
curl http://localhost:11434/api/tags

# If this fails, start Ollama:
ollama serve
```

2. **Verify model is downloaded**:
```bash
# List available models
ollama list

# If llama3.2:3b is not listed, download it:
ollama pull llama3.2:3b
```

3. **Test LLM connection**:
```bash
# Use the test script
python scripts/fine_tune_scripts/test_llm_connection.py

# Or manually test
ollama run llama3.2:3b "Hello, test message"
```

4. **Check Python Ollama package**:
```bash
# Verify ollama package is installed
pip list | grep ollama

# If not installed:
pip install ollama
```

5. **Port conflicts**: If port 11434 is in use:
```bash
# Check what's using the port
lsof -i :11434  # macOS/Linux
netstat -ano | findstr :11434  # Windows

# Change Ollama port (if needed)
export OLLAMA_HOST=0.0.0.0:11435
ollama serve
```

6. **Model download issues**: If model download fails:
```bash
# Try pulling again
ollama pull llama3.2:3b

# Check available disk space (models are ~2-4GB)
df -h  # macOS/Linux
```

**Note**: The fine-tuning script requires Ollama to be running. Without it, the LLM-based gold standard generation will fail.

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

### LLM Not Available

If you see warnings about LLM not being available:
- Run `python scripts/test_llm_connection.py` to diagnose
- Check that Mistral model can be loaded
- The script will continue using STT transcripts as gold standard (not ideal)

### Out of Memory

- Reduce `--batch_size` (try 2 or 1)
- Process fewer samples
- Use a smaller model
- Use LoRA instead of full fine-tuning

### Slow Processing

- Ensure you're using GPU if available
- Reduce number of epochs
- Process files in batches
- Use LoRA for faster training

---

## Performance Benchmarks

### LoRA vs Full Fine-Tuning

Typical performance on STT tasks:
- **LoRA**: WER/CER within 0.3-0.5% of full fine-tuning
- **Training time**: 3-5x faster with LoRA
- **Memory usage**: 3-5x less with LoRA
- **Model size**: LoRA adapters ~10-50MB vs full model ~300MB+

### LLM Inference Speed

With optimizations enabled (fast_mode=True, 4-bit quantization):
- **Target**: <1 second per transcript
- **Typical**: 0.5-2 seconds depending on transcript length and hardware
- **Without optimizations**: 3-10+ seconds per transcript

---

## Notes

- The script only fine-tunes on **error cases** (samples where STT transcript != LLM gold standard)
- WER/CER are calculated using `jiwer` library
- With LoRA: Only adapters are saved (much smaller files)
- With Full Fine-tuning: Complete model is saved
- Training history and logs are saved to `{output_dir}/logs/`
- LoRA adapters can be merged into base model for standalone inference if needed

---

## Support

- **Full Documentation:** See inline documentation in source files
- **Demo:** `experiments/demo_finetuning_orchestration.py`
- **Tests:** `tests/test_finetuning_*.py`

