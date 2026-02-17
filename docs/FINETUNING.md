# Fine-Tuning Guide

Complete guide covering Wav2Vec2 script fine-tuning and the orchestration system for automated model improvement.

---

## Quick Start

```bash
# Run comprehensive demo
python experiments/demo_finetuning_orchestration.py
```

```python
from src.data.data_manager import DataManager
from src.data.finetuning_coordinator import FinetuningCoordinator

data_manager = DataManager(use_gcs=False)
coordinator = FinetuningCoordinator(data_manager=data_manager, use_gcs=False)
coordinator.print_status()

trigger_result = coordinator.orchestrator.check_trigger_conditions()
if trigger_result['should_trigger']:
    workflow = coordinator.run_complete_workflow(force_trigger=True, auto_deploy=False)
```

---

## Part 1: Wav2Vec2 Script Fine-Tuning

Fine-tune the Wav2Vec2 STT model using LLM-generated gold standard transcripts.

### Overview

1. **Evaluation Phase**: Process 200 audio files (100 clean, 100 noisy), get STT transcripts, use LLM to generate gold standard, calculate baseline WER/CER
2. **Fine-Tuning Phase**: Fine-tune only on samples where STT made errors
3. **Re-evaluation Phase**: Evaluate fine-tuned model and show improvements

### Prerequisites

- Python 3.8+
- Audio files (200 total: 100 clean, 100 noisy)
- LLM (Mistral) connection working

### Run Fine-Tuning

```bash
# Test LLM connection first
python scripts/test_llm_connection.py

# Basic usage (uses LoRA by default)
python scripts/finetune_wav2vec2.py --audio_dir data/finetuning_audio

# Advanced options
python scripts/finetune_wav2vec2.py \
    --audio_dir data/finetuning_audio \
    --output_dir models/finetuned_wav2vec2 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --lora_rank 8 \
    --lora_alpha 16

# Full fine-tuning (no LoRA)
python scripts/finetune_wav2vec2.py --audio_dir data/finetuning_audio --no_lora
```

### LoRA vs Full Fine-Tuning

**LoRA (default)**: 3-5x faster, 3-5x less GPU memory, comparable accuracy (within 0.3-0.5% of full). Use for limited resources or fast iteration.

**Full Fine-Tuning**: Maximum accuracy potential. Use when maximum accuracy is critical and you have abundant compute.

### Audio File Structure

```
data/finetuning_audio/
├── clean/
│   └── audio_001.wav ... (100 files)
└── noisy/
    └── audio_101.wav ... (100 files)
```

---

## Part 2: Orchestration System

Automated pipeline for model fine-tuning, validation, deployment, and monitoring.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Fine-Tuning Coordinator                         │
└───────────────────────┬─────────────────────────────────────────┘
        ┌───────────────┼───────────────┬─────────────────┐
        ▼               ▼               ▼                 ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│  Fine-Tuning │ │    Model    │ │    Model    │ │  Regression  │
│ Orchestrator │ │  Validator  │ │  Deployer   │ │   Tester     │
└──────────────┘ └─────────────┘ └─────────────┘ └──────────────┘
```

### Components

1. **FinetuningOrchestrator** (`src/data/finetuning_orchestrator.py`) - Monitors errors, triggers fine-tuning
2. **ModelValidator** (`src/data/model_validator.py`) - Validates against baseline, statistical significance
3. **ModelDeployer** (`src/data/model_deployer.py`) - Version management, deployment, rollback
4. **RegressionTester** (`src/data/regression_tester.py`) - Prevents degradation
5. **FinetuningCoordinator** (`src/data/finetuning_coordinator.py`) - Central orchestration

### Configuration

```python
from src.data.finetuning_orchestrator import FinetuningConfig

# Development
config = FinetuningConfig(
    min_error_cases=10,
    auto_approve_finetuning=True
)

# Production
config = FinetuningConfig(
    min_error_cases=100,
    min_corrected_cases=50,
    trigger_on_error_rate=True,
    error_rate_threshold=0.15,
    auto_approve_finetuning=False
)
```

### Complete Workflow

```python
from src.data.finetuning_coordinator import FinetuningCoordinator
from src.data.data_manager import DataManager

data_manager = DataManager(use_gcs=True)
coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    use_gcs=True,
    project_id="your-project"
)

# Set callbacks
coordinator.set_training_callback(custom_training_function)
coordinator.set_baseline_transcribe_function(baseline_transcribe)

# Run workflow
workflow = coordinator.run_complete_workflow(force_trigger=True, auto_deploy=False)
```

### GCP Deployment

```bash
python scripts/deploy_finetuning_to_gcp.py \
    --create-vm --prepare-dataset --run-training \
    --dataset-id your_dataset_id
```

### Troubleshooting

- **Trigger not working**: `orchestrator.check_trigger_conditions()` → check `reasons` and `metrics`
- **Validation failing**: Review `result.failure_reason` and validation config
- **Deployment issues**: `deployer.print_status()`, `deployer.rollback()` if needed

---

## File Structure

```
src/data/
├── finetuning_orchestrator.py
├── model_validator.py
├── model_deployer.py
├── regression_tester.py
├── finetuning_coordinator.py
└── finetuning_pipeline.py

scripts/
├── finetune_wav2vec2.py
└── deploy_finetuning_to_gcp.py
```
