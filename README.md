# Adaptive Self-Learning Agentic AI System for Speech-to-Text

A production-ready, self-improving speech-to-text system with autonomous error detection, correction, and continuous learning capabilities. The system integrates baseline STT models (Whisper), intelligent agent-based error detection, comprehensive data management, and automated fine-tuning pipelines.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [System Components](#system-components)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Development Workflows](#development-workflows)
- [Testing](#testing)
- [Documentation](#documentation)
- [Tutorials](#tutorials)

## 🎯 Overview

This system provides:
1. **Baseline STT Model**: Optimized Whisper-based transcription with GPU acceleration
2. **Intelligent Agent**: Autonomous error detection with 8+ heuristics
3. **Data Management**: Comprehensive system for tracking failures, corrections, and performance
4. **Evaluation Framework**: Multi-metric evaluation with visualization
5. **Fine-tuning Pipeline**: Automated dataset preparation for model improvement
6. **Cloud Integration**: Seamless GCP integration with cost monitoring

## ✨ Features

### Core Capabilities
- ✅ **Real-time transcription** via REST API
- ✅ **Multi-heuristic error detection** (8+ error types)
- ✅ **Automatic correction** with learning feedback loop
- ✅ **Failed case tracking** and correction management
- ✅ **Performance monitoring** (WER, CER, latency, throughput)
- ✅ **Fine-tuning dataset preparation** with quality control
- ✅ **Version control** for datasets with checksums
- ✅ **GCP integration** with automated backup

### Error Detection Types
- Empty/too short transcripts
- Length anomalies (too long/short ratio)
- Repeated character patterns
- Special character overload
- Low model confidence
- Unusual word patterns
- All caps text
- Missing punctuation

## 📁 Project Structure

```
Adaptive-Self-Learning-Agentic-AI-System/
├── src/                          # Core source code
│   ├── baseline_model.py         # Whisper STT model wrapper
│   ├── inference_api.py          # Baseline transcription API
│   ├── agent_api.py              # Agent-integrated API
│   ├── model_selector.py         # Model comparison utilities
│   ├── benchmark.py              # Performance benchmarking
│   ├── agent/                    # Agent system
│   │   ├── agent.py              # Main agent orchestrator
│   │   ├── error_detector.py    # Multi-heuristic error detection
│   │   └── self_learner.py       # Learning and feedback system
│   ├── data/                     # Data management system
│   │   ├── data_manager.py       # Failed case storage
│   │   ├── metadata_tracker.py   # Performance tracking
│   │   ├── finetuning_pipeline.py # Dataset preparation
│   │   ├── version_control.py    # Data versioning
│   │   └── integration.py        # Unified interface
│   ├── evaluation/               # Evaluation tools
│   │   └── metrics.py            # WER/CER calculation
│   └── utils/                    # Utilities
│       └── gcs_utils.py          # Google Cloud Storage
│
├── experiments/                  # Testing and evaluation scripts
│   ├── test_baseline.py          # Test baseline model
│   ├── test_agent.py             # Test agent functionality
│   ├── test_api.py               # Test API endpoints
│   ├── test_data_management.py   # Test data management
│   ├── run_evaluation.py         # Evaluate dataset on baseline + improved models
│   ├── evaluate_models.py        # Model evaluation
│   ├── visualize_evaluation_results.py # Generate charts
│   └── example_usage.py          # Usage examples
│
├── scripts/                      # Setup and deployment
│   ├── setup_environment.py      # Environment setup
│   ├── verify_setup.py           # Verify installation
│   ├── quick_setup.sh            # Quick setup script
│   ├── setup_gcp_gpu.sh          # GCP GPU VM creation
│   ├── deploy_to_gcp.py          # Deploy to GCP
│   ├── monitor_gcp_costs.py      # Cost monitoring
│   ├── preprocess_data.py        # Data preprocessing
│   └── download_datasets.py      # Dataset downloads
│
├── data/                         # Data storage (created at runtime)
│   ├── raw/                      # Raw audio files
│   ├── processed/                # Processed data
│   ├── failed_cases/             # Error storage
│   ├── metadata/                 # Performance metrics
│   ├── finetuning/               # Training datasets
│   └── versions/                 # Dataset versions
│
├── docs/                         # Documentation
│   ├── SETUP_INSTRUCTIONS.md     # Setup guide
│   ├── TESTING_GUIDE.md          # Testing guide
│   ├── DATA_MANAGEMENT_SYSTEM.md # Data management
│   ├── FINETUNING.md             # Fine-tuning (script + orchestration)
│   ├── WANDB.md                  # Weights & Biases integration
│   ├── CONTROL_PANEL.md          # Control panel UI
│   ├── GCP.md                    # Google Cloud setup & deployment
│   ├── QUICK_REFERENCE.md        # Command reference
│   └── LLM_INTEGRATION.md        # Gemma LLM integration
│
├── environment.yml               # Conda environment specification
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image configuration
└── SETUP.md                      # Environment setup guide
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (optional, for faster inference)
- Google Cloud account (optional, for cloud integration)
- Git LFS (for downloading large datasets)

### Installation

**For detailed setup instructions, see [SETUP.md](SETUP.md)** which covers:
- Conda environment setup (recommended)
- Docker setup (for production)
- Manual installation (advanced)

**Quick start with Conda:**

```bash
# 1. Clone the repository
git clone <repository-url>
cd Adaptive-Self-Learning-Agentic-AI-System

# 2. Create conda environment (includes git-lfs, ffmpeg, and all dependencies)
conda env create -f environment.yml
conda activate stt-genai

# 3. Verify installation
python -c "import torch; print(torch.__version__)"
git lfs version
```

**Alternative: Manual installation:**

```bash
# 1. Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install git-lfs (if not already installed)
# macOS: brew install git-lfs && git lfs install
# Ubuntu: sudo apt install git-lfs && git lfs install

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python scripts/verify_setup.py
```

### Basic Usage

```python
from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent
from src.data.integration import IntegratedDataManagementSystem

# 1. Initialize components
baseline_model = BaselineSTTModel(model_name="whisper")
agent = STTAgent(baseline_model=baseline_model)
data_system = IntegratedDataManagementSystem(base_dir="data/production")

# 2. Transcribe with agent
result = agent.transcribe_with_agent(
    audio_path="data/test_audio/test_1.wav",
    enable_auto_correction=True
)

print(f"Transcript: {result['transcript']}")
print(f"Errors detected: {result['error_detection']['error_count']}")
print(f"Error types: {result['error_detection']['error_types']}")

# 3. Record failures for learning
if result['error_detection']['has_errors']:
    case_id = data_system.record_failed_transcription(
        audio_path="data/test_audio/test_1.wav",
        original_transcript=result['original_transcript'],
        corrected_transcript=None,  # Add correction later
        error_types=list(result['error_detection']['error_types'].keys()),
        error_score=result['error_detection']['error_score'],
        inference_time=result.get('inference_time_seconds', 0)
    )
    print(f"Recorded case: {case_id}")
```

## 🔧 System Components

### 1. Baseline STT Model (`src/baseline_model.py`)

GPU-optimized Whisper model wrapper for transcription.

**Features:**
- Automatic GPU/CPU detection
- TensorFloat-32 optimization for Ampere GPUs
- Beam search and KV cache optimization
- Model info and parameter reporting

**Usage:**
```python
from src.baseline_model import BaselineSTTModel

model = BaselineSTTModel(model_name="whisper")
result = model.transcribe("audio.wav")
info = model.get_model_info()
```

### 2. STT Agent (`src/agent/`)

Intelligent agent with error detection and self-learning.

**Components:**
- `error_detector.py`: Multi-heuristic error detection
- `self_learner.py`: Pattern tracking and feedback
- `agent.py`: Agent orchestration

**Usage:**
```python
from src.agent import STTAgent

agent = STTAgent(baseline_model=model, error_threshold=0.3)
result = agent.transcribe_with_agent("audio.wav", enable_auto_correction=True)

# Provide feedback
agent.record_user_feedback(
    transcript_id="123",
    user_feedback="Good transcription",
    is_correct=True
)

# Get statistics
stats = agent.get_agent_stats()
```

### 3. Data Management System (`src/data/`)

Production-ready data management with cloud integration.

**Components:**
- `data_manager.py`: Failed case storage
- `metadata_tracker.py`: Performance tracking
- `finetuning_pipeline.py`: Dataset preparation
- `version_control.py`: Versioning and quality control
- `integration.py`: Unified interface

**Usage:**
```python
from src.data.integration import IntegratedDataManagementSystem

system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=True  # Enable GCP sync
)

# Record failed case
case_id = system.record_failed_transcription(...)

# Add correction
system.add_correction(case_id, "corrected transcript")

# Prepare fine-tuning dataset
dataset_info = system.prepare_finetuning_dataset(
    min_error_score=0.5,
    max_samples=1000,
    create_version=True
)

# Track performance
system.record_training_performance(
    model_version="whisper_v2",
    wer=0.10,
    cer=0.05
)

# Generate report
report = system.generate_comprehensive_report()
```

### 4. Evaluation (`experiments/run_evaluation.py`)

Evaluate any dataset (audio + ground-truth references) on the baseline and on improved (fine-tuned) models if they exist.

**Features:**
- WER/CER for baseline (Whisper) and all fine-tuned Wav2Vec2 versions
- Optional latency/throughput benchmark (`--benchmark`)

**Usage:**
```bash
# Eval set file (JSON/JSONL/CSV with audio_path + reference)
python experiments/run_evaluation.py --eval-set path/to/eval_set.json

# Or audio directory + refs file
python experiments/run_evaluation.py --audio-dir data/recordings_for_test --refs path/to/refs.json
```

See **`docs/EVALUATION_SUMMARY.md`** for full options and examples.

## 🌐 Running the System

### 1. Baseline API (Simple Transcription)

Start the baseline API for simple transcription without agent features:

```bash
# Start API
uvicorn src.inference_api:app --reload --port 8000

# Test with curl
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@data/test_audio/test_1.wav"

# Get model info
curl "http://localhost:8000/model-info"

# Health check
curl "http://localhost:8000/health"
```

### 2. Agent API (Advanced Features)

Start the agent API with error detection and learning:

```bash
# Start API
uvicorn src.agent_api:app --reload --port 8000

# Transcribe with agent
curl -X POST "http://localhost:8000/agent/transcribe?auto_correction=true" \
  -F "file=@data/test_audio/test_1.wav"

# Submit feedback
curl -X POST "http://localhost:8000/agent/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_id": "123",
    "user_feedback": "Good transcription",
    "is_correct": true,
    "corrected_transcript": "This is the correct transcript"
  }'

# Get agent statistics
curl "http://localhost:8000/agent/stats"

# Get learning data
curl "http://localhost:8000/agent/learning-data"

# Baseline endpoints still work
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@data/test_audio/test_1.wav"
```

### 3. Evaluation

#### Run evaluation (baseline + improved models)
```bash
python experiments/run_evaluation.py --eval-set path/to/eval_set.json
```

With latency/throughput benchmark:
```bash
python experiments/run_evaluation.py --eval-set path/to/eval_set.json --benchmark
```

Output (in `experiments/evaluation_outputs/`):
- `evaluation_report.json` - Baseline and improved-model WER/CER
- `evaluation_report.txt` - Short summary
- `benchmark_report.json` - If `--benchmark` was used

See **`docs/EVALUATION_SUMMARY.md`** for input format (eval set, audio-dir + refs, or --gold-from-llm) and all options.

### 4. Testing Components

#### Test Baseline Model
```bash
python experiments/test_baseline.py
```

#### Test Agent System
```bash
python experiments/test_agent.py
```

#### Test API Endpoints
```bash
# Start API in one terminal
uvicorn src.agent_api:app --reload --port 8000

# Test in another terminal
python experiments/test_api.py
```

#### Test Data Management
```bash
python experiments/test_data_management.py
```

### 5. Data Management Workflows

#### View System Statistics
```python
from src.data.integration import IntegratedDataManagementSystem

system = IntegratedDataManagementSystem()
stats = system.get_system_statistics()
print(f"Total failed cases: {stats['data_management']['total_failed_cases']}")
print(f"Correction rate: {stats['data_management']['correction_rate']:.2%}")
```

#### Prepare Fine-tuning Dataset
```python
# Prepare when you have enough corrected cases (500+)
dataset_info = system.prepare_finetuning_dataset(
    min_error_score=0.5,
    max_samples=1000,
    balance_error_types=True,
    create_version=True
)

print(f"Dataset created: {dataset_info['local_path']}")
print(f"Train samples: {dataset_info['stats']['train_size']}")
print(f"Val samples: {dataset_info['stats']['val_size']}")
print(f"Test samples: {dataset_info['stats']['test_size']}")
```

#### Generate Comprehensive Report
```python
report = system.generate_comprehensive_report(
    output_path="data/production/reports/monthly_report.json"
)
```

### 6. Production Monitoring

#### Daily Monitoring
```bash
# Check system stats
python -c "
from src.data.integration import IntegratedDataManagementSystem
system = IntegratedDataManagementSystem()
stats = system.get_system_statistics()
print(f\"New cases today: {stats['data_management']['total_failed_cases']}\")
print(f\"Correction rate: {stats['data_management']['correction_rate']:.1%}\")
"
```

#### Weekly Tasks
```bash
# Generate report and prepare dataset if ready
python -c "
from src.data.integration import IntegratedDataManagementSystem
system = IntegratedDataManagementSystem()
report = system.generate_comprehensive_report()
stats = system.get_system_statistics()
if stats['data_management']['corrected_cases'] >= 500:
    print('Ready to prepare fine-tuning dataset!')
    dataset = system.prepare_finetuning_dataset(max_samples=1000)
"
```

## 📖 API Reference

### Baseline API Endpoints

#### POST /transcribe
Transcribe audio file (baseline model only).

**Request:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "transcript": "transcribed text",
  "model": "whisper",
  "inference_time_seconds": 0.5
}
```

#### GET /model-info
Get model information.

**Response:**
```json
{
  "name": "whisper",
  "parameters": 72593920,
  "device": "cpu",
  "trainable_params": 71825920
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "whisper"
}
```

### Agent API Endpoints

#### POST /agent/transcribe
Transcribe with agent error detection.

**Parameters:**
- `auto_correction` (optional): Enable automatic correction (default: false)

**Request:**
```bash
curl -X POST "http://localhost:8000/agent/transcribe?auto_correction=true" \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "transcript": "corrected transcript",
  "original_transcript": "original transcript",
  "error_detection": {
    "has_errors": true,
    "error_count": 2,
    "error_score": 0.65,
    "errors": [...],
    "error_types": {"all_caps": 1, "missing_punctuation": 1}
  },
  "corrections": {
    "applied": true,
    "count": 2,
    "details": [...]
  },
  "inference_time_seconds": 0.5
}
```

#### POST /agent/feedback
Submit user feedback for learning.

**Request:**
```bash
curl -X POST "http://localhost:8000/agent/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_id": "unique-id",
    "user_feedback": "feedback text",
    "is_correct": true,
    "corrected_transcript": "corrected version"
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Feedback recorded for learning"
}
```

#### GET /agent/stats
Get agent statistics.

**Response:**
```json
{
  "total_transcriptions": 100,
  "error_detection": {
    "threshold": 0.3,
    "total_errors_detected": 25,
    "error_rate": 0.25
  },
  "learning": {
    "total_errors_learned": 25,
    "total_corrections": 20,
    "feedback_count": 15
  }
}
```

#### GET /agent/learning-data
Get in-memory learning data (for external persistence).

**Response:**
```json
{
  "error_patterns": [...],
  "correction_history": [...],
  "feedback_records": [...]
}
```

## 🔄 Development Workflows

### Workflow 1: Production Transcription with Learning

```python
# Initialize system
from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent
from src.data.integration import IntegratedDataManagementSystem

baseline = BaselineSTTModel()
agent = STTAgent(baseline)
data_system = IntegratedDataManagementSystem()

# Process audio
result = agent.transcribe_with_agent("audio.wav", enable_auto_correction=True)

# Automatically record if errors detected
if result['error_detection']['has_errors']:
    case_id = data_system.record_failed_transcription(
        audio_path="audio.wav",
        original_transcript=result['original_transcript'],
        error_types=list(result['error_detection']['error_types'].keys()),
        error_score=result['error_detection']['error_score']
    )
    
# User provides correction
if user_correction:
    data_system.add_correction(case_id, user_correction)
```

### Workflow 2: Model Evaluation

```bash
# Run evaluation on a dataset (eval set = JSON/JSONL/CSV with audio_path + reference)
python experiments/run_evaluation.py --eval-set path/to/eval_set.json
```

Then read `experiments/evaluation_outputs/evaluation_report.json` for baseline and improved-model WER/CER. See `docs/EVALUATION_SUMMARY.md` for details.

### Workflow 3: Fine-tuning Pipeline

```python
from src.data.integration import IntegratedDataManagementSystem

system = IntegratedDataManagementSystem()

# 1. Check if ready for fine-tuning
stats = system.get_system_statistics()
if stats['data_management']['corrected_cases'] >= 500:
    
    # 2. Prepare dataset
    dataset_info = system.prepare_finetuning_dataset(
        min_error_score=0.5,
        max_samples=1000,
        balance_error_types=True,
        create_version=True
    )
    
    # 3. Fine-tune model (external script)
    # train_model(dataset_info['local_path'])
    
    # 4. Record training performance
    system.record_training_performance(
        model_version="whisper_finetuned_v1",
        wer=0.08,
        cer=0.04,
        training_metadata={"epochs": 10, "batch_size": 16}
    )
    
    # 5. Compare versions
    comparison = system.metadata_tracker.compare_model_versions(
        "whisper_base",
        "whisper_finetuned_v1"
    )
```

## 🧪 Testing

### Run All Tests
```bash
# Test baseline model
python experiments/test_baseline.py

# Test agent system
python experiments/test_agent.py

# Test data management
python experiments/test_data_management.py

# Test API (requires API to be running)
python experiments/test_api.py
```

### Example Test Output
```
✅ Testing baseline model transcription...
✅ Testing agent error detection...
✅ Testing data management system...
✅ Testing fine-tuning pipeline...
✅ All tests passed!
```

## ☁️ Google Cloud Platform Integration

### Setup GCP

```bash
# 1. Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# 2. Authenticate
gcloud auth login
gcloud config set project stt-agentic-ai-2025

# 3. Enable APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com

# 4. Create storage bucket
gsutil mb gs://stt-project-datasets

# 5. Verify setup
bash scripts/quick_setup.sh
```

### Create GPU VM

```bash
# Create GPU-enabled VM for faster inference
bash scripts/setup_gcp_gpu.sh
```

### Deploy to GCP

```bash
# Deploy code and run on GCP
python scripts/deploy_to_gcp.py
```

### Monitor Costs

```bash
# Check GCP usage and costs
python scripts/monitor_gcp_costs.py
```

## 📚 Documentation

- **[docs/SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md)** - Setup guide
- **[docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md)** - Testing guide
- **[docs/DATA_MANAGEMENT_SYSTEM.md](docs/DATA_MANAGEMENT_SYSTEM.md)** - Data management
- **[docs/FINETUNING.md](docs/FINETUNING.md)** - Fine-tuning
- **[docs/CONTROL_PANEL.md](docs/CONTROL_PANEL.md)** - Control panel
- **[docs/GCP.md](docs/GCP.md)** - GCP setup & deployment
- **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Command reference

## 🎥 Tutorials

### Complete System Overview Video

Watch our comprehensive tutorial video that covers everything you need to get started:

**[📹 Watch Complete Tutorial Video](docs/submission/Presentation_Demo_Tutorial.mp4)**

**What's included in the video:**
- 🎯 **Project Presentation** - Overview of the Adaptive Self-Learning Agentic AI System
- 🚀 **System Demo** - Live demonstration of transcription, error detection, and data management
- 🛠️ **Repository Setup** - Step-by-step guide to setting up the repository and dependencies
- 💡 **Key Features Walkthrough** - Deep dive into agent system, data management, and fine-tuning

### Additional Tutorial Resources

- **[docs/UI_TUTORIAL.md](docs/UI_TUTORIAL.md)** - Control Panel UI walkthrough
- **[docs/CONTROL_PANEL_GUIDE.md](docs/CONTROL_PANEL_GUIDE.md)** - Complete control panel guide
- **[docs/FINETUNING_QUICK_START.md](docs/FINETUNING_QUICK_START.md)** - Fine-tuning tutorial
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference

## 📊 Performance Metrics

### Baseline Performance (Whisper-base)
- **Model**: 72.6M parameters
- **WER**: 0.10 (10%)
- **CER**: 0.0227 (2.27%)
- **CPU Latency**: 5.29s per sample
- **GPU Latency**: ~0.1-0.2s per sample (estimated)
- **Throughput**: 2.65 samples/second (CPU)

### Agent Performance
- **Error Detection**: 8+ heuristic types
- **Detection Overhead**: ~5-10% additional processing
- **Correction Accuracy**: Tracked via user feedback

### Data Management
- **Storage**: ~1KB per failed case
- **Record Speed**: ~10ms (local) + ~100ms (GCS)
- **Scalability**: Handles 100,000+ cases
- **Dataset Prep**: ~1-5 seconds per 1000 samples

## 🎯 Key Achievements

✅ **Week 1**: Baseline model, evaluation framework, benchmarking, GCP integration  
✅ **Week 2**: Agent system, error detection, data management, fine-tuning pipeline

## 🚦 Current Status

- ✅ Baseline STT model with GPU optimization
- ✅ Real-time inference API (baseline + agent)
- ✅ Multi-heuristic error detection
- ✅ Self-learning feedback system
- ✅ Comprehensive data management
- ✅ Fine-tuning dataset preparation
- ✅ Version control and quality assurance
- ✅ Performance tracking and reporting
- ✅ Evaluation framework with visualizations
- ✅ GCP integration with cost monitoring
- 🔄 Model fine-tuning (in progress)
- 🔄 Automated retraining pipeline (planned)

## 🔮 Future Enhancements

- Automated model retraining based on collected data
- Multi-model support (Wav2Vec2, Conformer)
- Real-time streaming transcription
- Multi-language support
- Advanced error correction using LLMs
- Automated A/B testing framework
- Production deployment with load balancing

## 🤝 Contributing

This project follows standard Python development practices:
- Code style: Black formatter
- Linting: Flake8
- Testing: Pytest
- Documentation: Docstrings with type hints

## 📝 License

Part of the Adaptive Self-Learning Agentic AI System project.

## 👥 Team

- **Team Member 1**: Agent Integration & Error Detection
- **Team Member 2**: Data Management & Infrastructure
- **Team Member 3**: Evaluation Framework & Benchmarking

## 📞 Support

For issues, questions, or contributions:
1. Check the documentation in `docs/`
2. Run example scripts in `experiments/`
3. Review weekly deliverable reports

---

**Last Updated**: November 24, 2025  
**Version**: Week 2 Complete  
**Status**: Production Ready ✅
