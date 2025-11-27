# Setup Instructions - Adaptive Self-Learning Agentic AI System

Complete step-by-step guide to set up and run the Adaptive Self-Learning STT system.

## 📋 Table of Contents
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Quick Start Guide](#quick-start-guide)
- [Component Setup](#component-setup)
- [Running the System](#running-the-system)
- [Testing & Verification](#testing--verification)
- [GCP Setup (Optional)](#gcp-setup-optional)
- [Troubleshooting](#troubleshooting)

## 💻 System Requirements

### Minimum Requirements
- **OS**: macOS, Linux, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 10GB free space
- **Internet**: Required for model downloads

### Recommended for Production
- **GPU**: NVIDIA GPU with CUDA support (for faster inference)
- **RAM**: 16GB or more
- **Disk Space**: 50GB+ for datasets
- **Google Cloud Account**: For cloud integration and backups

### Software Dependencies
- Python 3.8+
- pip (Python package manager)
- Git
- CUDA Toolkit (optional, for GPU support)
- Google Cloud SDK (optional, for GCP integration)

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone <repository-url>
cd Adaptive-Self-Learning-Agentic-AI-System

# Or if you already have it
cd /path/to/Adaptive-Self-Learning-Agentic-AI-System
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show venv path)
which python
```

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation
where python
```

### Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This will install:
# - PyTorch and transformers (ML frameworks)
# - FastAPI and uvicorn (API server)
# - librosa and soundfile (audio processing)
# - google-cloud-storage (cloud integration)
# - pandas, numpy (data processing)
# - matplotlib, seaborn (visualization)
# - jiwer (evaluation metrics)
# And more...
```

**Note**: The first install may take 5-10 minutes as it downloads large packages.

### Step 4: Verify Installation

```bash
# Run verification script
python scripts/verify_setup.py

# Expected output:
# ✅ Python version: 3.x.x
# ✅ PyTorch installed: 2.x.x
# ✅ Transformers installed: 4.x.x
# ✅ FastAPI installed: 0.x.x
# ✅ All core dependencies satisfied
# ✅ GPU available: True/False
# ✅ CUDA version: 12.x (if GPU available)
```

### Step 5: Download Model (First Run)

The Whisper model will be automatically downloaded on first use (~300MB).

```bash
# Test model download
python -c "
from src.baseline_model import BaselineSTTModel
print('Downloading Whisper model...')
model = BaselineSTTModel(model_name='whisper')
print('✅ Model loaded successfully!')
info = model.get_model_info()
print(f'Model: {info[\"name\"]}')
print(f'Parameters: {info[\"parameters\"]:,}')
print(f'Device: {info[\"device\"]}')
"
```

## 🚀 Quick Start Guide

### Option 1: Run Everything with Python Script

```python
# quick_start.py
from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent
from src.data.integration import IntegratedDataManagementSystem

# 1. Initialize system
print("Initializing system...")
baseline = BaselineSTTModel(model_name="whisper")
agent = STTAgent(baseline_model=baseline, error_threshold=0.3)
data_system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=False  # Set to True if GCP is configured
)

# 2. Test transcription
print("\nTranscribing test audio...")
result = agent.transcribe_with_agent(
    audio_path="data/test_audio/test_1.wav",
    enable_auto_correction=True
)

print(f"\n✅ Transcript: {result['transcript']}")
print(f"   Errors detected: {result['error_detection']['error_count']}")
print(f"   Error score: {result['error_detection']['error_score']:.2f}")
print(f"   Error types: {list(result['error_detection']['error_types'].keys())}")

# 3. Record if errors found
if result['error_detection']['has_errors']:
    case_id = data_system.record_failed_transcription(
        audio_path="data/test_audio/test_1.wav",
        original_transcript=result['original_transcript'],
        error_types=list(result['error_detection']['error_types'].keys()),
        error_score=result['error_detection']['error_score']
    )
    print(f"\n✅ Recorded failed case: {case_id}")

# 4. Show statistics
stats = data_system.get_system_statistics()
print(f"\n📊 System Statistics:")
print(f"   Total cases: {stats['data_management']['total_failed_cases']}")
print(f"   Corrected: {stats['data_management']['corrected_cases']}")
print(f"   Correction rate: {stats['data_management']['correction_rate']:.1%}")

print("\n✅ Quick start complete!")
```

Save this as `quick_start.py` and run:
```bash
python quick_start.py
```

### Option 2: Run API Server

```bash
# Start the Agent API (recommended)
uvicorn src.agent_api:app --reload --port 8000

# Or start the Baseline API (simple transcription only)
uvicorn src.inference_api:app --reload --port 8000
```

Then test with:
```bash
# Test transcription
curl -X POST "http://localhost:8000/agent/transcribe" \
  -F "file=@data/test_audio/test_1.wav"

# Check health
curl "http://localhost:8000/health"

# Get statistics
curl "http://localhost:8000/agent/stats"
```

## 🔧 Component Setup

### 1. Baseline STT Model Setup

The baseline model is automatically configured. To customize:

```python
from src.baseline_model import BaselineSTTModel

# Use GPU if available
model = BaselineSTTModel(
    model_name="whisper",
    device="cuda"  # or "cpu" to force CPU
)

# Get model information
info = model.get_model_info()
print(f"Device: {info['device']}")
print(f"Parameters: {info['parameters']:,}")

# Test transcription
result = model.transcribe("data/test_audio/test_1.wav")
print(f"Transcript: {result['transcript']}")
```

### 2. Agent System Setup

Configure error detection thresholds:

```python
from src.agent import STTAgent

agent = STTAgent(
    baseline_model=model,
    error_threshold=0.3  # Lower = more sensitive (0.0 - 1.0)
)

# Transcribe with agent
result = agent.transcribe_with_agent(
    audio_path="audio.wav",
    enable_auto_correction=True  # Enable automatic corrections
)

# Provide feedback for learning
agent.record_user_feedback(
    transcript_id="123",
    user_feedback="Good transcription",
    is_correct=True,
    corrected_transcript="Corrected version if needed"
)
```

### 3. Data Management Setup

Initialize the integrated data management system:

```python
from src.data.integration import IntegratedDataManagementSystem

# Local-only mode (no cloud)
system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=False
)

# With Google Cloud Storage (requires GCP setup)
system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=True,
    gcs_bucket_name="stt-project-datasets",
    project_id="stt-agentic-ai-2025"
)

# Record failed transcription
case_id = system.record_failed_transcription(
    audio_path="audio.wav",
    original_transcript="original text",
    corrected_transcript=None,  # Add later
    error_types=["all_caps", "missing_punctuation"],
    error_score=0.75,
    inference_time=0.5
)

# Add correction
system.add_correction(case_id, "Corrected text with proper formatting.")

# Get statistics
stats = system.get_system_statistics()
print(f"Total cases: {stats['data_management']['total_failed_cases']}")
print(f"Correction rate: {stats['data_management']['correction_rate']:.1%}")
```

### 4. Evaluation Framework Setup

Set up and run comprehensive evaluation:

```python
from experiments.kavya_evaluation_framework import EvaluationFramework

# Initialize framework
framework = EvaluationFramework(
    model_name="whisper",
    output_dir="experiments/evaluation_outputs"
)

# Run evaluation (requires test dataset)
results = framework.run_comprehensive_evaluation(
    eval_datasets=["data/processed/test_dataset"],
    output_report=True,
    generate_visualizations=True
)

# Results saved to:
# - experiments/evaluation_outputs/evaluation_report.json
# - experiments/evaluation_outputs/evaluation_summary.json
# - experiments/evaluation_outputs/benchmark_report.json
# - experiments/evaluation_outputs/visualizations/*.png
```

## 🏃 Running the System

### Mode 1: API Server (Production)

**Start the Agent API:**
```bash
# Start server
uvicorn src.agent_api:app --host 0.0.0.0 --port 8000

# For development (auto-reload)
uvicorn src.agent_api:app --reload --port 8000
```

**API Endpoints:**
- `POST /agent/transcribe` - Transcribe with error detection
- `POST /agent/feedback` - Submit feedback
- `GET /agent/stats` - Get agent statistics
- `GET /agent/learning-data` - Get learning data
- `POST /transcribe` - Simple transcription (baseline)
- `GET /model-info` - Model information
- `GET /health` - Health check

**Example Usage:**
```bash
# Transcribe with agent
curl -X POST "http://localhost:8000/agent/transcribe?auto_correction=true" \
  -F "file=@data/test_audio/test_1.wav" | jq

# Submit feedback
curl -X POST "http://localhost:8000/agent/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_id": "unique-id",
    "user_feedback": "Excellent transcription",
    "is_correct": true
  }' | jq

# Get statistics
curl "http://localhost:8000/agent/stats" | jq
```

### Mode 2: Python Scripts (Development)

**Direct Python Usage:**
```python
from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent

# Initialize
model = BaselineSTTModel()
agent = STTAgent(model)

# Process audio files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

for audio_path in audio_files:
    result = agent.transcribe_with_agent(audio_path)
    print(f"File: {audio_path}")
    print(f"Transcript: {result['transcript']}")
    print(f"Errors: {result['error_detection']['error_count']}")
    print("-" * 50)
```

### Mode 3: Batch Processing

**Process Multiple Files:**
```python
from pathlib import Path
from src.agent import STTAgent
from src.baseline_model import BaselineSTTModel
from src.data.integration import IntegratedDataManagementSystem

# Initialize
model = BaselineSTTModel()
agent = STTAgent(model)
data_system = IntegratedDataManagementSystem()

# Get all audio files
audio_dir = Path("data/raw/audio_files")
audio_files = list(audio_dir.glob("*.wav"))

print(f"Processing {len(audio_files)} files...")

for audio_path in audio_files:
    # Transcribe
    result = agent.transcribe_with_agent(str(audio_path))
    
    # Record if errors
    if result['error_detection']['has_errors']:
        case_id = data_system.record_failed_transcription(
            audio_path=str(audio_path),
            original_transcript=result['transcript'],
            error_types=list(result['error_detection']['error_types'].keys()),
            error_score=result['error_detection']['error_score']
        )
        print(f"✅ Processed {audio_path.name} - Recorded case {case_id}")
    else:
        print(f"✅ Processed {audio_path.name} - No errors")

# Generate report
stats = data_system.get_system_statistics()
print(f"\n📊 Batch processing complete!")
print(f"   Total files: {len(audio_files)}")
print(f"   Failed cases: {stats['data_management']['total_failed_cases']}")
```

## ✅ Testing & Verification

### Run All Tests

```bash
# Test baseline model
echo "Testing baseline model..."
python experiments/test_baseline.py

# Test agent system
echo "Testing agent system..."
python experiments/test_agent.py

# Test data management
echo "Testing data management..."
python experiments/test_data_management.py

# Test API (requires API to be running in another terminal)
echo "Starting API for testing..."
uvicorn src.agent_api:app --port 8000 &
API_PID=$!
sleep 5
python experiments/test_api.py
kill $API_PID
```

### Individual Component Tests

**1. Test Baseline Model:**
```bash
python experiments/test_baseline.py
```

Expected output:
```
Testing BaselineSTTModel...
✅ Model loaded successfully
✅ Model info retrieved
✅ Transcription successful
✅ All baseline model tests passed!
```

**2. Test Agent System:**
```bash
python experiments/test_agent.py
```

Expected output:
```
Testing STT Agent...
✅ Agent initialized
✅ Error detection working
✅ Self-learning component active
✅ Feedback recording successful
✅ All agent tests passed!
```

**3. Test Data Management:**
```bash
python experiments/test_data_management.py
```

Expected output:
```
Testing Integrated Data Management System...
✅ System initialized
✅ Failed case recorded
✅ Correction added
✅ Fine-tuning dataset prepared
✅ Version control working
✅ All data management tests passed!
```

**4. Test API Endpoints:**
```bash
# Start API in background
uvicorn src.agent_api:app --port 8000 &

# Wait for startup
sleep 5

# Run tests
python experiments/test_api.py

# Stop API
pkill -f "uvicorn src.agent_api:app"
```

### Run Evaluation & Benchmarks

**Comprehensive Evaluation:**
```bash
cd experiments
python kavya_evaluation_framework.py
```

**Performance Benchmarking:**
```bash
python experiments/run_benchmark.py
```

**Generate Visualizations:**
```bash
python experiments/visualize_evaluation_results.py
```

## ☁️ GCP Setup (Optional)

Setting up Google Cloud Platform integration enables:
- Cloud storage backup
- GPU-accelerated inference
- Collaborative data sharing
- Cost monitoring

### Prerequisites
- Google Cloud account
- Credit card (free tier available)
- gcloud CLI installed

### Step-by-Step GCP Setup

#### 1. Install gcloud CLI

**On macOS:**
```bash
# Using Homebrew
brew install --cask google-cloud-sdk

# Or direct install
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**On Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**On Windows:**
Download from: https://cloud.google.com/sdk/docs/install

#### 2. Authenticate and Configure

```bash
# Login to GCP
gcloud auth login

# Set project (create one first in GCP Console)
gcloud config set project stt-agentic-ai-2025

# Verify configuration
gcloud config list

# Authenticate application default
gcloud auth application-default login
```

#### 3. Enable Required APIs

```bash
# Enable Compute Engine
gcloud services enable compute.googleapis.com

# Enable Cloud Storage
gcloud services enable storage-api.googleapis.com

# Verify enabled services
gcloud services list --enabled
```

#### 4. Create Storage Bucket

```bash
# Create bucket for datasets
gsutil mb -l us-central1 gs://stt-project-datasets

# Set lifecycle (optional - auto-delete old data)
gsutil lifecycle set lifecycle.json gs://stt-project-datasets

# Verify bucket
gsutil ls
```

#### 5. Configure Data Management System

```python
from src.data.integration import IntegratedDataManagementSystem

# Initialize with GCP
system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=True,
    gcs_bucket_name="stt-project-datasets",
    project_id="stt-agentic-ai-2025"
)

# Data is now automatically synced to GCS!
```

#### 6. Create GPU VM (Optional)

For faster inference, create a GPU-enabled VM:

```bash
# Run GPU VM setup script
bash scripts/setup_gcp_gpu.sh

# This creates a VM with:
# - NVIDIA T4 GPU
# - CUDA installed
# - All dependencies
# - Auto-shutdown after 4 hours
```

#### 7. Deploy to GCP

```bash
# Deploy code to GCP VM
python scripts/deploy_to_gcp.py

# This will:
# - Copy code to VM
# - Install dependencies
# - Run evaluation on GPU
# - Download results
```

#### 8. Monitor Costs

```bash
# Check GCP costs and usage
python scripts/monitor_gcp_costs.py

# Expected output:
# 📊 GCP Cost Summary
# - Compute: $X.XX
# - Storage: $X.XX
# - Network: $X.XX
# Total: $X.XX
```

### GCP Best Practices

**Cost Management:**
```bash
# Stop VM when not in use
gcloud compute instances stop stt-gpu-vm --zone=us-central1-a

# Start when needed
gcloud compute instances start stt-gpu-vm --zone=us-central1-a

# Delete when done
gcloud compute instances delete stt-gpu-vm --zone=us-central1-a
```

**Storage Management:**
```bash
# Check storage usage
gsutil du -sh gs://stt-project-datasets

# Clean old data
gsutil rm -r gs://stt-project-datasets/old_data/
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Module Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Make sure you're in the project root
cd /path/to/Adaptive-Self-Learning-Agentic-AI-System

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run with python -m
python -m src.baseline_model
```

#### 2. Model Download Issues

**Problem:**
```
OSError: Can't load model openai/whisper-base
```

**Solution:**
```bash
# Check internet connection
ping huggingface.co

# Clear cache and retry
rm -rf ~/.cache/huggingface/
python -c "from transformers import WhisperProcessor; WhisperProcessor.from_pretrained('openai/whisper-base')"
```

#### 3. GPU Not Detected

**Problem:**
```
Device: cpu (expected: cuda)
```

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 4. API Port Already in Use

**Problem:**
```
ERROR: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)

# Or use different port
uvicorn src.agent_api:app --port 8001
```

#### 5. Audio File Format Issues

**Problem:**
```
LibsndfileError: Error opening 'audio.mp3'
```

**Solution:**
```bash
# Convert to WAV using ffmpeg
ffmpeg -i audio.mp3 -ar 16000 -ac 1 audio.wav

# Or use librosa to load any format
python -c "import librosa; y, sr = librosa.load('audio.mp3', sr=16000)"
```

#### 6. Memory Errors

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Use CPU instead
model = BaselineSTTModel(device="cpu")

# Or use smaller batch size
# Or upgrade GPU memory
```

#### 7. GCS Authentication Errors

**Problem:**
```
google.auth.exceptions.DefaultCredentialsError
```

**Solution:**
```bash
# Re-authenticate
gcloud auth application-default login

# Or set credentials manually
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

#### 8. Permission Errors on Data Directories

**Problem:**
```
PermissionError: [Errno 13] Permission denied: 'data/production'
```

**Solution:**
```bash
# Fix permissions
chmod -R 755 data/

# Or run with sudo (not recommended)
sudo python script.py
```

### Getting Help

If you encounter issues not covered here:

1. **Check Logs**: Look for error messages in terminal output
2. **Review Documentation**: Check `docs/` folder
3. **Run Verification**: `python scripts/verify_setup.py`
4. **Check Examples**: Review `experiments/example_usage.py`
5. **Test Components**: Run individual test scripts

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all components will show debug output
```

## 📊 Verification Checklist

After setup, verify all components:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list` shows packages)
- [ ] Whisper model downloaded (automatic on first use)
- [ ] Baseline model works (`python experiments/test_baseline.py`)
- [ ] Agent system works (`python experiments/test_agent.py`)
- [ ] Data management works (`python experiments/test_data_management.py`)
- [ ] API starts successfully (`uvicorn src.agent_api:app`)
- [ ] API endpoints respond (curl tests work)
- [ ] Test audio files present in `test_audio/`
- [ ] GCP configured (optional)
- [ ] GPU detected (optional)

## 🎉 Next Steps

Once setup is complete:

1. **Run Quick Start**: `python quick_start.py`
2. **Start API Server**: `uvicorn src.agent_api:app --reload`
3. **Process Audio Files**: Use API or Python scripts
4. **Monitor Performance**: Check agent stats and data management reports
5. **Prepare Fine-tuning**: When you have 500+ corrected cases
6. **Deploy to Production**: Use GCP or your preferred platform

## 📚 Additional Resources

- **[README.md](README.md)** - Project overview and features
- **[docs/DATA_MANAGEMENT_SYSTEM.md](docs/DATA_MANAGEMENT_SYSTEM.md)** - Complete data management guide
- **[docs/QUICK_START_DATA_MANAGEMENT.md](docs/QUICK_START_DATA_MANAGEMENT.md)** - Quick start for data management
- **[docs/GCP_SETUP_GUIDE.md](docs/GCP_SETUP_GUIDE.md)** - Detailed GCP guide
- **[WEEK1_DELIVERABLES_REPORT.md](WEEK1_DELIVERABLES_REPORT.md)** - Week 1 report
- **[WEEK2_DELIVERABLES_REPORT.md](WEEK2_DELIVERABLES_REPORT.md)** - Week 2 report

---

**Setup Complete!** 🎉

You're now ready to use the Adaptive Self-Learning STT system. Start with the Quick Start guide and explore the different components.

For questions or issues, refer to the Troubleshooting section or check the documentation in `docs/`.
