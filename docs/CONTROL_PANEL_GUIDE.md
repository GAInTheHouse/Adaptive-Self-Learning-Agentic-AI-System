# Control Panel Guide

Complete guide to the STT Control Panel - user guide and implementation details.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [User Guide](#user-guide)
4. [API Reference](#api-reference)
5. [Implementation Details](#implementation-details)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The STT Control Panel is a comprehensive web-based interface for managing and controlling all aspects of the Adaptive Self-Learning Agentic AI System. It provides an intuitive dashboard to:

- Transcribe audio files with baseline or agent models
- Monitor system health and performance
- Manage failed cases and corrections
- Prepare fine-tuning datasets
- Trigger and monitor fine-tuning jobs
- View model versions and deployments
- Track performance metrics and trends

---

## Quick Start

### Prerequisites

Before starting the Control Panel, ensure you have:

1. **Python dependencies installed**:
```bash
pip install -r requirements.txt
```

2. **Ollama installed and running** (for LLM-based error correction):

The Control Panel uses Ollama with Llama models for intelligent error correction. Follow these steps:

#### Install Ollama

**macOS:**
```bash
# Download from https://ollama.ai/download
# Or use Homebrew:
brew install ollama
```

**Linux:**
```bash
# Download from https://ollama.ai/download
# Or use install script:
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
- Download installer from https://ollama.ai/download
- Run the installer

#### Start Ollama Server

**Important**: Ollama must be running before starting the Control Panel API.

```bash
# Start Ollama server in a separate terminal
ollama serve
```

The server runs on `http://localhost:11434` by default. Keep this terminal open.

#### Download Required Model

```bash
# Pull the default model (Llama 3.2 3B - recommended for speed)
ollama pull llama3.2:3b
```

**Alternative models** (better quality, slower):
```bash
# Llama 3.1 8B
ollama pull llama3.1:8b

# Llama 2 7B
ollama pull llama2:7b
```

#### Verify Ollama Setup

```bash
# Check if Ollama is running and models are available
ollama list

# Test the model
ollama run llama3.2:3b "Test message"

# Or use the test script
python scripts/fine_tune_scripts/test_llm_connection.py
```

**Note**: If Ollama is not running or the model is not downloaded, the Control Panel will still work but LLM-based error correction will be disabled (fallback to rule-based correction only).

### 1. Start the Control Panel API

```bash
# Navigate to project directory
cd Adaptive-Self-Learning-Agentic-AI-System

# Activate virtual environment (if using one)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the API server
uvicorn src.control_panel_api:app --reload --port 8000
```

**Important**: Make sure Ollama server is running in another terminal before starting the API.

### 2. Access the Web Interface

Open your web browser and navigate to:

```
http://localhost:8000/app
```

Or access the API documentation:

```
http://localhost:8000/docs
```

---

## User Guide

### Dashboard Tab

**Purpose**: Overview of system health and statistics

**What you'll see:**
- **System Health Card**: Shows baseline model status, agent status, and LLM availability
- **Agent Statistics Card**: 
  - Error detection threshold
  - Total errors detected
  - Corrections made
  - Feedback count
- **Data Statistics Card**:
  - Total failed cases
  - Corrected cases
  - Correction rate percentage
  - Average error score
- **Model Information Card**: Current model details (name, parameters, device)
- **Recent Activity**: Log of recent system activities

**How to use:**
- Click the refresh icon (🔄) on any card to update statistics
- Monitor system health indicators
- Check if all components are operational

### Transcription Tab

**Purpose**: Upload audio files and get transcriptions with error detection and correction

**Key Features:**
- Upload audio files (.wav, .mp3, .ogg)
- Select STT model version
- Choose transcription mode (Baseline or Agent)
- View side-by-side comparison of original vs. corrected transcripts

#### Step-by-Step Transcription Process:

1. **Select STT Model** (Dropdown):
   - **Wav2Vec2 Base**: Baseline model
   - **Fine-tuned Wav2Vec2**: Improved model after fine-tuning

2. **Choose Transcription Mode**:
   - **Agent (Recommended)**: Full pipeline with error detection and LLM correction
   - **Baseline (Fast)**: Simple transcription without error detection

3. **Agent Options** (only visible in Agent mode):
   - **Enable Auto-Correction**: LLM detects errors AND applies corrections
   - **Record Errors Automatically**: Failed cases are saved for future fine-tuning

4. **Upload Audio File**:
   - Click the upload area or drag and drop
   - Supported formats: WAV, MP3, OGG

5. **Click "Transcribe Audio"**:
   - Button shows loading state during processing
   - Results appear below when complete

#### Understanding Transcription Results:

**Side-by-Side Comparison:**
- **Left Column (Red border)**: STT Original Transcript
- **Right Column (Blue border)**: LLM Refined Transcript (Gold Standard)

**Additional Information:**
- **Model Information**: Selected model and mode
- **Error Detection**: Has Errors, Error Count, Error Score
- **Corrections Applied**: Number of corrections made
- **Case Recorded**: Case ID if errors were saved
- **Performance**: Inference time in seconds

### Data Management Tab

**Purpose**: View and manage failed transcription cases

**Features:**

#### Failed Cases Section:
- **Search Bar**: Filter cases by keywords
- **Filter Dropdown**: All Cases, Uncorrected, Corrected
- **Case List**: Shows case cards with ID, status, transcript preview, timestamp, error score
- **Pagination**: Navigate through cases

**Clicking a Case:**
- Opens a modal with full case details
- Shows original and corrected transcripts
- Displays error types
- Option to add manual corrections

#### Dataset Preparation Section:
- **Minimum Error Score**: Filter cases by error severity (0.0-1.0)
- **Max Samples**: Limit number of samples in dataset
- **Balance Error Types**: Ensure diverse error types
- **Create Version**: Create a new dataset version
- **Prepare Dataset Button**: Generate fine-tuning dataset

#### Available Datasets Section:
- Lists all prepared datasets
- Shows dataset IDs and status

### Fine-Tuning Tab

**Purpose**: Manage automated fine-tuning pipeline

**Features:**

#### Orchestrator Status:
- **Status**: Operational/Unavailable
- **Ready for Fine-tuning**: Yes/No indicator
- **Total Jobs**: Number of fine-tuning jobs

#### Trigger Fine-Tuning:
- **Force Trigger**: Bypass readiness checks
- **Trigger Fine-Tuning Button**: Manually start a fine-tuning job

#### Fine-Tuning Jobs:
- List of all fine-tuning jobs
- Shows job ID, status, creation time, and dataset used
- Click to view job details

### Models Tab

**Purpose**: View and manage model versions

**Features:**

#### Current Model:
- Model name and parameters
- Device information
- Trainable parameters

#### Deployed Model:
- Currently deployed model version
- Deployment timestamp
- Model metadata

#### Model Versions:
- List of all model versions
- Status badges (deployed/available)
- Creation timestamps
- Click to view version details

### Monitoring Tab

**Purpose**: Track system performance over time

**Features:**

#### Performance Metrics:
- Total inferences
- Average inference time
- Error detection rate
- Correction rate

#### Performance Trends:
- Select metric (WER or CER)
- Select time window (7/30/90 days)
- View trend data

---

## API Reference

### System Endpoints

```
GET  /                          # System overview
GET  /api/health                # Health check
GET  /api/system/stats          # System statistics
```

### Transcription Endpoints

```
POST /api/transcribe/baseline   # Baseline transcription
POST /api/transcribe/agent      # Agent transcription with error detection
```

### Agent Endpoints

```
POST /api/agent/feedback        # Submit user feedback
GET  /api/agent/stats           # Agent statistics
GET  /api/agent/learning-data   # Get learning data
```

### Data Management Endpoints

```
GET  /api/data/failed-cases      # List failed cases (paginated)
GET  /api/data/case/{case_id}    # Get case details
POST /api/data/correction        # Add correction to case
GET  /api/data/statistics        # Data statistics
POST /api/data/prepare-dataset   # Prepare fine-tuning dataset
GET  /api/data/datasets          # List available datasets
GET  /api/data/report            # Generate comprehensive report
GET  /api/data/sample-recordings # Get sample audio recordings
```

### Fine-Tuning Endpoints

```
GET    /api/finetuning/status      # Orchestrator status
POST   /api/finetuning/trigger     # Trigger fine-tuning job
GET    /api/finetuning/jobs        # List all jobs
GET    /api/finetuning/job/{id}    # Get job details
DELETE /api/finetuning/jobs        # Delete all jobs
GET    /api/finetuning/jobs/info   # Get jobs summary info
```

### Model Management Endpoints

```
GET  /api/models/info        # Current model information
GET  /api/models/versions    # List model versions
GET  /api/models/deployed    # Get deployed model
GET  /api/models/evaluation  # Get model evaluation results
GET  /api/models/available   # List available models
```

### Monitoring Endpoints

```
GET  /api/metadata/performance  # Performance metrics
GET  /api/metadata/trends       # Performance trends
```

---

## Implementation Details

### Backend API

**File:** `src/control_panel_api.py`

A comprehensive FastAPI server that integrates all system components:

**Key Features:**
- ✅ RESTful API with 30+ endpoints
- ✅ CORS enabled for frontend access
- ✅ Integration with baseline model, agent, data management, and fine-tuning systems
- ✅ Automatic component initialization
- ✅ Comprehensive error handling
- ✅ Interactive API documentation (OpenAPI/Swagger)

### Frontend

**Files:**
- `frontend/index.html` - Complete HTML structure
- `frontend/styles.css` - Professional styling with CSS variables
- `frontend/app.js` - Full application logic and API integration

**Features:**
- ✅ 6 main tabs: Dashboard, Transcribe, Data, Fine-Tuning, Models, Monitoring
- ✅ Real-time system health monitoring
- ✅ Drag-and-drop audio file upload
- ✅ Interactive data tables with pagination
- ✅ Modal dialogs for detailed views
- ✅ Toast notifications for user feedback
- ✅ Responsive design (desktop, tablet, mobile)
- ✅ Auto-refresh functionality
- ✅ Search and filter capabilities

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Web Browser                        │
│           (Control Panel Interface)                 │
└─────────────────┬───────────────────────────────────┘
                  │ HTTP/REST API
                  ▼
┌─────────────────────────────────────────────────────┐
│           Control Panel API Server                  │
│            (src/control_panel_api.py)               │
└─────┬─────┬─────┬─────┬─────┬─────────────────────┘
      │     │     │     │     │
      ▼     ▼     ▼     ▼     ▼
   ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
   │Base│ │Agnt│ │Data│ │Fine│ │Meta│
   │line│ │    │ │Mgmt│ │Tune│ │data│
   │Model│ │    │ │    │ │    │ │    │
   └────┘ └────┘ └────┘ └────┘ └────┘
```

---

## Configuration

### Backend Configuration

Edit `src/control_panel_api.py`:
```python
# Enable/disable GCS integration
data_system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=False  # Set to True for cloud storage
)

# Configure coordinator
coordinator = FinetuningCoordinator(
    data_manager=data_system.data_manager,
    use_gcs=False
)
```

### Frontend Configuration

Edit `frontend/app.js`:
```javascript
// Change API URL if needed
const API_BASE_URL = window.location.origin;

// Adjust pagination
const PAGE_SIZE = 20;

// Modify auto-refresh interval (in milliseconds)
setInterval(checkSystemHealth, 30000);  // 30 seconds
```

### Environment Variables

Set these before starting:

```bash
# Optional: Enable GCS integration
export USE_GCS=true
export GCS_BUCKET=your-bucket-name
export GCP_PROJECT=your-project-id

# Optional: Custom port
export PORT=8000

# Optional: Custom Ollama configuration
export OLLAMA_HOST=localhost:11434  # Ollama server URL
export OLLAMA_MODEL=llama3.2:3b      # Default model name
```

### Ollama Configuration

The Control Panel uses Ollama for LLM-based error correction. You can configure it:

**Change the default model** (in `src/control_panel_api.py`):
```python
# Change the default model
OLLAMA_MODEL = "llama3.1:8b"  # Use 8B model instead of 3B

# Or disable LLM correction by default
default_agent = STTAgent(
    baseline_model=default_baseline_model,
    use_llm_correction=False,  # Disable LLM
    ...
)
```

**Use a different Ollama server**:
```python
# If Ollama is running on a different host/port
from src.agent.fine_tuner.llm_corrector import LlamaLLMCorrector

corrector = LlamaLLMCorrector(
    model_name="llama3.2:3b",
    ollama_base_url="http://your-server:11434"  # Custom URL
)
```

---

## Troubleshooting

### Ollama Connection Issues

**Problem**: LLM correction not working, "Ollama server is not running" error

**Solutions**:

1. **Check if Ollama is running**:
```bash
# Check if Ollama server is running
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

3. **Test Ollama connection**:
```bash
# Use the test script
python scripts/fine_tune_scripts/test_llm_connection.py

# Or manually test
ollama run llama3.2:3b "Hello"
```

4. **Check Ollama installation**:
```bash
# Verify Ollama is installed
ollama --version

# If not installed, see installation instructions above
```

5. **Port conflicts**: If port 11434 is in use:
```bash
# Check what's using the port
lsof -i :11434  # macOS/Linux
netstat -ano | findstr :11434  # Windows

# Stop conflicting service or change Ollama port
export OLLAMA_HOST=0.0.0.0:11435
ollama serve
```

**Note**: The Control Panel will work without Ollama, but LLM-based error correction will be disabled. You'll see a warning in the system health status.

### Control Panel Won't Start

**Problem**: API server fails to start

**Solution**:
```bash
# Check if port is in use
lsof -ti:8000

# Kill existing process
kill -9 $(lsof -ti:8000)

# Restart API
uvicorn src.control_panel_api:app --reload --port 8000
```

### Frontend Not Loading

**Problem**: Blank page or 404 error

**Solution**:
1. Ensure API is running: `curl http://localhost:8000/api/health`
2. Check browser console for errors
3. Verify frontend files exist in `frontend/` directory
4. Clear browser cache and reload

### System Shows Offline

**Problem**: Red "System Offline" status

**Solution**:
1. Check API is running
2. Check network connection
3. Verify no firewall blocking localhost:8000
4. Check API logs for errors

### Fine-Tuning Not Available

**Problem**: "Fine-tuning coordinator not available" message

**Solution**:
- This is normal if fine-tuning components aren't initialized
- Check API startup logs for initialization errors
- Ensure all dependencies are installed
- Verify data directories exist

---

## Tips & Best Practices

### For Optimal Performance

1. **Use Agent Mode**: Enable error detection for better quality
2. **Record Errors**: Auto-record helps build training data
3. **Add Corrections**: Manual corrections improve fine-tuning quality
4. **Monitor Regularly**: Check dashboard for system health
5. **Prepare Datasets**: Aim for 500+ corrected cases before fine-tuning

### For Production Use

1. **Enable GCS**: Use cloud storage for data persistence
2. **Set Up Monitoring**: Track performance metrics regularly
3. **Regular Backups**: Export and backup data periodically
4. **Version Control**: Always create versions for datasets
5. **Test Before Deploy**: Validate models before deployment

### For Development

1. **Use Baseline for Testing**: Faster for quick tests
2. **Force Trigger**: Use force flag for testing fine-tuning
3. **Check Logs**: Monitor console logs for debugging
4. **API Docs**: Use `/docs` endpoint for API reference

---

## Support

For help with the control panel:
1. **User Guide**: This document
2. **API Docs**: http://localhost:8000/docs
3. **Frontend Docs**: frontend/README.md
4. **Main Project**: README.md

---

## Summary

The Control Panel provides a **production-ready, comprehensive web interface** for managing the entire Adaptive Self-Learning Agentic AI System. With **30+ API endpoints**, a **modern responsive UI**, and **extensive documentation**, it enables users to:

- 🎤 Transcribe audio with or without agent features
- 💾 Manage failed cases and corrections
- 📊 Monitor system health and performance
- 🔧 Orchestrate fine-tuning workflows
- 🧊 Track model versions and deployments
- 📈 Analyze performance trends

**Status**: ✅ Production Ready  
**Total Implementation**: ~3500+ lines of code  
**Documentation**: Complete  
**Testing**: Ready for use

