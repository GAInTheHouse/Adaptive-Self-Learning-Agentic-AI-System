# STT Control Panel - User Tutorial & Guide

Welcome to the Adaptive Self-Learning Agentic AI System Control Panel! This guide will help you navigate the UI and understand how to use all the features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [UI Overview](#ui-overview)
3. [Navigation Tabs](#navigation-tabs)
4. [Transcription Feature](#transcription-feature)
5. [Model Selection](#model-selection)
6. [Understanding Results](#understanding-results)
7. [Data Management](#data-management)
8. [Fine-Tuning](#fine-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Important Notes](#important-notes)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required dependencies installed (see `requirements.txt`)

### Starting the Control Panel

1. **Navigate to project directory:**
   ```bash
   cd Adaptive-Self-Learning-Agentic-AI-System
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Start the control panel:**
   ```bash
   ./start_control_panel.sh
   ```

4. **Access the UI:**
   - Open your browser and go to: `http://localhost:8000/app`
   - API documentation: `http://localhost:8000/docs`

---

## UI Overview

The Control Panel has a modern, dark-themed interface with the following main sections:

### Header
- **Logo & Title**: STT Control Panel
- **System Status Indicator**: Shows if the system is online/offline (green = online, red = offline)

### Navigation Tabs
Six main tabs for different functionalities:
1. **Dashboard** - System overview and statistics
2. **Transcribe** - Audio transcription interface
3. **Data Management** - Failed cases and dataset preparation
4. **Fine-Tuning** - Fine-tuning orchestration
5. **Models** - Model version management
6. **Monitoring** - Performance metrics and trends

---

## Navigation Tabs

### 1. Dashboard Tab

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

---

### 2. Transcribe Tab ⭐ (Main Feature)

**Purpose**: Upload audio files and get transcriptions with error detection and correction

**Key Features:**
- Upload audio files (.wav, .mp3, .ogg)
- Select STT model version
- Choose transcription mode (Baseline or Agent)
- View side-by-side comparison of original vs. corrected transcripts

#### Step-by-Step Transcription Process:

1. **Select STT Model** (Dropdown):
   - **Wav2Vec2 Base**: Baseline model (facebook/wav2vec2-base-960h)
   - **Fine-tuned Wav2Vec2**: Improved model after fine-tuning

2. **Choose Transcription Mode**:
   - **Agent (Recommended)**: Full pipeline with error detection and LLM correction
     - Processing time: 10-15 seconds (includes LLM processing)
     - Shows both original STT transcript and LLM-refined transcript
   - **Baseline (Fast)**: Simple transcription without error detection
     - Processing time: 1-2 seconds
     - No LLM correction

3. **Agent Options** (only visible in Agent mode):
   - **Enable Auto-Correction**: 
     - ✅ ON: LLM detects errors AND applies corrections
     - ❌ OFF: LLM only detects errors but doesn't correct them
   - **Record Errors Automatically**:
     - ✅ ON: Failed cases are saved for future fine-tuning
     - ❌ OFF: Errors detected but not saved

4. **Upload Audio File**:
   - Click the upload area or drag and drop
   - Supported formats: WAV, MP3, OGG
   - File info will display after selection

5. **Click "Transcribe Audio"**:
   - Button shows loading state during processing
   - Results appear below when complete

#### Understanding Transcription Results:

**Side-by-Side Comparison:**
- **Left Column (Red border)**: STT Original Transcript
  - Raw output from the selected STT model
  - May contain errors, especially with base model
- **Right Column (Blue border)**: LLM Refined Transcript (Gold Standard)
  - Corrected version after LLM analysis
  - Shows what the transcript should be

**Additional Information:**
- **Model Information**: Selected model and mode
- **Error Detection**: 
  - Has Errors: Yes/No badge
  - Error Count: Number of errors found
  - Error Score: Severity score (0-1)
- **Corrections Applied**: Number of corrections made
- **Case Recorded**: Case ID if errors were saved
- **Performance**: Inference time in seconds

---

### 3. Data Management Tab

**Purpose**: View and manage failed transcription cases

**Features:**

#### Failed Cases Section:
- **Search Bar**: Filter cases by keywords
- **Filter Dropdown**: 
  - All Cases
  - Uncorrected (need attention)
  - Corrected (already processed)
- **Case List**: Shows case cards with:
  - Case ID
  - Status badge (Corrected/Uncorrected)
  - Transcript preview
  - Timestamp
  - Error score
- **Pagination**: Navigate through cases (Previous/Next)

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

---

### 4. Fine-Tuning Tab

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

**Note**: Fine-tuning requires sufficient failed cases and proper configuration.

---

### 5. Models Tab

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

---

### 6. Monitoring Tab

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
- View trend data (visualization can be added)

---

## Model Selection Guide

### Understanding Model Versions

#### Wav2Vec2 Base (Baseline)
- **Model**: facebook/wav2vec2-base-960h
- **Framework**: PyTorch
- **Performance**: Baseline accuracy (~36% WER on real-world data)
- **Use Case**: Demonstrates baseline performance before fine-tuning
- **When to use**: Show the "before" state in your demo

#### Fine-tuned Wav2Vec2 (Improved)
- **Model**: Fine-tuned Wav2Vec2 (trained on failed cases)
- **Framework**: PyTorch
- **Performance**: Improved accuracy after fine-tuning
- **Use Case**: Shows improvement after fine-tuning on domain-specific data
- **When to use**: Demonstrate improved performance after fine-tuning

### Model Selection Strategy for Demo:

1. **Start with Baseline**: Upload audio → See baseline transcription
2. **Show Error Detection**: Notice errors in original transcript
3. **Show LLM Correction**: See refined transcript in right column
4. **Explain Fine-tuning**: Mention that errors are saved for training
5. **Switch to Fine-tuned v2/v3**: Upload same audio → See better results

---

## Understanding Results

### Transcript Comparison

**Original STT Transcript (Left):**
- Raw output from speech-to-text model
- May contain:
  - Spelling errors
  - Medical terminology mistakes
  - Grammar issues
  - Word substitutions

**LLM Refined Transcript (Right):**
- Corrected by Llama LLM (via Ollama)
- Improvements:
  - Fixed spelling errors
  - Corrected medical terms
  - Improved grammar
  - Better context understanding

### Error Detection Metrics

- **Has Errors**: Boolean indicating if errors were found
- **Error Count**: Number of individual errors detected
- **Error Score**: Overall quality score (0.0 = perfect, 1.0 = many errors)
- **Error Types**: Categories of errors (medical terminology, spelling, grammar)

### Case Recording

When errors are detected and "Record Errors Automatically" is enabled:
- Case is saved to data management system
- Gets a unique Case ID
- Original and corrected transcripts are stored
- Used for future fine-tuning dataset preparation

---

## Data Management

### Failed Cases Workflow

1. **Automatic Recording**: 
   - Errors detected during transcription
   - Cases automatically saved if "Record Errors Automatically" is ON

2. **Manual Review**:
   - View cases in Data Management tab
   - Filter by status (corrected/uncorrected)
   - Click case to view details

3. **Manual Correction**:
   - Open case details
   - Add correction if needed
   - Save correction

4. **Dataset Preparation**:
   - Set filters (error score, max samples)
   - Click "Prepare Dataset"
   - Dataset created for fine-tuning

### Dataset Preparation Tips

- **Minimum Error Score**: 
  - Lower (0.3): Include more cases, diverse errors
  - Higher (0.7): Only severe errors, focused training
- **Max Samples**: 
  - Start with 100-500 for testing
  - Use 1000+ for production fine-tuning
- **Balance Error Types**: 
  - ✅ Recommended: Ensures diverse training data
  - ❌ Off: May bias toward common error types

---

## Fine-Tuning

### When Fine-Tuning Triggers

The system automatically triggers fine-tuning when:
- Sufficient failed cases accumulated (threshold: configurable)
- Error rate is high enough
- System is ready (no ongoing jobs)

### Manual Trigger

You can manually trigger fine-tuning:
1. Go to Fine-Tuning tab
2. Check "Force Trigger" if needed (bypasses checks)
3. Click "Trigger Fine-Tuning"
4. Monitor job status

### Fine-Tuning Process

1. **Dataset Preparation**: Failed cases converted to training format
2. **Model Training**: Fine-tune on prepared dataset
3. **Validation**: Test against baseline
4. **Deployment**: Deploy if improvements validated
5. **Versioning**: New model version created

---

## Troubleshooting

### Common Issues

#### 1. "System Offline" Status
**Problem**: Red status indicator in header
**Solutions**:
- Check if server is running: `./start_control_panel.sh`
- Verify port 8000 is not in use
- Check server logs for errors

#### 2. Transcription Fails
**Problem**: Error message when transcribing
**Solutions**:
- Check audio file format (WAV, MP3, OGG supported)
- Ensure file is not corrupted
- Check server logs for detailed error
- Verify model is loaded (check Dashboard)

#### 3. "Fine-tuned model not found"
**Problem**: Fine-tuned model cannot be loaded
**Solutions**:
- Ensure fine-tuned model exists at `models/finetuned_wav2vec2/`
- Run fine-tuning script first if model doesn't exist
- Check server logs for detailed error messages

#### 4. Slow Transcription
**Problem**: Transcription takes too long
**Solutions**:
- Agent mode takes 10-15 seconds (normal for LLM processing)
- Use Baseline mode for faster results (1-2 seconds)
- Check system resources (CPU/GPU)
- Reduce audio file size if very large

#### 5. No Results Displayed
**Problem**: Transcription completes but no results shown
**Solutions**:
- Check browser console for JavaScript errors
- Refresh the page
- Check network tab for API errors
- Verify API is responding: `http://localhost:8000/api/health`

#### 6. Model Not Loading
**Problem**: Model fails to load
**Solutions**:
- Check internet connection (models download from Hugging Face)
- Ensure sufficient disk space (~2-4GB per model)
- Check model name is correct
- Review server logs for specific error

### Getting Help

1. **Check Logs**: Server logs show detailed error messages
2. **API Documentation**: Visit `http://localhost:8000/docs` for API details
3. **Health Check**: Visit `http://localhost:8000/api/health` for system status
4. **Browser Console**: Press F12 to see frontend errors

---

## Important Notes

### System Architecture

**Components:**
1. **STT Models**: Speech-to-text transcription (Wav2Vec2)
2. **LLM Corrector**: Llama LLM (via Ollama) for error detection and correction
3. **Error Detector**: Heuristic-based error detection
4. **Data Manager**: Stores failed cases and manages datasets
5. **Fine-tuning Coordinator**: Orchestrates model fine-tuning

### Processing Flow

1. **Audio Upload** → STT Model transcribes
2. **Error Detection** → Detects errors in transcript
3. **LLM Correction** → Llama LLM refines transcript
4. **Case Recording** → Saves errors if enabled
5. **Fine-tuning** → Uses cases to improve model

### Best Practices

1. **For Demos**:
   - Start with Base v1 to show poor performance
   - Use Agent mode to show full pipeline
   - Enable both auto-correction and error recording
   - Switch to Fine-tuned models to show improvement

2. **For Production**:
   - Use Fine-tuned v3 for best accuracy
   - Monitor error rates in Monitoring tab
   - Regularly review failed cases
   - Prepare datasets when sufficient cases accumulated

3. **Audio Files**:
   - Use clear audio (minimize background noise)
   - WAV format recommended for best quality
   - Keep files under 10MB for faster processing
   - Sample rate: 16kHz is optimal

### Performance Expectations

- **Base Model (Wav2Vec2 Base)**: 
  - Speed: ~1-2 seconds
  - Accuracy: ~36% WER on real-world data (demonstrates need for fine-tuning)
  
- **Fine-tuned Model (Fine-tuned Wav2Vec2)**:
  - Speed: ~1-2 seconds
  - Accuracy: Improved after fine-tuning on domain-specific data

- **LLM Correction**:
  - Processing time: <1 second (with Ollama)
  - Improves transcript quality significantly

### Security & Privacy

- All processing happens locally (if using local models)
- Audio files are temporarily stored during processing
- Failed cases stored in `data/production/` directory
- No data sent to external services (unless using cloud APIs)

### Limitations

1. **Ollama LLM**: Requires Ollama server running locally with Llama models installed
2. **Model Loading**: First load takes time (downloads from Hugging Face)
3. **Memory**: Large models require sufficient RAM
4. **Audio Length**: Very long audio files may timeout

---

## Quick Reference

### Keyboard Shortcuts
- **F12**: Open browser developer console
- **Ctrl+R / Cmd+R**: Refresh page
- **Ctrl+Shift+R / Cmd+Shift+R**: Hard refresh (clear cache)

### Important URLs
- **Control Panel**: `http://localhost:8000/app`
- **API Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/health`
- **API Root**: `http://localhost:8000/`

### File Locations
- **Audio Files**: Upload via UI (temporary storage)
- **Failed Cases**: `data/production/failed_cases/`
- **Datasets**: `data/production/finetuning/`
- **Model Versions**: `data/production/versions/`

---

## Demo Script Example

Here's a suggested flow for demonstrating the system:

1. **Introduction** (Dashboard Tab):
   - Show system health
   - Explain components

2. **Base Model Demo** (Transcribe Tab):
   - Select "Wav2Vec2 Base"
   - Upload audio file
   - Show baseline transcription in left column
   - Explain errors

3. **LLM Correction**:
   - Show refined transcript in right column
   - Highlight improvements
   - Explain error detection and correction

4. **Data Collection**:
   - Show case was recorded
   - Explain this feeds fine-tuning

5. **Fine-tuned Model** (Transcribe Tab):
   - Switch to "Fine-tuned Wav2Vec2"
   - Upload same audio
   - Show improved transcription
   - Compare with base model results

6. **System Overview**:
   - Show Data Management tab (failed cases)
   - Show Fine-tuning tab (jobs)
   - Show Monitoring tab (metrics)

---

## Support & Resources

- **Project Documentation**: See `docs/` directory
- **API Documentation**: Built-in at `/docs` endpoint
- **Setup Guide**: See `docs/SETUP_INSTRUCTIONS.md`

---

**Happy Transcribing! 🎤✨**

For questions or issues, check the troubleshooting section or review server logs.

