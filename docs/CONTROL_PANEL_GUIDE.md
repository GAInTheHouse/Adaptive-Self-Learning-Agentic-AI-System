# Control Panel User Guide

## 🎯 Overview

The STT Control Panel is a comprehensive web-based interface for managing and controlling all aspects of the Adaptive Self-Learning Agentic AI System. It provides an intuitive dashboard to:

- Transcribe audio files with baseline or agent models
- Monitor system health and performance
- Manage failed cases and corrections
- Prepare fine-tuning datasets
- Trigger and monitor fine-tuning jobs
- View model versions and deployments
- Track performance metrics and trends

---

## 🚀 Quick Start

### 1. Start the Control Panel API

```bash
# Navigate to project directory
cd Adaptive-Self-Learning-Agentic-AI-System

# Activate virtual environment (if using one)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the API server
uvicorn src.control_panel_api:app --reload --port 8000
```

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

## 📊 Dashboard Overview

### System Health Card

Displays real-time system status:
- **Baseline Model**: Current STT model and device
- **Agent Status**: Agent operational status
- **LLM Available**: Whether Gemma LLM is available for corrections
- **Last Check**: Last health check timestamp

### Agent Statistics Card

Shows agent performance metrics:
- **Error Threshold**: Configured error detection threshold
- **Total Errors Detected**: Cumulative errors found
- **Corrections Made**: Number of corrections applied
- **Feedback Count**: User feedback submissions

### Data Statistics Card

Displays data management metrics:
- **Total Failed Cases**: Number of transcription failures recorded
- **Corrected Cases**: Cases with corrections added
- **Correction Rate**: Percentage of cases corrected
- **Average Error Score**: Mean error confidence score

### Model Information Card

Shows current model details:
- **Model Name**: Active STT model
- **Parameters**: Total model parameters
- **Device**: CPU or CUDA GPU
- **Trainable Params**: Number of trainable parameters

---

## 🎤 Transcription Tab

### Upload Audio

1. **Click** the upload area or drag and drop an audio file
2. **Select** transcription mode:
   - **Baseline (Fast)**: Simple transcription without error detection
   - **Agent (Recommended)**: With error detection and auto-correction

### Agent Options

When using Agent mode, you can:
- **Enable Auto-Correction**: Automatically correct detected errors
- **Record Errors Automatically**: Save failed cases for learning

### View Results

After transcription, you'll see:
- **Transcript**: Final transcribed text
- **Error Detection**: Detected errors and confidence scores
- **Corrections Applied**: Original vs. corrected text
- **Performance**: Inference time

---

## 💾 Data Management Tab

### Failed Cases

Browse and manage transcription failures:
- **Search**: Filter cases by text
- **Filter**: Show all, corrected, or uncorrected cases
- **View Details**: Click on a case to see full details
- **Add Corrections**: Submit manual corrections for cases

### Dataset Preparation

Prepare fine-tuning datasets:
1. **Minimum Error Score**: Set threshold (0.0 - 1.0)
2. **Max Samples**: Limit dataset size
3. **Balance Error Types**: Ensure diverse error representation
4. **Create Version**: Enable dataset versioning
5. **Click** "Prepare Dataset" to generate

### Available Datasets

View all prepared datasets ready for fine-tuning.

---

## 🔧 Fine-Tuning Tab

### Orchestrator Status

Monitor fine-tuning system:
- **Status**: System operational status
- **Ready for Fine-tuning**: Whether conditions are met
- **Total Jobs**: Number of fine-tuning jobs run

### Trigger Fine-Tuning

Manually start a fine-tuning job:
1. **Check** "Force Trigger" to bypass readiness checks (optional)
2. **Click** "Trigger Fine-Tuning"
3. **Confirm** the action

### Fine-Tuning Jobs

View all fine-tuning jobs:
- **Job ID**: Unique identifier
- **Status**: Current job status
- **Created At**: Job creation timestamp
- **Dataset ID**: Associated dataset

---

## 🧊 Models Tab

### Current Model

View active model information:
- Model name and parameters
- Device (CPU/GPU)
- Trainable parameters

### Deployed Model

See currently deployed production model:
- Version ID
- Model name
- Deployment timestamp

### Model Versions

Browse all registered model versions:
- Version ID
- Status (deployed, registered, etc.)
- Creation date

---

## 📈 Monitoring Tab

### Performance Metrics

Track system performance:
- **Total Inferences**: Total transcriptions processed
- **Average Inference Time**: Mean processing time
- **Error Detection Rate**: Percentage of errors detected
- **Correction Rate**: Percentage of errors corrected

### Performance Trends

Visualize metrics over time:
1. **Select Metric**: WER or CER
2. **Select Time Window**: 7, 30, or 90 days
3. View trend data

---

## 🎨 Features

### Real-Time Updates

The dashboard automatically updates:
- System health every 30 seconds
- Manual refresh buttons on each card
- Instant feedback on actions

### Responsive Design

Works on:
- Desktop computers
- Tablets
- Mobile devices (responsive layout)

### Toast Notifications

Receive instant feedback:
- ✅ Success messages (green)
- ⚠️ Warning messages (yellow)
- ❌ Error messages (red)
- ℹ️ Info messages (blue)

### Modal Dialogs

Detailed views for:
- Case details with full transcripts
- Correction submissions
- Extended information

---

## 🔌 API Integration

### API Base URL

By default, the frontend connects to:
```
http://localhost:8000
```

### Available Endpoints

#### System
- `GET /` - System overview
- `GET /api/health` - Health check
- `GET /api/system/stats` - System statistics

#### Transcription
- `POST /api/transcribe/baseline` - Baseline transcription
- `POST /api/transcribe/agent` - Agent transcription

#### Agent
- `POST /api/agent/feedback` - Submit feedback
- `GET /api/agent/stats` - Agent statistics
- `GET /api/agent/learning-data` - Learning data

#### Data Management
- `GET /api/data/failed-cases` - List failed cases
- `GET /api/data/case/{case_id}` - Case details
- `POST /api/data/correction` - Add correction
- `GET /api/data/statistics` - Data statistics
- `POST /api/data/prepare-dataset` - Prepare dataset
- `GET /api/data/datasets` - List datasets

#### Fine-Tuning
- `GET /api/finetuning/status` - Orchestrator status
- `POST /api/finetuning/trigger` - Trigger job
- `GET /api/finetuning/jobs` - List jobs

#### Models
- `GET /api/models/info` - Model information
- `GET /api/models/versions` - Model versions
- `GET /api/models/deployed` - Deployed model

#### Monitoring
- `GET /api/metadata/performance` - Performance metrics
- `GET /api/metadata/trends` - Performance trends

---

## 🛠️ Configuration

### Environment Variables

Set these before starting:

```bash
# Optional: Enable GCS integration
export USE_GCS=true
export GCS_BUCKET=your-bucket-name
export GCP_PROJECT=your-project-id

# Optional: Custom port
export PORT=8000
```

### Backend Configuration

Edit `src/control_panel_api.py` to customize:
- Base directory for data storage
- GCS integration
- Model selection
- Error thresholds

### Frontend Configuration

Edit `frontend/app.js` to modify:
- `API_BASE_URL`: Backend API URL
- `PAGE_SIZE`: Cases per page
- Auto-refresh intervals

---

## 🔍 Troubleshooting

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

## 💡 Tips & Best Practices

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

## 📚 Additional Resources

- **Main README**: `README.md` - Project overview
- **API Docs**: `http://localhost:8000/docs` - Interactive API documentation
- **Quick Reference**: `QUICK_REFERENCE.md` - Command reference
- **Data Management**: `docs/DATA_MANAGEMENT_SYSTEM.md` - Detailed data guide
- **Fine-Tuning**: `docs/FINETUNING_ORCHESTRATION.md` - Fine-tuning guide

---

## 🐛 Known Issues

1. **Trend Visualization**: Currently shows text-based data. Integrate Chart.js for visual charts.
2. **Large Files**: Upload size limited by server configuration
3. **Long Transcriptions**: May timeout on very long audio files
4. **Mobile UX**: Some features optimized for desktop use

---

## 🔄 Future Enhancements

- [ ] Real-time audio recording
- [ ] Batch transcription upload
- [ ] Advanced visualization with Chart.js
- [ ] User authentication and roles
- [ ] WebSocket for real-time updates
- [ ] Export/import functionality
- [ ] Custom error type configuration
- [ ] A/B testing interface

---

## 📞 Support

For issues or questions:
1. Check this guide
2. Review API documentation at `/docs`
3. Check system logs
4. Refer to main project documentation

---

## 🎉 Quick Tips

- **Keyboard Shortcuts**: Tab to navigate, Enter to submit
- **Refresh**: Use refresh buttons to update data
- **Search**: Filter cases for quick access
- **Pagination**: Browse large datasets easily
- **Notifications**: Watch for toast messages in top-right

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready ✅

