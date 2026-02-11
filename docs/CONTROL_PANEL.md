# Control Panel User Guide

Web-based interface for managing the Adaptive Self-Learning Agentic AI System.

## Quick Start

```bash
./start_control_panel.sh
# Or: uvicorn src.control_panel_api:app --reload --port 8000
```

Open **http://localhost:8000/app** (API docs: http://localhost:8000/docs)

## Dashboard Overview

- **System Health**: Baseline model, agent status, LLM availability
- **Agent Statistics**: Errors detected, corrections made, feedback count
- **Data Statistics**: Failed cases, correction rate, average error score
- **Model Information**: Active STT model, parameters, device

## Tabs

### Transcription
- Upload audio (drag-and-drop or click)
- **Baseline (Fast)**: Simple transcription
- **Agent (Recommended)**: Error detection + auto-correction
- Options: Enable auto-correction, record errors for learning

### Data Management
- Browse failed cases with search/filter
- Add corrections, view case details
- Prepare fine-tuning datasets (min score, max samples, balance types)

### Fine-Tuning
- Orchestrator status and readiness
- Trigger fine-tuning (force option for testing)
- View job history

### Models
- Current model info
- Deployed model
- Version history

### Monitoring
- Performance metrics (inferences, latency, error rates)
- Trends (WER/CER over 7/30/90 days)

## API Endpoints

| Category | Endpoints |
|----------|-----------|
| System | `GET /api/health`, `/api/system/stats` |
| Transcription | `POST /api/transcribe/baseline`, `/api/transcribe/agent` |
| Agent | `POST /api/agent/feedback`, `GET /api/agent/stats` |
| Data | `GET /api/data/failed-cases`, `POST /api/data/correction`, `POST /api/data/prepare-dataset` |
| Fine-Tuning | `GET /api/finetuning/status`, `POST /api/finetuning/trigger` |
| Models | `GET /api/models/info`, `/versions`, `/deployed` |
| Monitoring | `GET /api/metadata/performance`, `/trends` |

## Configuration

```bash
export USE_GCS=true
export GCS_BUCKET=your-bucket
export GCP_PROJECT=your-project
```

Edit `src/control_panel_api.py` for backend config; `frontend/app.js` for `API_BASE_URL`, `PAGE_SIZE`.

## Troubleshooting

- **Port in use**: `kill -9 $(lsof -ti:8000)`
- **Frontend not loading**: Verify API at `curl http://localhost:8000/api/health`
- **Fine-tuning unavailable**: Normal if coordinator init fails; check logs

## Additional Resources

- [Setup](SETUP_INSTRUCTIONS.md) | [Data Management](DATA_MANAGEMENT_SYSTEM.md) | [Fine-Tuning](FINETUNING.md)
