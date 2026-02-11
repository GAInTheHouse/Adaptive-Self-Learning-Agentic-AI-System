# Quick Reference

## Installation

```bash
git clone <repo-url>
cd Adaptive-Self-Learning-Agentic-AI-System
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/verify_setup.py
```

## Running the System

```bash
# Agent API (recommended)
uvicorn src.agent_api:app --reload --port 8000

# Baseline API
uvicorn src.inference_api:app --reload --port 8000

# Control Panel
./start_control_panel.sh
```

## API Endpoints

```bash
curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.wav"
curl -X POST "http://localhost:8000/agent/transcribe?auto_correction=true" -F "file=@audio.wav"
curl "http://localhost:8000/agent/stats"
```

## Testing

```bash
python experiments/test_baseline.py
python experiments/test_agent.py
python experiments/test_data_management.py
python experiments/kavya_evaluation_framework.py
pytest tests/
```

## Data Management

```python
from src.data.integration import IntegratedDataManagementSystem

system = IntegratedDataManagementSystem()
case_id = system.record_failed_transcription(...)
system.add_correction(case_id, "Corrected text")
dataset_info = system.prepare_finetuning_dataset(max_samples=1000, create_version=True)
stats = system.get_system_statistics()
```

## GCP

```bash
gcloud auth login
gcloud config set project your-project-id
bash scripts/setup_gcp_gpu.sh
python scripts/deploy_to_gcp.py
python scripts/monitor_gcp_costs.py
```

## Adaptive Scheduling (Week 3)

```python
agent = STTAgent(
    baseline_model=baseline_model,
    enable_adaptive_fine_tuning=True,
    scheduler_history_path="data/processed/scheduler_history.json"
)
stats = agent.get_adaptive_scheduler_stats()
```

## Integration & Testing (Week 4)

```python
from src.integration.unified_system import UnifiedSTTSystem
from src.integration.end_to_end_testing import EndToEndTester

system = UnifiedSTTSystem(
    model_name="whisper",
    enable_error_detection=True,
    enable_llm_correction=True,
    enable_adaptive_fine_tuning=True
)
tester = EndToEndTester(system)
results = tester.run_full_test_suite(audio_files, reference_transcripts)
```

## Troubleshooting

```bash
python scripts/verify_setup.py
kill -9 $(lsof -ti:8000)
```
