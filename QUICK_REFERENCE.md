# Quick Reference Guide - STT System

Fast lookup for common commands and operations.

## 🚀 Installation (One-Time)

```bash
# Clone and setup
git clone <repo-url>
cd Adaptive-Self-Learning-Agentic-AI-System
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/verify_setup.py
```

## 🎯 Running the System

### Start API Server

```bash
# Agent API (recommended - includes error detection)
uvicorn src.agent_api:app --reload --port 8000

# Baseline API (simple transcription only)
uvicorn src.inference_api:app --reload --port 8000
```

### Quick Python Script

```python
from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent
from src.data.integration import IntegratedDataManagementSystem

# Initialize
model = BaselineSTTModel()
agent = STTAgent(model)
data_system = IntegratedDataManagementSystem()

# Transcribe
result = agent.transcribe_with_agent("audio.wav", enable_auto_correction=True)
print(result['transcript'])

# Record failures
if result['error_detection']['has_errors']:
    case_id = data_system.record_failed_transcription(
        audio_path="audio.wav",
        original_transcript=result['transcript'],
        error_types=list(result['error_detection']['error_types'].keys()),
        error_score=result['error_detection']['error_score']
    )
```

## 🌐 API Endpoints

### Baseline Endpoints

```bash
# Transcribe
curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.wav"

# Model info
curl "http://localhost:8000/model-info"

# Health
curl "http://localhost:8000/health"
```

### Agent Endpoints

```bash
# Agent transcribe with auto-correction
curl -X POST "http://localhost:8000/agent/transcribe?auto_correction=true" \
  -F "file=@audio.wav"

# Submit feedback
curl -X POST "http://localhost:8000/agent/feedback" \
  -H "Content-Type: application/json" \
  -d '{"transcript_id":"123","is_correct":true,"corrected_transcript":"text"}'

# Get statistics
curl "http://localhost:8000/agent/stats"

# Get learning data
curl "http://localhost:8000/agent/learning-data"
```

## 🧪 Testing

```bash
# Test all components
python experiments/test_baseline.py
python experiments/test_agent.py
python experiments/test_data_management.py
python experiments/test_api.py  # Requires API running

# Run evaluation
python experiments/kavya_evaluation_framework.py

# Run benchmarks
python experiments/run_benchmark.py

# Generate visualizations
python experiments/visualize_evaluation_results.py
```

## 📊 Data Management

```python
from src.data.integration import IntegratedDataManagementSystem

system = IntegratedDataManagementSystem()

# Record failed case
case_id = system.record_failed_transcription(
    audio_path="audio.wav",
    original_transcript="text",
    error_types=["all_caps"],
    error_score=0.75
)

# Add correction
system.add_correction(case_id, "Corrected text")

# Prepare fine-tuning dataset
dataset_info = system.prepare_finetuning_dataset(
    min_error_score=0.5,
    max_samples=1000,
    create_version=True
)

# Get statistics
stats = system.get_system_statistics()
print(f"Cases: {stats['data_management']['total_failed_cases']}")
print(f"Correction rate: {stats['data_management']['correction_rate']:.1%}")

# Generate report
report = system.generate_comprehensive_report("report.json")
```

## ☁️ GCP Commands

```bash
# Setup
gcloud auth login
gcloud config set project stt-agentic-ai-2025
gcloud services enable compute.googleapis.com storage-api.googleapis.com
gsutil mb gs://stt-project-datasets

# Create GPU VM
bash scripts/gcp_scripts/setup_gcp_gpu.sh

# Deploy to GCP
python scripts/gcp_scripts/deploy_to_gcp.py

# Monitor costs
python scripts/gcp_scripts/monitor_gcp_costs.py

# VM control
gcloud compute instances stop stt-gpu-vm --zone=us-central1-a
gcloud compute instances start stt-gpu-vm --zone=us-central1-a
gcloud compute instances delete stt-gpu-vm --zone=us-central1-a
```

## 🔧 Common Operations

### Process Single Audio File

```python
from src.agent import STTAgent
from src.baseline_model import BaselineSTTModel

model = BaselineSTTModel()
agent = STTAgent(model)
result = agent.transcribe_with_agent("audio.wav")
print(result['transcript'])
```

### Process Multiple Files

```python
from pathlib import Path
from src.agent import STTAgent
from src.baseline_model import BaselineSTTModel

model = BaselineSTTModel()
agent = STTAgent(model)

for audio_file in Path("data/raw").glob("*.wav"):
    result = agent.transcribe_with_agent(str(audio_file))
    print(f"{audio_file.name}: {result['transcript']}")
```

### Check System Status

```python
from src.data.integration import IntegratedDataManagementSystem

system = IntegratedDataManagementSystem()
stats = system.get_system_statistics()

print(f"Total cases: {stats['data_management']['total_failed_cases']}")
print(f"Corrected: {stats['data_management']['corrected_cases']}")
print(f"Correction rate: {stats['data_management']['correction_rate']:.1%}")
```

### Export Data for Analysis

```python
from src.data.integration import IntegratedDataManagementSystem

system = IntegratedDataManagementSystem()

# Export to DataFrame
df = system.data_manager.to_dataframe()
print(df.head())

# Export learning data
learning_data = system.data_manager.export_learning_data("export.json")
```

## 📈 Evaluation & Benchmarking

```python
from experiments.kavya_evaluation_framework import EvaluationFramework

framework = EvaluationFramework(model_name="whisper")
results = framework.run_comprehensive_evaluation(
    eval_datasets=["data/processed/test_dataset"]
)

print(f"WER: {results['overall_metrics']['mean_wer']:.4f}")
print(f"CER: {results['overall_metrics']['mean_cer']:.4f}")
```

## 🐛 Troubleshooting

```bash
# Check installation
python scripts/verify_setup.py

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Clear model cache
rm -rf ~/.cache/huggingface/

# Kill process on port
kill -9 $(lsof -ti:8000)

# Fix permissions
chmod -R 755 data/

# Enable debug logging
export PYTHONLOGLEVEL=DEBUG
```

## 📁 Important Files

```
src/baseline_model.py          # Baseline STT model
src/agent_api.py               # Agent API server
src/agent/agent.py             # Agent orchestrator
src/data/integration.py        # Data management system
experiments/test_*.py          # Test scripts
experiments/kavya_evaluation_framework.py  # Evaluation
requirements.txt               # Dependencies
README.md                      # Full documentation
SETUP_INSTRUCTIONS.md          # Detailed setup
```

## 🔑 Key Concepts

- **Baseline Model**: Whisper-based STT without agent features
- **Agent**: Adds error detection and auto-correction
- **Data Management**: Tracks failures and prepares training data
- **Error Types**: 8+ heuristics (all_caps, missing_punctuation, etc.)
- **Fine-tuning Dataset**: Prepared from corrected failed cases
- **Version Control**: Tracks dataset versions with checksums

## 💡 Tips

- Use agent API for production (includes error detection)
- Record failures automatically for continuous learning
- Prepare fine-tuning dataset when you have 500+ corrected cases
- Enable GCS for backup and collaboration
- Use GPU for 3-7x faster inference
- Monitor agent statistics to track system health
- Generate regular reports for performance tracking

## 📞 Help

```bash
# Verify setup
python scripts/verify_setup.py

# Run examples
python experiments/example_usage.py

# Check documentation
ls docs/
```

---

For detailed information, see:
- **README.md** - Full project documentation
- **SETUP_INSTRUCTIONS.md** - Detailed setup guide
- **docs/** - Component-specific guides

