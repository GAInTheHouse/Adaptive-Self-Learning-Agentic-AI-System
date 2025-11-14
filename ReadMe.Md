# Baseline STT Inference

This repo contains a baseline speech-to-text inference server and evaluation scripts for the course final project.

Files added:
- `model_loader.py` - loads Whisper or Wav2Vec2 and provides transcription helpers.
- `server.py` - FastAPI server exposing `/transcribe` and `/health`.
- `evaluate_models.py` - simple script to measure load and inference time for test audio.
- `baseline.py` - entrypoint to run the server with `python baseline.py`.
- `requirements.txt` - Python dependencies for the baseline.

Quick start:

1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the server (default: Whisper small):

```bash
MODEL_TYPE=whisper python baseline.py
```

3. Transcribe an audio file using curl (replace `audio.wav`):

```bash
curl -F "file=@audio.wav" http://localhost:8000/transcribe
```

Notes:
- By default the server uses `MODEL_TYPE=whisper`. Set `MODEL_TYPE=wav2vec2` to use the Hugging Face model.
- The evaluation script `evaluate_models.py` compares load and inference times.
