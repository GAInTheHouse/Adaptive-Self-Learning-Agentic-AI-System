# Baseline STT Architecture

## Model Selection
- **Selected Model**: Whisper-base (or Wav2Vec2)
- **Reasoning**: [Latency, cost, accuracy tradeoffs from Task 1]
- **Parameters**: [from model_info]

## Inference Environment
- **Framework**: PyTorch + Transformers
- **Device**: GPU (CUDA)
- **Input Format**: 16kHz WAV files
- **Output**: JSON with transcript + metadata

## API Endpoint
- **Base URL**: `http://localhost:8000`
- **POST /transcribe**: Upload audio → get transcript
- **GET /health**: Check service status
- **GET /model-info**: Get model metadata

## Start API (in VSCode terminal):
`cd project/`  
`pip install -r requirements.txt`  
`uvicorn src.inference_api:app --reload --port 8000`

To test:
`curl -X POST "http://localhost:8000/transcribe" -F "file=@test_audio.wav"`

## Performance Baseline
- **Latency**: [from benchmark]
- **Throughput**: [samples/sec]
- **Model Size**: [MB]
- **Estimated Cost**: [$/hour transcribed]

## Dependencies
[List from requirements.txt]
