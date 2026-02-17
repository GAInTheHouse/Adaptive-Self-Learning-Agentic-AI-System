"""
Task 3: Real-time inference API using FastAPI
Run this to start the API server for your team to use
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from src.baseline_model import BaselineSTTModel
from src.utils.api_helpers import handle_audio_upload, transcribe_with_timing
from src.utils.file_utils import cleanup_temp_file

app = FastAPI(title="STT Baseline API")

# Load model at startup
model = BaselineSTTModel(model_name="whisper")

@app.on_event("startup")
async def startup():
    """Log startup info"""
    info = model.get_model_info()
    print(f"✅ Loaded {info['name']} model with {info['parameters']:,} parameters")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe uploaded audio file
    
    Usage:
        curl -X POST "http://localhost:8000/transcribe" \\
          -F "file=@audio.wav"
    """
    tmp_path = None
    try:
        # Save uploaded file temporarily
        tmp_path = await handle_audio_upload(file)
        
        # Transcribe with timing
        result = transcribe_with_timing(model, tmp_path)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if tmp_path:
            cleanup_temp_file(tmp_path)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": model.model_name,
        "device": model.device
    }

@app.get("/model-info")
async def model_info():
    """Get model metadata"""
    return model.get_model_info()

# To run: uvicorn inference_api:app --reload --port 8000
