"""
Task 3: Real-time inference API using FastAPI
Run this to start the API server for your team to use
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
from baseline_model import BaselineSTTModel
import time

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
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Transcribe
        start = time.time()
        result = model.transcribe(tmp_path)
        result["inference_time_seconds"] = time.time() - start
        
        # Cleanup
        os.remove(tmp_path)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
