from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import time
import os
from typing import Optional

from .model_loader import (
    load_whisper_model,
    load_wav2vec2_model,
    transcribe_whisper,
    transcribe_wav2vec2,
)

app = FastAPI(title="STT Inference API")

# Simple config via env var MODEL_TYPE: 'whisper' or 'wav2vec2'
MODEL_TYPE = os.environ.get("MODEL_TYPE", "whisper")
MODEL_NAME = os.environ.get("MODEL_NAME", "small")

_model = None
_model_meta = {}


def ensure_model():
    global _model, _model_meta
    if _model is not None:
        return
    if MODEL_TYPE == "whisper":
        m, load = load_whisper_model(MODEL_NAME)
        _model = m
        _model_meta = {"type": "whisper", "name": MODEL_NAME, "load_time": load}
    else:
        (proc, m), load = load_wav2vec2_model(MODEL_NAME)
        _model = (proc, m)
        _model_meta = {"type": "wav2vec2", "name": MODEL_NAME, "load_time": load}


@app.on_event("startup")
def startup():
    ensure_model()


@app.get("/health")
def health():
    return {"status": "ok", "model": _model_meta}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = None):
    """Accepts an audio file upload and returns transcription and timing."""
    ensure_model()
    tmp_path = f"/tmp/{int(time.time()*1000)}_{file.filename}"
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        if _model_meta.get("type") == "whisper":
            text, elapsed = transcribe_whisper(_model, tmp_path, language=language) if language else transcribe_whisper(_model, tmp_path)
        else:
            text, elapsed = transcribe_wav2vec2(_model, tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return JSONResponse({"transcription": text, "elapsed_seconds": elapsed, "model": _model_meta})
