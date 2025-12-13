"""
Agent API - Week 2
FastAPI endpoints for agent-integrated STT with error detection and self-learning
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from typing import Optional

from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent
from src.utils.api_helpers import handle_audio_upload, transcribe_with_timing, transcribe_agent_with_timing
from src.utils.file_utils import cleanup_temp_file

app = FastAPI(title="STT Agent API", version="2.0.0")

# Initialize baseline model and agent with Gemma LLM support
baseline_model = BaselineSTTModel(model_name="whisper")
# Initialize agent with LLM correction enabled (will fallback to rule-based if LLM unavailable)
agent = STTAgent(
    baseline_model=baseline_model,
    use_llm_correction=True,  # Enable Gemma LLM for intelligent correction
    use_quantization=False  # Set to True to save memory (requires bitsandbytes)
)


class FeedbackRequest(BaseModel):
    """Feedback request model"""
    transcript_id: str
    user_feedback: str
    is_correct: bool
    corrected_transcript: Optional[str] = None


@app.on_event("startup")
async def startup():
    """Log startup info"""
    info = baseline_model.get_model_info()
    agent_stats = agent.get_agent_stats()
    print(f"✅ Loaded {info['name']} model with {info['parameters']:,} parameters")
    print(f"✅ Agent initialized with error detection and self-learning")
    print(f"   - Error threshold: {agent_stats['error_detection']['threshold']}")
    print(f"   - Total errors learned: {agent_stats['learning']['total_errors_learned']}")
    
    # Show LLM status
    if 'llm_info' in agent_stats:
        llm_info = agent_stats['llm_info']
        if llm_info.get('status') == 'loaded':
            print(f"✅ Gemma LLM ({llm_info.get('model', 'unknown')}) loaded on {llm_info.get('device', 'unknown')}")
        else:
            print(f"⚠️  Gemma LLM not available, using rule-based correction only")
    else:
        print(f"⚠️  Gemma LLM not initialized, using rule-based correction only")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe uploaded audio file (baseline endpoint)
    
    Usage:
        curl -X POST "http://localhost:8000/transcribe" \\
          -F "file=@audio.wav"
    """
    tmp_path = None
    try:
        # Save uploaded file temporarily
        tmp_path = await handle_audio_upload(file)
        
        # Transcribe with timing
        result = transcribe_with_timing(baseline_model, tmp_path)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if tmp_path:
            cleanup_temp_file(tmp_path)


@app.post("/agent/transcribe")
async def agent_transcribe(
    file: UploadFile = File(...),
    auto_correction: bool = True
):
    """
    Transcribe with agent-based error detection and correction
    
    Args:
        file: Audio file to transcribe
        auto_correction: Whether to apply automatic corrections
    
    Usage:
        curl -X POST "http://localhost:8000/agent/transcribe?auto_correction=true" \\
          -F "file=@audio.wav"
    """
    tmp_path = None
    try:
        # Save uploaded file temporarily
        tmp_path = await handle_audio_upload(file)
        
        # Transcribe with agent (includes audio length calculation)
        result = transcribe_agent_with_timing(
            agent,
            tmp_path,
            enable_auto_correction=auto_correction
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if tmp_path:
            cleanup_temp_file(tmp_path)


@app.post("/agent/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for agent learning
    
    Args:
        feedback: Feedback request with transcript_id, feedback, and correctness
    
    Usage:
        curl -X POST "http://localhost:8000/agent/feedback" \\
          -H "Content-Type: application/json" \\
          -d '{"transcript_id": "123", "user_feedback": "Good", "is_correct": true}'
    """
    try:
        agent.submit_feedback(
            transcript_id=feedback.transcript_id,
            user_feedback=feedback.user_feedback,
            is_correct=feedback.is_correct,
            corrected_transcript=feedback.corrected_transcript
        )
        return {
            "status": "success",
            "message": "Feedback recorded",
            "transcript_id": feedback.transcript_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/stats")
async def get_agent_stats():
    """
    Get agent statistics and learning insights
    
    Usage:
        curl "http://localhost:8000/agent/stats"
    """
    try:
        return agent.get_agent_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/learning-data")
async def get_learning_data():
    """
    Get in-memory learning data for external persistence.
    Note: Data persistence handled by data management layer.
    
    Usage:
        curl "http://localhost:8000/agent/learning-data"
    """
    try:
        return agent.get_learning_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    agent_stats = agent.get_agent_stats()
    llm_available = agent_stats.get('llm_info', {}).get('status') == 'loaded' if 'llm_info' in agent_stats else False
    
    return {
        "status": "healthy",
        "model": baseline_model.model_name,
        "device": baseline_model.device,
        "agent_enabled": True,
        "llm_enabled": llm_available
    }


@app.get("/model-info")
async def model_info():
    """Get model metadata"""
    return baseline_model.get_model_info()


# To run: uvicorn src.agent_api:app --reload --port 8000

