"""
Unified Control Panel API for Adaptive Self-Learning Agentic AI System
Provides comprehensive endpoints for frontend control panel
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import os
import time
import librosa
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from src.baseline_model import BaselineSTTModel
from src.agent.agent import STTAgent
from src.data.integration import IntegratedDataManagementSystem
from src.data.finetuning_coordinator import FinetuningCoordinator
from src.data.finetuning_orchestrator import FinetuningConfig

# Initialize FastAPI
app = FastAPI(
    title="STT Control Panel API",
    version="3.0.0",
    description="Unified API for controlling all aspects of the STT system"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
print("🚀 Initializing STT Control Panel...")
baseline_model = BaselineSTTModel(model_name="whisper")
agent = STTAgent(
    baseline_model=baseline_model,
    use_llm_correction=True,
    use_quantization=False
)
data_system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=False  # Set to True for GCS integration
)

# Initialize fine-tuning coordinator
coordinator = None
try:
    coordinator = FinetuningCoordinator(
        data_manager=data_system.data_manager,
        use_gcs=False
    )
except Exception as e:
    print(f"⚠️  Fine-tuning coordinator initialization failed: {e}")

print("✅ Control Panel API initialized successfully")


# ==================== PYDANTIC MODELS ====================

class FeedbackRequest(BaseModel):
    transcript_id: str
    user_feedback: str
    is_correct: bool
    corrected_transcript: Optional[str] = None


class CorrectionRequest(BaseModel):
    case_id: str
    corrected_transcript: str
    correction_method: str = "manual"


class DatasetPrepRequest(BaseModel):
    min_error_score: float = 0.5
    max_samples: Optional[int] = 1000
    balance_error_types: bool = True
    create_version: bool = True


# ==================== SYSTEM STATUS ====================

@app.get("/")
async def root():
    """Root endpoint - system overview"""
    return {
        "name": "Adaptive Self-Learning Agentic AI System - Control Panel",
        "version": "3.0.0",
        "status": "operational",
        "components": {
            "baseline_model": "loaded",
            "agent": "loaded",
            "data_management": "loaded",
            "finetuning_coordinator": "loaded" if coordinator else "unavailable"
        },
        "documentation": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Comprehensive health check"""
    agent_stats = agent.get_agent_stats()
    
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "baseline_model": {
                "status": "operational",
                "model": baseline_model.model_name,
                "device": baseline_model.device
            },
            "agent": {
                "status": "operational",
                "error_threshold": agent_stats['error_detection']['threshold'],
                "llm_available": agent_stats.get('llm_info', {}).get('status') == 'loaded'
            },
            "data_management": {
                "status": "operational"
            },
            "finetuning": {
                "status": "operational" if coordinator else "unavailable"
            }
        }
    }
    return health


@app.get("/api/system/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # Get data system stats
        data_stats = data_system.get_system_statistics()
        
        # Get agent stats
        agent_stats = agent.get_agent_stats()
        
        # Get coordinator stats if available
        coordinator_stats = {}
        if coordinator:
            try:
                coordinator_stats = coordinator.get_system_status()
            except:
                coordinator_stats = {"status": "unavailable"}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data_management": data_stats,
            "agent": agent_stats,
            "finetuning": coordinator_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TRANSCRIPTION ====================

@app.post("/api/transcribe/baseline")
async def transcribe_baseline(file: UploadFile = File(...)):
    """
    Transcribe audio with baseline model only
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        start = time.time()
        result = baseline_model.transcribe(tmp_path)
        result["inference_time_seconds"] = time.time() - start
        
        os.remove(tmp_path)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcribe/agent")
async def transcribe_agent(
    file: UploadFile = File(...),
    auto_correction: bool = True,
    record_if_error: bool = True
):
    """
    Transcribe with agent error detection and optional auto-recording
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Get audio length
        try:
            audio, sr = librosa.load(tmp_path, sr=16000)
            audio_length = len(audio) / sr
        except:
            audio_length = None
        
        # Transcribe with agent
        result = agent.transcribe_with_agent(
            audio_path=tmp_path,
            audio_length_seconds=audio_length,
            enable_auto_correction=auto_correction
        )
        
        # Auto-record if errors detected and enabled
        case_id = None
        if record_if_error and result['error_detection']['has_errors']:
            try:
                case_id = data_system.record_failed_transcription(
                    audio_path=tmp_path,
                    original_transcript=result['original_transcript'],
                    corrected_transcript=result['transcript'] if auto_correction else None,
                    error_types=list(result['error_detection']['error_types'].keys()),
                    error_score=result['error_detection']['error_score'],
                    inference_time=result['inference_time_seconds']
                )
                result['case_id'] = case_id
            except Exception as e:
                print(f"Failed to record error: {e}")
        
        os.remove(tmp_path)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== AGENT MANAGEMENT ====================

@app.post("/api/agent/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for agent learning"""
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


@app.get("/api/agent/stats")
async def get_agent_stats():
    """Get agent statistics"""
    try:
        return agent.get_agent_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agent/learning-data")
async def get_learning_data():
    """Get agent learning data"""
    try:
        return agent.get_learning_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DATA MANAGEMENT ====================

@app.get("/api/data/failed-cases")
async def get_failed_cases(
    limit: int = Query(100, description="Maximum number of cases to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get list of failed cases"""
    try:
        # Get all failed cases
        all_cases = data_system.data_manager.failed_cases
        
        # Paginate
        cases_list = list(all_cases.values())[offset:offset + limit]
        
        return {
            "total": len(all_cases),
            "limit": limit,
            "offset": offset,
            "cases": cases_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/case/{case_id}")
async def get_case_details(case_id: str):
    """Get details of a specific case"""
    try:
        case = data_system.data_manager.failed_cases.get(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        return case
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/correction")
async def add_correction(correction: CorrectionRequest):
    """Add correction to a failed case"""
    try:
        success = data_system.add_correction(
            case_id=correction.case_id,
            corrected_transcript=correction.corrected_transcript,
            correction_method=correction.correction_method
        )
        
        if success:
            return {
                "status": "success",
                "message": "Correction added",
                "case_id": correction.case_id
            }
        else:
            raise HTTPException(status_code=404, detail="Case not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/statistics")
async def get_data_statistics():
    """Get data management statistics"""
    try:
        return data_system.data_manager.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/prepare-dataset")
async def prepare_finetuning_dataset(request: DatasetPrepRequest):
    """Prepare fine-tuning dataset"""
    try:
        dataset_info = data_system.prepare_finetuning_dataset(
            min_error_score=request.min_error_score,
            max_samples=request.max_samples,
            balance_error_types=request.balance_error_types,
            create_version=request.create_version
        )
        
        if 'error' in dataset_info:
            raise HTTPException(status_code=400, detail=dataset_info['error'])
        
        return dataset_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/datasets")
async def list_datasets():
    """List available fine-tuning datasets"""
    try:
        datasets = data_system.finetuning_pipeline.list_datasets()
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/report")
async def generate_report():
    """Generate comprehensive data management report"""
    try:
        report = data_system.generate_comprehensive_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== FINE-TUNING ORCHESTRATION ====================

@app.get("/api/finetuning/status")
async def get_finetuning_status():
    """Get fine-tuning orchestrator status"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Fine-tuning coordinator not available")
    
    try:
        return coordinator.get_system_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/finetuning/trigger")
async def trigger_finetuning(force: bool = False):
    """Manually trigger fine-tuning"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Fine-tuning coordinator not available")
    
    try:
        job = coordinator.orchestrator.trigger_finetuning(force=force)
        
        if not job:
            return {
                "status": "not_triggered",
                "message": "Conditions not met for fine-tuning"
            }
        
        return {
            "status": "triggered",
            "job_id": job.job_id,
            "job": job.__dict__
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/finetuning/jobs")
async def list_finetuning_jobs():
    """List all fine-tuning jobs"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Fine-tuning coordinator not available")
    
    try:
        jobs = coordinator.orchestrator.jobs
        return {
            "jobs": [job.__dict__ for job in jobs.values()]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/finetuning/job/{job_id}")
async def get_job_details(job_id: str):
    """Get details of a specific fine-tuning job"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Fine-tuning coordinator not available")
    
    try:
        job = coordinator.orchestrator.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.__dict__
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MODEL MANAGEMENT ====================

@app.get("/api/models/info")
async def get_model_info():
    """Get current model information"""
    try:
        return baseline_model.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/versions")
async def list_model_versions():
    """List all model versions"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Model management not available")
    
    try:
        versions = coordinator.deployer.versions
        return {
            "versions": [v.__dict__ for v in versions.values()]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/deployed")
async def get_deployed_model():
    """Get currently deployed model"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Model management not available")
    
    try:
        deployed = coordinator.deployer.get_deployed_model()
        if not deployed:
            return {"deployed": None}
        return {"deployed": deployed.__dict__}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== METADATA & TRACKING ====================

@app.get("/api/metadata/performance")
async def get_performance_metrics():
    """Get performance metrics history"""
    try:
        report = data_system.metadata_tracker.generate_performance_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metadata/trends")
async def get_performance_trends(
    metric: str = Query("wer", description="Metric to get trend for (wer, cer)"),
    days: int = Query(30, description="Number of days to look back")
):
    """Get performance trends"""
    try:
        trend = data_system.metadata_tracker.get_performance_trend(
            metric=metric,
            time_window_days=days
        )
        return {"metric": metric, "days": days, "trend": trend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    info = baseline_model.get_model_info()
    agent_stats = agent.get_agent_stats()
    
    print("\n" + "="*60)
    print("🎯 STT CONTROL PANEL API STARTED")
    print("="*60)
    print(f"📊 Model: {info['name']} ({info['parameters']:,} parameters)")
    print(f"💻 Device: {info['device']}")
    print(f"🤖 Agent: Initialized (threshold: {agent_stats['error_detection']['threshold']})")
    print(f"📦 Data Management: Operational")
    print(f"🔧 Fine-tuning: {'Operational' if coordinator else 'Unavailable'}")
    print(f"🌐 API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")


# Mount static files for frontend (will be created next)
static_dir = Path(__file__).parent.parent / "frontend"
if static_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(static_dir), html=True), name="frontend")
    
    @app.get("/app")
    async def serve_frontend():
        """Serve the frontend application"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend not found")


# To run: uvicorn src.control_panel_api:app --reload --port 8000

