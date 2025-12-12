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
import asyncio
import random
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
import logging

logger = logging.getLogger(__name__)

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
# We'll create model instances dynamically based on selection
OLLAMA_MODEL = "llama3.2:3b"  # Default Ollama Llama model

# For now, initialize a default one (baseline = wav2vec2 base)
default_baseline_model = BaselineSTTModel(model_name="wav2vec2-base")
default_agent = STTAgent(
    baseline_model=default_baseline_model,
    use_llm_correction=False,  # disable LLM by default to avoid UI hangs
    llm_model_name=OLLAMA_MODEL,
    use_quantization=False  # Not used for Ollama, kept for compatibility
)

# Store model instances for different versions (lazy loading)
model_instances = {}
agent_instances = {}

def get_model_and_agent(model_name: str):
    """
    Get or create model and agent instances for the specified model version.
    Uses lazy loading to avoid loading all models at startup.
    """
    if model_name not in model_instances:
        print(f"🔄 Loading STT model: {model_name}")
        try:
            model = BaselineSTTModel(model_name=model_name)
            agent = STTAgent(
                baseline_model=model,
                use_llm_correction=False,  # disable LLM to keep UI responsive
                llm_model_name=OLLAMA_MODEL,
                use_quantization=False  # Not used for Ollama
            )
            model_instances[model_name] = model
            agent_instances[model_name] = agent
            print(f"✅ Model {model_name} loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            # Fallback to default
            return default_baseline_model, default_agent
    
    return model_instances[model_name], agent_instances[model_name]
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

def compute_error_score(orig: str, corrected: str) -> Dict[str, Any]:
    """Compute simple word-diff based error metrics."""
    o = (orig or "").strip().lower()
    c = (corrected or "").strip().lower()
    if o == c:
        return {"has_errors": False, "error_count": 0, "error_score": 0.0, "error_types": {}}
    ow = o.split()
    cw = c.split()
    diff_count = sum(1 for x, y in zip(ow, cw) if x != y) + abs(len(ow) - len(cw))
    error_score = min(1.0, max(0.0, diff_count / max(1, len(ow) or 1)))
    return {
        "has_errors": True,
        "error_count": diff_count,
        "error_score": error_score,
        "error_types": {"diff": diff_count},
    }

# Simple in-memory performance counters
perf_counters = {
    "total_inferences": 0,
    "total_inference_time": 0.0,
    "sum_error_scores": 0.0,
}


def _normalize_case(case: Dict) -> Dict:
    """Ensure case fields are JSON-serializable and have string timestamps."""
    c = dict(case)
    ts = c.get("timestamp")
    if isinstance(ts, (datetime,)):
        c["timestamp"] = ts.isoformat()
    elif ts is None:
        c["timestamp"] = datetime.now().isoformat()
    return c


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
    agent_stats = default_agent.get_agent_stats()
    
    # Check Ollama availability
    llm_available = False
    try:
        from src.agent.ollama_llm import OllamaLLM
        ollama_llm = OllamaLLM(model_name="llama3.2:3b")
        llm_available = ollama_llm.is_available()
    except Exception:
        llm_available = False
    
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "baseline_model": {
                "status": "operational",
                "model": default_baseline_model.model_name,
                "device": default_baseline_model.device
            },
            "agent": {
                "status": "operational",
                "error_threshold": agent_stats['error_detection']['threshold'],
                "llm_available": llm_available
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
        agent_stats = default_agent.get_agent_stats()
        
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

# ==================== SAMPLE RECORDINGS ====================

@app.get("/api/data/sample-recordings")
async def list_sample_recordings():
    """
    List files under data/sample_recordings for UI display.
    """
    try:
        sample_dir = Path("data/sample_recordings")
        if not sample_dir.exists():
            return {"files": []}
        
        files = []
        for f in sample_dir.iterdir():
            if f.is_file():
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size
                })
        
        return {"files": sorted(files, key=lambda x: x["name"].lower())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sample recordings: {e}")


# ==================== MOCK DATA FOR DEMO ====================

def get_mock_transcription_result(model: str, mode: str, auto_correction: bool = True) -> Dict[str, Any]:
    """
    Generate mock transcription results for demo purposes.
    Shows how different model versions perform differently on health sector audio.
    """
    import random
    
    # Health sector example transcript
    gold_standard = "The patient presents with chest pain and shortness of breath. Blood pressure is 140 over 90. Recommend ECG and chest X-ray to rule out cardiac issues."
    
    # For base model, introduce subtle errors (2-3 words differing)
    if model == "gemma-base-v1":
        # Subtle differences: "presents" -> "presenting", "ECG" -> "EKG", "rule out" -> "rules out"
        original_text = "The patient presenting with chest pain and shortness of breath. Blood pressure is 140 over 90. Recommend EKG and chest X-ray to rules out cardiac issues."
        refined_text = gold_standard
        has_errors = True
        error_score = 0.65
        error_count = 3
    elif model == "gemma-finetuned-v2":
        # Very minor error: just "EKG" instead of "ECG"
        original_text = "The patient presents with chest pain and shortness of breath. Blood pressure is 140 over 90. Recommend EKG and chest X-ray to rule out cardiac issues."
        refined_text = gold_standard
        has_errors = True
        error_score = 0.25
        error_count = 1
    else:  # v3 - best performance, fine-tuned on health sector data
        original_text = gold_standard
        refined_text = gold_standard
        has_errors = False
        error_score = 0.05
        error_count = 0
    
    result = {
        "transcript": refined_text if (auto_correction and has_errors) else original_text,
        "original_transcript": original_text,
        "corrected_transcript": refined_text if has_errors else None,
        "model_used": model,
        "inference_time_seconds": round(random.uniform(0.5, 2.0), 2),
        "error_detection": {
            "has_errors": has_errors,
            "error_score": error_score,
            "error_count": error_count,
            "error_types": {
                "medical_terminology": error_count if has_errors else 0,
                "spelling": 0,
                "grammar": 0
            }
        }
    }
    
    if auto_correction and has_errors:
        result["corrections"] = {
            "applied": True,
            "count": error_count
        }
    
    # Generate a mock case_id if errors detected
    if has_errors:
        result["case_id"] = f"case_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return result


# ==================== TRANSCRIPTION ====================

@app.post("/api/transcribe/baseline")
async def transcribe_baseline(
    file: UploadFile = File(...),
    model: str = Query("wav2vec2-base", description="STT model version to use")
):
    """
    Transcribe audio with baseline model only (no LLM correction)
    Faster than agent mode since no LLM processing is involved
    """
    try:
        # Get the appropriate model instance
        stt_model, _ = get_model_and_agent(model)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        start = time.time()
        result = stt_model.transcribe(tmp_path)
        result["inference_time_seconds"] = time.time() - start
        result["model_used"] = model
        result["original_transcript"] = result.get("transcript", "")
        
        # Update perf counters
        perf_counters["total_inferences"] += 1
        perf_counters["total_inference_time"] += result.get("inference_time_seconds", 0.0)
        
        # Track error score for average calculation
        error_score = result.get("error_detection", {}).get("error_score", 0.0)
        perf_counters["sum_error_scores"] += error_score
        
        os.remove(tmp_path)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/transcribe/agent")
async def transcribe_agent(
    file: UploadFile = File(...),
    model: str = Query("wav2vec2-base", description="STT model version to use"),
    auto_correction: bool = True,
    record_if_error: bool = True
):
    """
    Transcribe with agent error detection and optional auto-recording
    Uses real STT models and LLM for correction
    """
    try:
        # Get the appropriate model and agent instances
        stt_model, stt_agent = get_model_and_agent(model)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Get audio length
        try:
            audio, sr = librosa.load(tmp_path, sr=16000)
            audio_length = len(audio) / sr
        except Exception as e:
            print(f"Warning: Could not load audio for length calculation: {e}")
            audio_length = None
        
        # Transcribe with agent (this includes STT + LLM correction)
        start = time.time()
        result = stt_agent.transcribe_with_agent(
            audio_path=tmp_path,
            audio_length_seconds=audio_length,
            enable_auto_correction=auto_correction
        )
        result["inference_time_seconds"] = result.get("inference_time_seconds", time.time() - start)
        
        result["model_used"] = model
        
        # Ensure we have original_transcript and corrected_transcript
        if "original_transcript" not in result:
            result["original_transcript"] = result.get("transcript", "")
        
        # If auto_correction was enabled and LLM made corrections, use the corrected version
        if auto_correction and result.get("corrections", {}).get("applied"):
            result["corrected_transcript"] = result.get("transcript", "")
        elif not result.get("error_detection", {}).get("has_errors", False):
            # No errors detected, so original and corrected are the same
            result["corrected_transcript"] = result.get("transcript", "")
        else:
            # Errors detected but correction not applied
            result["corrected_transcript"] = result.get("transcript", result.get("original_transcript", ""))

        # Derive error_detection based on STT vs LLM transcript to avoid hardcoded values
        orig_raw = (result.get("original_transcript") or result.get("transcript") or "")
        corrected_raw = (result.get("corrected_transcript") or result.get("transcript") or "")
        orig = orig_raw.strip()
        corrected = corrected_raw.strip()
        orig_clean = orig.lower()
        corrected_clean = corrected.lower()

        if orig_clean == corrected_clean:
            result["error_detection"] = {
                "has_errors": False,
                "error_count": 0,
                "error_score": 0.0,
                "error_types": {}
            }
        else:
            orig_words = orig_clean.split()
            corr_words = corrected_clean.split()
            diff_count = sum(1 for o, c in zip(orig_words, corr_words) if o != c) + abs(len(orig_words) - len(corr_words))
            error_score = min(1.0, max(0.0, diff_count / max(1, len(orig_words) or 1)))
            result["error_detection"] = {
                "has_errors": True,
                "error_count": diff_count,
                "error_score": error_score,
                "error_types": {"diff": diff_count}
            }

        # Auto-record if errors detected and enabled
        case_id = None
        if record_if_error and result.get('error_detection', {}).get('has_errors', False):
            try:
                case_id = data_system.record_failed_transcription(
                    audio_path=tmp_path,
                    original_transcript=result['original_transcript'],
                    corrected_transcript=result.get('corrected_transcript') if auto_correction else None,
                    error_types=list(result.get('error_detection', {}).get('error_types', {}).keys()),
                    error_score=result.get('error_detection', {}).get('error_score', 0.0),
                    inference_time=result.get('inference_time_seconds', 0.0)
                )
                result['case_id'] = case_id
            except Exception as e:
                print(f"Failed to record error: {e}")
        
        os.remove(tmp_path)
        
        # Update perf counters
        perf_counters["total_inferences"] += 1
        perf_counters["total_inference_time"] += result.get("inference_time_seconds", 0.0)
        
        # Track error score for average calculation
        error_score = result.get("error_detection", {}).get("error_score", 0.0)
        perf_counters["sum_error_scores"] += error_score

        return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ==================== AGENT MANAGEMENT ====================

@app.post("/api/agent/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for agent learning"""
    try:
        default_agent.submit_feedback(
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
        return default_agent.get_agent_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agent/learning-data")
async def get_learning_data():
    """Get agent learning data"""
    try:
        return default_agent.get_learning_data()
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
        all_cases = getattr(data_system, "data_manager", None)
        all_cases = getattr(all_cases, "failed_cases", {}) if all_cases else {}
        if not isinstance(all_cases, dict):
            all_cases = {}

        cases_list = list(all_cases.values())[offset:offset + limit]
        cases_list = [_normalize_case(c) for c in cases_list]
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
        case = getattr(data_system, "data_manager", None)
        case = getattr(case, "failed_cases", {}) if case else {}
        if not isinstance(case, dict):
            case = {}
        case = case.get(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        return _normalize_case(case)
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
        # Return mock status if coordinator not available
        return {
            "status": "unavailable",
            "message": "Fine-tuning coordinator not initialized",
            "orchestrator": {
                "status": "disabled",
                "active_jobs": 0,
                "total_jobs": 0,
                "error_cases_count": 0
            }
        }
    
    try:
        status = coordinator.get_system_status()
        
        # Get real error cases count
        if hasattr(data_system, 'data_manager') and hasattr(data_system.data_manager, 'failed_cases'):
            error_count = len(data_system.data_manager.failed_cases) if isinstance(data_system.data_manager.failed_cases, list) else 0
            if isinstance(status, dict):
                if 'orchestrator' in status:
                    status['orchestrator']['error_cases_count'] = error_count
                else:
                    status['orchestrator'] = {"error_cases_count": error_count}
        
        return status
    except Exception as e:
        logger.error(f"Error getting fine-tuning status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "orchestrator": {
                "status": "error",
                "active_jobs": 0,
                "total_jobs": 0
            }
        }


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
        # Return empty list if coordinator not available
        return {
            "jobs": [],
            "message": "Fine-tuning coordinator not available"
        }
    
    try:
        jobs = coordinator.orchestrator.jobs if hasattr(coordinator, 'orchestrator') and hasattr(coordinator.orchestrator, 'jobs') else {}
        
        # Convert jobs to dict format
        jobs_list = []
        for job in jobs.values():
            if hasattr(job, 'to_dict'):
                jobs_list.append(job.to_dict())
            elif hasattr(job, '__dict__'):
                jobs_list.append(job.__dict__)
            else:
                jobs_list.append(str(job))
        
        return {
            "jobs": jobs_list
        }
    except Exception as e:
        logger.error(f"Error listing fine-tuning jobs: {e}")
        return {
            "jobs": [],
            "error": str(e)
        }


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
async def get_model_info(model: str = Query("wav2vec2-base", description="Model to get info for")):
    """Get model information for specified model"""
    try:
        # Handle model name variations from UI
        if model == "Fine-tuned Wav2Vec2":
            model = "wav2vec2-finetuned"
        elif model == "Wav2Vec2 Base":
            model = "wav2vec2-base"
            
        stt_model, _ = get_model_and_agent(model)
        return stt_model.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/evaluation")
async def get_model_evaluation():
    """Get evaluation results (WER/CER) from fine-tuning"""
    try:
        eval_file = Path("models/finetuned_wav2vec2/evaluation_results.json")
        if not eval_file.exists():
            # Return default values if no evaluation results found
            return {
                "baseline": {
                    "wer": 0.36,
                    "cer": 0.13
                },
                "finetuned": {
                    "wer": 0.36,
                    "cer": 0.13
                },
                "improvement": {
                    "wer_improvement": 0.0,
                    "cer_improvement": 0.0
                },
                "available": False
            }
        
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        
        # Extract WER/CER from evaluation results
        baseline_wer = eval_data.get("baseline_wer", 0.36)
        baseline_cer = eval_data.get("baseline_cer", 0.13)
        finetuned_wer = eval_data.get("finetuned_wer", baseline_wer)
        finetuned_cer = eval_data.get("finetuned_cer", baseline_cer)
        
        return {
            "baseline": {
                "wer": baseline_wer,
                "cer": baseline_cer
            },
            "finetuned": {
                "wer": finetuned_wer,
                "cer": finetuned_cer
            },
            "improvement": {
                "wer_improvement": baseline_wer - finetuned_wer,
                "cer_improvement": baseline_cer - finetuned_cer
            },
            "available": True,
            "evaluation_date": eval_data.get("evaluation_date", None)
        }
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        return {
            "baseline": {"wer": 0.36, "cer": 0.13},
            "finetuned": {"wer": 0.36, "cer": 0.13},
            "improvement": {"wer_improvement": 0.0, "cer_improvement": 0.0},
            "available": False,
            "error": str(e)
        }


@app.get("/api/models/versions")
async def list_model_versions():
    """List all model versions - detect from models folder"""
    try:
        versions = []
        
        # Always include baseline
        baseline_model, _ = get_model_and_agent("wav2vec2-base")
        baseline_info = baseline_model.get_model_info()
        versions.append({
            "version_id": "baseline",
            "model_name": baseline_info["name"],
            "parameters": baseline_info["parameters"],
            "is_current": False,
            "created_at": None
        })
        
        # Check for fine-tuned model
        finetuned_path = Path("models/finetuned_wav2vec2")
        if finetuned_path.exists():
            try:
                from src.agent.fine_tuner import FineTuner
                if FineTuner.model_exists(str(finetuned_path)):
                    # Load metadata if available
                    metadata_file = finetuned_path / "model_metadata.json"
                    created_at = None
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            created_at = metadata.get("saved_at")
                    
                    finetuned_model, _ = get_model_and_agent("wav2vec2-finetuned")
                    finetuned_info = finetuned_model.get_model_info()
                    versions.append({
                        "version_id": "finetuned",
                        "model_name": finetuned_info["name"],
                        "parameters": finetuned_info["parameters"],
                        "is_current": True,  # Fine-tuned is current
                        "created_at": created_at
                    })
            except Exception as e:
                logger.warning(f"Could not load fine-tuned model info: {e}")
        
        # Mark baseline as not current if fine-tuned exists
        if len(versions) > 1:
            versions[0]["is_current"] = False
        
        return {
            "versions": versions
        }
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
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
    """Get performance metrics history with real WER/CER from evaluation"""
    try:
        # Get evaluation results (real WER/CER from fine-tuning)
        eval_file = Path("models/finetuned_wav2vec2/evaluation_results.json")
        baseline_wer = 0.36
        baseline_cer = 0.13
        finetuned_wer = 0.36
        finetuned_cer = 0.13
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                baseline_wer = eval_data.get("baseline_wer", 0.36)
                baseline_cer = eval_data.get("baseline_cer", 0.13)
                finetuned_wer = eval_data.get("finetuned_wer", baseline_wer)
                finetuned_cer = eval_data.get("finetuned_cer", baseline_cer)
            except Exception as e:
                logger.warning(f"Could not read evaluation results: {e}")
        
        # Get live performance counters
        report = data_system.metadata_tracker.generate_performance_report() if hasattr(data_system, 'metadata_tracker') else {}
        # Overlay live perf counters
        overall = report.get("overall_stats", {})
        total_inf = max(overall.get("total_inferences", 0), perf_counters["total_inferences"])
        total_time = perf_counters["total_inference_time"] or overall.get("total_inference_time", 0.0)
        avg_time = (total_time / total_inf) if total_inf > 0 else overall.get("avg_inference_time", 0.0)
        
        # Calculate average error score from actual counters
        sum_error_scores = perf_counters.get("sum_error_scores", 0.0)
        avg_error_score = (sum_error_scores / total_inf) if total_inf > 0 else 0.0
        
        # Get real WER/CER from evaluation results
        eval_file = Path("models/finetuned_wav2vec2/evaluation_results.json")
        baseline_wer = 0.36
        baseline_cer = 0.13
        finetuned_wer = 0.36
        finetuned_cer = 0.13
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                baseline_wer = eval_data.get("baseline_wer", 0.36)
                baseline_cer = eval_data.get("baseline_cer", 0.13)
                finetuned_wer = eval_data.get("finetuned_wer", baseline_wer)
                finetuned_cer = eval_data.get("finetuned_cer", baseline_cer)
            except Exception as e:
                logger.warning(f"Could not read evaluation results: {e}")
        
        overall.update({
            "total_inferences": total_inf,
            "total_inference_time": total_time,
            "avg_inference_time": avg_time,
            "avg_error_score": avg_error_score,
            # Add WER/CER from evaluation results
            "baseline_wer": baseline_wer,
            "baseline_cer": baseline_cer,
            "finetuned_wer": finetuned_wer,
            "finetuned_cer": finetuned_cer,
        })
        report["overall_stats"] = overall
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metadata/trends")
async def get_performance_trends(
    metric: str = Query("wer", description="Metric to get trend for (wer, cer)"),
    days: int = Query(30, description="Number of days to look back")
):
    """Get performance trends - returns baseline and fine-tuned WER/CER"""
    try:
        # Get real evaluation results
        eval_file = Path("models/finetuned_wav2vec2/evaluation_results.json")
        baseline_wer = 0.36
        baseline_cer = 0.13
        finetuned_wer = 0.36
        finetuned_cer = 0.13
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                baseline_wer = eval_data.get("baseline_wer", 0.36)
                baseline_cer = eval_data.get("baseline_cer", 0.13)
                finetuned_wer = eval_data.get("finetuned_wer", baseline_wer)
                finetuned_cer = eval_data.get("finetuned_cer", baseline_cer)
            except Exception as e:
                logger.warning(f"Could not read evaluation results: {e}")
        
        # Return simple two-point trend (baseline vs fine-tuned)
        if metric.lower() == "wer":
            trend = [
                {"date": "baseline", "value": baseline_wer},
                {"date": "fine-tuned", "value": finetuned_wer}
            ]
        elif metric.lower() == "cer":
            trend = [
                {"date": "baseline", "value": baseline_cer},
                {"date": "fine-tuned", "value": finetuned_cer}
            ]
        else:
            trend = []
        
        return {
            "metric": metric,
            "days": days,
            "trend": trend,
            "baseline": {"wer": baseline_wer, "cer": baseline_cer},
            "finetuned": {"wer": finetuned_wer, "cer": finetuned_cer}
        }
    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    info = default_baseline_model.get_model_info()
    agent_stats = default_agent.get_agent_stats()
    
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


# Mount static files for frontend
static_dir = Path(__file__).parent.parent / "frontend"
if static_dir.exists():
    # Mount CSS and JS files at root level for proper access
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/app", response_class=FileResponse)
    async def serve_frontend():
        """Serve the frontend application"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend not found")
    
    @app.get("/styles.css", response_class=FileResponse)
    async def serve_css():
        """Serve CSS file"""
        css_file = static_dir / "styles.css"
        if css_file.exists():
            return FileResponse(str(css_file), media_type="text/css")
        raise HTTPException(status_code=404, detail="CSS not found")
    
    @app.get("/app.js", response_class=FileResponse)
    async def serve_js():
        """Serve JavaScript file"""
        js_file = static_dir / "app.js"
        if js_file.exists():
            return FileResponse(str(js_file), media_type="application/javascript")
        raise HTTPException(status_code=404, detail="JavaScript not found")


# To run: uvicorn src.control_panel_api:app --reload --port 8000

