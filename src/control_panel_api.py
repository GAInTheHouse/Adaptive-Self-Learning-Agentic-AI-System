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
import re
from datetime import datetime

from src.baseline_model import BaselineSTTModel
from src.agent.agent import STTAgent
from src.data.integration import IntegratedDataManagementSystem
from src.data.finetuning_coordinator import FinetuningCoordinator
from src.data.finetuning_orchestrator import FinetuningConfig
from src.constants import (
    MIN_SAMPLES_FOR_FINETUNING,
    RECOMMENDED_SAMPLES_FOR_FINETUNING,
    SMALL_DATASET_THRESHOLD
)
from src.utils.model_versioning import (
    get_next_model_version,
    get_model_version_name,
    migrate_legacy_models,
    get_all_model_versions,
    get_best_model_version,
    set_current_model,
    get_current_model_path
)
import logging
import torch

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

def get_model_and_agent(model_name: str, use_llm: bool = False):
    """
    Get or create model and agent instances for the specified model version.
    Uses lazy loading to avoid loading all models at startup.
    
    Args:
        model_name: Name of the model to load
        use_llm: Whether to enable LLM correction (default: False for performance)
    """
    cache_key = f"{model_name}_{use_llm}"
    if cache_key not in model_instances:
        print(f"🔄 Loading STT model: {model_name} (LLM: {use_llm})")
        try:
            model = BaselineSTTModel(model_name=model_name)
            
            # Verify which model was actually loaded
            print(f"📊 Model loaded - Name: {model.model_name}, Path: {model.model_path}")
            if hasattr(model, 'is_finetuned'):
                print(f"   Is Fine-tuned: {model.is_finetuned}")
            
            agent = STTAgent(
                baseline_model=model,
                use_llm_correction=use_llm,  # Enable LLM only when explicitly requested
                llm_model_name=OLLAMA_MODEL,
                use_quantization=False  # Not used for Ollama
            )
            model_instances[cache_key] = model
            agent_instances[cache_key] = agent
            print(f"✅ Model {model_name} loaded successfully (actual: {model.model_name})")
            print(f"   Model path: {model.model_path}")
            if hasattr(model, 'is_finetuned') and model.is_finetuned:
                print(f"   ✅ Confirmed: This is a FINE-TUNED model")
            elif "finetuned" in model.model_path.lower() if model.model_path else False:
                print(f"   ✅ Confirmed: This is a FINE-TUNED model (from path)")
            else:
                print(f"   ⚠️  Warning: This appears to be a BASELINE model")
        except FileNotFoundError as e:
            # If fine-tuned model not found, don't silently fall back - return error
            print(f"❌ Fine-tuned model not found for {model_name}: {e}")
            logger.error(f"Fine-tuned model requested but not found: {model_name}")
            raise  # Re-raise to let API return proper error
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            # Only fallback to default for non-finetuned models
            if "finetuned" not in model_name.lower():
                return default_baseline_model, default_agent
            else:
                raise  # Don't silently fallback for fine-tuned models
    
    cached_model = model_instances[cache_key]
    print(f"🔍 Using cached model: {cached_model.model_name} (requested: {model_name})")
    return model_instances[cache_key], agent_instances[cache_key]

data_system = IntegratedDataManagementSystem(
    base_dir="data/production",
    use_gcs=False  # Set to True for GCS integration
)

# Migrate legacy model names on startup
logger.info("🔄 Checking for legacy model names...")
migrations = migrate_legacy_models()
if migrations:
    logger.info(f"✅ Migrated {len(migrations)} legacy models: {migrations}")

# Set current model to best WER model on startup
try:
    best_model_path = get_best_model_version()
    if best_model_path:
        set_current_model(model_path=best_model_path)
        logger.info(f"✅ Set best model (lowest WER) as current on startup: {best_model_path}")
    else:
        logger.info("ℹ️  No fine-tuned models found, baseline will be used")
except Exception as e:
    logger.warning(f"Could not set current model on startup: {e}")

# Initialize fine-tuning coordinator
coordinator = None
try:
    coordinator = FinetuningCoordinator(
        data_manager=data_system.data_manager,
        use_gcs=False
    )
    
    # Set up training callback to actually run fine-tuning
    def training_callback(job, training_params=None):
        """Callback function to run actual fine-tuning."""
        try:
            from src.agent.fine_tuner import FineTuner
            
            logger.info(f"Starting fine-tuning for job {job.job_id}")
            
            # Get dataset path from job
            # Try to get from job_info first
            job_info = coordinator.orchestrator.get_job_info(job.job_id)
            dataset_path = None
            
            if job_info:
                # Try dataset_info.local_path
                if 'dataset_info' in job_info and job_info['dataset_info']:
                    dataset_info = job_info['dataset_info']
                    if 'local_path' in dataset_info:
                        dataset_path = Path(dataset_info['local_path'])
                
                # Fallback: try dataset_path directly
                if not dataset_path and 'dataset_path' in job_info:
                    dataset_path = Path(job_info['dataset_path'])
            
            # If still not found, try constructing from dataset_id
            if not dataset_path and job.dataset_id:
                # Construct path from dataset_id
                dataset_dir = coordinator.orchestrator.dataset_pipeline.output_dir / job.dataset_id
                if dataset_dir.exists():
                    dataset_path = dataset_dir
                    logger.info(f"Constructed dataset path from dataset_id: {dataset_path}")
            
            if not dataset_path:
                logger.error(f"Dataset path not found for job {job.job_id}. Job has dataset_id: {job.dataset_id}")
                logger.error(f"Job info keys: {list(job_info.keys()) if job_info else 'No job_info'}")
                if job_info and 'dataset_info' in job_info:
                    logger.error(f"Dataset info keys: {list(job_info['dataset_info'].keys()) if job_info['dataset_info'] else 'No dataset_info'}")
                return False
            
            if not dataset_path.exists():
                logger.error(f"Dataset path does not exist: {dataset_path}")
                return False
            
            logger.info(f"Using dataset path: {dataset_path}")
            
            # Load error samples from dataset JSONL files
            error_samples = []
            train_file = dataset_path / "train.jsonl"
            val_file = dataset_path / "val.jsonl"
            test_file = dataset_path / "test.jsonl"
            
            # Load from all available splits
            for jsonl_file in [train_file, val_file, test_file]:
                if jsonl_file.exists():
                    with open(jsonl_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    sample = json.loads(line)
                                    # Handle different field names that might be in the dataset
                                    audio_path = sample.get('audio_path') or sample.get('input_path')
                                    # For corrected transcript, prefer corrected_transcript, fallback to target_text
                                    corrected_transcript = (
                                        sample.get('corrected_transcript') or 
                                        sample.get('target_text') or 
                                        sample.get('output_text')
                                    )
                                    
                                    if audio_path and corrected_transcript:
                                        error_samples.append({
                                            'audio_path': audio_path,
                                            'corrected_transcript': corrected_transcript
                                        })
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse JSON line in {jsonl_file}: {e}")
                                    continue
            
            if len(error_samples) < RECOMMENDED_SAMPLES_FOR_FINETUNING:
                logger.warning(f"Insufficient error samples ({len(error_samples)}) for fine-tuning")
                logger.warning(f"Recommended {RECOMMENDED_SAMPLES_FOR_FINETUNING} samples (minimum: {MIN_SAMPLES_FOR_FINETUNING}). Found {len(error_samples)} samples in dataset")
                logger.warning(f"Train file exists: {train_file.exists()}, Val file exists: {val_file.exists()}, Test file exists: {test_file.exists()}")
                # For very small datasets, we can still try but it may not work well
                if len(error_samples) < MIN_SAMPLES_FOR_FINETUNING:
                    logger.error(f"Too few samples ({len(error_samples)}), cannot proceed with fine-tuning (minimum: {MIN_SAMPLES_FOR_FINETUNING})")
                    return False
                else:
                    logger.warning(f"Proceeding with only {len(error_samples)} samples (may not work well)")
            
            logger.info(f"Loaded {len(error_samples)} error samples from dataset at {dataset_path}")
            
            # Initialize fine-tuner
            device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
            finetuner = FineTuner(
                model_name="facebook/wav2vec2-base-960h",
                device=device,
                use_lora=False  # LoRA disabled for Wav2Vec2 stability
            )
            
            # Fine-tune the model - use versioned naming
            next_version = get_next_model_version()
            version_name = get_model_version_name(next_version)
            output_path = Path(f"models/{version_name}")
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📦 Saving fine-tuned model to: {output_path} (version {next_version})")
            
            # Prepare training parameters
            # For small datasets, use smaller batch size and fewer epochs to avoid overfitting
            num_samples = len(error_samples)
            if num_samples < SMALL_DATASET_THRESHOLD:
                default_epochs = 2  # Fewer epochs for small datasets
                default_batch_size = min(2, num_samples)  # Small batch size
            else:
                default_epochs = 3
                default_batch_size = 4
            
            train_params = {
                'num_epochs': training_params.get('epochs', default_epochs) if training_params else default_epochs,
                'batch_size': training_params.get('batch_size', default_batch_size) if training_params else default_batch_size,
                'learning_rate': training_params.get('learning_rate', 5e-6) if training_params else 5e-6,
            }
            
            # Use minimum samples constant
            min_samples = MIN_SAMPLES_FOR_FINETUNING if num_samples < SMALL_DATASET_THRESHOLD else RECOMMENDED_SAMPLES_FOR_FINETUNING
            
            result = finetuner.fine_tune(
                error_samples=error_samples,
                min_samples=min_samples,  # Allow small datasets
                **train_params
            )
            
            if result and result.get('success', False):
                logger.info(f"Fine-tuning completed successfully for job {job.job_id}")
                # Save model to output path
                finetuner._save_model(str(output_path))
                
                # Note: Evaluation results will be saved by the fine-tuning script if run separately
                # For now, we'll rely on evaluation_results.json that may exist from previous runs
                # The orchestrator will check for evaluation results when selecting the best model
                
                # Complete the training in orchestrator
                coordinator.orchestrator.complete_training(
                    job_id=job.job_id,
                    model_path=str(output_path),
                    training_metrics=result.get('metrics', {})
                )
                
                # Find best model by WER and set as current
                best_model_path = get_best_model_version()
                if best_model_path:
                    set_current_model(model_path=best_model_path)
                    logger.info(f"✅ Set best model (lowest WER) as current: {best_model_path}")
                else:
                    # If no best model found, set the newly trained one as current
                    set_current_model(model_path=str(output_path))
                    logger.info(f"✅ Set newly trained model as current: {output_path}")
                
                return True
            else:
                logger.error(f"Fine-tuning failed for job {job.job_id}: {result.get('reason', 'unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"Training callback error: {e}", exc_info=True)
            return False
    
    # Register the callback
    coordinator.set_training_callback(training_callback)
    logger.info("✅ Training callback registered for fine-tuning orchestrator")
    
except Exception as e:
    print(f"⚠️  Fine-tuning coordinator initialization failed: {e}")
    import traceback
    traceback.print_exc()

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
    List files under data/sample_recordings_for_UI for UI display.
    """
    try:
        sample_dir = Path("data/sample_recordings_for_UI")
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
        
        # Verify which model is actually being used
        logger.info(f"📝 Transcribing with model: {model} -> Actual: {stt_model.model_name}, Path: {stt_model.model_path}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        start = time.time()
        result = stt_model.transcribe(tmp_path)
        result["inference_time_seconds"] = time.time() - start
        result["model_used"] = model
        result["model_name"] = stt_model.model_name  # Include actual model name
        result["model_path"] = stt_model.model_path  # Include model path for verification
        result["original_transcript"] = result.get("transcript", "")
        
        logger.info(f"✅ Transcription complete. Model: {stt_model.model_name}, Transcript: {result.get('transcript', '')[:50]}...")
        
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
        # Enable LLM if auto_correction is enabled
        stt_model, stt_agent = get_model_and_agent(model, use_llm=auto_correction)
        
        # Verify which model is actually being used
        logger.info(f"📝 Transcribing with agent. Model: {model} -> Actual: {stt_model.model_name}, Path: {stt_model.model_path}")
        logger.info(f"   Agent's baseline_model: {stt_agent.baseline_model.model_name}, Path: {stt_agent.baseline_model.model_path}")
        
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
        
        # Transcribe with agent (this includes STT + LLM correction if enabled)
        start = time.time()
        result = stt_agent.transcribe_with_agent(
            audio_path=tmp_path,
            audio_length_seconds=audio_length,
            enable_auto_correction=auto_correction
        )
        result["inference_time_seconds"] = result.get("inference_time_seconds", time.time() - start)
        
        result["model_used"] = model
        result["model_name"] = stt_model.model_name  # Include actual model name for verification
        result["model_path"] = stt_model.model_path  # Include model path for verification
        result["agent_model_name"] = stt_agent.baseline_model.model_name  # Verify agent's model
        
        logger.info(f"✅ Agent transcription complete. Model: {stt_model.model_name}, Original: {result.get('original_transcript', '')[:50]}...")
        
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

        # Derive error_detection based on STT vs LLM transcript
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
    """Get list of failed cases from JSONL file"""
    try:
        failed_cases_file = Path("data/production/failed_cases/failed_cases.jsonl")
        all_cases = []
        
        if failed_cases_file.exists():
            with open(failed_cases_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            case_data = json.loads(line)
                            all_cases.append(case_data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line in failed_cases.jsonl: {e}")
                            continue
        else:
            logger.warning(f"Failed cases file not found: {failed_cases_file}")
        
        # Sort by timestamp (newest first) if available
        all_cases.sort(
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        # Apply pagination
        total = len(all_cases)
        cases_list = all_cases[offset:offset + limit]
        cases_list = [_normalize_case(c) for c in cases_list]
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "cases": cases_list
        }
    except Exception as e:
        logger.error(f"Error loading failed cases: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/case/{case_id}")
async def get_case_details(case_id: str):
    """Get details of a specific case from JSONL file"""
    try:
        failed_cases_file = Path("data/production/failed_cases/failed_cases.jsonl")
        case = None
        
        if failed_cases_file.exists():
            with open(failed_cases_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            case_data = json.loads(line)
                            if case_data.get('case_id') == case_id:
                                case = case_data
                                break
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line in failed_cases.jsonl: {e}")
                            continue
        
        if not case:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
        
        return _normalize_case(case)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading case {case_id}: {e}")
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
        # Get system status from coordinator
        system_status = coordinator.get_system_status()
        orchestrator_status = system_status.get('orchestrator', {})
        
        # Get real error cases count from data manager
        error_count = 0
        try:
            stats = data_system.data_manager.get_statistics()
            error_count = stats.get('total_failed_cases', 0)
        except Exception as e:
            logger.warning(f"Could not get error cases count: {e}")
            # Fallback: try reading from failed cases file directly
            failed_cases_file = Path("data/production/failed_cases/failed_cases.jsonl")
            if failed_cases_file.exists():
                with open(failed_cases_file, 'r') as f:
                    error_count = sum(1 for line in f if line.strip())
        
        # Extract trigger conditions
        trigger_conditions = orchestrator_status.get('trigger_conditions', {})
        should_trigger = trigger_conditions.get('should_trigger', False)
        trigger_metrics = trigger_conditions.get('metrics', {})
        
        # Calculate active jobs (running, training, preparing)
        jobs_by_status = orchestrator_status.get('jobs_by_status', {})
        active_jobs = (
            jobs_by_status.get('running', 0) +
            jobs_by_status.get('training', 0) +
            jobs_by_status.get('preparing', 0) +
            jobs_by_status.get('in_progress', 0)
        )
        
        # Get configuration
        config = orchestrator_status.get('config', {})
        min_error_cases = config.get('min_error_cases', 100)
        min_corrected_cases = config.get('min_corrected_cases', 50)
        
        # Calculate how many more error cases are needed
        current_error_cases = error_count
        cases_needed = max(0, min_error_cases - current_error_cases)
        
        # Determine overall status
        if should_trigger:
            overall_status = "ready"
        elif active_jobs > 0:
            overall_status = "active"
        elif orchestrator_status.get('total_jobs', 0) > 0:
            overall_status = "operational"
        else:
            overall_status = "idle"
        
        # Format response for frontend
        return {
            "status": overall_status,
            "orchestrator": {
                "status": overall_status,
                "error_cases_count": error_count,
                "total_jobs": orchestrator_status.get('total_jobs', 0),
                "active_jobs": active_jobs,
                "jobs_by_status": jobs_by_status,
                "should_trigger": should_trigger,
                "trigger_reasons": trigger_conditions.get('reasons', []),
                "min_error_cases": min_error_cases,
                "min_corrected_cases": min_corrected_cases,
                "cases_needed": cases_needed,
                "cases_needed_message": f"Need {cases_needed} more error cases" if cases_needed > 0 else "Threshold met",
                "corrected_cases": trigger_metrics.get('corrected_cases', 0)
            },
            "timestamp": orchestrator_status.get('timestamp', datetime.now().isoformat())
        }
    except Exception as e:
        logger.error(f"Error getting fine-tuning status: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "orchestrator": {
                "status": "error",
                "active_jobs": 0,
                "total_jobs": 0,
                "error_cases_count": 0
            }
        }


@app.post("/api/finetuning/trigger")
async def trigger_finetuning(force: bool = False):
    """Manually trigger fine-tuning"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Fine-tuning coordinator not available")
    
    try:
        # Trigger fine-tuning job
        job = coordinator.orchestrator.trigger_finetuning(force=force)
        
        if not job:
            return {
                "status": "not_triggered",
                "message": "Conditions not met for fine-tuning. Use force=true to trigger anyway."
            }
        
        # If job is ready, start training automatically
        if job.status == 'ready':
            logger.info(f"Job {job.job_id} is ready, starting training...")
            # Start training (this will call the training callback)
            training_started = coordinator.orchestrator.start_training(job.job_id)
            if not training_started:
                logger.warning(f"Failed to start training for job {job.job_id}")
        
        return {
            "status": "triggered",
            "job_id": job.job_id,
            "job_status": job.status,
            "message": f"Fine-tuning job {job.job_id} created and training started"
        }
    except Exception as e:
        logger.error(f"Error triggering fine-tuning: {e}", exc_info=True)
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
        
        # Sort by creation time (newest first) - reverse sort
        try:
            jobs_list.sort(key=lambda x: x.get('created_at', '') if isinstance(x, dict) else '', reverse=True)
        except Exception as e:
            logger.warning(f"Could not sort jobs: {e}")
        
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


@app.delete("/api/finetuning/jobs")
async def clear_finetuning_jobs():
    """Clear all fine-tuning job history"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Fine-tuning coordinator not available")
    
    try:
        orchestrator = coordinator.orchestrator
        jobs_file = orchestrator.jobs_file
        
        # Clear in-memory jobs
        orchestrator.jobs = {}
        
        # Clear the jobs file
        if jobs_file.exists():
            # Backup the old file before clearing
            backup_file = jobs_file.with_suffix('.jsonl.backup')
            if backup_file.exists():
                backup_file.unlink()
            jobs_file.rename(backup_file)
            logger.info(f"Backed up jobs file to {backup_file}")
            
            # Create empty file
            jobs_file.touch()
            logger.info(f"Cleared fine-tuning jobs file: {jobs_file}")
        
        return {
            "success": True,
            "message": "Fine-tuning jobs cleared",
            "jobs_file": str(jobs_file),
            "backup_file": str(backup_file) if jobs_file.exists() or backup_file.exists() else None
        }
    except Exception as e:
        logger.error(f"Error clearing fine-tuning jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/finetuning/jobs/info")
async def get_jobs_info():
    """Get information about fine-tuning jobs storage"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Fine-tuning coordinator not available")
    
    try:
        orchestrator = coordinator.orchestrator
        jobs_file = orchestrator.jobs_file
        
        job_count = len(orchestrator.jobs)
        file_exists = jobs_file.exists()
        file_size = jobs_file.stat().st_size if file_exists else 0
        
        return {
            "jobs_file": str(jobs_file),
            "jobs_file_exists": file_exists,
            "jobs_file_size_bytes": file_size,
            "jobs_in_memory": job_count,
            "absolute_path": str(jobs_file.absolute())
        }
    except Exception as e:
        logger.error(f"Error getting jobs info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MODEL MANAGEMENT ====================

@app.get("/api/models/info")
async def get_model_info(model: str = Query(None, description="Model to get info for. If not specified, returns current model (fine-tuned if available, else base)")):
    """Get model information for specified model, or current model if not specified"""
    try:
        # If model not specified, detect current model (best fine-tuned if available, else base)
        if model is None:
            current_model_path = get_current_model_path()
            if current_model_path:
                # Extract version from path
                path_obj = Path(current_model_path)
                version_match = re.match(r'finetuned_wav2vec2_v(\d+)', path_obj.name)
                if version_match:
                    version_num = version_match.group(1)
                    model = f"wav2vec2-finetuned-v{version_num}"
                elif path_obj.name in ["finetuned_wav2vec2", "finetuned"]:
                    model = "wav2vec2-finetuned"  # Legacy
                else:
                    model = "wav2vec2-base"
            else:
                model = "wav2vec2-base"
        
        # Handle model name variations from UI
        if model == "Fine-tuned Wav2Vec2":
            # Use current model or default to latest
            current_model_path = get_current_model_path()
            if current_model_path:
                path_obj = Path(current_model_path)
                version_match = re.match(r'finetuned_wav2vec2_v(\d+)', path_obj.name)
                if version_match:
                    version_num = version_match.group(1)
                    model = f"wav2vec2-finetuned-v{version_num}"
                else:
                    model = "wav2vec2-finetuned"
            else:
                model = "wav2vec2-finetuned"
        elif model == "Wav2Vec2 Base":
            model = "wav2vec2-base"
            
        stt_model, _ = get_model_and_agent(model)
        return stt_model.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/evaluation")
async def get_model_evaluation():
    """Get evaluation results (WER/CER) - uses current model's evaluation results"""
    try:
        # Get current model path and use its evaluation results
        current_model_path = get_current_model_path()
        
        # Default to baseline values
        baseline_wer = 0.36
        baseline_cer = 0.13
        finetuned_wer = baseline_wer
        finetuned_cer = baseline_cer
        available = False
        
        if current_model_path:
            eval_file = Path(current_model_path) / "evaluation_results.json"
        else:
            # Fallback to legacy path
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
        
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            # Extract WER/CER from evaluation results
            # The structure is: baseline_metrics['wer'], fine_tuned_metrics['wer'], etc.
            baseline_metrics = eval_data.get("baseline_metrics", {})
            fine_tuned_metrics = eval_data.get("fine_tuned_metrics", {})
            
            baseline_wer = baseline_metrics.get("wer", 0.36)
            baseline_cer = baseline_metrics.get("cer", 0.13)
            finetuned_wer = fine_tuned_metrics.get("wer", baseline_wer)
            finetuned_cer = fine_tuned_metrics.get("cer", baseline_cer)
            
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
                "evaluation_date": eval_data.get("timestamp") or eval_data.get("evaluation_date"),
                "model_path": str(eval_file.parent) if current_model_path else None
            }
        except Exception as e:
            logger.warning(f"Error reading evaluation file {eval_file}: {e}")
            # Return defaults on error
            return {
                "baseline": {"wer": 0.36, "cer": 0.13},
                "finetuned": {"wer": 0.36, "cer": 0.13},
                "improvement": {"wer_improvement": 0.0, "cer_improvement": 0.0},
                "available": False,
                "error": str(e)
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


@app.get("/api/models/available")
async def list_available_models():
    """List all available models for transcription - includes all versioned fine-tuned models"""
    try:
        models = []
        
        # Always include baseline with clear display name
        models.append({
            "id": "wav2vec2-base",
            "name": "Wav2Vec2 Base",
            "display_name": "Wav2Vec2 Base",
            "is_available": True,
            "is_finetuned": False,
            "is_current": False
        })
        
        # Get all fine-tuned model versions
        current_model_path = get_current_model_path()
        all_versions = get_all_model_versions()
        
        # Mark current model
        for version in all_versions:
            version['is_current'] = (version['path'] == current_model_path) if current_model_path else False
        
        # Sort by version number (newest first) and add to models list
        # Only include models that have been evaluated (have WER)
        for version in all_versions:
            from src.agent.fine_tuner import FineTuner
            if FineTuner.model_exists(version['path']):
                # Skip models without WER (not evaluated yet)
                if version.get('wer') is None:
                    logger.debug(f"Skipping model {version['version_name']} - no WER (not evaluated)")
                    continue
                
                display_name = f"Fine-tuned Wav2Vec2 v{version['version_num']}"
                display_name += f" (WER: {version['wer']:.2%})"
                
                models.append({
                    "id": f"wav2vec2-finetuned-v{version['version_num']}",
                    "name": version['version_name'],
                    "display_name": display_name,
                    "version_num": version['version_num'],
                    "path": version['path'],
                    "wer": version.get('wer'),
                    "cer": version.get('cer'),
                    "is_available": True,
                    "is_current": version.get('is_current', False),
                    "is_finetuned": True,
                    "created_at": version.get('created_at')
                })
        
        # Determine default (current model or latest)
        default_model = "wav2vec2-base"
        current_model = next((m for m in models if m.get("is_current")), None)
        if current_model:
            default_model = current_model["id"]
        elif all_versions:
            # Use latest version if no current is set
            latest = models[-1] if models else None
            if latest and latest.get("is_finetuned"):
                default_model = latest["id"]
        
        return {
            "models": models,
            "default": default_model
        }
    except Exception as e:
        logger.error(f"Error listing available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/versions")
async def list_model_versions():
    """List all model versions - includes baseline and all fine-tuned versions"""
    try:
        versions = []
        
        # Always include baseline
        baseline_model, _ = get_model_and_agent("wav2vec2-base")
        baseline_info = baseline_model.get_model_info()
        versions.append({
            "version_id": "wav2vec2-base",
            "model_id": "wav2vec2-base",
            "model_name": baseline_info["name"],
            "parameters": baseline_info["parameters"],
            "is_current": False,
            "created_at": None,
            "wer": None,
            "cer": None
        })
        
        # Get all fine-tuned model versions
        current_model_path = get_current_model_path()
        all_versions = get_all_model_versions()
        
        # Mark current model and add to versions list
        # Only include models that have been evaluated (have WER)
        for version in all_versions:
            try:
                from src.agent.fine_tuner import FineTuner
                if FineTuner.model_exists(version['path']):
                    # Skip models without WER (not evaluated yet)
                    if version.get('wer') is None:
                        logger.debug(f"Skipping model {version['version_name']} - no WER (not evaluated)")
                        continue
                    
                    # Load model info
                    try:
                        baseline_model_test = BaselineSTTModel(model_name=f"wav2vec2-finetuned-v{version['version_num']}")
                        model_info = baseline_model_test.get_model_info()
                        parameters = model_info.get("parameters", "unknown")
                    except Exception as e:
                        logger.warning(f"Could not get model info for {version['version_name']}: {e}")
                        parameters = "unknown"
                    
                    versions.append({
                        "version_id": f"wav2vec2-finetuned-v{version['version_num']}",
                        "model_id": version['version_name'],
                        "model_name": f"Fine-tuned Wav2Vec2 v{version['version_num']}",
                        "parameters": parameters,
                        "is_current": (version['path'] == current_model_path) if current_model_path else False,
                        "created_at": version.get('created_at'),
                        "wer": version.get('wer'),
                        "cer": version.get('cer'),
                        "path": version['path']
                    })
            except Exception as e:
                logger.warning(f"Could not load fine-tuned model info for {version['version_name']}: {e}")
        
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
        # Get evaluation results (real WER/CER from current model's evaluation)
        current_model_path = get_current_model_path()
        if current_model_path:
            eval_file = Path(current_model_path) / "evaluation_results.json"
        else:
            eval_file = Path("models/finetuned_wav2vec2/evaluation_results.json")
        
        baseline_wer = 0.36
        baseline_cer = 0.13
        finetuned_wer = 0.36
        finetuned_cer = 0.13
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                # The structure is: baseline_metrics['wer'], fine_tuned_metrics['wer'], etc.
                baseline_metrics = eval_data.get("baseline_metrics", {})
                fine_tuned_metrics = eval_data.get("fine_tuned_metrics", {})
                
                baseline_wer = baseline_metrics.get("wer", 0.36)
                baseline_cer = baseline_metrics.get("cer", 0.13)
                finetuned_wer = fine_tuned_metrics.get("wer", baseline_wer)
                finetuned_cer = fine_tuned_metrics.get("cer", baseline_cer)
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
        
        # Get real WER/CER from current model's evaluation results
        current_model_path = get_current_model_path()
        if current_model_path:
            eval_file = Path(current_model_path) / "evaluation_results.json"
        else:
            eval_file = Path("models/finetuned_wav2vec2/evaluation_results.json")
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                # The structure is: baseline_metrics['wer'], fine_tuned_metrics['wer'], etc.
                baseline_metrics = eval_data.get("baseline_metrics", {})
                fine_tuned_metrics = eval_data.get("fine_tuned_metrics", {})
                
                baseline_wer = baseline_metrics.get("wer", 0.36)
                baseline_cer = baseline_metrics.get("cer", 0.13)
                finetuned_wer = fine_tuned_metrics.get("wer", baseline_wer)
                finetuned_cer = fine_tuned_metrics.get("cer", baseline_cer)
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
    """Get performance trends - returns baseline and current model WER/CER"""
    try:
        # Get real evaluation results from current model
        current_model_path = get_current_model_path()
        if current_model_path:
            eval_file = Path(current_model_path) / "evaluation_results.json"
        else:
            eval_file = Path("models/finetuned_wav2vec2/evaluation_results.json")
        
        baseline_wer = 0.36
        baseline_cer = 0.13
        current_wer = 0.36
        current_cer = 0.13
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                # The structure is: baseline_metrics['wer'], fine_tuned_metrics['wer'], etc.
                baseline_metrics = eval_data.get("baseline_metrics", {})
                fine_tuned_metrics = eval_data.get("fine_tuned_metrics", {})
                
                baseline_wer = baseline_metrics.get("wer", 0.36)
                baseline_cer = baseline_metrics.get("cer", 0.13)
                current_wer = fine_tuned_metrics.get("wer", baseline_wer)
                current_cer = fine_tuned_metrics.get("cer", baseline_cer)
            except Exception as e:
                logger.warning(f"Could not read evaluation results: {e}")
        
        # Return simple two-point trend (baseline vs current model)
        if metric.lower() == "wer":
            trend = [
                {"date": "baseline", "value": baseline_wer},
                {"date": "current", "value": current_wer}
            ]
        elif metric.lower() == "cer":
            trend = [
                {"date": "baseline", "value": baseline_cer},
                {"date": "current", "value": current_cer}
            ]
        else:
            trend = []
        
        return {
            "metric": metric,
            "days": days,
            "trend": trend,
            "baseline": {"wer": baseline_wer, "cer": baseline_cer},
            "finetuned": {"wer": current_wer, "cer": current_cer},
            "current": {"wer": current_wer, "cer": current_cer}
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

