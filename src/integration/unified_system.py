"""
Unified System Architecture - Week 4
Integrates all components into a single cohesive system.
"""

import logging
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
import time
import json
from datetime import datetime

from ..baseline_model import BaselineSTTModel
from ..agent import STTAgent
from ..evaluation.metrics import STTEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedSTTSystem:
    """
    Unified system that integrates all components:
    - Baseline STT Model
    - Agent (Error Detection, Self-Learning, LLM Correction)
    - Adaptive Scheduler & Fine-Tuner
    - Evaluation Metrics
    """
    
    def __init__(
        self,
        model_name: str = "whisper",
        enable_error_detection: bool = True,
        enable_llm_correction: bool = True,
        enable_adaptive_fine_tuning: bool = True,
        error_threshold: float = 0.3,
        scheduler_history_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize unified STT system with all components.
        
        Args:
            model_name: Baseline model name ("whisper")
            enable_error_detection: Enable error detection component
            enable_llm_correction: Enable LLM-based correction
            enable_adaptive_fine_tuning: Enable adaptive fine-tuning (Week 3)
            error_threshold: Error detection threshold
            scheduler_history_path: Path for scheduler history
            config: Additional configuration dictionary
        """
        self.config = config or {}
        self.model_name = model_name
        
        # Component flags
        self.enable_error_detection = enable_error_detection
        self.enable_llm_correction = enable_llm_correction
        self.enable_adaptive_fine_tuning = enable_adaptive_fine_tuning
        
        # Initialize baseline model
        logger.info(f"Initializing baseline model: {model_name}")
        self.baseline_model = BaselineSTTModel(model_name=model_name)
        
        # Initialize agent with all components
        logger.info("Initializing STT Agent with integrated components...")
        self.agent = STTAgent(
            baseline_model=self.baseline_model,
            error_threshold=error_threshold,
            use_llm_correction=enable_llm_correction,
            enable_adaptive_fine_tuning=enable_adaptive_fine_tuning,
            scheduler_history_path=scheduler_history_path
        )
        
        # Initialize evaluator
        self.evaluator = STTEvaluator()
        
        # System statistics
        self.system_stats = {
            'initialization_time': datetime.now().isoformat(),
            'total_transcriptions': 0,
            'total_errors_detected': 0,
            'total_corrections_applied': 0,
            'fine_tuning_events': 0,
            'component_status': self._get_component_status()
        }
        
        logger.info("✅ Unified STT System initialized successfully")
    
    def _get_component_status(self) -> Dict[str, bool]:
        """Get status of all system components."""
        return {
            'baseline_model': self.baseline_model is not None,
            'error_detection': self.enable_error_detection and self.agent.error_detector is not None,
            'self_learning': self.agent.self_learner is not None,
            'llm_correction': self.enable_llm_correction and self.agent.llm_corrector is not None,
            'adaptive_scheduler': self.enable_adaptive_fine_tuning and self.agent.adaptive_scheduler is not None,
            'fine_tuner': self.enable_adaptive_fine_tuning and self.agent.fine_tuner is not None
        }
    
    def transcribe(
        self,
        audio_path: str,
        reference_transcript: Optional[str] = None,
        enable_auto_correction: bool = True
    ) -> Dict:
        """
        Transcribe audio with full system pipeline.
        
        Args:
            audio_path: Path to audio file
            reference_transcript: Ground truth transcript (for evaluation)
            enable_auto_correction: Whether to apply automatic corrections
        
        Returns:
            Dictionary with transcription results and metadata
        """
        start_time = time.time()
        
        # Get audio length for error detection
        import librosa
        audio_length = librosa.get_duration(filename=audio_path)
        
        # Transcribe with agent (includes error detection, correction, learning)
        result = self.agent.transcribe_with_agent(
            audio_path=audio_path,
            audio_length_seconds=audio_length,
            enable_auto_correction=enable_auto_correction
        )
        
        # Evaluate if reference provided
        evaluation_results = None
        if reference_transcript:
            transcript = result.get('transcript', '')
            wer = self.evaluator.calculate_wer(reference_transcript, transcript)
            cer = self.evaluator.calculate_cer(reference_transcript, transcript)
            
            evaluation_results = {
                'wer': wer,
                'cer': cer,
                'reference': reference_transcript,
                'hypothesis': transcript
            }
        
        # Update system statistics
        self.system_stats['total_transcriptions'] += 1
        if result.get('error_detection', {}).get('has_errors', False):
            self.system_stats['total_errors_detected'] += result['error_detection']['error_count']
        if result.get('corrections', {}).get('applied', False):
            self.system_stats['total_corrections_applied'] += result['corrections']['count']
        if result.get('agent_metadata', {}).get('fine_tuning_triggered', False):
            self.system_stats['fine_tuning_events'] += 1
        
        processing_time = time.time() - start_time
        
        # Compile full result
        full_result = {
            **result,
            'system_metadata': {
                'processing_time': processing_time,
                'components_enabled': self._get_component_status(),
                'system_stats': self.system_stats.copy()
            }
        }
        
        if evaluation_results:
            full_result['evaluation'] = evaluation_results
        
        return full_result
    
    def evaluate_batch(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        enable_auto_correction: bool = True
    ) -> Dict:
        """
        Evaluate system on a batch of audio files.
        
        Args:
            audio_files: List of audio file paths
            reference_transcripts: List of reference transcripts
            enable_auto_correction: Whether to apply corrections
        
        Returns:
            Dictionary with batch evaluation results
        """
        assert len(audio_files) == len(reference_transcripts), \
            "Audio files and reference transcripts must have same length"
        
        logger.info(f"Evaluating batch of {len(audio_files)} files...")
        
        results = []
        total_time = 0.0
        
        for i, (audio_path, reference) in enumerate(zip(audio_files, reference_transcripts)):
            logger.info(f"Processing {i+1}/{len(audio_files)}: {Path(audio_path).name}")
            
            result = self.transcribe(
                audio_path=audio_path,
                reference_transcript=reference,
                enable_auto_correction=enable_auto_correction
            )
            
            results.append(result)
            total_time += result['system_metadata']['processing_time']
        
        # Calculate aggregate metrics
        wers = [r['evaluation']['wer'] for r in results if 'evaluation' in r]
        cers = [r['evaluation']['cer'] for r in results if 'evaluation' in r]
        
        batch_results = {
            'num_samples': len(results),
            'average_wer': sum(wers) / len(wers) if wers else None,
            'average_cer': sum(cers) / len(cers) if cers else None,
            'total_processing_time': total_time,
            'average_processing_time': total_time / len(results) if results else 0,
            'total_errors_detected': sum(r.get('error_detection', {}).get('error_count', 0) for r in results),
            'total_corrections_applied': sum(r.get('corrections', {}).get('count', 0) for r in results),
            'fine_tuning_events': sum(1 for r in results if r.get('agent_metadata', {}).get('fine_tuning_triggered', False)),
            'detailed_results': results
        }
        
        return batch_results
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        agent_stats = self.agent.get_agent_stats()
        scheduler_stats = self.agent.get_adaptive_scheduler_stats()
        
        return {
            'component_status': self._get_component_status(),
            'system_statistics': self.system_stats,
            'agent_statistics': agent_stats,
            'scheduler_statistics': scheduler_stats,
            'model_info': self.baseline_model.get_model_info()
        }
    
    def get_component_contributions(self) -> Dict:
        """
        Get contribution of each component to system performance.
        Useful for ablation studies.
        """
        return {
            'baseline_model': {
                'enabled': True,
                'description': 'Core STT transcription'
            },
            'error_detection': {
                'enabled': self.enable_error_detection,
                'description': 'Multi-heuristic error detection'
            },
            'self_learning': {
                'enabled': True,
                'description': 'Error pattern tracking and learning'
            },
            'llm_correction': {
                'enabled': self.enable_llm_correction,
                'description': 'LLM-based intelligent correction'
            },
            'adaptive_scheduler': {
                'enabled': self.enable_adaptive_fine_tuning,
                'description': 'Adaptive fine-tuning scheduling'
            },
            'fine_tuner': {
                'enabled': self.enable_adaptive_fine_tuning,
                'description': 'Automated model fine-tuning'
            }
        }
    
    def save_system_state(self, output_path: str):
        """Save system state for later restoration."""
        state = {
            'config': self.config,
            'system_stats': self.system_stats,
            'component_status': self._get_component_status(),
            'timestamp': datetime.now().isoformat()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"System state saved to {output_path}")
