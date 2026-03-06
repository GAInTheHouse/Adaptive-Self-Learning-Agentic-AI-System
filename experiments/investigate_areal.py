"""
Investigate AReal (RealtimeSTT) model performance on available data.
AReal is a real-time speech-to-text library using Faster_Whisper.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
import logging

from src.evaluation.metrics import STTEvaluator
from src.baseline_model import BaselineSTTModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_audio_files(data_dir: str = "data") -> List[str]:
    """Find all audio files in data directory."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} not found")
        return audio_files
    
    for ext in audio_extensions:
        audio_files.extend(data_path.rglob(f"*{ext}"))
    
    return [str(f) for f in audio_files]


def transcribe_with_areal(audio_file: str) -> Dict:
    """
    Transcribe audio using RealtimeSTT (AReal).
    
    Note: RealtimeSTT is designed for real-time streaming.
    For file transcription, we'll use Faster_Whisper directly.
    """
    try:
        # Try importing RealtimeSTT
        try:
            from realtimestt import RealtimeSTT
            logger.info("Using RealtimeSTT library")
            # RealtimeSTT is for streaming, so we'll use Faster_Whisper directly
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(audio_file, beam_size=5)
            transcript = " ".join([segment.text for segment in segments])
            return {
                'transcript': transcript,
                'language': info.language,
                'language_probability': info.language_probability
            }
        except ImportError:
            logger.info("RealtimeSTT not available, using Faster_Whisper directly")
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(audio_file, beam_size=5)
            transcript = " ".join([segment.text for segment in segments])
            return {
                'transcript': transcript,
                'language': info.language,
                'language_probability': info.language_probability
            }
    except Exception as e:
        logger.error(f"Error transcribing {audio_file}: {e}")
        return {'transcript': '', 'error': str(e)}


def evaluate_areal_performance(
    audio_files: List[str],
    reference_transcripts: Optional[List[str]] = None,
    output_dir: str = "experiments/evaluation_outputs"
) -> Dict:
    """
    Evaluate AReal (RealtimeSTT) performance on available data.
    
    Args:
        audio_files: List of audio file paths
        reference_transcripts: Optional ground truth transcripts
        output_dir: Directory to save results
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating AReal on {len(audio_files)} audio files")
    
    evaluator = STTEvaluator()
    transcripts = []
    latencies = []
    errors = []
    
    for i, audio_file in enumerate(audio_files):
        logger.info(f"Processing {i+1}/{len(audio_files)}: {audio_file}")
        
        start_time = time.time()
        result = transcribe_with_areal(audio_file)
        latency = time.time() - start_time
        
        latencies.append(latency)
        transcripts.append(result.get('transcript', ''))
        
        if 'error' in result:
            errors.append({
                'file': audio_file,
                'error': result['error']
            })
    
    results = {
        'model': 'AReal (RealtimeSTT/Faster_Whisper)',
        'num_files': len(audio_files),
        'num_errors': len(errors),
        'errors': errors,
        'latency': {
            'mean': sum(latencies) / len(latencies) if latencies else 0,
            'min': min(latencies) if latencies else 0,
            'max': max(latencies) if latencies else 0,
            'values': latencies
        },
        'transcripts': transcripts
    }
    
    # Evaluate if reference transcripts available
    if reference_transcripts and len(reference_transcripts) == len(transcripts):
        logger.info("Evaluating with reference transcripts")
        eval_results = evaluator.evaluate_batch(
            references=reference_transcripts,
            hypotheses=transcripts,
            include_verb_rate=True,
            include_domain_rate=True
        )
        results['metrics'] = eval_results
        
        # Save detailed results
        output_path = Path(output_dir) / "areal_evaluation_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        logger.warning("No reference transcripts provided, skipping metric calculation")
    
    return results


def main():
    """Main function to investigate AReal performance."""
    logger.info("=" * 60)
    logger.info("Investigating AReal (RealtimeSTT) Performance")
    logger.info("=" * 60)
    
    # Find audio files
    audio_files = find_audio_files()
    logger.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        logger.error("No audio files found. Please check data directory.")
        return
    
    # Limit to first 10 files for testing
    audio_files = audio_files[:10]
    logger.info(f"Processing {len(audio_files)} files")
    
    # Evaluate performance
    results = evaluate_areal_performance(audio_files)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("AReal Performance Summary")
    logger.info("=" * 60)
    logger.info(f"Files processed: {results['num_files']}")
    logger.info(f"Errors: {results['num_errors']}")
    if results['latency']['mean'] > 0:
        logger.info(f"Mean latency: {results['latency']['mean']:.2f}s")
    
    if 'metrics' in results:
        logger.info(f"WER: {results['metrics']['wer']:.4f}")
        logger.info(f"CER: {results['metrics']['cer']:.4f}")
        if 'verb_error_rate' in results['metrics']:
            logger.info(f"Verb Error Rate: {results['metrics']['verb_error_rate']:.4f}")
        if 'domain_error_rates' in results['metrics']:
            logger.info(f"Domain Error Rates: {results['metrics']['domain_error_rates']}")


if __name__ == "__main__":
    main()
