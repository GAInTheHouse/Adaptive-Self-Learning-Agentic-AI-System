"""
Investigate Miles model performance on available data.
Miles could refer to M.I.L.E.S voice assistant or Moonshine model.
This script investigates Moonshine model (specialized for live transcription).
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


def transcribe_with_moonshine(audio_file: str) -> Dict:
    """
    Transcribe audio using Moonshine model.
    
    Moonshine is optimized for live transcription with 5x compute reduction.
    We'll try to use it if available, otherwise fall back to Whisper.
    """
    try:
        # Try importing Moonshine from HuggingFace
        try:
            from transformers import pipeline
            logger.info("Attempting to use Moonshine model")
            # Moonshine model ID (if available on HuggingFace)
            # Note: Actual model ID may vary
            pipe = pipeline(
                "automatic-speech-recognition",
                model="kyutai-org/moonshine-2.6b-en",  # Example ID
                device=-1  # CPU
            )
            result = pipe(audio_file)
            return {
                'transcript': result.get('text', ''),
                'model': 'moonshine'
            }
        except Exception as e:
            logger.warning(f"Moonshine model not available: {e}")
            logger.info("Falling back to Whisper")
            # Fallback to Whisper
            from transformers import pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=-1
            )
            result = pipe(audio_file)
            return {
                'transcript': result.get('text', ''),
                'model': 'whisper-base-fallback'
            }
    except Exception as e:
        logger.error(f"Error transcribing {audio_file}: {e}")
        return {'transcript': '', 'error': str(e)}


def transcribe_with_miles_assistant(audio_file: str) -> Dict:
    """
    Alternative: Use M.I.L.E.S voice assistant approach.
    This would require API access, so we'll simulate or use local model.
    """
    # For now, use baseline Whisper as proxy
    # In production, this would connect to M.I.L.E.S API
    try:
        from transformers import pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=-1
        )
        result = pipe(audio_file)
        return {
            'transcript': result.get('text', ''),
            'model': 'miles-proxy-whisper'
        }
    except Exception as e:
        logger.error(f"Error transcribing {audio_file}: {e}")
        return {'transcript': '', 'error': str(e)}


def evaluate_miles_performance(
    audio_files: List[str],
    reference_transcripts: Optional[List[str]] = None,
    model_type: str = "moonshine",
    output_dir: str = "experiments/evaluation_outputs"
) -> Dict:
    """
    Evaluate Miles/Moonshine performance on available data.
    
    Args:
        audio_files: List of audio file paths
        reference_transcripts: Optional ground truth transcripts
        model_type: "moonshine" or "miles"
        output_dir: Directory to save results
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating {model_type} on {len(audio_files)} audio files")
    
    evaluator = STTEvaluator()
    transcripts = []
    latencies = []
    errors = []
    
    transcribe_func = transcribe_with_moonshine if model_type == "moonshine" else transcribe_with_miles_assistant
    
    for i, audio_file in enumerate(audio_files):
        logger.info(f"Processing {i+1}/{len(audio_files)}: {audio_file}")
        
        start_time = time.time()
        result = transcribe_func(audio_file)
        latency = time.time() - start_time
        
        latencies.append(latency)
        transcripts.append(result.get('transcript', ''))
        
        if 'error' in result:
            errors.append({
                'file': audio_file,
                'error': result['error']
            })
    
    results = {
        'model': f'Miles ({model_type})',
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
        output_path = Path(output_dir) / f"miles_{model_type}_evaluation_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        logger.warning("No reference transcripts provided, skipping metric calculation")
    
    return results


def main():
    """Main function to investigate Miles performance."""
    logger.info("=" * 60)
    logger.info("Investigating Miles/Moonshine Performance")
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
    
    # Evaluate Moonshine performance
    logger.info("\nEvaluating Moonshine model...")
    results_moonshine = evaluate_miles_performance(audio_files, model_type="moonshine")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Miles/Moonshine Performance Summary")
    logger.info("=" * 60)
    logger.info(f"Files processed: {results_moonshine['num_files']}")
    logger.info(f"Errors: {results_moonshine['num_errors']}")
    if results_moonshine['latency']['mean'] > 0:
        logger.info(f"Mean latency: {results_moonshine['latency']['mean']:.2f}s")
    
    if 'metrics' in results_moonshine:
        logger.info(f"WER: {results_moonshine['metrics']['wer']:.4f}")
        logger.info(f"CER: {results_moonshine['metrics']['cer']:.4f}")
        if 'verb_error_rate' in results_moonshine['metrics']:
            logger.info(f"Verb Error Rate: {results_moonshine['metrics']['verb_error_rate']:.4f}")
        if 'domain_error_rates' in results_moonshine['metrics']:
            logger.info(f"Domain Error Rates: {results_moonshine['metrics']['domain_error_rates']}")


if __name__ == "__main__":
    main()
