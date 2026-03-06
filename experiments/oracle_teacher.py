"""
Oracle Teacher Script: Generate synthetic "Gold" transcripts using GPT-4o/Llama 3 API.
This script creates high-quality reference transcripts for audio files without ground truth.
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
import argparse

from src.baseline_model import BaselineSTTModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OracleTeacher:
    """Generate synthetic gold transcripts using LLM APIs."""
    
    def __init__(self, api_type: str = "openai", model: str = "gpt-4o"):
        """
        Initialize Oracle Teacher.
        
        Args:
            api_type: "openai" or "llama" (via Ollama)
            model: Model name (e.g., "gpt-4o", "gpt-4", "llama3")
        """
        self.api_type = api_type
        self.model = model
        self.baseline_model = BaselineSTTModel("whisper")
        
        if api_type == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("Initialized OpenAI client")
            except ImportError:
                logger.error("OpenAI library not installed. Install with: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                raise
        elif api_type == "llama":
            try:
                import ollama
                self.client = ollama
                logger.info("Initialized Ollama client")
            except ImportError:
                logger.error("Ollama library not installed. Install with: pip install ollama")
                raise
    
    def get_baseline_transcript(self, audio_file: str) -> str:
        """Get baseline transcript from Whisper."""
        try:
            transcript = self.baseline_model.transcribe(audio_file)
            return transcript
        except Exception as e:
            logger.error(f"Error getting baseline transcript: {e}")
            return ""
    
    def refine_with_llm(self, baseline_transcript: str, audio_file: str) -> str:
        """
        Refine baseline transcript using LLM to create gold-standard transcript.
        
        Args:
            baseline_transcript: Initial transcript from Whisper
            audio_file: Path to audio file (for context)
        
        Returns:
            Refined gold transcript
        """
        prompt = f"""You are an expert transcriptionist. Please refine the following speech-to-text transcript to create a high-quality, accurate transcription.

Consider:
1. Fix any obvious speech recognition errors
2. Add proper punctuation and capitalization
3. Correct grammar while preserving the speaker's intended meaning
4. Maintain natural speech patterns (don't over-formalize)
5. Preserve technical terms, names, and domain-specific vocabulary

Original transcript:
{baseline_transcript}

Please provide ONLY the refined transcript without any additional commentary:"""

        if self.api_type == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert transcriptionist who creates accurate, polished transcripts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=2000
                )
                refined = response.choices[0].message.content.strip()
                return refined
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {e}")
                return baseline_transcript
        
        elif self.api_type == "llama":
            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=f"System: You are an expert transcriptionist.\n\nUser: {prompt}\n\nAssistant:",
                    options={
                        "temperature": 0.3,
                        "num_predict": 2000
                    }
                )
                refined = response.get('response', baseline_transcript).strip()
                return refined
            except Exception as e:
                logger.error(f"Error calling Ollama API: {e}")
                return baseline_transcript
    
    def generate_gold_transcript(self, audio_file: str, use_baseline: bool = True) -> Dict:
        """
        Generate gold transcript for audio file.
        
        Args:
            audio_file: Path to audio file
            use_baseline: Whether to use baseline transcript as starting point
        
        Returns:
            Dictionary with gold transcript and metadata
        """
        logger.info(f"Generating gold transcript for {audio_file}")
        
        # Get baseline transcript
        baseline_transcript = ""
        if use_baseline:
            baseline_transcript = self.get_baseline_transcript(audio_file)
            logger.info(f"Baseline transcript: {baseline_transcript[:100]}...")
        
        # Refine with LLM
        start_time = time.time()
        gold_transcript = self.refine_with_llm(baseline_transcript, audio_file)
        refinement_time = time.time() - start_time
        
        return {
            'audio_file': audio_file,
            'baseline_transcript': baseline_transcript,
            'gold_transcript': gold_transcript,
            'refinement_time': refinement_time,
            'model': self.model,
            'api_type': self.api_type
        }


def process_audio_files(
    audio_files: List[str],
    output_file: str,
    api_type: str = "openai",
    model: str = "gpt-4o",
    use_baseline: bool = True
) -> List[Dict]:
    """
    Process multiple audio files to generate gold transcripts.
    
    Args:
        audio_files: List of audio file paths
        output_file: Path to save results JSON
        api_type: "openai" or "llama"
        model: Model name
        use_baseline: Whether to use baseline transcript
    
    Returns:
        List of results dictionaries
    """
    oracle = OracleTeacher(api_type=api_type, model=model)
    results = []
    
    for i, audio_file in enumerate(audio_files):
        logger.info(f"Processing {i+1}/{len(audio_files)}: {audio_file}")
        
        try:
            result = oracle.generate_gold_transcript(audio_file, use_baseline=use_baseline)
            results.append(result)
            
            # Save intermediate results
            if (i + 1) % 5 == 0:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved intermediate results ({i+1}/{len(audio_files)})")
            
            # Rate limiting for API calls
            time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            results.append({
                'audio_file': audio_file,
                'error': str(e)
            })
    
    # Save final results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Final results saved to {output_path}")
    
    return results


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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Oracle Teacher: Generate gold transcripts using LLM")
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="data",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/evaluation_outputs/oracle_gold_transcripts.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--api-type",
        type=str,
        choices=["openai", "llama"],
        default="openai",
        help="API type: openai or llama"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name (e.g., gpt-4o, gpt-4, llama3)"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Don't use baseline transcript, generate from scratch"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Oracle Teacher: Generating Gold Transcripts")
    logger.info("=" * 60)
    logger.info(f"API Type: {args.api_type}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Using baseline: {not args.no_baseline}")
    
    # Check API key
    if args.api_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.info("Set it with: export OPENAI_API_KEY='your-key'")
        return
    
    # Find audio files
    audio_files = find_audio_files(args.audio_dir)
    logger.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        logger.error("No audio files found")
        return
    
    # Limit files if specified
    if args.limit:
        audio_files = audio_files[:args.limit]
        logger.info(f"Processing {len(audio_files)} files (limited)")
    
    # Process files
    results = process_audio_files(
        audio_files=audio_files,
        output_file=args.output,
        api_type=args.api_type,
        model=args.model,
        use_baseline=not args.no_baseline
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Files processed: {len(results)}")
    successful = sum(1 for r in results if 'gold_transcript' in r)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(results) - successful}")
    
    if successful > 0:
        avg_time = sum(r.get('refinement_time', 0) for r in results if 'refinement_time' in r) / successful
        logger.info(f"Average refinement time: {avg_time:.2f}s")
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
