"""
Main Agent Evaluation Script - Week 2

Comprehensive evaluation of agent correction accuracy, false positives,
ablation testing, and latency benchmarking.

This script:
1. Creates a test dataset with ground truth transcripts
2. Runs agent evaluation on the dataset
3. Generates comprehensive reports

Usage:
    python -m experiments.run_agent_evaluation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import Optional, List, Dict
import librosa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_evaluation_dataset() -> Dict:
    """
    Create a test evaluation dataset with ground truth transcripts.
    
    This is based on the create_test_evaluation.py pattern but specifically
    designed for agent evaluation with multiple reference samples.
    
    Returns:
        Dictionary with 'audio_paths' and 'reference_transcripts'
    """
    
    logger.info("Creating test evaluation dataset...")
    
    # Define test samples with ground truth transcripts
    # UPDATE THESE with your actual ground truth transcriptions!
    test_samples = [
        {
            "audio_path": "data/test_audio/addf8-Alaw-GW.wav",
            "reference": "add the sum to the product of these three",
            "id": "test_001"
        },
        # Add more samples as needed:
        # {
        #     "audio_path": "data/test_audio/another_file.wav",
        #     "reference": "your ground truth transcription",
        #     "id": "test_002"
        # }
    ]
    
    audio_paths = []
    reference_transcripts = []
    
    for sample in test_samples:
        audio_path = Path(sample["audio_path"])
        
        # Check if file exists
        if not audio_path.exists():
            logger.warning(f"⚠️  Audio file not found: {audio_path}")
            continue
        
        try:
            # Verify audio can be loaded
            audio, sr = librosa.load(str(audio_path), sr=16000)
            logger.info(f"✅ Loaded {audio_path} ({len(audio)/sr:.2f}s audio)")
            
            audio_paths.append(str(audio_path))
            reference_transcripts.append(sample["reference"])
            
        except Exception as e:
            logger.error(f"❌ Error loading {audio_path}: {e}")
            continue
    
    if not audio_paths:
        logger.error("❌ No valid audio files found!")
        return None
    
    logger.info(f"✅ Created test dataset with {len(audio_paths)} samples")
    
    return {
        "audio_paths": audio_paths,
        "reference_transcripts": reference_transcripts
    }


def run_evaluation(audio_paths: List[str], reference_transcripts: List[str]):
    """Run complete agent evaluation"""
    
    print("\n" + "="*70)
    print("AGENT EVALUATION FRAMEWORK - WEEK 2")
    print("="*70 + "\n")
    
    # Import after path setup
    from src.baseline_model import BaselineSTTModel
    from src.agent import STTAgent
    from src.agent_evaluation import (
        AgentEvaluator,
        AblationTester,
        AgentBenchmark,
        FalsePositiveDetector
    )
    
    # Initialize models
    logger.info("Initializing models...")
    baseline_model = BaselineSTTModel(model_name="whisper")
    agent = STTAgent(baseline_model=baseline_model)
    
    output_dir = Path("experiments/evaluation_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # 1. CORRECTION ACCURACY EVALUATION
    # ============================================================
    
    logger.info("\n" + "="*70)
    logger.info("1. CORRECTION ACCURACY EVALUATION")
    logger.info("="*70)
    logger.info(f"✅ Running WITH {len(reference_transcripts)} reference transcripts")
    
    evaluator = AgentEvaluator(
        agent=agent,
        baseline_model=baseline_model,
        output_dir=str(output_dir)
    )
    
    batch_results = evaluator.evaluate_batch(
        audio_paths=audio_paths,
        reference_transcripts=reference_transcripts,
        enable_correction=True
    )
    
    evaluator.print_summary()
    evaluator.save_results(filename="agent_evaluator_results.json")
    
    # ============================================================
    # 2. ABLATION TESTING
    # ============================================================
    
    logger.info("\n" + "="*70)
    logger.info("2. ABLATION TESTING")
    logger.info("="*70)
    
    ablation_tester = AblationTester(
        agent=agent,
        baseline_model=baseline_model,
        output_dir=str(output_dir)
    )
    
    ablation_tester.run_full_ablation(audio_paths=audio_paths)
    ablation_tester.print_summary()
    ablation_tester.save_ablation_report(filename="ablation_study_results.json")
    
    # ============================================================
    # 3. LATENCY BENCHMARKING
    # ============================================================
    
    logger.info("\n" + "="*70)
    logger.info("3. LATENCY BENCHMARKING")
    logger.info("="*70)
    
    benchmarker = AgentBenchmark(
        agent=agent,
        baseline_model=baseline_model,
        output_dir=str(output_dir)
    )
    
    benchmarker.benchmark_batch(audio_paths=audio_paths, verbose=True)
    benchmarker.print_summary()
    benchmarker.save_benchmark_report(filename="agent_benchmark_results.json")
    
    # ============================================================
    # 4. FALSE POSITIVE DETECTION
    # ============================================================
    
    logger.info("\n" + "="*70)
    logger.info("4. FALSE POSITIVE DETECTION")
    logger.info("="*70)
    
    fp_detector = FalsePositiveDetector(output_dir=str(output_dir))
    fp_count = 0
    
    for idx, (audio_path, ref) in enumerate(zip(audio_paths, reference_transcripts)):
        try:
            agent_result = agent.transcribe_with_agent(audio_path)
            
            # Detect false positives for each error detected
            for error in agent_result['error_detection'].get('errors', []):
                fp = fp_detector.detect_false_positive(
                    original_transcript=agent_result['original_transcript'],
                    corrected_transcript=agent_result['transcript'],
                    reference_transcript=ref,
                    error_type=error.get('type', 'unknown'),
                    error_confidence=error.get('confidence', 0.0),
                    audio_path=str(audio_path)
                )
                if fp:
                    fp_count += 1
            
            logger.info(f"  Processed {idx+1}/{len(audio_paths)} samples for false positive detection")
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            continue
    
    fp_detector.print_summary()
    fp_detector.save_analysis(filename="false_positives_analysis.json")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print("\n" + "="*70)
    print("✅ AGENT EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📁 All results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. agent_evaluator_results.json        - Correction accuracy metrics")
    print("  2. ablation_study_results.json         - Ablation test breakdown")
    print("  3. agent_benchmark_results.json        - Latency performance")
    print("  4. false_positives_analysis.json       - False positive detection")
    print("\nNext steps:")
    print("  - Review the JSON files in experiments/evaluation_outputs/")
    print("  - Add more test samples to expand evaluation")
    print("  - Compare metrics with baseline model")
    print("\n" + "="*70 + "\n")


def main():
    """Main execution"""
    
    logger.info("\n" + "="*70)
    logger.info("AGENT EVALUATION PIPELINE")
    logger.info("="*70)
    
    # Step 1: Create test evaluation dataset
    logger.info("\n📊 Step 1: Creating test evaluation dataset...")
    dataset = create_test_evaluation_dataset()
    
    if dataset is None or not dataset.get("audio_paths"):
        logger.error("❌ Failed to create test dataset")
        return
    
    audio_paths = dataset["audio_paths"]
    reference_transcripts = dataset["reference_transcripts"]
    
    logger.info(f"\n✅ Ready to evaluate:")
    logger.info(f"   - {len(audio_paths)} audio file(s)")
    logger.info(f"   - {len(reference_transcripts)} reference transcript(s)")
    
    # Step 2: Run evaluation
    logger.info("\n📈 Step 2: Running agent evaluation...")
    run_evaluation(
        audio_paths=audio_paths,
        reference_transcripts=reference_transcripts
    )
    
    logger.info("\n🎉 Evaluation pipeline complete!")


if __name__ == "__main__":
    main()
