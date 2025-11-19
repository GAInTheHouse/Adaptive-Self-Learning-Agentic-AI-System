"""
Kavya Evaluation Framework - Week 1
Comprehensive evaluation framework for STT baseline models
Generates all relevant outputs for Week 1 deliverables
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, Dataset

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.baseline_model import BaselineSTTModel
from src.evaluation.metrics import STTEvaluator
from src.benchmark import BaselineBenchmark
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationFramework:
    """
    Comprehensive evaluation framework for STT models.
    Handles dataset evaluation, error analysis, and report generation.
    """
    
    def __init__(self, model_name: str = "whisper", output_dir: str = "experiments/evaluation_outputs"):
        """
        Initialize evaluation framework.
        
        Args:
            model_name: Name of model to evaluate
            output_dir: Directory to save evaluation outputs
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info(f"Loading {model_name} model...")
        self.model = BaselineSTTModel(model_name=model_name)
        self.evaluator = STTEvaluator()
        self.benchmark = BaselineBenchmark(model_name=model_name)
        
        # Results storage
        self.evaluation_results = {
            "model_info": self.model.get_model_info(),
            "evaluation_date": datetime.now().isoformat(),
            "datasets": {},
            "summary": {}
        }
    
    def evaluate_dataset(
        self,
        dataset_path: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        audio_column: str = "audio",
        text_column: str = "text"
    ) -> Dict:
        """
        Evaluate model on a dataset split.
        
        Args:
            dataset_path: Path to dataset
            split: Dataset split to evaluate (train/dev/test)
            max_samples: Maximum number of samples to evaluate (None for all)
            audio_column: Name of audio column in dataset
            text_column: Name of text column in dataset
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Loading dataset from {dataset_path} (split: {split})...")
        
        try:
            # Load dataset
            dataset = load_from_disk(dataset_path)
            
            # Handle DatasetDict vs Dataset
            if isinstance(dataset, dict):
                if split not in dataset:
                    logger.warning(f"Split '{split}' not found. Available: {list(dataset.keys())}")
                    split = list(dataset.keys())[0]
                dataset = dataset[split]
            
            # Limit samples if specified
            if max_samples and len(dataset) > max_samples:
                logger.info(f"Limiting evaluation to {max_samples} samples")
                dataset = dataset.select(range(max_samples))
            
            logger.info(f"Evaluating on {len(dataset)} samples...")
            
            # Run evaluation
            references = []
            hypotheses = []
            inference_times = []
            errors = []
            
            for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
                try:
                    # Get reference text
                    reference = sample.get(text_column, "")
                    if not reference:
                        logger.warning(f"Sample {idx} missing text column")
                        continue
                    
                    # Get audio
                    audio_data = sample.get(audio_column)
                    if audio_data is None:
                        logger.warning(f"Sample {idx} missing audio column")
                        continue
                    
                    # Save audio temporarily if it's an Audio object
                    import tempfile
                    import soundfile as sf
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        if hasattr(audio_data, 'array'):
                            # HuggingFace Audio object
                            sf.write(tmp.name, audio_data['array'], audio_data['sampling_rate'])
                        elif isinstance(audio_data, dict):
                            sf.write(tmp.name, audio_data['array'], audio_data['sampling_rate'])
                        else:
                            # Assume it's a path
                            tmp.name = audio_data
                        
                        # Transcribe
                        start_time = time.time()
                        result = self.model.transcribe(tmp.name)
                        inference_time = time.time() - start_time
                        
                        # Cleanup
                        import os
                        if os.path.exists(tmp.name):
                            os.remove(tmp.name)
                    
                    hypothesis = result['transcript']
                    
                    references.append(reference)
                    hypotheses.append(hypothesis)
                    inference_times.append(inference_time)
                    
                except Exception as e:
                    logger.error(f"Error processing sample {idx}: {e}")
                    errors.append({"sample_idx": idx, "error": str(e)})
                    continue
            
            # Calculate metrics
            if len(references) == 0:
                logger.error("No valid samples processed")
                return {}
            
            metrics = self.evaluator.evaluate_batch(references, hypotheses)
            
            # Add inference statistics
            metrics.update({
                "num_samples": len(references),
                "num_errors": len(errors),
                "inference_stats": {
                    "mean_time_seconds": np.mean(inference_times),
                    "std_time_seconds": np.std(inference_times),
                    "min_time_seconds": np.min(inference_times),
                    "max_time_seconds": np.max(inference_times),
                    "total_time_seconds": sum(inference_times)
                }
            })
            
            logger.info(f"Evaluation complete: WER={metrics['wer']:.4f}, CER={metrics['cer']:.4f}")
            
            return {
                "metrics": metrics,
                "references": references,
                "hypotheses": hypotheses,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error evaluating dataset: {e}")
            return {}
    
    def perform_error_analysis(
        self,
        references: List[str],
        hypotheses: List[str],
        top_n: int = 20
    ) -> Dict:
        """
        Perform detailed error analysis on predictions.
        
        Args:
            references: List of reference transcriptions
            hypotheses: List of predicted transcriptions
            top_n: Number of worst errors to analyze
        
        Returns:
            Dictionary with error analysis results
        """
        logger.info("Performing error analysis...")
        
        # Calculate per-sample errors
        sample_errors = []
        for ref, hyp in zip(references, hypotheses):
            wer = self.evaluator.calculate_wer(ref, hyp)
            cer = self.evaluator.calculate_cer(ref, hyp)
            sample_errors.append({
                "reference": ref,
                "hypothesis": hyp,
                "wer": wer,
                "cer": cer,
                "ref_length": len(ref.split()),
                "hyp_length": len(hyp.split()),
                "length_diff": len(hyp.split()) - len(ref.split())
            })
        
        # Sort by WER (worst first)
        sample_errors.sort(key=lambda x: x['wer'], reverse=True)
        
        # Analyze error patterns
        error_analysis = {
            "worst_errors": sample_errors[:top_n],
            "error_statistics": {
                "high_wer_count": sum(1 for e in sample_errors if e['wer'] > 0.5),
                "medium_wer_count": sum(1 for e in sample_errors if 0.2 < e['wer'] <= 0.5),
                "low_wer_count": sum(1 for e in sample_errors if e['wer'] <= 0.2),
                "avg_ref_length": np.mean([e['ref_length'] for e in sample_errors]),
                "avg_hyp_length": np.mean([e['hyp_length'] for e in sample_errors]),
                "avg_length_diff": np.mean([e['length_diff'] for e in sample_errors])
            }
        }
        
        return error_analysis
    
    def evaluate_multiple_datasets(
        self,
        dataset_configs: List[Dict],
        max_samples_per_dataset: Optional[int] = 100
    ) -> Dict:
        """
        Evaluate model on multiple datasets.
        
        Args:
            dataset_configs: List of dataset configuration dicts
            max_samples_per_dataset: Max samples to evaluate per dataset
        
        Returns:
            Dictionary with results for all datasets
        """
        logger.info(f"Evaluating on {len(dataset_configs)} datasets...")
        
        all_results = {}
        
        for config in dataset_configs:
            dataset_name = config.get("name", "unknown")
            dataset_path = config.get("path")
            split = config.get("split", "test")
            
            if not dataset_path or not Path(dataset_path).exists():
                logger.warning(f"Dataset {dataset_name} not found at {dataset_path}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {dataset_name}")
            logger.info(f"{'='*60}")
            
            results = self.evaluate_dataset(
                dataset_path=dataset_path,
                split=split,
                max_samples=max_samples_per_dataset,
                audio_column=config.get("audio_column", "audio"),
                text_column=config.get("text_column", "text")
            )
            
            if results:
                # Perform error analysis
                if "references" in results and "hypotheses" in results:
                    error_analysis = self.perform_error_analysis(
                        results["references"],
                        results["hypotheses"]
                    )
                    results["error_analysis"] = error_analysis
                
                all_results[dataset_name] = results
                self.evaluation_results["datasets"][dataset_name] = {
                    "metrics": results["metrics"],
                    "error_analysis_summary": results.get("error_analysis", {}).get("error_statistics", {})
                }
        
        return all_results
    
    def generate_summary_report(self, all_results: Dict) -> Dict:
        """
        Generate summary report across all datasets.
        
        Args:
            all_results: Results from evaluate_multiple_datasets
        
        Returns:
            Summary report dictionary
        """
        logger.info("Generating summary report...")
        
        # Aggregate metrics
        all_wers = []
        all_cers = []
        total_samples = 0
        
        for dataset_name, results in all_results.items():
            if "metrics" in results:
                metrics = results["metrics"]
                all_wers.append(metrics["wer"])
                all_cers.append(metrics["cer"])
                total_samples += metrics.get("num_samples", 0)
        
        summary = {
            "model": self.model_name,
            "model_info": self.model.get_model_info(),
            "evaluation_date": datetime.now().isoformat(),
            "total_datasets": len(all_results),
            "total_samples_evaluated": total_samples,
            "overall_metrics": {
                "mean_wer": np.mean(all_wers) if all_wers else None,
                "std_wer": np.std(all_wers) if all_wers else None,
                "mean_cer": np.mean(all_cers) if all_cers else None,
                "std_cer": np.std(all_cers) if all_cers else None,
                "best_wer": np.min(all_wers) if all_wers else None,
                "worst_wer": np.max(all_wers) if all_wers else None
            },
            "per_dataset_metrics": {
                name: {
                    "wer": results["metrics"]["wer"],
                    "cer": results["metrics"]["cer"],
                    "num_samples": results["metrics"]["num_samples"]
                }
                for name, results in all_results.items()
                if "metrics" in results
            }
        }
        
        self.evaluation_results["summary"] = summary
        return summary
    
    def save_evaluation_report(
        self,
        all_results: Dict,
        summary: Dict,
        filename: str = "evaluation_report.json"
    ):
        """
        Save comprehensive evaluation report.
        
        Args:
            all_results: Detailed results from all datasets
            summary: Summary report
            filename: Output filename
        """
        report = {
            "summary": summary,
            "detailed_results": all_results,
            "full_evaluation": self.evaluation_results
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Evaluation report saved to {output_path}")
        
        # Also save summary as separate file
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Summary saved to {summary_path}")
    
    def print_evaluation_summary(self, summary: Dict):
        """
        Print formatted evaluation summary.
        
        Args:
            summary: Summary report dictionary
        """
        print("\n" + "="*70)
        print("EVALUATION FRAMEWORK SUMMARY REPORT")
        print("="*70)
        
        print(f"\nModel: {summary['model']}")
        print(f"Evaluation Date: {summary['evaluation_date']}")
        print(f"Total Datasets: {summary['total_datasets']}")
        print(f"Total Samples Evaluated: {summary['total_samples_evaluated']}")
        
        if summary['overall_metrics']['mean_wer'] is not None:
            print("\n" + "-"*70)
            print("OVERALL METRICS")
            print("-"*70)
            print(f"Mean WER: {summary['overall_metrics']['mean_wer']:.4f} ± {summary['overall_metrics']['std_wer']:.4f}")
            print(f"Mean CER: {summary['overall_metrics']['mean_cer']:.4f} ± {summary['overall_metrics']['std_cer']:.4f}")
            print(f"Best WER: {summary['overall_metrics']['best_wer']:.4f}")
            print(f"Worst WER: {summary['overall_metrics']['worst_wer']:.4f}")
        
        if summary['per_dataset_metrics']:
            print("\n" + "-"*70)
            print("PER-DATASET METRICS")
            print("-"*70)
            for dataset_name, metrics in summary['per_dataset_metrics'].items():
                print(f"\n{dataset_name}:")
                print(f"  WER: {metrics['wer']:.4f}")
                print(f"  CER: {metrics['cer']:.4f}")
                print(f"  Samples: {metrics['num_samples']}")
        
        print("\n" + "="*70)


def main():
    """
    Main evaluation framework execution.
    """
    print("="*70)
    print("KAVYA EVALUATION FRAMEWORK - WEEK 1")
    print("="*70)
    
    # Initialize framework
    framework = EvaluationFramework(
        model_name="whisper",
        output_dir="experiments/evaluation_outputs"
    )
    
    # Define dataset configurations
    # Note: Update these paths based on your actual dataset locations
    dataset_configs = [
        {
            "name": "test_dataset",
            "path": "data/evaluation/test_dataset",
            "split": "test",
            "audio_column": "audio",
            "text_column": "text"
        },
        {
            "name": "common_voice_test",
            "path": "data/evaluation/common_voice_accents",
            "split": "test",
            "audio_column": "audio",
            "text_column": "sentence"
        },
        {
            "name": "librispeech_test",
            "path": "data/evaluation/librispeech_clean",
            "split": "test",
            "audio_column": "audio",
            "text_column": "text"
        }
    ]
    
    # Filter to only existing datasets
    existing_configs = [
        config for config in dataset_configs
        if Path(config["path"]).exists()
    ]
    
    if not existing_configs:
        logger.warning("No evaluation datasets found. Using test audio for demonstration...")
        # Fallback: evaluate on test audio files
        test_audio_files = [
            "test_audio/addf8-Alaw-GW.wav",
            "test_audio/test_1.wav"
        ]
        
        # Create a simple evaluation with test files
        print("\nRunning benchmark on test audio files...")
        benchmark_report = framework.benchmark.generate_report(
            [f for f in test_audio_files if Path(f).exists()]
        )
        
        # Save benchmark report
        framework.benchmark.save_report(
            benchmark_report,
            str(framework.output_dir / "benchmark_report.json")
        )
        
        print("\n✅ Benchmark complete. Check evaluation_outputs/ for results.")
        return
    
    # Evaluate on all datasets
    all_results = framework.evaluate_multiple_datasets(
        dataset_configs=existing_configs,
        max_samples_per_dataset=100  # Adjust as needed
    )
    
    # Generate summary
    summary = framework.generate_summary_report(all_results)
    
    # Print summary
    framework.print_evaluation_summary(summary)
    
    # Save reports
    framework.save_evaluation_report(all_results, summary)
    
    print("\n✅ Evaluation framework complete!")
    print(f"📁 All outputs saved to: {framework.output_dir}")
    print("\nGenerated files:")
    print("  - evaluation_report.json (full detailed report)")
    print("  - evaluation_summary.json (summary metrics)")
    print("  - benchmark_report.json (performance benchmarks)")


if __name__ == "__main__":
    main()
