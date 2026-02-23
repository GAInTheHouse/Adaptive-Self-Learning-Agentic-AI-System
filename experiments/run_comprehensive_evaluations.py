#!/usr/bin/env python3
"""
Comprehensive Evaluation Script
Runs actual evaluations to get measured numbers for the report.
"""

import sys
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.baseline_model import BaselineSTTModel
from src.integration import UnifiedSTTSystem, StatisticalAnalyzer, AblationStudy
from src.evaluation.metrics import STTEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluator that runs all evaluation types."""
    
    def __init__(self, output_dir: str = "experiments/evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def find_test_audio_files(self) -> List[str]:
        """Find available test audio files."""
        audio_files = []
        
        # Check test_audio directory
        test_audio_dir = Path("data/test_audio")
        if test_audio_dir.exists():
            audio_files.extend(list(test_audio_dir.glob("*.wav")))
        
        # Check recordings_for_test directory
        recordings_dir = Path("data/recordings_for_test")
        if recordings_dir.exists():
            audio_files.extend(list(recordings_dir.glob("*.wav"))[:5])  # Limit to 5 for speed
        
        return [str(f) for f in audio_files if f.exists()]
    
    def create_reference_transcripts(self, audio_files: List[str]) -> List[str]:
        """Create placeholder reference transcripts for evaluation."""
        # In real scenario, these would be actual ground truth transcripts
        # For now, we'll use baseline transcription as reference
        logger.info("Creating reference transcripts from baseline model...")
        baseline = BaselineSTTModel(model_name="whisper")
        references = []
        
        for audio_file in audio_files:
            try:
                result = baseline.transcribe(audio_file)
                references.append(result.get('transcript', ''))
                logger.info(f"  Created reference for {Path(audio_file).name}")
            except Exception as e:
                logger.warning(f"  Failed to transcribe {audio_file}: {e}")
                references.append("")  # Empty reference
        
        return references
    
    def evaluate_baseline(self, audio_files: List[str], references: List[str]) -> Dict:
        """Evaluate baseline model."""
        logger.info("="*70)
        logger.info("EVALUATING BASELINE MODEL")
        logger.info("="*70)
        
        baseline = BaselineSTTModel(model_name="whisper")
        evaluator = STTEvaluator()
        
        wers = []
        cers = []
        latencies = []
        
        for i, (audio_file, reference) in enumerate(zip(audio_files, references)):
            if not reference.strip():
                continue
                
            logger.info(f"Processing {i+1}/{len(audio_files)}: {Path(audio_file).name}")
            
            start_time = time.time()
            result = baseline.transcribe(audio_file)
            latency = time.time() - start_time
            
            transcript = result.get('transcript', '')
            if transcript:
                wer = evaluator.calculate_wer(reference, transcript)
                cer = evaluator.calculate_cer(reference, transcript)
                wers.append(wer)
                cers.append(cer)
                latencies.append(latency)
        
        results = {
            'num_samples': len(wers),
            'wer': {
                'mean': np.mean(wers) if wers else None,
                'std': np.std(wers) if wers else None,
                'min': np.min(wers) if wers else None,
                'max': np.max(wers) if wers else None,
                'values': wers
            },
            'cer': {
                'mean': np.mean(cers) if cers else None,
                'std': np.std(cers) if cers else None,
                'values': cers
            },
            'latency': {
                'mean': np.mean(latencies) if latencies else None,
                'std': np.std(latencies) if latencies else None,
                'values': latencies
            }
        }
        
        logger.info(f"Baseline WER: {results['wer']['mean']:.4f} ({results['wer']['mean']*100:.2f}%)")
        logger.info(f"Baseline CER: {results['cer']['mean']:.4f} ({results['cer']['mean']*100:.2f}%)")
        logger.info(f"Mean Latency: {results['latency']['mean']:.2f}s")
        
        return results
    
    def evaluate_full_system(self, audio_files: List[str], references: List[str]) -> Dict:
        """Evaluate full system."""
        logger.info("="*70)
        logger.info("EVALUATING FULL SYSTEM")
        logger.info("="*70)
        
        system = UnifiedSTTSystem(
            model_name="whisper",
            enable_error_detection=True,
            enable_llm_correction=True,
            enable_adaptive_fine_tuning=False  # Disable for faster evaluation
        )
        
        wers = []
        cers = []
        latencies = []
        errors_detected = []
        corrections_applied = []
        
        for i, (audio_file, reference) in enumerate(zip(audio_files, references)):
            if not reference.strip():
                continue
                
            logger.info(f"Processing {i+1}/{len(audio_files)}: {Path(audio_file).name}")
            
            start_time = time.time()
            result = system.transcribe(audio_file, reference_transcript=reference)
            latency = time.time() - start_time
            
            if 'evaluation' in result:
                wers.append(result['evaluation']['wer'])
                cers.append(result['evaluation']['cer'])
                latencies.append(latency)
                
                # Track error detection and correction
                if result.get('error_detection', {}).get('has_errors', False):
                    errors_detected.append(result['error_detection']['error_count'])
                else:
                    errors_detected.append(0)
                
                if result.get('corrections', {}).get('applied', False):
                    corrections_applied.append(result['corrections']['count'])
                else:
                    corrections_applied.append(0)
        
        results = {
            'num_samples': len(wers),
            'wer': {
                'mean': np.mean(wers) if wers else None,
                'std': np.std(wers) if wers else None,
                'values': wers
            },
            'cer': {
                'mean': np.mean(cers) if cers else None,
                'std': np.std(cers) if cers else None,
                'values': cers
            },
            'latency': {
                'mean': np.mean(latencies) if latencies else None,
                'std': np.std(latencies) if latencies else None,
                'values': latencies
            },
            'errors_detected': {
                'total': sum(errors_detected),
                'mean': np.mean(errors_detected) if errors_detected else None,
                'values': errors_detected
            },
            'corrections_applied': {
                'total': sum(corrections_applied),
                'mean': np.mean(corrections_applied) if corrections_applied else None,
                'values': corrections_applied
            }
        }
        
        logger.info(f"Full System WER: {results['wer']['mean']:.4f} ({results['wer']['mean']*100:.2f}%)")
        logger.info(f"Full System CER: {results['cer']['mean']:.4f} ({results['cer']['mean']*100:.2f}%)")
        logger.info(f"Mean Latency: {results['latency']['mean']:.2f}s")
        logger.info(f"Total Errors Detected: {results['errors_detected']['total']}")
        logger.info(f"Total Corrections Applied: {results['corrections_applied']['total']}")
        
        return results
    
    def run_statistical_analysis(self, baseline_wers: List[float], full_system_wers: List[float]) -> Dict:
        """Run statistical analysis comparing baseline vs full system."""
        logger.info("="*70)
        logger.info("RUNNING STATISTICAL ANALYSIS")
        logger.info("="*70)
        
        if len(baseline_wers) != len(full_system_wers) or len(baseline_wers) < 2:
            logger.warning("Insufficient data for statistical analysis")
            return {}
        
        analyzer = StatisticalAnalyzer()
        
        # Paired t-test
        t_test_result = analyzer.paired_t_test(baseline_wers, full_system_wers)
        
        # System comparison
        comparison = analyzer.compare_systems(
            baseline_wers,
            full_system_wers,
            "Baseline",
            "Full System"
        )
        
        results = {
            'paired_t_test': t_test_result,
            'system_comparison': comparison
        }
        
        logger.info(f"Mean Baseline WER: {t_test_result['mean_baseline']:.4f}")
        logger.info(f"Mean Full System WER: {t_test_result['mean_treatment']:.4f}")
        logger.info(f"Mean Difference: {t_test_result['mean_difference']:.4f}")
        logger.info(f"p-value: {t_test_result['p_value']:.4f}")
        logger.info(f"Statistically Significant: {t_test_result['is_significant']}")
        logger.info(f"Cohen's d: {t_test_result['cohens_d']:.4f}")
        
        return results
    
    def run_ablation_study(self, audio_files: List[str], references: List[str]) -> Dict:
        """Run ablation study."""
        logger.info("="*70)
        logger.info("RUNNING ABLATION STUDY")
        logger.info("="*70)
        
        try:
            study = AblationStudy()
            results = study.run_ablation_study(
                audio_files=audio_files,
                reference_transcripts=references,
                model_name="whisper"
            )
            
            logger.info("Ablation study completed")
            if 'summary' in results:
                summary = results['summary']
                logger.info(f"Baseline WER: {summary.get('baseline_performance', 'N/A')}")
                logger.info(f"Full System WER: {summary.get('full_system_performance', 'N/A')}")
            
            return results
        except Exception as e:
            logger.error(f"Error running ablation study: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report."""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("COMPREHENSIVE EVALUATION REPORT")
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Baseline Results
        if 'baseline' in self.results:
            baseline = self.results['baseline']
            report_lines.append("BASELINE MODEL RESULTS")
            report_lines.append("-"*70)
            report_lines.append(f"Number of Samples: {baseline['num_samples']}")
            if baseline['wer']['mean'] is not None:
                report_lines.append(f"WER: {baseline['wer']['mean']:.4f} ({baseline['wer']['mean']*100:.2f}%)")
                report_lines.append(f"  Std: {baseline['wer']['std']:.4f}")
                report_lines.append(f"  Range: [{baseline['wer']['min']:.4f}, {baseline['wer']['max']:.4f}]")
            if baseline['cer']['mean'] is not None:
                report_lines.append(f"CER: {baseline['cer']['mean']:.4f} ({baseline['cer']['mean']*100:.2f}%)")
            if baseline['latency']['mean'] is not None:
                report_lines.append(f"Mean Latency: {baseline['latency']['mean']:.2f}s")
            report_lines.append("")
        
        # Full System Results
        if 'full_system' in self.results:
            full = self.results['full_system']
            report_lines.append("FULL SYSTEM RESULTS")
            report_lines.append("-"*70)
            report_lines.append(f"Number of Samples: {full['num_samples']}")
            if full['wer']['mean'] is not None:
                report_lines.append(f"WER: {full['wer']['mean']:.4f} ({full['wer']['mean']*100:.2f}%)")
            if full['cer']['mean'] is not None:
                report_lines.append(f"CER: {full['cer']['mean']:.4f} ({full['cer']['mean']*100:.2f}%)")
            if full['latency']['mean'] is not None:
                report_lines.append(f"Mean Latency: {full['latency']['mean']:.2f}s")
            if 'errors_detected' in full:
                report_lines.append(f"Total Errors Detected: {full['errors_detected']['total']}")
                report_lines.append(f"Total Corrections Applied: {full['corrections_applied']['total']}")
            report_lines.append("")
        
        # Statistical Analysis
        if 'statistical' in self.results:
            stats = self.results['statistical']
            report_lines.append("STATISTICAL ANALYSIS")
            report_lines.append("-"*70)
            if 'paired_t_test' in stats:
                t_test = stats['paired_t_test']
                report_lines.append(f"Mean Difference: {t_test['mean_difference']:.4f}")
                report_lines.append(f"p-value: {t_test['p_value']:.4f}")
                report_lines.append(f"Statistically Significant: {t_test['is_significant']}")
                report_lines.append(f"Cohen's d: {t_test['cohens_d']:.4f}")
                report_lines.append(f"95% CI: [{t_test['confidence_interval'][0]:.4f}, {t_test['confidence_interval'][1]:.4f}]")
            report_lines.append("")
        
        # Ablation Study
        if 'ablation' in self.results:
            ablation = self.results['ablation']
            report_lines.append("ABLATION STUDY RESULTS")
            report_lines.append("-"*70)
            if 'summary' in ablation:
                summary = ablation['summary']
                report_lines.append(f"Baseline WER: {summary.get('baseline_performance', 'N/A')}")
                report_lines.append(f"Full System WER: {summary.get('full_system_performance', 'N/A')}")
            report_lines.append("")
        
        # Comparison
        if 'baseline' in self.results and 'full_system' in self.results:
            baseline_wer = self.results['baseline']['wer']['mean']
            full_wer = self.results['full_system']['wer']['mean']
            if baseline_wer and full_wer:
                improvement = ((baseline_wer - full_wer) / baseline_wer) * 100
                report_lines.append("IMPROVEMENT SUMMARY")
                report_lines.append("-"*70)
                report_lines.append(f"WER Improvement: {improvement:.2f}% relative reduction")
                report_lines.append(f"  ({baseline_wer*100:.2f}% → {full_wer*100:.2f}%)")
        
        report_lines.append("")
        report_lines.append("="*70)
        
        return "\n".join(report_lines)
    
    def run_all_evaluations(self):
        """Run all evaluation types."""
        logger.info("Starting comprehensive evaluation...")
        
        # Find test files
        audio_files = self.find_test_audio_files()
        if not audio_files:
            logger.error("No audio files found!")
            return
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Create references (using baseline as proxy)
        references = self.create_reference_transcripts(audio_files)
        
        # Filter out files without references
        valid_pairs = [(a, r) for a, r in zip(audio_files, references) if r.strip()]
        audio_files = [a for a, r in valid_pairs]
        references = [r for a, r in valid_pairs]
        
        logger.info(f"Evaluating {len(audio_files)} files with references")
        
        # Run evaluations
        self.results['baseline'] = self.evaluate_baseline(audio_files, references)
        self.results['full_system'] = self.evaluate_full_system(audio_files, references)
        
        # Statistical analysis
        if (self.results['baseline']['wer']['values'] and 
            self.results['full_system']['wer']['values']):
            self.results['statistical'] = self.run_statistical_analysis(
                self.results['baseline']['wer']['values'],
                self.results['full_system']['wer']['values']
            )
        
        # Ablation study (may take longer)
        logger.info("\nRunning ablation study (this may take a while)...")
        self.results['ablation'] = self.run_ablation_study(audio_files, references)
        
        # Save results
        results_file = self.output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {results_file}")
        
        # Generate report
        report = self.generate_report()
        report_file = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_file}")
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(report)
        
        return self.results


def main():
    """Main function."""
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_all_evaluations()
    return results


if __name__ == "__main__":
    main()


