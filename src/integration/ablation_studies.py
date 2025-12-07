"""
Ablation Studies Framework - Week 4
Evaluate individual component contributions through systematic ablation.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import json
from datetime import datetime
import numpy as np

from .unified_system import UnifiedSTTSystem
from .statistical_analysis import StatisticalAnalyzer
from ..evaluation.metrics import STTEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Ablation study framework for evaluating component contributions.
    Systematically removes components to measure their individual impact.
    """
    
    def __init__(self, base_config: Optional[Dict] = None):
        """
        Initialize ablation study framework.
        
        Args:
            base_config: Base configuration for system initialization
        """
        self.base_config = base_config or {}
        self.study_results = []
        self.analyzer = StatisticalAnalyzer()
        self.evaluator = STTEvaluator()
    
    def run_ablation_study(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        model_name: str = "whisper"
    ) -> Dict:
        """
        Run complete ablation study.
        
        Args:
            audio_files: List of audio file paths
            reference_transcripts: List of reference transcripts
            model_name: Model name to use
        
        Returns:
            Dictionary with ablation study results
        """
        logger.info("="*60)
        logger.info("Running Ablation Study")
        logger.info("="*60)
        logger.info(f"Testing {len(audio_files)} files")
        logger.info("")
        
        # Define all system configurations to test
        configurations = self._define_configurations()
        
        all_results = {}
        
        for config_name, config in configurations.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Configuration: {config_name}")
            logger.info(f"{'='*60}")
            logger.info(f"Components enabled: {config}")
            
            # Initialize system with this configuration
            system = UnifiedSTTSystem(
                model_name=model_name,
                enable_error_detection=config.get('error_detection', False),
                enable_llm_correction=config.get('llm_correction', False),
                enable_adaptive_fine_tuning=config.get('adaptive_fine_tuning', False),
                **self.base_config
            )
            
            # Evaluate this configuration
            results = self._evaluate_configuration(
                system=system,
                audio_files=audio_files,
                reference_transcripts=reference_transcripts,
                config_name=config_name
            )
            
            all_results[config_name] = results
            
            logger.info(f"  Average WER: {results['average_wer']:.4f}")
            logger.info(f"  Average CER: {results['average_cer']:.4f}")
            logger.info(f"  Processing time: {results['total_time']:.2f}s")
        
        # Analyze component contributions
        contribution_analysis = self._analyze_contributions(all_results)
        
        return {
            'study_type': 'ablation',
            'timestamp': datetime.now().isoformat(),
            'num_files': len(audio_files),
            'configurations_tested': list(all_results.keys()),
            'results': all_results,
            'contribution_analysis': contribution_analysis,
            'summary': self._generate_ablation_summary(all_results, contribution_analysis)
        }
    
    def _define_configurations(self) -> Dict[str, Dict]:
        """
        Define all system configurations for ablation study.
        Each configuration represents a different combination of components.
        """
        return {
            # Baseline: Only baseline model
            'baseline_only': {
                'error_detection': False,
                'llm_correction': False,
                'adaptive_fine_tuning': False,
                'description': 'Baseline STT model only'
            },
            
            # Baseline + Error Detection
            'baseline_error_detection': {
                'error_detection': True,
                'llm_correction': False,
                'adaptive_fine_tuning': False,
                'description': 'Baseline + Error Detection'
            },
            
            # Baseline + Error Detection + Self-Learning
            'baseline_error_self_learning': {
                'error_detection': True,
                'llm_correction': False,
                'adaptive_fine_tuning': False,
                'description': 'Baseline + Error Detection + Self-Learning'
            },
            
            # Baseline + Error Detection + LLM Correction
            'baseline_error_llm': {
                'error_detection': True,
                'llm_correction': True,
                'adaptive_fine_tuning': False,
                'description': 'Baseline + Error Detection + LLM Correction'
            },
            
            # Full system without fine-tuning
            'full_no_finetuning': {
                'error_detection': True,
                'llm_correction': True,
                'adaptive_fine_tuning': False,
                'description': 'Full system without adaptive fine-tuning'
            },
            
            # Full system with all components
            'full_system': {
                'error_detection': True,
                'llm_correction': True,
                'adaptive_fine_tuning': True,
                'description': 'Full system with all components'
            }
        }
    
    def _evaluate_configuration(
        self,
        system: UnifiedSTTSystem,
        audio_files: List[str],
        reference_transcripts: List[str],
        config_name: str
    ) -> Dict:
        """
        Evaluate a specific system configuration.
        
        Args:
            system: UnifiedSTTSystem instance
            audio_files: List of audio file paths
            reference_transcripts: List of reference transcripts
            config_name: Name of configuration
        
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        wers = []
        cers = []
        error_counts = []
        correction_counts = []
        
        for audio_path, reference in zip(audio_files, reference_transcripts):
            result = system.transcribe(
                audio_path=audio_path,
                reference_transcript=reference,
                enable_auto_correction=True
            )
            
            if 'evaluation' in result:
                wers.append(result['evaluation']['wer'])
                cers.append(result['evaluation']['cer'])
            
            error_counts.append(result.get('error_detection', {}).get('error_count', 0))
            correction_counts.append(result.get('corrections', {}).get('count', 0))
        
        total_time = time.time() - start_time
        
        return {
            'config_name': config_name,
            'average_wer': np.mean(wers) if wers else None,
            'average_cer': np.mean(cers) if cers else None,
            'std_wer': np.std(wers) if wers else None,
            'std_cer': np.std(cers) if cers else None,
            'total_errors_detected': sum(error_counts),
            'total_corrections_applied': sum(correction_counts),
            'total_time': total_time,
            'average_time_per_file': total_time / len(audio_files) if audio_files else 0,
            'wer_scores': wers,
            'cer_scores': cers
        }
    
    def _analyze_contributions(self, all_results: Dict[str, Dict]) -> Dict:
        """
        Analyze contribution of each component.
        
        Args:
            all_results: Dictionary mapping config names to results
        
        Returns:
            Dictionary with contribution analysis
        """
        baseline_results = all_results.get('baseline_only', {})
        baseline_wers = baseline_results.get('wer_scores', [])
        
        if not baseline_wers:
            return {'status': 'insufficient_baseline_data'}
        
        contributions = {}
        
        # Error Detection contribution
        error_detection_results = all_results.get('baseline_error_detection', {})
        if error_detection_results.get('wer_scores'):
            error_detection_contribution = self.analyzer.paired_t_test(
                baseline_scores=baseline_wers,
                treatment_scores=error_detection_results['wer_scores']
            )
            contributions['error_detection'] = {
                'improvement': error_detection_contribution['mean_difference'],
                'p_value': error_detection_contribution['p_value'],
                'is_significant': error_detection_contribution['is_significant'],
                'effect_size': error_detection_contribution['cohens_d']
            }
        
        # LLM Correction contribution
        llm_results = all_results.get('baseline_error_llm', {})
        error_only_results = all_results.get('baseline_error_detection', {})
        if llm_results.get('wer_scores') and error_only_results.get('wer_scores'):
            llm_contribution = self.analyzer.paired_t_test(
                baseline_scores=error_only_results['wer_scores'],
                treatment_scores=llm_results['wer_scores']
            )
            contributions['llm_correction'] = {
                'improvement': llm_contribution['mean_difference'],
                'p_value': llm_contribution['p_value'],
                'is_significant': llm_contribution['is_significant'],
                'effect_size': llm_contribution['cohens_d']
            }
        
        # Adaptive Fine-Tuning contribution
        full_results = all_results.get('full_system', {})
        no_finetuning_results = all_results.get('full_no_finetuning', {})
        if full_results.get('wer_scores') and no_finetuning_results.get('wer_scores'):
            finetuning_contribution = self.analyzer.paired_t_test(
                baseline_scores=no_finetuning_results['wer_scores'],
                treatment_scores=full_results['wer_scores']
            )
            contributions['adaptive_fine_tuning'] = {
                'improvement': finetuning_contribution['mean_difference'],
                'p_value': finetuning_contribution['p_value'],
                'is_significant': finetuning_contribution['is_significant'],
                'effect_size': finetuning_contribution['cohens_d']
            }
        
        # Overall system improvement
        if full_results.get('wer_scores'):
            overall_improvement = self.analyzer.paired_t_test(
                baseline_scores=baseline_wers,
                treatment_scores=full_results['wer_scores']
            )
            contributions['overall_system'] = {
                'improvement': overall_improvement['mean_difference'],
                'p_value': overall_improvement['p_value'],
                'is_significant': overall_improvement['is_significant'],
                'effect_size': overall_improvement['cohens_d'],
                'relative_improvement': (
                    overall_improvement['mean_difference'] / np.mean(baseline_wers) * 100
                    if np.mean(baseline_wers) > 0 else 0
                )
            }
        
        return {
            'baseline_wer': np.mean(baseline_wers),
            'component_contributions': contributions,
            'significant_components': [
                name for name, contrib in contributions.items()
                if contrib.get('is_significant', False)
            ]
        }
    
    def _generate_ablation_summary(
        self,
        all_results: Dict[str, Dict],
        contribution_analysis: Dict
    ) -> Dict:
        """Generate summary of ablation study."""
        baseline_wer = all_results.get('baseline_only', {}).get('average_wer')
        full_system_wer = all_results.get('full_system', {}).get('average_wer')
        
        summary = {
            'baseline_performance': baseline_wer,
            'full_system_performance': full_system_wer,
            'overall_improvement': baseline_wer - full_system_wer if (baseline_wer and full_system_wer) else None,
            'configurations_tested': len(all_results),
            'component_contributions': {}
        }
        
        if 'component_contributions' in contribution_analysis:
            for component, contrib in contribution_analysis['component_contributions'].items():
                summary['component_contributions'][component] = {
                    'improvement': contrib.get('improvement', 0),
                    'is_significant': contrib.get('is_significant', False),
                    'effect_size': contrib.get('effect_size', 0)
                }
        
        return summary
    
    def generate_ablation_report(
        self,
        study_results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed ablation study report.
        
        Args:
            study_results: Results from ablation study
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("Ablation Study Report")
        report_lines.append("="*60)
        report_lines.append("")
        
        # Summary
        summary = study_results.get('summary', {})
        report_lines.append("SUMMARY")
        report_lines.append("-"*60)
        report_lines.append(f"Baseline WER: {summary.get('baseline_performance', 'N/A'):.4f}")
        report_lines.append(f"Full System WER: {summary.get('full_system_performance', 'N/A'):.4f}")
        report_lines.append(f"Overall Improvement: {summary.get('overall_improvement', 'N/A'):.4f}")
        report_lines.append("")
        
        # Component Contributions
        report_lines.append("COMPONENT CONTRIBUTIONS")
        report_lines.append("-"*60)
        contributions = summary.get('component_contributions', {})
        for component, contrib in contributions.items():
            report_lines.append(f"{component}:")
            report_lines.append(f"  Improvement: {contrib.get('improvement', 0):.4f}")
            report_lines.append(f"  Significant: {contrib.get('is_significant', False)}")
            report_lines.append(f"  Effect Size: {contrib.get('effect_size', 0):.4f}")
            report_lines.append("")
        
        # Configuration Results
        report_lines.append("CONFIGURATION RESULTS")
        report_lines.append("-"*60)
        results = study_results.get('results', {})
        for config_name, config_results in results.items():
            report_lines.append(f"{config_name}:")
            report_lines.append(f"  WER: {config_results.get('average_wer', 'N/A'):.4f}")
            report_lines.append(f"  CER: {config_results.get('average_cer', 'N/A'):.4f}")
            report_lines.append(f"  Time: {config_results.get('total_time', 0):.2f}s")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Ablation report saved to {output_path}")
        
        return report
