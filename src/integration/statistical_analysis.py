"""
Statistical Analysis Module - Week 4
Quantitative analysis with paired t-tests for statistical significance.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Statistical analysis module for evaluating system performance.
    Implements paired t-tests and other statistical methods.
    """
    
    def __init__(self):
        """Initialize statistical analyzer."""
        self.analysis_results = []
    
    def paired_t_test(
        self,
        baseline_scores: List[float],
        treatment_scores: List[float],
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Perform paired t-test to compare baseline vs treatment.
        
        Args:
            baseline_scores: List of baseline performance scores
            treatment_scores: List of treatment performance scores
            alpha: Significance level (default: 0.05)
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            Dictionary with t-test results
        """
        assert len(baseline_scores) == len(treatment_scores), \
            "Baseline and treatment scores must have same length"
        
        baseline_array = np.array(baseline_scores)
        treatment_array = np.array(treatment_scores)
        
        # Calculate differences
        differences = treatment_array - baseline_array
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        
        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(baseline_array, treatment_array, alternative=alternative)
        
        # Calculate effect size (Cohen's d for paired samples)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Determine significance
        is_significant = p_value < alpha
        
        # Calculate confidence interval
        se_diff = std_diff / np.sqrt(n)
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1) if alternative == 'two-sided' else stats.t.ppf(1 - alpha, df=n-1)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        result = {
            'test_type': 'paired_t_test',
            'n_samples': n,
            'mean_baseline': np.mean(baseline_array),
            'mean_treatment': np.mean(treatment_array),
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            't_statistic': t_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'is_significant': is_significant,
            'cohens_d': cohens_d,
            'confidence_interval': (ci_lower, ci_upper),
            'alternative': alternative,
            'interpretation': self._interpret_t_test_result(mean_diff, p_value, alpha, alternative)
        }
        
        return result
    
    def compare_systems(
        self,
        system_a_scores: List[float],
        system_b_scores: List[float],
        system_a_name: str = "System A",
        system_b_name: str = "System B",
        alpha: float = 0.05
    ) -> Dict:
        """
        Compare two systems using paired t-test.
        
        Args:
            system_a_scores: Performance scores for system A
            system_b_scores: Performance scores for system B
            system_a_name: Name of system A
            system_b_name: Name of system B
            alpha: Significance level
        
        Returns:
            Dictionary with comparison results
        """
        result = self.paired_t_test(
            baseline_scores=system_a_scores,
            treatment_scores=system_b_scores,
            alpha=alpha
        )
        
        result['system_a_name'] = system_a_name
        result['system_b_name'] = system_b_name
        
        # Determine which system is better (assuming lower scores are better, e.g., WER)
        if result['mean_difference'] < 0:
            better_system = system_b_name
            improvement = abs(result['mean_difference'])
        else:
            better_system = system_a_name
            improvement = result['mean_difference']
        
        result['better_system'] = better_system
        result['improvement'] = improvement
        
        return result
    
    def analyze_component_contributions(
        self,
        baseline_scores: List[float],
        component_scores: Dict[str, List[float]],
        alpha: float = 0.05
    ) -> Dict:
        """
        Analyze contribution of individual components.
        
        Args:
            baseline_scores: Baseline performance scores
            component_scores: Dictionary mapping component names to their scores
            alpha: Significance level
        
        Returns:
            Dictionary with component contribution analysis
        """
        contributions = {}
        
        for component_name, scores in component_scores.items():
            comparison = self.paired_t_test(
                baseline_scores=baseline_scores,
                treatment_scores=scores,
                alpha=alpha
            )
            
            contributions[component_name] = {
                'mean_improvement': comparison['mean_difference'],
                'p_value': comparison['p_value'],
                'is_significant': comparison['is_significant'],
                'effect_size': comparison['cohens_d'],
                'interpretation': comparison['interpretation']
            }
        
        return {
            'baseline_mean': np.mean(baseline_scores),
            'component_contributions': contributions,
            'significant_components': [
                name for name, contrib in contributions.items()
                if contrib['is_significant']
            ]
        }
    
    def analyze_trajectory(
        self,
        iteration_scores: List[List[float]],
        alpha: float = 0.05
    ) -> Dict:
        """
        Analyze performance trajectory across iterations.
        
        Args:
            iteration_scores: List of score lists, one per iteration
            alpha: Significance level
        
        Returns:
            Dictionary with trajectory analysis
        """
        if len(iteration_scores) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate mean scores per iteration
        mean_scores = [np.mean(scores) for scores in iteration_scores]
        
        # Test if final iteration is significantly better than first
        first_iteration = iteration_scores[0]
        final_iteration = iteration_scores[-1]
        
        improvement_test = self.paired_t_test(
            baseline_scores=first_iteration,
            treatment_scores=final_iteration,
            alpha=alpha,
            alternative='less'  # Testing if final is better (lower scores)
        )
        
        # Calculate trend
        x = np.arange(len(mean_scores))
        slope, intercept, r_value, p_value_trend, std_err = stats.linregress(x, mean_scores)
        
        return {
            'mean_scores_per_iteration': mean_scores,
            'total_improvement': mean_scores[0] - mean_scores[-1],
            'improvement_test': improvement_test,
            'trend_slope': slope,
            'trend_p_value': p_value_trend,
            'trend_r_squared': r_value ** 2,
            'is_improving': slope < 0 and improvement_test['is_significant']
        }
    
    def _interpret_t_test_result(
        self,
        mean_diff: float,
        p_value: float,
        alpha: float,
        alternative: str
    ) -> str:
        """Interpret t-test result in plain language."""
        if p_value < alpha:
            if alternative == 'two-sided':
                if mean_diff < 0:
                    return f"Treatment is significantly better (p={p_value:.4f} < {alpha})"
                else:
                    return f"Treatment is significantly worse (p={p_value:.4f} < {alpha})"
            elif alternative == 'less':
                return f"Treatment is significantly better (p={p_value:.4f} < {alpha})"
            else:  # greater
                return f"Treatment is significantly worse (p={p_value:.4f} < {alpha})"
        else:
            return f"No significant difference (p={p_value:.4f} >= {alpha})"
    
    def generate_report(
        self,
        analysis_results: List[Dict],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate statistical analysis report.
        
        Args:
            analysis_results: List of analysis result dictionaries
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("Statistical Analysis Report")
        report_lines.append("="*60)
        report_lines.append("")
        
        for i, result in enumerate(analysis_results, 1):
            report_lines.append(f"Analysis {i}: {result.get('test_type', 'unknown')}")
            report_lines.append("-"*60)
            
            if 'mean_baseline' in result:
                report_lines.append(f"  Baseline Mean: {result['mean_baseline']:.4f}")
                report_lines.append(f"  Treatment Mean: {result['mean_treatment']:.4f}")
                report_lines.append(f"  Mean Difference: {result['mean_difference']:.4f}")
            
            if 'p_value' in result:
                report_lines.append(f"  p-value: {result['p_value']:.4f}")
                report_lines.append(f"  Significant: {result['is_significant']}")
                report_lines.append(f"  Effect Size (Cohen's d): {result.get('cohens_d', 0):.4f}")
            
            if 'interpretation' in result:
                report_lines.append(f"  Interpretation: {result['interpretation']}")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
