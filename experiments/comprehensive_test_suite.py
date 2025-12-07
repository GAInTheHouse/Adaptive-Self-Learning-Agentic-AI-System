"""
Comprehensive Test Suite - Week 4
Complete testing framework integrating all Week 4 components.
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration import UnifiedSTTSystem
from src.integration.end_to_end_testing import EndToEndTester
from src.integration.statistical_analysis import StatisticalAnalyzer
from src.integration.ablation_studies import AblationStudy
from src.baseline_model import BaselineSTTModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveTestSuite:
    """
    Comprehensive test suite that integrates all testing components.
    """
    
    def __init__(self, output_dir: str = "experiments/test_outputs"):
        """
        Initialize comprehensive test suite.
        
        Args:
            output_dir: Directory to save test outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
    
    def run_all_tests(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        model_name: str = "whisper"
    ) -> Dict:
        """
        Run complete test suite.
        
        Args:
            audio_files: List of audio file paths
            reference_transcripts: List of reference transcripts
            model_name: Model name to use
        
        Returns:
            Dictionary with all test results
        """
        logger.info("="*70)
        logger.info("COMPREHENSIVE TEST SUITE - WEEK 4")
        logger.info("="*70)
        logger.info(f"Testing {len(audio_files)} audio files")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'num_files': len(audio_files),
            'model_name': model_name
        }
        
        # Test 1: System Integration
        logger.info("\n" + "="*70)
        logger.info("TEST 1: System Integration")
        logger.info("="*70)
        integration_results = self._test_system_integration(audio_files, reference_transcripts, model_name)
        all_results['integration'] = integration_results
        
        # Test 2: End-to-End Testing
        logger.info("\n" + "="*70)
        logger.info("TEST 2: End-to-End Testing")
        logger.info("="*70)
        e2e_results = self._test_end_to_end(audio_files, reference_transcripts, model_name)
        all_results['end_to_end'] = e2e_results
        
        # Test 3: Statistical Analysis
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Statistical Analysis")
        logger.info("="*70)
        statistical_results = self._test_statistical_analysis(audio_files, reference_transcripts, model_name)
        all_results['statistical_analysis'] = statistical_results
        
        # Test 4: Ablation Studies
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Ablation Studies")
        logger.info("="*70)
        ablation_results = self._test_ablation_studies(audio_files, reference_transcripts, model_name)
        all_results['ablation_studies'] = ablation_results
        
        # Generate comprehensive report
        logger.info("\n" + "="*70)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("="*70)
        report = self._generate_comprehensive_report(all_results)
        
        # Save results
        results_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to: {results_file}")
        
        report_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_file}")
        
        return all_results
    
    def _test_system_integration(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        model_name: str
    ) -> Dict:
        """Test system integration."""
        logger.info("Initializing unified system...")
        system = UnifiedSTTSystem(
            model_name=model_name,
            enable_error_detection=True,
            enable_llm_correction=True,
            enable_adaptive_fine_tuning=True
        )
        
        # Test system status
        status = system.get_system_status()
        
        # Test batch evaluation
        logger.info("Running batch evaluation...")
        batch_results = system.evaluate_batch(
            audio_files[:min(5, len(audio_files))],  # Test on subset
            reference_transcripts[:min(5, len(reference_transcripts))]
        )
        
        return {
            'system_status': status,
            'batch_evaluation': batch_results,
            'components_enabled': status['component_status']
        }
    
    def _test_end_to_end(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        model_name: str
    ) -> Dict:
        """Test end-to-end functionality."""
        system = UnifiedSTTSystem(model_name=model_name)
        tester = EndToEndTester(system)
        
        # Run full test suite
        test_files = audio_files[:min(10, len(audio_files))]
        test_refs = reference_transcripts[:min(10, len(reference_transcripts))]
        
        results = tester.run_full_test_suite(test_files, test_refs)
        
        return results
    
    def _test_statistical_analysis(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        model_name: str
    ) -> Dict:
        """Test statistical analysis."""
        analyzer = StatisticalAnalyzer()
        
        # Create baseline and full system
        baseline_system = UnifiedSTTSystem(
            model_name=model_name,
            enable_error_detection=False,
            enable_llm_correction=False,
            enable_adaptive_fine_tuning=False
        )
        
        full_system = UnifiedSTTSystem(
            model_name=model_name,
            enable_error_detection=True,
            enable_llm_correction=True,
            enable_adaptive_fine_tuning=True
        )
        
        # Evaluate both systems
        test_files = audio_files[:min(20, len(audio_files))]
        test_refs = reference_transcripts[:min(20, len(reference_transcripts))]
        
        baseline_scores = []
        full_system_scores = []
        
        for audio_path, reference in zip(test_files, test_refs):
            baseline_result = baseline_system.transcribe(audio_path, reference)
            full_result = full_system.transcribe(audio_path, reference)
            
            if 'evaluation' in baseline_result:
                baseline_scores.append(baseline_result['evaluation']['wer'])
            if 'evaluation' in full_result:
                full_system_scores.append(full_result['evaluation']['wer'])
        
        # Perform paired t-test
        if len(baseline_scores) == len(full_system_scores) and len(baseline_scores) > 0:
            comparison = analyzer.compare_systems(
                baseline_scores,
                full_system_scores,
                "Baseline System",
                "Full System"
            )
            
            return {
                'baseline_scores': baseline_scores,
                'full_system_scores': full_system_scores,
                'statistical_comparison': comparison
            }
        else:
            return {'status': 'insufficient_data'}
    
    def _test_ablation_studies(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        model_name: str
    ) -> Dict:
        """Test ablation studies."""
        study = AblationStudy()
        
        # Run ablation study on subset
        test_files = audio_files[:min(15, len(audio_files))]
        test_refs = reference_transcripts[:min(15, len(reference_transcripts))]
        
        results = study.run_ablation_study(test_files, test_refs, model_name)
        
        # Generate report
        report = study.generate_ablation_report(results)
        
        return {
            'ablation_results': results,
            'report': report
        }
    
    def _generate_comprehensive_report(self, all_results: Dict) -> str:
        """Generate comprehensive test report."""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("COMPREHENSIVE TEST SUITE REPORT - WEEK 4")
        report_lines.append("="*70)
        report_lines.append(f"Generated: {all_results.get('timestamp', 'Unknown')}")
        report_lines.append(f"Files Tested: {all_results.get('num_files', 0)}")
        report_lines.append("")
        
        # Integration Results
        if 'integration' in all_results:
            report_lines.append("1. SYSTEM INTEGRATION")
            report_lines.append("-"*70)
            integration = all_results['integration']
            components = integration.get('components_enabled', {})
            for component, enabled in components.items():
                status = "✅" if enabled else "❌"
                report_lines.append(f"  {status} {component}: {enabled}")
            report_lines.append("")
        
        # End-to-End Results
        if 'end_to_end' in all_results:
            report_lines.append("2. END-TO-END TESTING")
            report_lines.append("-"*70)
            e2e = all_results['end_to_end']
            results = e2e.get('results', {})
            if 'feedback_loop' in results:
                fb = results['feedback_loop']
                summary = fb.get('summary', {})
                report_lines.append(f"  Feedback Loop Iterations: {summary.get('num_iterations', 0)}")
                report_lines.append(f"  Total Errors Detected: {summary.get('total_errors_detected', 0)}")
                report_lines.append(f"  Total Corrections Applied: {summary.get('total_corrections_applied', 0)}")
            report_lines.append("")
        
        # Statistical Analysis
        if 'statistical_analysis' in all_results:
            report_lines.append("3. STATISTICAL ANALYSIS")
            report_lines.append("-"*70)
            stats = all_results['statistical_analysis']
            if 'statistical_comparison' in stats:
                comp = stats['statistical_comparison']
                report_lines.append(f"  Baseline Mean WER: {comp.get('mean_baseline', 0):.4f}")
                report_lines.append(f"  Full System Mean WER: {comp.get('mean_treatment', 0):.4f}")
                report_lines.append(f"  Improvement: {comp.get('mean_difference', 0):.4f}")
                report_lines.append(f"  p-value: {comp.get('p_value', 0):.4f}")
                report_lines.append(f"  Significant: {comp.get('is_significant', False)}")
            report_lines.append("")
        
        # Ablation Studies
        if 'ablation_studies' in all_results:
            report_lines.append("4. ABLATION STUDIES")
            report_lines.append("-"*70)
            ablation = all_results['ablation_studies']
            if 'ablation_results' in ablation:
                results = ablation['ablation_results']
                summary = results.get('summary', {})
                report_lines.append(f"  Baseline WER: {summary.get('baseline_performance', 0):.4f}")
                report_lines.append(f"  Full System WER: {summary.get('full_system_performance', 0):.4f}")
                report_lines.append(f"  Overall Improvement: {summary.get('overall_improvement', 0):.4f}")
                
                contributions = summary.get('component_contributions', {})
                if contributions:
                    report_lines.append("  Component Contributions:")
                    for component, contrib in contributions.items():
                        sig = "✅" if contrib.get('is_significant', False) else "❌"
                        report_lines.append(f"    {sig} {component}: {contrib.get('improvement', 0):.4f}")
            report_lines.append("")
        
        report_lines.append("="*70)
        report_lines.append("END OF REPORT")
        report_lines.append("="*70)
        
        return "\n".join(report_lines)


def main():
    """Main function to run comprehensive test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive test suite')
    parser.add_argument('--audio-dir', type=str, help='Directory containing audio files')
    parser.add_argument('--references', type=str, help='JSON file with reference transcripts')
    parser.add_argument('--output-dir', type=str, default='experiments/test_outputs',
                       help='Output directory for test results')
    parser.add_argument('--model', type=str, default='whisper', help='Model name')
    
    args = parser.parse_args()
    
    # For demonstration, use test audio if available
    test_audio_dir = Path("src/test_audio")
    audio_files = []
    reference_transcripts = []
    
    if test_audio_dir.exists():
        audio_files = list(test_audio_dir.glob("*.wav"))[:10]  # Limit for testing
        # Create dummy references for testing
        reference_transcripts = ["Test transcript"] * len(audio_files)
    
    if not audio_files:
        logger.warning("No audio files found. Please provide audio files for testing.")
        logger.info("Usage: python comprehensive_test_suite.py --audio-dir <dir> --references <json>")
        return
    
    # Run test suite
    suite = ComprehensiveTestSuite(output_dir=args.output_dir)
    results = suite.run_all_tests(
        audio_files=[str(f) for f in audio_files],
        reference_transcripts=reference_transcripts,
        model_name=args.model
    )
    
    logger.info("\n✅ Comprehensive test suite completed!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
