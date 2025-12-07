"""
End-to-End Testing Framework - Week 4
Tests the complete feedback loop from transcription to fine-tuning.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import json
from datetime import datetime
import numpy as np

from .unified_system import UnifiedSTTSystem
from ..evaluation.metrics import STTEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndToEndTester:
    """
    End-to-end testing framework for the complete feedback loop.
    Tests the full pipeline: transcription → error detection → correction → learning → fine-tuning
    """
    
    def __init__(self, system: UnifiedSTTSystem):
        """
        Initialize end-to-end tester.
        
        Args:
            system: UnifiedSTTSystem instance to test
        """
        self.system = system
        self.evaluator = STTEvaluator()
        self.test_results = []
    
    def test_feedback_loop(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        num_iterations: int = 3,
        enable_corrections: bool = True
    ) -> Dict:
        """
        Test the complete feedback loop over multiple iterations.
        
        Args:
            audio_files: List of audio file paths
            reference_transcripts: List of reference transcripts
            num_iterations: Number of feedback loop iterations
            enable_corrections: Whether to enable corrections
        
        Returns:
            Dictionary with feedback loop test results
        """
        logger.info(f"Starting feedback loop test: {len(audio_files)} files, {num_iterations} iterations")
        
        iteration_results = []
        
        for iteration in range(num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*60}")
            
            iteration_start = time.time()
            
            # Transcribe all files
            batch_results = []
            for i, (audio_path, reference) in enumerate(zip(audio_files, reference_transcripts)):
                logger.info(f"  Processing {i+1}/{len(audio_files)}: {Path(audio_path).name}")
                
                result = self.system.transcribe(
                    audio_path=audio_path,
                    reference_transcript=reference,
                    enable_auto_correction=enable_corrections
                )
                
                batch_results.append(result)
            
            # Calculate metrics for this iteration
            wers = [r['evaluation']['wer'] for r in batch_results if 'evaluation' in r]
            cers = [r['evaluation']['cer'] for r in batch_results if 'evaluation' in r]
            
            iteration_metrics = {
                'iteration': iteration + 1,
                'average_wer': np.mean(wers) if wers else None,
                'average_cer': np.mean(cers) if cers else None,
                'total_errors_detected': sum(r.get('error_detection', {}).get('error_count', 0) for r in batch_results),
                'total_corrections_applied': sum(r.get('corrections', {}).get('count', 0) for r in batch_results),
                'fine_tuning_triggered': any(r.get('agent_metadata', {}).get('fine_tuning_triggered', False) for r in batch_results),
                'processing_time': time.time() - iteration_start,
                'detailed_results': batch_results
            }
            
            iteration_results.append(iteration_metrics)
            
            # Get system status after iteration
            system_status = self.system.get_system_status()
            iteration_metrics['system_status'] = system_status
            
            logger.info(f"  Average WER: {iteration_metrics['average_wer']:.4f}" if iteration_metrics['average_wer'] else "  Average WER: N/A")
            logger.info(f"  Errors detected: {iteration_metrics['total_errors_detected']}")
            logger.info(f"  Corrections applied: {iteration_metrics['total_corrections_applied']}")
            logger.info(f"  Fine-tuning triggered: {iteration_metrics['fine_tuning_triggered']}")
        
        # Analyze feedback loop effectiveness
        feedback_analysis = self._analyze_feedback_loop(iteration_results)
        
        return {
            'test_type': 'feedback_loop',
            'num_files': len(audio_files),
            'num_iterations': num_iterations,
            'iteration_results': iteration_results,
            'feedback_analysis': feedback_analysis,
            'summary': self._generate_summary(iteration_results, feedback_analysis)
        }
    
    def test_error_detection_accuracy(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        known_errors: Optional[List[List[Dict]]] = None
    ) -> Dict:
        """
        Test accuracy of error detection component.
        
        Args:
            audio_files: List of audio file paths
            reference_transcripts: List of reference transcripts
            known_errors: Optional list of known errors per file
        
        Returns:
            Dictionary with error detection test results
        """
        logger.info(f"Testing error detection accuracy: {len(audio_files)} files")
        
        detection_results = []
        
        for audio_path, reference in zip(audio_files, reference_transcripts):
            result = self.system.transcribe(
                audio_path=audio_path,
                reference_transcript=reference,
                enable_auto_correction=False  # Don't correct, just detect
            )
            
            errors_detected = result.get('error_detection', {})
            
            detection_result = {
                'audio_file': str(audio_path),
                'errors_detected': errors_detected.get('error_count', 0),
                'error_types': errors_detected.get('error_types', {}),
                'has_errors': errors_detected.get('has_errors', False),
                'error_score': errors_detected.get('error_score', 0.0)
            }
            
            # Compare with known errors if provided
            if known_errors:
                file_index = audio_files.index(audio_path)
                known = known_errors[file_index] if file_index < len(known_errors) else []
                detection_result['known_errors'] = len(known)
                detection_result['detection_accuracy'] = self._calculate_detection_accuracy(
                    errors_detected, known
                )
            
            detection_results.append(detection_result)
        
        return {
            'test_type': 'error_detection_accuracy',
            'num_files': len(audio_files),
            'detection_results': detection_results,
            'summary': {
                'total_errors_detected': sum(r['errors_detected'] for r in detection_results),
                'files_with_errors': sum(1 for r in detection_results if r['has_errors']),
                'average_error_score': np.mean([r['error_score'] for r in detection_results])
            }
        }
    
    def test_correction_effectiveness(
        self,
        audio_files: List[str],
        reference_transcripts: List[str]
    ) -> Dict:
        """
        Test effectiveness of correction mechanisms.
        
        Args:
            audio_files: List of audio file paths
            reference_transcripts: List of reference transcripts
        
        Returns:
            Dictionary with correction effectiveness results
        """
        logger.info(f"Testing correction effectiveness: {len(audio_files)} files")
        
        # Test without corrections
        results_no_correction = []
        for audio_path, reference in zip(audio_files, reference_transcripts):
            result = self.system.transcribe(
                audio_path=audio_path,
                reference_transcript=reference,
                enable_auto_correction=False
            )
            results_no_correction.append(result)
        
        # Test with corrections
        results_with_correction = []
        for audio_path, reference in zip(audio_files, reference_transcripts):
            result = self.system.transcribe(
                audio_path=audio_path,
                reference_transcript=reference,
                enable_auto_correction=True
            )
            results_with_correction.append(result)
        
        # Compare results
        comparison = []
        for no_corr, with_corr in zip(results_no_correction, results_with_correction):
            wer_no = no_corr.get('evaluation', {}).get('wer', float('inf'))
            wer_with = with_corr.get('evaluation', {}).get('wer', float('inf'))
            
            improvement = wer_no - wer_with  # Positive means improvement
            
            comparison.append({
                'wer_without_correction': wer_no,
                'wer_with_correction': wer_with,
                'wer_improvement': improvement,
                'relative_improvement': (improvement / wer_no * 100) if wer_no > 0 else 0,
                'corrections_applied': with_corr.get('corrections', {}).get('count', 0)
            })
        
        return {
            'test_type': 'correction_effectiveness',
            'num_files': len(audio_files),
            'comparison': comparison,
            'summary': {
                'average_wer_without': np.mean([c['wer_without_correction'] for c in comparison]),
                'average_wer_with': np.mean([c['wer_with_correction'] for c in comparison]),
                'average_improvement': np.mean([c['wer_improvement'] for c in comparison]),
                'average_relative_improvement': np.mean([c['relative_improvement'] for c in comparison]),
                'total_corrections': sum(c['corrections_applied'] for c in comparison)
            }
        }
    
    def test_fine_tuning_impact(
        self,
        audio_files: List[str],
        reference_transcripts: List[str],
        trigger_fine_tuning: bool = True
    ) -> Dict:
        """
        Test impact of fine-tuning on system performance.
        
        Args:
            audio_files: List of audio file paths
            reference_transcripts: List of reference transcripts
            trigger_fine_tuning: Whether to trigger fine-tuning during test
        
        Returns:
            Dictionary with fine-tuning impact results
        """
        logger.info(f"Testing fine-tuning impact: {len(audio_files)} files")
        
        # Baseline performance (before fine-tuning)
        baseline_results = []
        for audio_path, reference in zip(audio_files, reference_transcripts):
            result = self.system.transcribe(
                audio_path=audio_path,
                reference_transcript=reference
            )
            baseline_results.append(result)
        
        baseline_wer = np.mean([r['evaluation']['wer'] for r in baseline_results if 'evaluation' in r])
        
        # Trigger fine-tuning if enabled
        if trigger_fine_tuning and self.system.enable_adaptive_fine_tuning:
            logger.info("Triggering fine-tuning...")
            fine_tuning_result = self.system.agent.manually_trigger_fine_tuning()
            logger.info(f"Fine-tuning result: {fine_tuning_result.get('success', False)}")
        
        # Performance after fine-tuning
        post_fine_tuning_results = []
        for audio_path, reference in zip(audio_files, reference_transcripts):
            result = self.system.transcribe(
                audio_path=audio_path,
                reference_transcript=reference
            )
            post_fine_tuning_results.append(result)
        
        post_wer = np.mean([r['evaluation']['wer'] for r in post_fine_tuning_results if 'evaluation' in r])
        
        return {
            'test_type': 'fine_tuning_impact',
            'num_files': len(audio_files),
            'baseline_wer': baseline_wer,
            'post_fine_tuning_wer': post_wer,
            'wer_improvement': baseline_wer - post_wer,
            'relative_improvement': ((baseline_wer - post_wer) / baseline_wer * 100) if baseline_wer > 0 else 0,
            'fine_tuning_triggered': trigger_fine_tuning
        }
    
    def _analyze_feedback_loop(self, iteration_results: List[Dict]) -> Dict:
        """Analyze feedback loop effectiveness."""
        wers = [r['average_wer'] for r in iteration_results if r['average_wer'] is not None]
        
        if len(wers) < 2:
            return {'status': 'insufficient_data'}
        
        # Check if performance is improving
        improvement_trend = wers[-1] < wers[0]  # Lower WER is better
        
        # Calculate improvement rate
        if len(wers) >= 2:
            total_improvement = wers[0] - wers[-1]
            improvement_rate = total_improvement / len(wers) if len(wers) > 1 else 0
        else:
            improvement_rate = 0
        
        return {
            'initial_wer': wers[0],
            'final_wer': wers[-1],
            'improvement_trend': improvement_trend,
            'total_improvement': total_improvement,
            'improvement_rate': improvement_rate,
            'wer_trajectory': wers
        }
    
    def _calculate_detection_accuracy(
        self,
        detected_errors: Dict,
        known_errors: List[Dict]
    ) -> Dict:
        """Calculate error detection accuracy metrics."""
        detected_count = detected_errors.get('error_count', 0)
        known_count = len(known_errors)
        
        if known_count == 0:
            return {
                'precision': 1.0 if detected_count == 0 else 0.0,
                'recall': 1.0,
                'f1_score': 1.0 if detected_count == 0 else 0.0
            }
        
        # Simplified: assume detected errors match known errors
        # In practice, would need more sophisticated matching
        true_positives = min(detected_count, known_count)
        precision = true_positives / detected_count if detected_count > 0 else 0.0
        recall = true_positives / known_count if known_count > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': max(0, detected_count - known_count),
            'false_negatives': max(0, known_count - detected_count)
        }
    
    def _generate_summary(
        self,
        iteration_results: List[Dict],
        feedback_analysis: Dict
    ) -> Dict:
        """Generate summary of feedback loop test."""
        return {
            'num_iterations': len(iteration_results),
            'average_wer_per_iteration': [r['average_wer'] for r in iteration_results if r['average_wer'] is not None],
            'total_errors_detected': sum(r['total_errors_detected'] for r in iteration_results),
            'total_corrections_applied': sum(r['total_corrections_applied'] for r in iteration_results),
            'fine_tuning_events': sum(1 for r in iteration_results if r['fine_tuning_triggered']),
            'feedback_effectiveness': feedback_analysis.get('improvement_trend', False),
            'total_improvement': feedback_analysis.get('total_improvement', 0)
        }
    
    def run_full_test_suite(
        self,
        audio_files: List[str],
        reference_transcripts: List[str]
    ) -> Dict:
        """
        Run complete end-to-end test suite.
        
        Args:
            audio_files: List of audio file paths
            reference_transcripts: List of reference transcripts
        
        Returns:
            Dictionary with all test results
        """
        logger.info("="*60)
        logger.info("Running Full End-to-End Test Suite")
        logger.info("="*60)
        
        all_results = {}
        
        # Test 1: Feedback loop
        logger.info("\n1. Testing feedback loop...")
        all_results['feedback_loop'] = self.test_feedback_loop(
            audio_files, reference_transcripts, num_iterations=3
        )
        
        # Test 2: Error detection accuracy
        logger.info("\n2. Testing error detection accuracy...")
        all_results['error_detection'] = self.test_error_detection_accuracy(
            audio_files, reference_transcripts
        )
        
        # Test 3: Correction effectiveness
        logger.info("\n3. Testing correction effectiveness...")
        all_results['correction_effectiveness'] = self.test_correction_effectiveness(
            audio_files, reference_transcripts
        )
        
        # Test 4: Fine-tuning impact
        logger.info("\n4. Testing fine-tuning impact...")
        all_results['fine_tuning_impact'] = self.test_fine_tuning_impact(
            audio_files, reference_transcripts
        )
        
        return {
            'test_suite': 'end_to_end',
            'timestamp': datetime.now().isoformat(),
            'num_files': len(audio_files),
            'results': all_results,
            'system_status': self.system.get_system_status()
        }
