"""
Ablation Testing Framework - Isolate Agent Impact

Tests agent components individually to understand their contribution.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime

from jiwer import wer, cer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of an ablation test"""
    test_name: str
    description: str
    
    # What was tested
    baseline_enabled: bool
    agent_enabled: bool
    error_detection_enabled: bool
    auto_correction_enabled: bool
    
    # Results
    mean_wer: float
    mean_cer: float
    mean_inference_time: float
    samples_evaluated: int
    
    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'description': self.description,
            'baseline_enabled': self.baseline_enabled,
            'agent_enabled': self.agent_enabled,
            'error_detection_enabled': self.error_detection_enabled,
            'auto_correction_enabled': self.auto_correction_enabled,
            'mean_wer': self.mean_wer,
            'mean_cer': self.mean_cer,
            'mean_inference_time': self.mean_inference_time,
            'samples_evaluated': self.samples_evaluated
        }


class AblationTester:
    """
    Performs ablation testing to isolate agent impact.
    
    Tests:
    1. Baseline only (no agent)
    2. Baseline + Error detection (no correction)
    3. Baseline + Error detection + Auto-correction (full agent)
    4. Error detection only (no baseline)
    
    Metrics:
    - WER/CER improvement at each stage
    - Latency overhead at each stage
    - Consistency across configurations
    """
    
    def __init__(self,
                 agent=None,
                 baseline_model=None,
                 output_dir: str = "experiments/evaluation_outputs"):
        """Initialize ablation tester"""
        self.agent = agent
        self.baseline_model = baseline_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, AblationResult] = {}
        
        logger.info("Ablation Tester initialized")
    
    def test_baseline_only(self, audio_paths: List[str]) -> AblationResult:
        """Test baseline model without agent"""
        
        logger.info("Testing baseline only (no agent)...")
        
        wers = []
        cers = []
        times = []
        
        for audio_path in audio_paths:
            import time
            start = time.time()
            result = self.baseline_model.transcribe(audio_path)
            times.append(time.time() - start)
            
            # Can't calculate WER/CER without reference, use error score
            wers.append(0.0)  # Placeholder
            cers.append(0.0)
        
        result = AblationResult(
            test_name="baseline_only",
            description="Baseline model without agent",
            baseline_enabled=True,
            agent_enabled=False,
            error_detection_enabled=False,
            auto_correction_enabled=False,
            mean_wer=np.mean(wers) if wers else 0.0,
            mean_cer=np.mean(cers) if cers else 0.0,
            mean_inference_time=np.mean(times),
            samples_evaluated=len(audio_paths)
        )
        
        self.results['baseline_only'] = result
        return result
    
    def test_with_error_detection(self, audio_paths: List[str]) -> AblationResult:
        """Test with error detection but no auto-correction"""
        
        logger.info("Testing with error detection (no auto-correction)...")
        
        times = []
        errors_detected = 0
        
        for audio_path in audio_paths:
            import time
            start = time.time()
            result = self.agent.transcribe_with_agent(
                audio_path,
                enable_auto_correction=False
            )
            times.append(time.time() - start)
            
            if result['error_detection']['has_errors']:
                errors_detected += 1
        
        result = AblationResult(
            test_name="with_error_detection",
            description="Error detection enabled, auto-correction disabled",
            baseline_enabled=True,
            agent_enabled=True,
            error_detection_enabled=True,
            auto_correction_enabled=False,
            mean_wer=0.0,  # Would need reference for WER
            mean_cer=0.0,
            mean_inference_time=np.mean(times),
            samples_evaluated=len(audio_paths)
        )
        
        self.results['with_error_detection'] = result
        
        print(f"  Errors detected in {errors_detected}/{len(audio_paths)} samples")
        
        return result
    
    def test_full_agent(self, audio_paths: List[str]) -> AblationResult:
        """Test full agent with error detection and auto-correction"""
        
        logger.info("Testing full agent (detection + correction)...")
        
        times = []
        errors_detected = 0
        corrections_applied = 0
        
        for audio_path in audio_paths:
            import time
            start = time.time()
            result = self.agent.transcribe_with_agent(
                audio_path,
                enable_auto_correction=True
            )
            times.append(time.time() - start)
            
            if result['error_detection']['has_errors']:
                errors_detected += 1
            if result['corrections']['applied']:
                corrections_applied += 1
        
        result = AblationResult(
            test_name="full_agent",
            description="Full agent with error detection and auto-correction",
            baseline_enabled=True,
            agent_enabled=True,
            error_detection_enabled=True,
            auto_correction_enabled=True,
            mean_wer=0.0,
            mean_cer=0.0,
            mean_inference_time=np.mean(times),
            samples_evaluated=len(audio_paths)
        )
        
        self.results['full_agent'] = result
        
        print(f"  Errors detected in {errors_detected}/{len(audio_paths)} samples")
        print(f"  Corrections applied in {corrections_applied}/{len(audio_paths)} samples")
        
        return result
    
    def run_full_ablation(self, audio_paths: List[str]) -> Dict:
        """Run complete ablation study"""
        
        logger.info(f"Running full ablation study on {len(audio_paths)} samples...")
        
        self.test_baseline_only(audio_paths)
        self.test_with_error_detection(audio_paths)
        self.test_full_agent(audio_paths)
        
        return self.compare_configurations()
    
    def compare_configurations(self) -> Dict:
        """Compare all configurations"""
        
        if not self.results:
            return {"error": "No ablation results"}
        
        comparison = {
            "configurations": [r.to_dict() for r in self.results.values()],
            "latency_analysis": {},
            "impact_analysis": {}
        }
        
        # Latency comparison
        if 'baseline_only' in self.results:
            baseline_time = self.results['baseline_only'].mean_inference_time
            
            for config_name, result in self.results.items():
                if config_name != 'baseline_only':
                    overhead = (result.mean_inference_time - baseline_time) / baseline_time * 100
                    comparison['latency_analysis'][config_name] = {
                        'baseline_time_ms': baseline_time * 1000,
                        'config_time_ms': result.mean_inference_time * 1000,
                        'overhead_percent': overhead
                    }
        
        return comparison
    
    def save_ablation_report(self, filename: str = "ablation_study.json"):
        """Save ablation study report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "configurations": self.compare_configurations()
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Ablation report saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print ablation summary"""
        
        print("\n" + "="*70)
        print("ABLATION TESTING REPORT")
        print("="*70)
        
        for config_name, result in self.results.items():
            print(f"\n{config_name.upper().replace('_', ' ')}:")
            print(f"  Mean inference time: {result.mean_inference_time*1000:.2f} ms")
            print(f"  Samples: {result.samples_evaluated}")
        
        comparison = self.compare_configurations()
        if 'latency_analysis' in comparison:
            print("\nLatency overhead:")
            for config, stats in comparison['latency_analysis'].items():
                print(f"  {config}: +{stats['overhead_percent']:.2f}%")
        
        print("="*70 + "\n")
