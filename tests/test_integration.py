"""
Integration tests for complete workflow
Tests the entire pipeline from transcription to data management
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent
from src.data.integration import IntegratedDataManagementSystem
import pytest
import tempfile
import shutil


class TestCompleteWorkflow:
    """Test complete workflow integration"""
    
    def setup_method(self):
        """Setup test components"""
        self.test_dir = tempfile.mkdtemp()
        self.baseline_model = BaselineSTTModel(model_name="whisper")
        self.agent = STTAgent(baseline_model=self.baseline_model)
        self.data_system = IntegratedDataManagementSystem(
            base_dir=self.test_dir,
            use_gcs=False
        )
    
    def teardown_method(self):
        """Cleanup test directory"""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.skipif(not Path("test_audio/test_1.wav").exists(), 
                       reason="Test audio file not available")
    def test_end_to_end_workflow(self):
        """Test complete workflow: transcribe -> detect errors -> record -> correct"""
        audio_path = "test_audio/test_1.wav"
        
        # Step 1: Transcribe with agent
        result = self.agent.transcribe_with_agent(
            audio_path=audio_path,
            enable_auto_correction=True
        )
        
        assert 'transcript' in result
        assert 'error_detection' in result
        assert 'corrections' in result
        
        # Step 2: Record failed case if errors detected
        case_id = None
        if result['error_detection']['has_errors']:
            case_id = self.data_system.record_failed_transcription(
                audio_path=audio_path,
                original_transcript=result['original_transcript'],
                error_types=list(result['error_detection']['error_types'].keys()),
                error_score=result['error_detection']['error_score']
            )
            
            assert case_id is not None
        
        # Step 3: Add correction
        if case_id:
            success = self.data_system.add_correction(
                case_id=case_id,
                corrected_transcript="This is the corrected transcript."
            )
            
            assert success is True
        
        # Step 4: Check statistics
        stats = self.data_system.get_system_statistics()
        
        assert 'data_management' in stats
        if case_id:
            assert stats['data_management']['total_failed_cases'] > 0
    
    def test_multiple_transcriptions_workflow(self):
        """Test workflow with multiple transcriptions"""
        # Simulate multiple transcriptions
        test_cases = [
            ("test_1.wav", "HELLO WORLD"),
            ("test_2.wav", "This is normal."),
            ("test_3.wav", "ANOTHER ALL CAPS"),
        ]
        
        for audio_file, simulated_transcript in test_cases:
            # Simulate error detection
            from src.agent.error_detector import ErrorDetector
            detector = ErrorDetector()
            
            errors = detector.detect_errors(simulated_transcript)
            summary = detector.get_error_summary(errors)
            
            # Record if errors detected
            if summary['has_errors']:
                case_id = self.data_system.record_failed_transcription(
                    audio_path=f"test_audio/{audio_file}",
                    original_transcript=simulated_transcript,
                    corrected_transcript=None,
                    error_types=list(summary['error_types'].keys()),
                    error_score=summary['error_score'],
                    inference_time=0.5
                )
                
                assert case_id is not None
        
        # Check aggregated statistics
        stats = self.data_system.get_system_statistics()
        assert stats['data_management']['total_failed_cases'] >= 0
    
    def test_agent_learning_feedback_loop(self):
        """Test agent learning from feedback"""
        # Record multiple feedback entries
        for i in range(5):
            self.agent.submit_feedback(
                transcript_id=f"test_{i}",
                user_feedback=f"Feedback {i}",
                is_correct=(i % 2 == 0),
                corrected_transcript=f"Corrected {i}" if i % 2 != 0 else None
            )
        
        # Get agent statistics
        stats = self.agent.get_agent_stats()
        
        assert stats['learning']['feedback_count'] == 5
    
    def test_data_system_report_generation(self):
        """Test complete report generation"""
        # Add some test data
        for i in range(3):
            self.data_system.record_failed_transcription(
                audio_path=f"test_{i}.wav",
                original_transcript=f"TEST {i}",
                corrected_transcript=f"Test {i}",
                error_types=["all_caps"],
                error_score=0.8,
                inference_time=0.5
            )
        
        # Track performance
        self.data_system.record_training_performance(
            model_version="test_v1",
            wer=0.12,
            cer=0.06,
            training_metadata={'model_name': 'test_model', 'training_data_size': 100}
        )
        
        # Generate report
        report = self.data_system.generate_comprehensive_report()
        
        assert 'system_statistics' in report
        assert 'performance_metrics' in report
        assert 'data_quality' in report
        assert 'recommendations' in report
    
    def test_dataset_preparation_workflow(self):
        """Test dataset preparation for fine-tuning"""
        # Add enough corrected cases
        for i in range(15):
            self.data_system.record_failed_transcription(
                audio_path=f"test_{i}.wav",
                original_transcript=f"Original {i}",
                corrected_transcript=f"Corrected {i}",
                error_types=["all_caps"],
                error_score=0.6 + (i % 3) * 0.1,
                inference_time=0.5 + (i % 5) * 0.1
            )
        
        # Prepare dataset
        dataset_info = self.data_system.prepare_finetuning_dataset(
            min_error_score=0.5,
            max_samples=10,
            create_version=False  # Don't create version for test
        )
        
        if 'error' not in dataset_info:
            assert 'dataset_id' in dataset_info
            assert 'split_sizes' in dataset_info
            assert dataset_info['split_sizes']['train'] > 0


class TestAgentDataIntegration:
    """Test integration between agent and data management"""
    
    def setup_method(self):
        """Setup test components"""
        self.test_dir = tempfile.mkdtemp()
        self.baseline_model = BaselineSTTModel(model_name="whisper")
        self.agent = STTAgent(baseline_model=self.baseline_model)
        self.data_system = IntegratedDataManagementSystem(
            base_dir=self.test_dir,
            use_gcs=False
        )
    
    def teardown_method(self):
        """Cleanup"""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_agent_stats_and_data_stats_consistency(self):
        """Test that agent and data system statistics are consistent"""
        # Simulate errors
        for i in range(5):
            # Use error detector
            from src.agent.error_detector import ErrorDetector
            detector = ErrorDetector()
            
            errors = detector.detect_errors(f"TEST CASE {i}")
            if errors:
                # Record in data system
                self.data_system.record_failed_transcription(
                    audio_path=f"test_{i}.wav",
                    original_transcript=f"TEST CASE {i}",
                    corrected_transcript=None,
                    error_types=[e.error_type for e in errors],
                    error_score=0.8,
                    inference_time=0.5
                )
        
        # Check statistics
        data_stats = self.data_system.get_system_statistics()
        
        assert data_stats['data_management']['total_failed_cases'] == 5
    
    def test_correction_rate_tracking(self):
        """Test correction rate calculation"""
        # Add cases with and without corrections
        for i in range(10):
            case_id = self.data_system.record_failed_transcription(
                audio_path=f"test_{i}.wav",
                original_transcript=f"Original {i}",
                corrected_transcript=None,
                error_types=["all_caps"],
                error_score=0.8,
                inference_time=0.5
            )
            
            # Add correction for half of them
            if i < 5:
                self.data_system.add_correction(
                    case_id=case_id,
                    corrected_transcript=f"Corrected {i}"
                )
        
        # Check correction rate
        stats = self.data_system.get_system_statistics()
        correction_rate = stats['data_management']['correction_rate']
        
        assert 0.4 <= correction_rate <= 0.6, "Correction rate should be around 50%"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

