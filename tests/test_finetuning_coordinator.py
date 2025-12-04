"""
Unit tests for Fine-Tuning Coordinator
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.finetuning_coordinator import FinetuningCoordinator
from src.data.finetuning_orchestrator import FinetuningConfig
from src.data.model_validator import ValidationConfig
from src.data.model_deployer import DeploymentConfig
from src.data.regression_tester import RegressionConfig
from src.data.data_manager import DataManager


@pytest.fixture
def mock_data_manager(temp_directory):
    """Create mock data manager."""
    manager = Mock(spec=DataManager)
    manager.get_statistics.return_value = {
        'total_failed_cases': 100,
        'corrected_cases': 50,
        'uncorrected_cases': 50,
        'correction_rate': 0.5,
        'error_type_distribution': {}
    }
    manager.get_corrected_cases.return_value = []
    manager.failed_cases_cache = {}
    return manager


@pytest.fixture
def coordinator(mock_data_manager, temp_directory):
    """Create coordinator instance."""
    with patch('src.data.finetuning_orchestrator.GCSManager'), \
         patch('src.data.model_validator.GCSManager'), \
         patch('src.data.model_deployer.GCSManager'), \
         patch('src.data.regression_tester.GCSManager'):
        
        coordinator = FinetuningCoordinator(
            data_manager=mock_data_manager,
            finetuning_config=FinetuningConfig(
                min_error_cases=10,
                auto_approve_finetuning=True
            ),
            validation_config=ValidationConfig(
                min_wer_improvement=0.0,
                require_significance=False
            ),
            deployment_config=DeploymentConfig(
                keep_previous_versions=3
            ),
            regression_config=RegressionConfig(
                fail_on_critical_degradation=False
            ),
            use_gcs=False,
            storage_dir=str(Path(temp_directory) / "coordinator")
        )
    
    return coordinator


class TestFinetuningCoordinator:
    """Test FinetuningCoordinator."""
    
    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator is not None
        assert coordinator.orchestrator is not None
        assert coordinator.validator is not None
        assert coordinator.deployer is not None
        assert coordinator.regression_tester is not None
    
    def test_set_training_callback(self, coordinator):
        """Test setting training callback."""
        mock_callback = Mock()
        coordinator.set_training_callback(mock_callback)
        
        assert coordinator.training_callback == mock_callback
        assert coordinator.orchestrator.training_callback == mock_callback
    
    def test_set_baseline_transcribe_function(self, coordinator):
        """Test setting baseline transcribe function."""
        mock_fn = Mock()
        coordinator.set_baseline_transcribe_function(mock_fn)
        
        assert coordinator.baseline_transcribe_fn == mock_fn
    
    def test_set_model_transcribe_function_factory(self, coordinator):
        """Test setting model transcribe function factory."""
        mock_factory = Mock()
        coordinator.set_model_transcribe_function_factory(mock_factory)
        
        assert coordinator.model_transcribe_fn_factory == mock_factory
    
    @patch('src.data.finetuning_orchestrator.FinetuningDatasetPipeline')
    def test_run_complete_workflow_not_triggered(
        self, mock_pipeline, coordinator, mock_data_manager
    ):
        """Test workflow when trigger conditions not met."""
        # Set low error count
        mock_data_manager.get_statistics.return_value = {
            'total_failed_cases': 5,
            'corrected_cases': 2,
            'uncorrected_cases': 3,
            'correction_rate': 0.4,
            'error_type_distribution': {}
        }
        
        workflow = coordinator.run_complete_workflow(force_trigger=False)
        
        assert workflow['status'] == 'not_triggered'
    
    @patch('src.data.finetuning_orchestrator.FinetuningDatasetPipeline')
    def test_run_complete_workflow_triggered(
        self, mock_pipeline, coordinator, mock_data_manager
    ):
        """Test workflow when triggered."""
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.prepare_dataset.return_value = {
            'dataset_id': 'test_dataset',
            'split_sizes': {'train': 80, 'val': 10, 'test': 10}
        }
        mock_pipeline_instance.validate_dataset.return_value = {
            'is_valid': True,
            'issues': []
        }
        coordinator.orchestrator.dataset_pipeline = mock_pipeline_instance
        
        # Mock version control
        coordinator.orchestrator.version_control = Mock()
        coordinator.orchestrator.version_control.create_version.return_value = "version_1"
        
        workflow = coordinator.run_complete_workflow(
            force_trigger=True,
            skip_validation=True,
            skip_regression=True
        )
        
        assert workflow['status'] in ['completed', 'training_pending', 'waiting_training']
        assert 'stages' in workflow
        assert 'trigger' in workflow['stages']
    
    def test_get_system_status(self, coordinator):
        """Test getting system status."""
        status = coordinator.get_system_status()
        
        assert 'timestamp' in status
        assert 'orchestrator' in status
        assert 'deployer' in status
        assert 'validation' in status
        assert 'regression' in status
        assert 'workflows' in status
    
    @patch('src.data.finetuning_orchestrator.FinetuningDatasetPipeline')
    def test_deploy_job_model(
        self, mock_pipeline, coordinator, temp_directory, mock_data_manager
    ):
        """Test deploying a model from a job."""
        # Create a job first
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.prepare_dataset.return_value = {
            'dataset_id': 'test_dataset',
            'split_sizes': {'train': 80, 'val': 10, 'test': 10}
        }
        mock_pipeline_instance.validate_dataset.return_value = {
            'is_valid': True,
            'issues': []
        }
        coordinator.orchestrator.dataset_pipeline = mock_pipeline_instance
        coordinator.orchestrator.version_control = Mock()
        coordinator.orchestrator.version_control.create_version.return_value = "version_1"
        
        job = coordinator.orchestrator.trigger_finetuning(force=True)
        
        # Create mock model path
        model_path = str(Path(temp_directory) / "test_model")
        Path(model_path).mkdir(parents=True)
        (Path(model_path) / "model.pt").write_text("mock model")
        
        # Deploy without validation/regression
        success = coordinator.deploy_job_model(
            job_id=job.job_id,
            model_path=model_path,
            run_validation=False,
            run_regression=False
        )
        
        assert coordinator.orchestrator.jobs[job.job_id].status == 'completed'


@pytest.mark.unit
def test_coordinator_integration(temp_directory):
    """Integration test for coordinator."""
    # Create real data manager
    from src.data.data_manager import DataManager
    
    data_manager = DataManager(
        local_storage_dir=str(Path(temp_directory) / "data"),
        use_gcs=False
    )
    
    # Add test cases
    for i in range(15):
        data_manager.store_failed_case(
            audio_path=f"test_{i}.wav",
            original_transcript=f"original {i}",
            corrected_transcript=f"corrected {i}",
            error_types=["test_error"],
            error_score=0.8
        )
    
    # Create coordinator
    with patch('src.data.finetuning_orchestrator.GCSManager'), \
         patch('src.data.model_validator.GCSManager'), \
         patch('src.data.model_deployer.GCSManager'), \
         patch('src.data.regression_tester.GCSManager'):
        
        coordinator = FinetuningCoordinator(
            data_manager=data_manager,
            finetuning_config=FinetuningConfig(
                min_error_cases=10,
                auto_approve_finetuning=True
            ),
            use_gcs=False,
            storage_dir=str(Path(temp_directory) / "coordinator")
        )
    
    # Check status
    status = coordinator.get_system_status()
    assert status is not None
    assert status['orchestrator']['trigger_conditions']['should_trigger'] is True
    
    # Test callbacks
    def mock_training(job, params):
        return True
    
    def mock_baseline_fn(audio_path):
        return "baseline transcript"
    
    def mock_model_factory(model_path):
        def transcribe(audio_path):
            return "model transcript"
        return transcribe
    
    coordinator.set_training_callback(mock_training)
    coordinator.set_baseline_transcribe_function(mock_baseline_fn)
    coordinator.set_model_transcribe_function_factory(mock_model_factory)
    
    assert coordinator.training_callback is not None
    assert coordinator.baseline_transcribe_fn is not None
    assert coordinator.model_transcribe_fn_factory is not None


@pytest.mark.unit  
def test_coordinator_workflow_stages(temp_directory):
    """Test individual workflow stages."""
    from src.data.data_manager import DataManager
    
    # Setup
    data_manager = DataManager(
        local_storage_dir=str(Path(temp_directory) / "data"),
        use_gcs=False
    )
    
    for i in range(20):
        data_manager.store_failed_case(
            audio_path=f"test_{i}.wav",
            original_transcript=f"original {i}",
            corrected_transcript=f"corrected {i}",
            error_types=["test"],
            error_score=0.8
        )
    
    with patch('src.data.finetuning_orchestrator.GCSManager'), \
         patch('src.data.model_validator.GCSManager'), \
         patch('src.data.model_deployer.GCSManager'), \
         patch('src.data.regression_tester.GCSManager'):
        
        coordinator = FinetuningCoordinator(
            data_manager=data_manager,
            finetuning_config=FinetuningConfig(
                min_error_cases=10,
                auto_approve_finetuning=True
            ),
            validation_config=ValidationConfig(
                min_wer_improvement=0.0,
                require_significance=False
            ),
            use_gcs=False,
            storage_dir=str(Path(temp_directory) / "coordinator")
        )
    
    # Stage 1: Check trigger
    trigger_result = coordinator.orchestrator.check_trigger_conditions()
    assert trigger_result['should_trigger'] is True
    
    # Stage 2: Register test for regression testing
    test_file = Path(temp_directory) / "test_data.jsonl"
    with open(test_file, 'w') as f:
        for i in range(5):
            f.write(json.dumps({
                'audio_path': f'test{i}.wav',
                'reference': f'test {i}'
            }) + '\n')
    
    test_id = coordinator.regression_tester.register_test(
        test_name="Test",
        test_type="benchmark",
        test_data_path=str(test_file),
        baseline_wer=0.15,
        baseline_cer=0.08,
        baseline_version="baseline"
    )
    
    assert test_id in coordinator.regression_tester.tests
    
    # Stage 3: Test deployment functions
    model_dir = Path(temp_directory) / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("model")
    
    version_id = coordinator.deployer.register_model(
        model_name="test",
        model_path=str(model_dir),
        validation_result={'passed': True}
    )
    
    assert version_id in coordinator.deployer.versions

