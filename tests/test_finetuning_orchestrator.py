"""
Unit tests for Fine-Tuning Orchestrator
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.data.finetuning_orchestrator import (
    FinetuningOrchestrator,
    FinetuningConfig,
    FinetuningJob
)
from src.data.data_manager import DataManager


@pytest.fixture
def mock_data_manager(temp_directory):
    """Create mock data manager."""
    manager = Mock(spec=DataManager)
    manager.get_statistics.return_value = {
        'total_failed_cases': 50,
        'corrected_cases': 30,
        'uncorrected_cases': 20,
        'correction_rate': 0.6,
        'error_type_distribution': {'word_substitution': 25, 'missing_word': 15}
    }
    manager.get_corrected_cases.return_value = []
    manager.failed_cases_cache = {}
    return manager


@pytest.fixture
def orchestrator(mock_data_manager, temp_directory):
    """Create orchestrator instance."""
    config = FinetuningConfig(
        min_error_cases=10,
        min_corrected_cases=5,
        auto_approve_finetuning=True
    )
    
    with patch('src.data.finetuning_orchestrator.GCSManager'):
        orch = FinetuningOrchestrator(
            data_manager=mock_data_manager,
            config=config,
            storage_dir=str(Path(temp_directory) / "orchestration"),
            use_gcs=False
        )
    
    return orch


class TestFinetuningConfig:
    """Test FinetuningConfig."""
    
    def test_config_creation(self):
        """Test config creation with default values."""
        config = FinetuningConfig()
        assert config.min_error_cases == 100
        assert config.min_corrected_cases == 50
        assert config.auto_approve_finetuning is False
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = FinetuningConfig(min_error_cases=50)
        config_dict = config.to_dict()
        assert config_dict['min_error_cases'] == 50
        assert 'auto_approve_finetuning' in config_dict


class TestFinetuningJob:
    """Test FinetuningJob."""
    
    def test_job_creation(self):
        """Test job creation."""
        job = FinetuningJob(
            job_id="test_job_1",
            status="pending",
            trigger_reason="manual"
        )
        assert job.job_id == "test_job_1"
        assert job.status == "pending"
    
    def test_job_to_dict(self):
        """Test job serialization."""
        job = FinetuningJob(
            job_id="test_job_1",
            status="pending",
            trigger_reason="manual",
            trigger_metrics={'total_cases': 100}
        )
        job_dict = job.to_dict()
        assert job_dict['job_id'] == "test_job_1"
        assert job_dict['trigger_metrics']['total_cases'] == 100


class TestFinetuningOrchestrator:
    """Test FinetuningOrchestrator."""
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert orchestrator.config.min_error_cases == 10
        assert len(orchestrator.jobs) == 0
    
    def test_check_trigger_conditions_met(self, orchestrator, mock_data_manager):
        """Test trigger conditions when met."""
        mock_data_manager.get_statistics.return_value = {
            'total_failed_cases': 100,
            'corrected_cases': 50,
            'uncorrected_cases': 50,
            'correction_rate': 0.5,
            'error_type_distribution': {}
        }
        
        result = orchestrator.check_trigger_conditions()
        
        assert result['should_trigger'] is True
        assert len(result['reasons']) > 0
        assert 'metrics' in result
    
    def test_check_trigger_conditions_not_met(self, orchestrator, mock_data_manager):
        """Test trigger conditions when not met."""
        mock_data_manager.get_statistics.return_value = {
            'total_failed_cases': 5,
            'corrected_cases': 2,
            'uncorrected_cases': 3,
            'correction_rate': 0.4,
            'error_type_distribution': {}
        }
        
        result = orchestrator.check_trigger_conditions()
        
        assert result['should_trigger'] is False
    
    def test_create_finetuning_job(self, orchestrator):
        """Test job creation."""
        job = orchestrator.create_finetuning_job(
            trigger_reason="test",
            trigger_metrics={'total_cases': 100}
        )
        
        assert job is not None
        assert job.status == "pending"
        assert job.trigger_reason == "test"
        assert job.job_id in orchestrator.jobs
    
    @patch('src.data.finetuning_orchestrator.FinetuningDatasetPipeline')
    def test_prepare_dataset_for_job(self, mock_pipeline, orchestrator):
        """Test dataset preparation."""
        # Create a job first
        job = orchestrator.create_finetuning_job("test")
        
        # Mock the pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.prepare_dataset.return_value = {
            'dataset_id': 'test_dataset',
            'split_sizes': {'train': 80, 'val': 10, 'test': 10}
        }
        mock_pipeline_instance.validate_dataset.return_value = {
            'is_valid': True,
            'issues': []
        }
        orchestrator.dataset_pipeline = mock_pipeline_instance
        
        # Mock version control
        orchestrator.version_control = Mock()
        orchestrator.version_control.create_version.return_value = "version_1"
        
        success = orchestrator.prepare_dataset_for_job(job.job_id)
        
        assert success is True
        assert orchestrator.jobs[job.job_id].dataset_id == 'test_dataset'
        assert orchestrator.jobs[job.job_id].status == 'ready'
    
    def test_trigger_finetuning_conditions_not_met(self, orchestrator, mock_data_manager):
        """Test trigger when conditions not met."""
        mock_data_manager.get_statistics.return_value = {
            'total_failed_cases': 5,
            'corrected_cases': 2,
            'uncorrected_cases': 3,
            'correction_rate': 0.4,
            'error_type_distribution': {}
        }
        
        job = orchestrator.trigger_finetuning(force=False)
        
        assert job is None
    
    @patch('src.data.finetuning_orchestrator.FinetuningDatasetPipeline')
    def test_trigger_finetuning_force(self, mock_pipeline, orchestrator, mock_data_manager):
        """Test forced trigger."""
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
        orchestrator.dataset_pipeline = mock_pipeline_instance
        
        # Mock version control
        orchestrator.version_control = Mock()
        orchestrator.version_control.create_version.return_value = "version_1"
        
        job = orchestrator.trigger_finetuning(force=True)
        
        assert job is not None
        assert job.status == 'ready'
    
    def test_get_job_info(self, orchestrator):
        """Test getting job info."""
        job = orchestrator.create_finetuning_job("test")
        
        info = orchestrator.get_job_info(job.job_id)
        
        assert info is not None
        assert info['job_id'] == job.job_id
    
    def test_list_jobs(self, orchestrator):
        """Test listing jobs."""
        # Create some jobs
        job1 = orchestrator.create_finetuning_job("test1")
        job2 = orchestrator.create_finetuning_job("test2")
        
        jobs = orchestrator.list_jobs(limit=10)
        
        assert len(jobs) == 2
        assert any(j['job_id'] == job1.job_id for j in jobs)
    
    def test_list_jobs_with_status_filter(self, orchestrator):
        """Test listing jobs with status filter."""
        job1 = orchestrator.create_finetuning_job("test1")
        job2 = orchestrator.create_finetuning_job("test2")
        job2.status = 'completed'
        
        pending_jobs = orchestrator.list_jobs(status='pending')
        
        assert len(pending_jobs) == 1
        assert pending_jobs[0]['job_id'] == job1.job_id
    
    def test_get_orchestration_status(self, orchestrator):
        """Test getting orchestration status."""
        status = orchestrator.get_orchestration_status()
        
        assert 'trigger_conditions' in status
        assert 'config' in status
        assert 'total_jobs' in status
    
    def test_start_training_no_callback(self, orchestrator):
        """Test starting training without callback."""
        job = orchestrator.create_finetuning_job("test")
        job.status = 'ready'
        
        success = orchestrator.start_training(job.job_id)
        
        assert success is True
    
    def test_start_training_with_callback(self, orchestrator):
        """Test starting training with callback."""
        # Set callback
        mock_callback = Mock(return_value=True)
        orchestrator.set_training_callback(mock_callback)
        
        job = orchestrator.create_finetuning_job("test")
        job.status = 'ready'
        
        success = orchestrator.start_training(job.job_id)
        
        assert success is True
        assert mock_callback.called
    
    def test_complete_training(self, orchestrator):
        """Test completing training."""
        job = orchestrator.create_finetuning_job("test")
        
        success = orchestrator.complete_training(
            job.job_id,
            model_path="/path/to/model",
            training_metrics={'loss': 0.5}
        )
        
        assert success is True
        assert orchestrator.jobs[job.job_id].status == 'completed'


@pytest.mark.unit
def test_orchestrator_integration(temp_directory):
    """Integration test for orchestrator workflow."""
    # Create real data manager with test data
    from src.data.data_manager import DataManager
    
    data_manager = DataManager(
        local_storage_dir=str(Path(temp_directory) / "data"),
        use_gcs=False
    )
    
    # Add some test cases
    for i in range(15):
        data_manager.store_failed_case(
            audio_path=f"test_{i}.wav",
            original_transcript=f"original {i}",
            corrected_transcript=f"corrected {i}",
            error_types=["test_error"],
            error_score=0.8
        )
    
    # Create orchestrator
    config = FinetuningConfig(
        min_error_cases=10,
        auto_approve_finetuning=True
    )
    
    with patch('src.data.finetuning_orchestrator.GCSManager'):
        orchestrator = FinetuningOrchestrator(
            data_manager=data_manager,
            config=config,
            storage_dir=str(Path(temp_directory) / "orchestration"),
            use_gcs=False
        )
    
    # Check trigger conditions
    result = orchestrator.check_trigger_conditions()
    assert result['should_trigger'] is True
    
    # Check that jobs can be created
    job = orchestrator.create_finetuning_job("integration_test")
    assert job is not None
    assert job.job_id in orchestrator.jobs

