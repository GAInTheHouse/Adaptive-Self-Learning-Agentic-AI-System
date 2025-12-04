"""
Unit tests for Model Deployer
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.model_deployer import (
    ModelDeployer,
    DeploymentConfig,
    ModelVersion
)


@pytest.fixture
def deployer(temp_directory):
    """Create deployer instance."""
    config = DeploymentConfig(
        keep_previous_versions=3,
        auto_backup_before_deploy=True
    )
    
    with patch('src.data.model_deployer.GCSManager'):
        deployer = ModelDeployer(
            config=config,
            storage_dir=str(Path(temp_directory) / "deployed_models"),
            use_gcs=False
        )
    
    return deployer


@pytest.fixture
def mock_model_path(temp_directory):
    """Create mock model path."""
    model_dir = Path(temp_directory) / "test_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy model file
    model_file = model_dir / "model.pt"
    model_file.write_text("mock model data")
    
    return str(model_dir)


class TestDeploymentConfig:
    """Test DeploymentConfig."""
    
    def test_config_creation(self):
        """Test config creation."""
        config = DeploymentConfig()
        assert config.deployment_strategy == "replace"
        assert config.keep_previous_versions == 5
        assert config.auto_backup_before_deploy is True
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = DeploymentConfig(keep_previous_versions=3)
        config_dict = config.to_dict()
        assert config_dict['keep_previous_versions'] == 3


class TestModelVersion:
    """Test ModelVersion."""
    
    def test_version_creation(self):
        """Test version creation."""
        version = ModelVersion(
            version_id="v1",
            model_name="test-model",
            model_path="/path/to/model",
            status="deployed",
            wer=0.15,
            cer=0.08
        )
        
        assert version.version_id == "v1"
        assert version.status == "deployed"
        assert version.wer == 0.15
    
    def test_version_to_dict(self):
        """Test version serialization."""
        version = ModelVersion(
            version_id="v1",
            model_name="test-model",
            model_path="/path/to/model",
            status="deployed"
        )
        
        version_dict = version.to_dict()
        assert version_dict['version_id'] == "v1"
        assert version_dict['status'] == "deployed"


class TestModelDeployer:
    """Test ModelDeployer."""
    
    def test_initialization(self, deployer):
        """Test deployer initialization."""
        assert deployer is not None
        assert deployer.config.keep_previous_versions == 3
        assert len(deployer.versions) == 0
    
    def test_register_model(self, deployer, mock_model_path):
        """Test model registration."""
        version_id = deployer.register_model(
            model_name="test-model",
            model_path=mock_model_path,
            validation_result={'passed': True, 'model_wer': 0.15}
        )
        
        assert version_id is not None
        assert version_id in deployer.versions
        assert deployer.versions[version_id].model_name == "test-model"
    
    def test_register_model_with_metrics(self, deployer, mock_model_path):
        """Test model registration with metrics."""
        validation_result = {
            'passed': True,
            'model_wer': 0.15,
            'model_cer': 0.08,
            'timestamp': '2024-01-01T00:00:00'
        }
        
        version_id = deployer.register_model(
            model_name="test-model",
            model_path=mock_model_path,
            training_job_id="job123",
            training_dataset_id="dataset456",
            validation_result=validation_result
        )
        
        version = deployer.versions[version_id]
        assert version.wer == 0.15
        assert version.cer == 0.08
        assert version.validation_passed is True
    
    def test_deploy_model(self, deployer, mock_model_path):
        """Test model deployment."""
        # Register model first
        version_id = deployer.register_model(
            model_name="test-model",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        
        # Deploy
        success = deployer.deploy_model(version_id)
        
        assert success is True
        assert deployer.active_version_id == version_id
        assert deployer.versions[version_id].status == "deployed"
    
    def test_deploy_model_without_validation(self, deployer, mock_model_path):
        """Test deployment without validation should fail."""
        # Register model without validation
        version_id = deployer.register_model(
            model_name="test-model",
            model_path=mock_model_path,
            validation_result={'passed': False}
        )
        
        # Try to deploy (should fail without force)
        success = deployer.deploy_model(version_id, force=False)
        
        assert success is False
    
    def test_deploy_model_with_force(self, deployer, mock_model_path):
        """Test forced deployment."""
        # Register model without validation
        version_id = deployer.register_model(
            model_name="test-model",
            model_path=mock_model_path,
            validation_result={'passed': False}
        )
        
        # Force deploy
        success = deployer.deploy_model(version_id, force=True)
        
        assert success is True
        assert deployer.active_version_id == version_id
    
    def test_get_active_version(self, deployer, mock_model_path):
        """Test getting active version."""
        # Initially no active version
        active = deployer.get_active_version()
        assert active is None
        
        # Deploy a model
        version_id = deployer.register_model(
            model_name="test-model",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        deployer.deploy_model(version_id)
        
        # Now should have active version
        active = deployer.get_active_version()
        assert active is not None
        assert active.version_id == version_id
    
    def test_list_versions(self, deployer, mock_model_path):
        """Test listing versions."""
        # Register multiple versions
        v1 = deployer.register_model(
            model_name="model-v1",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        v2 = deployer.register_model(
            model_name="model-v2",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        
        versions = deployer.list_versions()
        
        assert len(versions) == 2
        assert any(v.version_id == v1 for v in versions)
        assert any(v.version_id == v2 for v in versions)
    
    def test_list_versions_with_status_filter(self, deployer, mock_model_path):
        """Test listing versions with status filter."""
        # Register and deploy one
        v1 = deployer.register_model(
            model_name="model-v1",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        deployer.deploy_model(v1)
        
        # Register another without deploying
        v2 = deployer.register_model(
            model_name="model-v2",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        
        deployed_versions = deployer.list_versions(status='deployed')
        
        assert len(deployed_versions) == 1
        assert deployed_versions[0].version_id == v1
    
    def test_rollback(self, deployer, mock_model_path):
        """Test rollback functionality."""
        # Deploy first version
        v1 = deployer.register_model(
            model_name="model-v1",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        deployer.deploy_model(v1)
        
        # Deploy second version
        v2 = deployer.register_model(
            model_name="model-v2",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        deployer.deploy_model(v2)
        
        # Rollback
        success = deployer.rollback()
        
        assert success is True
        assert deployer.active_version_id == v1
    
    def test_rollback_with_target(self, deployer, mock_model_path):
        """Test rollback to specific version."""
        # Deploy multiple versions
        v1 = deployer.register_model(
            model_name="model-v1",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        deployer.deploy_model(v1)
        
        v2 = deployer.register_model(
            model_name="model-v2",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        deployer.deploy_model(v2)
        
        v3 = deployer.register_model(
            model_name="model-v3",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        deployer.deploy_model(v3)
        
        # Rollback to v1
        success = deployer.rollback(target_version_id=v1)
        
        assert success is True
        assert deployer.active_version_id == v1
    
    def test_get_deployment_history(self, deployer, mock_model_path):
        """Test getting deployment history."""
        # Deploy multiple versions
        for i in range(3):
            v = deployer.register_model(
                model_name=f"model-v{i}",
                model_path=mock_model_path,
                validation_result={'passed': True}
            )
            deployer.deploy_model(v)
        
        history = deployer.get_deployment_history()
        
        assert len(history) == 3
        assert all('version_id' in h for h in history)
        assert all('deployed_at' in h for h in history)
    
    def test_generate_deployment_report(self, deployer, mock_model_path):
        """Test deployment report generation."""
        # Deploy a model
        v1 = deployer.register_model(
            model_name="test-model",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        deployer.deploy_model(v1)
        
        report = deployer.generate_deployment_report()
        
        assert 'generated_at' in report
        assert 'active_version' in report
        assert 'total_versions' in report
        assert report['active_version'] is not None
    
    def test_get_version(self, deployer, mock_model_path):
        """Test getting specific version."""
        version_id = deployer.register_model(
            model_name="test-model",
            model_path=mock_model_path,
            validation_result={'passed': True}
        )
        
        version = deployer.get_version(version_id)
        
        assert version is not None
        assert version.version_id == version_id


@pytest.mark.unit
def test_deployer_integration(temp_directory):
    """Integration test for deployer."""
    # Create real model files
    model_dir = Path(temp_directory) / "models"
    model_dir.mkdir(parents=True)
    
    model_v1 = model_dir / "model_v1"
    model_v1.mkdir()
    (model_v1 / "model.pt").write_text("model v1 data")
    
    model_v2 = model_dir / "model_v2"
    model_v2.mkdir()
    (model_v2 / "model.pt").write_text("model v2 data")
    
    # Create deployer
    config = DeploymentConfig(
        keep_previous_versions=2,
        auto_backup_before_deploy=True
    )
    
    with patch('src.data.model_deployer.GCSManager'):
        deployer = ModelDeployer(
            config=config,
            storage_dir=str(Path(temp_directory) / "deployed"),
            use_gcs=False
        )
    
    # Register and deploy v1
    v1 = deployer.register_model(
        model_name="integration-model",
        model_path=str(model_v1),
        validation_result={'passed': True, 'model_wer': 0.20}
    )
    deployer.deploy_model(v1)
    
    assert deployer.active_version_id == v1
    
    # Register and deploy v2
    v2 = deployer.register_model(
        model_name="integration-model",
        model_path=str(model_v2),
        validation_result={'passed': True, 'model_wer': 0.15}
    )
    deployer.deploy_model(v2)
    
    assert deployer.active_version_id == v2
    
    # Rollback to v1
    deployer.rollback()
    assert deployer.active_version_id == v1

