"""
Model Deployment System
Manages model versioning, deployment, and switching between model versions.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .metadata_tracker import MetadataTracker
from ..utils.gcs_utils import GCSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a deployed model version."""
    version_id: str
    model_name: str
    model_path: str
    status: str  # 'deployed', 'archived', 'testing', 'failed'
    
    # Performance metrics
    wer: Optional[float] = None
    cer: Optional[float] = None
    
    # Deployment metadata
    deployed_at: Optional[str] = None
    deployed_by: str = "system"
    parent_version: Optional[str] = None
    
    # Training info
    training_job_id: Optional[str] = None
    training_dataset_id: Optional[str] = None
    training_samples: int = 0
    
    # Validation info
    validation_passed: bool = False
    validation_timestamp: Optional[str] = None
    
    # Additional metadata
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        if self.metadata is None:
            result['metadata'] = {}
        return result


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    # Deployment strategy
    deployment_strategy: str = "replace"  # 'replace', 'canary', 'blue_green'
    
    # Backup settings
    keep_previous_versions: int = 5
    auto_backup_before_deploy: bool = True
    
    # Rollback settings
    enable_auto_rollback: bool = True
    rollback_on_error_threshold: float = 0.5  # Rollback if error rate > 50%
    
    # GCS settings
    sync_to_gcs: bool = True
    gcs_models_prefix: str = "deployed_models"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ModelDeployer:
    """
    Manages model deployment, versioning, and switching.
    
    Features:
    - Deploy new model versions
    - Switch between versions
    - Rollback to previous versions
    - Version history tracking
    - Backup and archival
    - GCS integration
    """
    
    def __init__(
        self,
        config: Optional[DeploymentConfig] = None,
        storage_dir: str = "data/deployed_models",
        use_gcs: bool = True,
        project_id: str = "stt-agentic-ai-2025"
    ):
        """
        Initialize model deployer.
        
        Args:
            config: Deployment configuration
            storage_dir: Local storage directory for models
            use_gcs: Whether to use Google Cloud Storage
            project_id: GCP project ID
        """
        self.config = config or DeploymentConfig()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Directories
        self.active_dir = self.storage_dir / "active"
        self.archive_dir = self.storage_dir / "archive"
        self.backup_dir = self.storage_dir / "backups"
        
        for dir_path in [self.active_dir, self.archive_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-systems
        self.metadata_tracker = MetadataTracker(
            use_gcs=use_gcs,
            project_id=project_id
        )
        
        # GCS integration
        self.use_gcs = use_gcs
        self.gcs_manager = None
        if use_gcs:
            try:
                self.gcs_manager = GCSManager(project_id, "stt-project-models")
                logger.info("GCS integration enabled for deployment")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS: {e}")
                self.use_gcs = False
        
        # Version tracking
        self.versions_file = self.storage_dir / "model_versions.json"
        self.versions: Dict[str, ModelVersion] = {}
        self.active_version_id: Optional[str] = None
        
        self._load_versions()
        
        logger.info("Model Deployer initialized")
        logger.info(f"  Active version: {self.active_version_id or 'None'}")
    
    def _load_versions(self):
        """Load version registry."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load versions
                    for version_id, version_data in data.get('versions', {}).items():
                        self.versions[version_id] = ModelVersion(**version_data)
                    
                    # Load active version
                    self.active_version_id = data.get('active_version')
                    
                logger.info(f"Loaded {len(self.versions)} model versions")
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
    
    def _save_versions(self):
        """Save version registry."""
        data = {
            'active_version': self.active_version_id,
            'versions': {
                version_id: version.to_dict()
                for version_id, version in self.versions.items()
            },
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            try:
                gcs_path = f"{self.config.gcs_models_prefix}/model_versions.json"
                self.gcs_manager.upload_file(str(self.versions_file), gcs_path)
            except Exception as e:
                logger.error(f"Failed to sync versions to GCS: {e}")
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        version_id: Optional[str] = None,
        training_job_id: Optional[str] = None,
        training_dataset_id: Optional[str] = None,
        validation_result: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Model name
            model_path: Path to model files
            version_id: Version ID (auto-generated if not provided)
            training_job_id: Training job ID
            training_dataset_id: Training dataset ID
            validation_result: Validation result
            metadata: Additional metadata
            
        Returns:
            Version ID
        """
        # Generate version ID if not provided
        if version_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_id = f"{model_name}_v{timestamp}"
        
        # Extract validation info
        validation_passed = False
        validation_timestamp = None
        wer = None
        cer = None
        
        if validation_result:
            validation_passed = validation_result.get('passed', False)
            validation_timestamp = validation_result.get('timestamp')
            wer = validation_result.get('model_wer')
            cer = validation_result.get('model_cer')
        
        # Create model version
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            model_path=model_path,
            status='registered',
            wer=wer,
            cer=cer,
            training_job_id=training_job_id,
            training_dataset_id=training_dataset_id,
            validation_passed=validation_passed,
            validation_timestamp=validation_timestamp,
            metadata=metadata or {}
        )
        
        # Register version
        self.versions[version_id] = version
        self._save_versions()
        
        # Record in metadata tracker
        self.metadata_tracker.record_model_version(
            version_id=version_id,
            model_name=model_name,
            training_data_size=version.training_samples,
            training_metadata={
                'training_job_id': training_job_id,
                'training_dataset_id': training_dataset_id,
                'model_path': model_path
            },
            performance_metrics={
                'wer': wer,
                'cer': cer,
                'validation_passed': validation_passed
            }
        )
        
        logger.info(f"Registered model version: {version_id}")
        return version_id
    
    def deploy_model(
        self,
        version_id: str,
        force: bool = False
    ) -> bool:
        """
        Deploy a model version.
        
        Args:
            version_id: Version to deploy
            force: Force deployment even if validation failed
            
        Returns:
            True if successful
        """
        version = self.versions.get(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False
        
        # Check validation
        if not version.validation_passed and not force:
            logger.error(f"Version {version_id} did not pass validation. Use force=True to deploy anyway.")
            return False
        
        try:
            logger.info(f"Deploying model version: {version_id}")
            
            # Backup current active version if exists
            if self.active_version_id and self.config.auto_backup_before_deploy:
                self._backup_version(self.active_version_id)
            
            # Copy model to active directory
            source_path = Path(version.model_path)
            if source_path.is_file():
                # Single file model
                dest_path = self.active_dir / source_path.name
                shutil.copy2(source_path, dest_path)
            elif source_path.is_dir():
                # Directory model
                dest_path = self.active_dir / version_id
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
            else:
                raise ValueError(f"Model path does not exist: {source_path}")
            
            # Update version status
            if self.active_version_id:
                prev_version = self.versions.get(self.active_version_id)
                if prev_version:
                    prev_version.status = 'archived'
            
            version.status = 'deployed'
            version.deployed_at = datetime.now().isoformat()
            version.parent_version = self.active_version_id
            
            # Update active version
            prev_active = self.active_version_id
            self.active_version_id = version_id
            self._save_versions()
            
            # Sync to GCS if enabled
            if self.use_gcs and self.config.sync_to_gcs:
                self._sync_model_to_gcs(version_id)
            
            # Record deployment in metadata
            self.metadata_tracker.record_learning_progress(
                stage='model_deployed',
                metrics={
                    'version_id': version_id,
                    'wer': version.wer,
                    'cer': version.cer
                },
                metadata={
                    'previous_version': prev_active,
                    'deployment_timestamp': version.deployed_at
                }
            )
            
            logger.info(f"✅ Successfully deployed version: {version_id}")
            
            # Clean up old versions
            self._cleanup_old_versions()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            version.status = 'failed'
            self._save_versions()
            return False
    
    def _backup_version(self, version_id: str):
        """Backup a model version."""
        version = self.versions.get(version_id)
        if not version:
            return
        
        try:
            backup_path = self.backup_dir / f"{version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            source_path = Path(version.model_path)
            
            if source_path.is_file():
                backup_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, backup_path / source_path.name)
            elif source_path.is_dir():
                shutil.copytree(source_path, backup_path)
            
            logger.info(f"Backed up version {version_id} to {backup_path}")
            
        except Exception as e:
            logger.warning(f"Failed to backup version {version_id}: {e}")
    
    def _cleanup_old_versions(self):
        """Clean up old archived versions beyond retention limit."""
        archived = [v for v in self.versions.values() if v.status == 'archived']
        archived.sort(key=lambda v: v.deployed_at or '', reverse=True)
        
        if len(archived) > self.config.keep_previous_versions:
            to_remove = archived[self.config.keep_previous_versions:]
            
            for version in to_remove:
                try:
                    # Move to archive directory
                    source_path = Path(version.model_path)
                    if source_path.exists():
                        archive_path = self.archive_dir / version.version_id
                        if source_path.is_dir():
                            if archive_path.exists():
                                shutil.rmtree(archive_path)
                            shutil.move(str(source_path), str(archive_path))
                        else:
                            archive_path.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(source_path), str(archive_path / source_path.name))
                    
                    logger.info(f"Archived old version: {version.version_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to archive version {version.version_id}: {e}")
    
    def rollback(self, target_version_id: Optional[str] = None) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            target_version_id: Version to rollback to (previous if not specified)
            
        Returns:
            True if successful
        """
        if not self.active_version_id:
            logger.error("No active version to rollback from")
            return False
        
        current_version = self.versions[self.active_version_id]
        
        # Determine target version
        if target_version_id is None:
            target_version_id = current_version.parent_version
        
        if not target_version_id:
            logger.error("No previous version to rollback to")
            return False
        
        target_version = self.versions.get(target_version_id)
        if not target_version:
            logger.error(f"Target version not found: {target_version_id}")
            return False
        
        logger.warning(f"🔄 Rolling back from {self.active_version_id} to {target_version_id}")
        
        # Deploy target version
        success = self.deploy_model(target_version_id, force=True)
        
        if success:
            # Mark current as failed
            current_version.status = 'failed'
            self._save_versions()
            
            logger.info(f"✅ Rollback successful to version: {target_version_id}")
        else:
            logger.error("❌ Rollback failed")
        
        return success
    
    def _sync_model_to_gcs(self, version_id: str):
        """Sync model to GCS."""
        if not self.gcs_manager:
            return
        
        version = self.versions.get(version_id)
        if not version:
            return
        
        try:
            gcs_prefix = f"{self.config.gcs_models_prefix}/{version_id}"
            source_path = Path(version.model_path)
            
            if source_path.is_file():
                gcs_path = f"{gcs_prefix}/{source_path.name}"
                self.gcs_manager.upload_file(str(source_path), gcs_path)
            elif source_path.is_dir():
                # Upload directory
                for file_path in source_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(source_path)
                        gcs_path = f"{gcs_prefix}/{relative_path}"
                        self.gcs_manager.upload_file(str(file_path), gcs_path)
            
            logger.info(f"Synced model {version_id} to GCS: gs://{self.gcs_manager.bucket_name}/{gcs_prefix}")
            
        except Exception as e:
            logger.error(f"Failed to sync model to GCS: {e}")
    
    def get_active_version(self) -> Optional[ModelVersion]:
        """Get currently active model version."""
        if self.active_version_id:
            return self.versions.get(self.active_version_id)
        return None
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        return self.versions.get(version_id)
    
    def list_versions(
        self,
        status: Optional[str] = None,
        limit: int = 20
    ) -> List[ModelVersion]:
        """
        List model versions.
        
        Args:
            status: Filter by status
            limit: Maximum number of versions
            
        Returns:
            List of ModelVersions
        """
        versions = list(self.versions.values())
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        # Sort by deployment time (newest first)
        versions.sort(key=lambda v: v.deployed_at or '', reverse=True)
        
        return versions[:limit]
    
    def get_deployment_history(self) -> List[Dict]:
        """Get deployment history."""
        deployed = [v for v in self.versions.values() if v.deployed_at]
        deployed.sort(key=lambda v: v.deployed_at, reverse=True)
        
        return [
            {
                'version_id': v.version_id,
                'model_name': v.model_name,
                'deployed_at': v.deployed_at,
                'status': v.status,
                'wer': v.wer,
                'cer': v.cer,
                'validation_passed': v.validation_passed
            }
            for v in deployed
        ]
    
    def generate_deployment_report(self) -> Dict:
        """Generate deployment status report."""
        active = self.get_active_version()
        
        versions_by_status = {}
        for version in self.versions.values():
            versions_by_status[version.status] = versions_by_status.get(version.status, 0) + 1
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'active_version': active.to_dict() if active else None,
            'total_versions': len(self.versions),
            'versions_by_status': versions_by_status,
            'deployment_history': self.get_deployment_history()[:10],
            'config': self.config.to_dict()
        }
        
        return report
    
    def print_status(self):
        """Print deployment status."""
        active = self.get_active_version()
        
        print("\n" + "="*80)
        print("MODEL DEPLOYMENT STATUS")
        print("="*80)
        
        if active:
            print(f"\n🟢 Active Version: {active.version_id}")
            print(f"   Model: {active.model_name}")
            print(f"   Deployed: {active.deployed_at}")
            print(f"   WER: {active.wer:.4f}" if active.wer else "   WER: N/A")
            print(f"   CER: {active.cer:.4f}" if active.cer else "   CER: N/A")
            print(f"   Validation: {'Passed ✅' if active.validation_passed else 'Failed ❌'}")
        else:
            print("\n⚠️  No active version")
        
        print(f"\nTotal Versions: {len(self.versions)}")
        
        versions_by_status = {}
        for version in self.versions.values():
            versions_by_status[version.status] = versions_by_status.get(version.status, 0) + 1
        
        print("\nVersions by Status:")
        for status, count in versions_by_status.items():
            print(f"  {status}: {count}")
        
        print("\nRecent Deployments:")
        for deployment in self.get_deployment_history()[:5]:
            print(f"  {deployment['version_id']}: {deployment['deployed_at']} ({deployment['status']})")
        
        print("="*80 + "\n")


