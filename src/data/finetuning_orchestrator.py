"""
Fine-Tuning Orchestration System
Automated pipeline that triggers fine-tuning when error cases accumulate.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
import time
from dataclasses import dataclass, asdict

from .data_manager import DataManager
from .finetuning_pipeline import FinetuningDatasetPipeline
from .version_control import DataVersionControl
from .metadata_tracker import MetadataTracker
from ..utils.gcs_utils import GCSManager
from .wandb_tracker import WandbTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinetuningConfig:
    """Configuration for fine-tuning orchestration."""
    # Trigger settings
    min_error_cases: int = 100  # Minimum cases before triggering
    min_corrected_cases: int = 50  # Minimum corrected cases
    trigger_on_error_rate: bool = True  # Trigger if error rate exceeds threshold
    error_rate_threshold: float = 0.15  # 15% error rate
    
    # Dataset settings
    min_error_score: float = 0.5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    balance_error_types: bool = True
    
    # Quality control
    min_dataset_quality: float = 0.90
    
    # Auto-approval
    auto_approve_finetuning: bool = False  # Require manual approval by default
    
    # W&B tracking
    use_wandb: bool = True  # Enable Weights & Biases tracking
    wandb_project: str = "stt-finetuning"  # W&B project name
    wandb_entity: Optional[str] = None  # W&B entity (username/team)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FinetuningJob:
    """Represents a fine-tuning job."""
    job_id: str
    status: str  # 'pending', 'preparing', 'ready', 'training', 'completed', 'failed'
    dataset_id: Optional[str] = None
    version_id: Optional[str] = None
    trigger_reason: str = ""
    trigger_metrics: Dict = None
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    config: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        if self.trigger_metrics is None:
            result['trigger_metrics'] = {}
        if self.config is None:
            result['config'] = {}
        return result


class FinetuningOrchestrator:
    """
    Orchestrates automated fine-tuning pipeline.
    
    Features:
    - Monitors error case accumulation
    - Automatically triggers fine-tuning when thresholds are met
    - Prepares training datasets
    - Manages fine-tuning jobs
    - Integrates with version control and metadata tracking
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        config: Optional[FinetuningConfig] = None,
        storage_dir: str = "data/finetuning_orchestration",
        use_gcs: bool = True,
        project_id: str = "stt-agentic-ai-2025"
    ):
        """
        Initialize fine-tuning orchestrator.
        
        Args:
            data_manager: DataManager instance
            config: Fine-tuning configuration
            storage_dir: Directory for orchestrator data
            use_gcs: Whether to use Google Cloud Storage
            project_id: GCP project ID
        """
        self.data_manager = data_manager
        self.config = config or FinetuningConfig()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-systems
        self.dataset_pipeline = FinetuningDatasetPipeline(
            data_manager=data_manager,
            use_gcs=use_gcs,
            project_id=project_id
        )
        
        self.version_control = DataVersionControl(
            use_gcs=use_gcs,
            project_id=project_id
        )
        
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
                logger.info("GCS integration enabled for orchestration")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS: {e}")
                self.use_gcs = False
        
        # Job tracking
        self.jobs_file = self.storage_dir / "finetuning_jobs.jsonl"
        self.jobs: Dict[str, FinetuningJob] = {}
        self._load_jobs()
        
        # Callbacks for custom training integration
        self.training_callback: Optional[Callable] = None
        
        # W&B tracking
        self.wandb_tracker = None
        if self.config.use_wandb:
            try:
                self.wandb_tracker = WandbTracker(
                    project_name=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    enabled=True,
                    config=self.config.to_dict()
                )
                logger.info(f"W&B tracking enabled for project: {self.config.wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.config.use_wandb = False
        
        logger.info("Fine-tuning Orchestrator initialized")
        logger.info(f"  Trigger threshold: {self.config.min_error_cases} error cases")
        logger.info(f"  Auto-approve: {self.config.auto_approve_finetuning}")
        logger.info(f"  W&B tracking: {self.config.use_wandb}")
    
    def _load_jobs(self):
        """Load job history."""
        if self.jobs_file.exists():
            with open(self.jobs_file, 'r') as f:
                for line in f:
                    if line.strip():
                        job_data = json.loads(line)
                        job = FinetuningJob(**job_data)
                        self.jobs[job.job_id] = job
            logger.info(f"Loaded {len(self.jobs)} job records")
    
    def _save_job(self, job: FinetuningJob):
        """Save job to persistent storage."""
        with open(self.jobs_file, 'a') as f:
            f.write(json.dumps(job.to_dict()) + '\n')
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            try:
                gcs_path = "orchestration/finetuning_jobs.jsonl"
                self.gcs_manager.upload_file(str(self.jobs_file), gcs_path)
            except Exception as e:
                logger.error(f"Failed to sync jobs to GCS: {e}")
    
    def check_trigger_conditions(self) -> Dict:
        """
        Check if fine-tuning should be triggered.
        
        Returns:
            Dictionary with trigger decision and metrics
        """
        stats = self.data_manager.get_statistics()
        
        # Calculate metrics
        total_cases = stats['total_failed_cases']
        corrected_cases = stats['corrected_cases']
        error_rate = total_cases / max(total_cases + 100, 1)  # Estimate
        
        # Check trigger conditions
        should_trigger = False
        reasons = []
        
        if total_cases >= self.config.min_error_cases:
            should_trigger = True
            reasons.append(f"Error cases ({total_cases}) >= threshold ({self.config.min_error_cases})")
        
        if corrected_cases >= self.config.min_corrected_cases:
            if not should_trigger:
                should_trigger = True
            reasons.append(f"Corrected cases ({corrected_cases}) >= threshold ({self.config.min_corrected_cases})")
        
        if self.config.trigger_on_error_rate and error_rate >= self.config.error_rate_threshold:
            if not should_trigger:
                should_trigger = True
            reasons.append(f"Error rate ({error_rate:.2%}) >= threshold ({self.config.error_rate_threshold:.2%})")
        
        result = {
            'should_trigger': should_trigger,
            'reasons': reasons,
            'metrics': {
                'total_error_cases': total_cases,
                'corrected_cases': corrected_cases,
                'uncorrected_cases': stats['uncorrected_cases'],
                'estimated_error_rate': error_rate,
                'error_type_distribution': stats['error_type_distribution']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if should_trigger:
            logger.info(f"✅ Fine-tuning trigger conditions met: {', '.join(reasons)}")
        else:
            logger.info(f"⏸️  Fine-tuning not triggered. Need {self.config.min_error_cases - total_cases} more error cases")
        
        return result
    
    def create_finetuning_job(
        self,
        trigger_reason: str = "manual",
        trigger_metrics: Optional[Dict] = None
    ) -> FinetuningJob:
        """
        Create a new fine-tuning job.
        
        Args:
            trigger_reason: Reason for triggering
            trigger_metrics: Metrics at trigger time
            
        Returns:
            FinetuningJob instance
        """
        job_id = f"ft_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = FinetuningJob(
            job_id=job_id,
            status='pending',
            trigger_reason=trigger_reason,
            trigger_metrics=trigger_metrics or {},
            created_at=datetime.now().isoformat(),
            config=self.config.to_dict()
        )
        
        self.jobs[job_id] = job
        self._save_job(job)
        
        logger.info(f"Created fine-tuning job: {job_id}")
        return job
    
    def prepare_dataset_for_job(self, job_id: str) -> bool:
        """
        Prepare training dataset for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if successful
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False
        
        try:
            logger.info(f"Preparing dataset for job {job_id}...")
            job.status = 'preparing'
            self._save_job(job)
            
            # Prepare dataset using pipeline
            dataset_info = self.dataset_pipeline.prepare_dataset(
                min_error_score=self.config.min_error_score,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                balance_error_types=self.config.balance_error_types
            )
            
            if 'error' in dataset_info:
                raise ValueError(f"Dataset preparation failed: {dataset_info['error']}")
            
            dataset_id = dataset_info['dataset_id']
            job.dataset_id = dataset_id
            
            # Validate dataset quality
            logger.info("Validating dataset quality...")
            validation_report = self.dataset_pipeline.validate_dataset(dataset_id)
            
            if not validation_report.get('is_valid'):
                logger.warning(f"Dataset validation issues: {validation_report.get('issues')}")
            
            # Create dataset version
            logger.info("Creating dataset version...")
            dataset_path = self.dataset_pipeline.output_dir / dataset_id
            version_id = self.version_control.create_version(
                dataset_path=str(dataset_path),
                version_name=f"finetuning_{job_id}",
                metadata={
                    'job_id': job_id,
                    'dataset_info': dataset_info,
                    'validation_report': validation_report
                },
                run_quality_check=True
            )
            
            job.version_id = version_id
            job.status = 'ready'
            self._save_job(job)
            
            logger.info(f"✅ Dataset prepared: {dataset_id} (version: {version_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            job.status = 'failed'
            job.error_message = str(e)
            self._save_job(job)
            return False
    
    def trigger_finetuning(
        self,
        force: bool = False,
        manual_approval: bool = None
    ) -> Optional[FinetuningJob]:
        """
        Check conditions and trigger fine-tuning if appropriate.
        
        Args:
            force: Force trigger regardless of conditions
            manual_approval: Override auto-approval setting
            
        Returns:
            FinetuningJob if triggered, None otherwise
        """
        # Check trigger conditions
        trigger_result = self.check_trigger_conditions()
        
        if not force and not trigger_result['should_trigger']:
            logger.info("Fine-tuning not triggered - conditions not met")
            return None
        
        # Check approval requirement
        needs_approval = manual_approval if manual_approval is not None else not self.config.auto_approve_finetuning
        
        if needs_approval and not force:
            logger.warning("⚠️  Fine-tuning requires manual approval!")
            logger.warning("   Call trigger_finetuning(force=True) to approve")
            return None
        
        # Create job
        trigger_reason = "forced" if force else ", ".join(trigger_result['reasons'])
        job = self.create_finetuning_job(
            trigger_reason=trigger_reason,
            trigger_metrics=trigger_result['metrics']
        )
        
        # Prepare dataset
        if not self.prepare_dataset_for_job(job.job_id):
            return job
        
        # Record in metadata tracker
        self.metadata_tracker.record_learning_progress(
            stage='finetuning_triggered',
            metrics=trigger_result['metrics'],
            metadata={'job_id': job.job_id}
        )
        
        # Start W&B run
        if self.wandb_tracker:
            self.wandb_tracker.start_run(
                run_name=f"finetuning_{job.job_id}",
                job_id=job.job_id,
                tags=['finetuning', 'automated'],
                config=self.config.to_dict()
            )
            
            # Log dataset info if available
            job_info = self.get_job_info(job.job_id)
            if job_info and 'dataset_info' in job_info:
                dataset_info = job_info['dataset_info']
                self.wandb_tracker.log_dataset_info(
                    dataset_id=job.dataset_id,
                    split_sizes=dataset_info.get('split_sizes', {}),
                    error_type_distribution=dataset_info.get('error_type_distribution', {})
                )
            
            # Log system metrics
            stats = self.data_manager.get_statistics()
            self.wandb_tracker.log_system_metrics(
                error_cases=stats['total_failed_cases'],
                corrected_cases=stats['corrected_cases'],
                correction_rate=stats['correction_rate'],
                models_deployed=len(self.jobs)
            )
        
        logger.info(f"🚀 Fine-tuning triggered! Job ID: {job.job_id}")
        
        return job
    
    def start_training(self, job_id: str, training_params: Optional[Dict] = None) -> bool:
        """
        Start training for a job.
        
        Args:
            job_id: Job ID
            training_params: Training parameters (model-specific)
            
        Returns:
            True if training started successfully
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False
        
        if job.status != 'ready':
            logger.error(f"Job not ready for training. Status: {job.status}")
            return False
        
        try:
            job.status = 'training'
            job.started_at = datetime.now().isoformat()
            self._save_job(job)
            
            logger.info(f"Starting training for job {job_id}...")
            
            # If custom training callback is provided, use it
            if self.training_callback:
                logger.info("Using custom training callback")
                result = self.training_callback(job, training_params)
                if result:
                    job.status = 'completed'
                    job.completed_at = datetime.now().isoformat()
                else:
                    job.status = 'failed'
                    job.error_message = "Training callback returned failure"
            else:
                # Default: Mark as ready for external training
                logger.info("⚠️  No training callback configured")
                logger.info("   Job is ready for training. Use get_job_info() to get dataset path")
                logger.info("   Then call complete_training() after training finishes")
            
            self._save_job(job)
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            job.status = 'failed'
            job.error_message = str(e)
            self._save_job(job)
            return False
    
    def complete_training(
        self,
        job_id: str,
        model_path: str,
        training_metrics: Optional[Dict] = None
    ) -> bool:
        """
        Mark training as complete and register model.
        
        Args:
            job_id: Job ID
            model_path: Path to trained model
            training_metrics: Training metrics
            
        Returns:
            True if successful
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False
        
        try:
            job.status = 'completed'
            job.completed_at = datetime.now().isoformat()
            self._save_job(job)
            
            # Record model version in metadata tracker
            model_version_id = f"model_{job_id}"
            self.metadata_tracker.record_model_version(
                version_id=model_version_id,
                model_name="fine-tuned-stt",
                training_data_size=len(self.data_manager.get_corrected_cases()),
                training_metadata={
                    'job_id': job_id,
                    'dataset_id': job.dataset_id,
                    'version_id': job.version_id,
                    'model_path': model_path
                },
                performance_metrics=training_metrics
            )
            
            # Log training completion to W&B
            if self.wandb_tracker and training_metrics:
                self.wandb_tracker.log_metrics({
                    'training/final_loss': training_metrics.get('loss', 0),
                    'training/final_wer': training_metrics.get('wer', 0),
                    'training/final_cer': training_metrics.get('cer', 0),
                    'training/epochs': training_metrics.get('epochs', 0),
                    'training/duration': training_metrics.get('duration', 0)
                })
                
                # Log model artifact
                self.wandb_tracker.log_model_artifact(
                    model_path=model_path,
                    model_name=f"model_{job_id}",
                    metadata={
                        'job_id': job_id,
                        'dataset_id': job.dataset_id,
                        'training_metrics': training_metrics
                    }
                )
            
            logger.info(f"✅ Training completed for job {job_id}")
            logger.info(f"   Model: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete training: {e}")
            return False
    
    def get_job_info(self, job_id: str) -> Optional[Dict]:
        """Get detailed information about a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        info = job.to_dict()
        
        # Add dataset info if available
        if job.dataset_id:
            dataset_info = self.dataset_pipeline.get_dataset_info(job.dataset_id)
            info['dataset_info'] = dataset_info
        
        # Add version info if available
        if job.version_id:
            version = self.version_control.get_version(job.version_id)
            if version:
                info['version_info'] = version.to_dict()
        
        return info
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        List fine-tuning jobs.
        
        Args:
            status: Filter by status
            limit: Maximum number of jobs to return
            
        Returns:
            List of job dictionaries
        """
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return [j.to_dict() for j in jobs[:limit]]
    
    def get_orchestration_status(self) -> Dict:
        """Get overall orchestration status."""
        trigger_result = self.check_trigger_conditions()
        
        jobs_by_status = {}
        for job in self.jobs.values():
            jobs_by_status[job.status] = jobs_by_status.get(job.status, 0) + 1
        
        return {
            'trigger_conditions': trigger_result,
            'config': self.config.to_dict(),
            'total_jobs': len(self.jobs),
            'jobs_by_status': jobs_by_status,
            'latest_jobs': self.list_jobs(limit=5),
            'timestamp': datetime.now().isoformat()
        }
    
    def set_training_callback(self, callback: Callable):
        """
        Set custom training callback function.
        
        Args:
            callback: Function(job, training_params) -> bool
        """
        self.training_callback = callback
        logger.info("Training callback registered")
    
    def run_monitoring_loop(
        self,
        check_interval_seconds: int = 3600,
        max_iterations: Optional[int] = None
    ):
        """
        Run continuous monitoring loop to auto-trigger fine-tuning.
        
        Args:
            check_interval_seconds: How often to check conditions
            max_iterations: Maximum iterations (None for infinite)
        """
        logger.info("🔄 Starting fine-tuning monitoring loop...")
        logger.info(f"   Check interval: {check_interval_seconds}s")
        
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            try:
                logger.info(f"Checking trigger conditions (iteration {iteration + 1})...")
                
                # Check and potentially trigger
                job = self.trigger_finetuning(force=False)
                
                if job:
                    logger.info(f"✅ Triggered fine-tuning job: {job.job_id}")
                    
                    # Start training if callback is configured
                    if self.training_callback:
                        self.start_training(job.job_id)
                
                # Wait for next check
                logger.info(f"Waiting {check_interval_seconds}s until next check...")
                time.sleep(check_interval_seconds)
                
                iteration += 1
                
            except KeyboardInterrupt:
                logger.info("Monitoring loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval_seconds)
                continue
        
        logger.info("Monitoring loop terminated")


