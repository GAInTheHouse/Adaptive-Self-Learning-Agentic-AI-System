"""
Fine-Tuning Coordinator
Central coordinator that orchestrates the entire fine-tuning lifecycle:
1. Monitor and trigger fine-tuning
2. Validate trained models
3. Deploy validated models
4. Run regression tests
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime

from .finetuning_orchestrator import (
    FinetuningOrchestrator,
    FinetuningConfig,
    FinetuningJob
)
from .model_validator import (
    ModelValidator,
    ValidationConfig,
    ValidationResult
)
from .model_deployer import (
    ModelDeployer,
    DeploymentConfig,
    ModelVersion
)
from .regression_tester import (
    RegressionTester,
    RegressionConfig,
    RegressionTestResult
)
from .data_manager import DataManager
from .metadata_tracker import MetadataTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinetuningCoordinator:
    """
    Central coordinator for the complete fine-tuning lifecycle.
    
    Workflow:
    1. Monitor error cases → Trigger fine-tuning when threshold met
    2. Prepare dataset and train model
    3. Validate model against baseline
    4. Run regression tests
    5. Deploy if all checks pass
    6. Continuous monitoring
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        finetuning_config: Optional[FinetuningConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
        deployment_config: Optional[DeploymentConfig] = None,
        regression_config: Optional[RegressionConfig] = None,
        use_gcs: bool = True,
        project_id: str = "stt-agentic-ai-2025",
        storage_dir: str = "data/orchestration"
    ):
        """
        Initialize fine-tuning coordinator.
        
        Args:
            data_manager: DataManager instance
            finetuning_config: Fine-tuning configuration
            validation_config: Validation configuration
            deployment_config: Deployment configuration
            regression_config: Regression testing configuration
            use_gcs: Whether to use Google Cloud Storage
            project_id: GCP project ID
            storage_dir: Storage directory for coordinator data
        """
        self.data_manager = data_manager
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize subsystems
        self.orchestrator = FinetuningOrchestrator(
            data_manager=data_manager,
            config=finetuning_config,
            use_gcs=use_gcs,
            project_id=project_id
        )
        
        self.validator = ModelValidator(
            config=validation_config,
            use_gcs=use_gcs,
            project_id=project_id
        )
        
        self.deployer = ModelDeployer(
            config=deployment_config,
            use_gcs=use_gcs,
            project_id=project_id
        )
        
        self.regression_tester = RegressionTester(
            config=regression_config,
            use_gcs=use_gcs,
            project_id=project_id
        )
        
        self.metadata_tracker = MetadataTracker(
            use_gcs=use_gcs,
            project_id=project_id
        )
        
        # Workflow state
        self.workflows_file = self.storage_dir / "workflows.jsonl"
        self.workflows: List[Dict] = []
        self._load_workflows()
        
        # Callbacks
        self.training_callback: Optional[Callable] = None
        self.baseline_transcribe_fn: Optional[Callable] = None
        self.model_transcribe_fn_factory: Optional[Callable] = None
        
        logger.info("="*80)
        logger.info("Fine-Tuning Coordinator Initialized")
        logger.info("="*80)
        logger.info("Components:")
        logger.info("  ✓ Fine-Tuning Orchestrator")
        logger.info("  ✓ Model Validator")
        logger.info("  ✓ Model Deployer")
        logger.info("  ✓ Regression Tester")
        logger.info("  ✓ Metadata Tracker")
        logger.info("="*80)
    
    def _load_workflows(self):
        """Load workflow history."""
        if self.workflows_file.exists():
            with open(self.workflows_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.workflows.append(json.loads(line))
            logger.info(f"Loaded {len(self.workflows)} workflow records")
    
    def _save_workflow(self, workflow: Dict):
        """Save workflow record."""
        with open(self.workflows_file, 'a') as f:
            f.write(json.dumps(workflow) + '\n')
    
    def set_training_callback(self, callback: Callable):
        """
        Set custom training callback.
        
        Args:
            callback: Function(job, training_params) -> bool
        """
        self.training_callback = callback
        self.orchestrator.set_training_callback(callback)
        logger.info("Training callback registered")
    
    def set_baseline_transcribe_function(self, func: Callable):
        """
        Set baseline transcription function for validation.
        
        Args:
            func: Function(audio_path) -> str or Dict
        """
        self.baseline_transcribe_fn = func
        logger.info("Baseline transcription function registered")
    
    def set_model_transcribe_function_factory(self, factory: Callable):
        """
        Set factory function that creates transcription functions for models.
        
        Args:
            factory: Function(model_path) -> transcribe_function
        """
        self.model_transcribe_fn_factory = factory
        logger.info("Model transcription function factory registered")
    
    def run_complete_workflow(
        self,
        force_trigger: bool = False,
        model_path: Optional[str] = None,
        training_params: Optional[Dict] = None,
        skip_validation: bool = False,
        skip_regression: bool = False,
        auto_deploy: bool = False
    ) -> Dict:
        """
        Run complete fine-tuning workflow.
        
        Args:
            force_trigger: Force trigger even if conditions not met
            model_path: Path to trained model (if training done externally)
            training_params: Training parameters
            skip_validation: Skip validation step
            skip_regression: Skip regression testing
            auto_deploy: Automatically deploy if all checks pass
            
        Returns:
            Workflow result summary
        """
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("="*80)
        logger.info(f"🚀 Starting Fine-Tuning Workflow: {workflow_id}")
        logger.info("="*80)
        
        workflow = {
            'workflow_id': workflow_id,
            'started_at': datetime.now().isoformat(),
            'stages': {},
            'status': 'running'
        }
        
        try:
            # Stage 1: Trigger and prepare dataset
            logger.info("\n📊 STAGE 1: Triggering Fine-Tuning...")
            job = self.orchestrator.trigger_finetuning(force=force_trigger)
            
            if not job:
                workflow['status'] = 'not_triggered'
                workflow['message'] = 'Trigger conditions not met'
                self._save_workflow(workflow)
                logger.info("⏸️  Workflow terminated: Trigger conditions not met")
                return workflow
            
            workflow['stages']['trigger'] = {
                'status': 'completed',
                'job_id': job.job_id,
                'dataset_id': job.dataset_id,
                'version_id': job.version_id
            }
            
            # Stage 2: Train model (if not provided)
            if model_path is None:
                logger.info("\n🔧 STAGE 2: Training Model...")
                
                if self.training_callback is None:
                    logger.warning("⚠️  No training callback configured")
                    logger.warning("   Provide model_path or set training callback")
                    workflow['status'] = 'waiting_training'
                    workflow['message'] = 'Waiting for model training'
                    workflow['job_id'] = job.job_id
                    self._save_workflow(workflow)
                    return workflow
                
                # Start training
                training_started = self.orchestrator.start_training(
                    job.job_id,
                    training_params
                )
                
                if not training_started:
                    raise ValueError("Failed to start training")
                
                # Get model path from job (training callback should update this)
                job_info = self.orchestrator.get_job_info(job.job_id)
                model_path = job_info.get('model_path')
                
                if not model_path:
                    workflow['status'] = 'training_pending'
                    workflow['message'] = 'Training in progress'
                    workflow['job_id'] = job.job_id
                    self._save_workflow(workflow)
                    return workflow
                
                workflow['stages']['training'] = {
                    'status': 'completed',
                    'model_path': model_path
                }
            else:
                logger.info(f"\n✓ Using provided model: {model_path}")
                workflow['stages']['training'] = {
                    'status': 'skipped',
                    'model_path': model_path
                }
            
            # Stage 3: Validate model
            if not skip_validation:
                logger.info("\n✅ STAGE 3: Validating Model...")
                
                if self.baseline_transcribe_fn is None:
                    logger.warning("⚠️  No baseline transcription function set")
                    skip_validation = True
                elif self.model_transcribe_fn_factory is None:
                    logger.warning("⚠️  No model transcription function factory set")
                    skip_validation = True
                
                if not skip_validation:
                    # Create model transcription function
                    model_transcribe_fn = self.model_transcribe_fn_factory(model_path)
                    
                    # Run validation
                    validation_result = self.validator.validate_model(
                        model_id=f"finetuned_{job.job_id}",
                        model_transcribe_fn=model_transcribe_fn,
                        baseline_id="baseline_v1",
                        baseline_transcribe_fn=self.baseline_transcribe_fn,
                        validation_set_name=f"workflow_{workflow_id}"
                    )
                    
                    workflow['stages']['validation'] = {
                        'status': 'completed',
                        'passed': validation_result.passed,
                        'wer': validation_result.model_wer,
                        'cer': validation_result.model_cer,
                        'wer_improvement': validation_result.wer_improvement
                    }
                    
                    if not validation_result.passed:
                        logger.warning("❌ Model validation failed!")
                        workflow['status'] = 'validation_failed'
                        workflow['completed_at'] = datetime.now().isoformat()
                        self._save_workflow(workflow)
                        return workflow
                else:
                    workflow['stages']['validation'] = {
                        'status': 'skipped',
                        'reason': 'Missing transcription functions'
                    }
            else:
                logger.info("\n⏭️  STAGE 3: Validation skipped")
                workflow['stages']['validation'] = {'status': 'skipped'}
            
            # Stage 4: Regression tests
            if not skip_regression and not skip_validation:
                logger.info("\n🔬 STAGE 4: Running Regression Tests...")
                
                if self.model_transcribe_fn_factory:
                    model_transcribe_fn = self.model_transcribe_fn_factory(model_path)
                    
                    # Run regression test suite
                    regression_results = self.regression_tester.run_test_suite(
                        model_version=f"finetuned_{job.job_id}",
                        model_transcribe_fn=model_transcribe_fn
                    )
                    
                    workflow['stages']['regression'] = {
                        'status': 'completed',
                        'all_passed': regression_results['all_passed'],
                        'pass_rate': regression_results['pass_rate'],
                        'total_tests': regression_results['total_tests']
                    }
                    
                    if not regression_results['all_passed']:
                        logger.warning("⚠️  Some regression tests failed")
                        
                        if not auto_deploy:
                            workflow['status'] = 'regression_failed'
                            workflow['completed_at'] = datetime.now().isoformat()
                            self._save_workflow(workflow)
                            return workflow
                else:
                    workflow['stages']['regression'] = {
                        'status': 'skipped',
                        'reason': 'No model transcription function'
                    }
            else:
                logger.info("\n⏭️  STAGE 4: Regression tests skipped")
                workflow['stages']['regression'] = {'status': 'skipped'}
            
            # Stage 5: Register and deploy model
            if auto_deploy:
                logger.info("\n🚀 STAGE 5: Deploying Model...")
                
                # Register model
                version_id = self.deployer.register_model(
                    model_name="fine-tuned-stt",
                    model_path=model_path,
                    training_job_id=job.job_id,
                    training_dataset_id=job.dataset_id,
                    validation_result=workflow['stages'].get('validation', {})
                )
                
                # Deploy model
                deployed = self.deployer.deploy_model(version_id)
                
                workflow['stages']['deployment'] = {
                    'status': 'completed' if deployed else 'failed',
                    'version_id': version_id,
                    'deployed': deployed
                }
                
                if deployed:
                    logger.info("✅ Model deployed successfully!")
                else:
                    logger.error("❌ Model deployment failed")
            else:
                logger.info("\n⏭️  STAGE 5: Deployment skipped (auto_deploy=False)")
                workflow['stages']['deployment'] = {
                    'status': 'skipped',
                    'reason': 'Manual deployment required'
                }
            
            # Workflow completed
            workflow['status'] = 'completed'
            workflow['completed_at'] = datetime.now().isoformat()
            self._save_workflow(workflow)
            
            logger.info("\n" + "="*80)
            logger.info("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            
            return workflow
            
        except Exception as e:
            logger.error(f"\n❌ Workflow failed with error: {e}")
            workflow['status'] = 'failed'
            workflow['error'] = str(e)
            workflow['completed_at'] = datetime.now().isoformat()
            self._save_workflow(workflow)
            return workflow
    
    def deploy_job_model(
        self,
        job_id: str,
        model_path: str,
        training_metrics: Optional[Dict] = None,
        run_validation: bool = True,
        run_regression: bool = True
    ) -> bool:
        """
        Deploy a model from a completed training job.
        
        Args:
            job_id: Training job ID
            model_path: Path to trained model
            training_metrics: Training metrics
            run_validation: Run validation before deployment
            run_regression: Run regression tests before deployment
            
        Returns:
            True if deployed successfully
        """
        logger.info(f"Deploying model from job: {job_id}")
        
        # Complete training job
        self.orchestrator.complete_training(
            job_id=job_id,
            model_path=model_path,
            training_metrics=training_metrics
        )
        
        # Get job info
        job_info = self.orchestrator.get_job_info(job_id)
        if not job_info:
            logger.error(f"Job not found: {job_id}")
            return False
        
        # Validate if requested
        validation_result = None
        if run_validation and self.baseline_transcribe_fn and self.model_transcribe_fn_factory:
            logger.info("Running validation...")
            model_transcribe_fn = self.model_transcribe_fn_factory(model_path)
            
            validation_result = self.validator.validate_model(
                model_id=f"finetuned_{job_id}",
                model_transcribe_fn=model_transcribe_fn,
                baseline_id="baseline_v1",
                baseline_transcribe_fn=self.baseline_transcribe_fn
            )
            
            if not validation_result.passed:
                logger.warning("❌ Validation failed. Model not deployed.")
                return False
        
        # Regression tests if requested
        if run_regression and self.model_transcribe_fn_factory:
            logger.info("Running regression tests...")
            model_transcribe_fn = self.model_transcribe_fn_factory(model_path)
            
            regression_results = self.regression_tester.run_test_suite(
                model_version=f"finetuned_{job_id}",
                model_transcribe_fn=model_transcribe_fn
            )
            
            if not regression_results['all_passed']:
                logger.warning("⚠️  Some regression tests failed")
        
        # Register model
        version_id = self.deployer.register_model(
            model_name="fine-tuned-stt",
            model_path=model_path,
            training_job_id=job_id,
            training_dataset_id=job_info.get('dataset_id'),
            validation_result=validation_result.to_dict() if validation_result else None,
            metadata=training_metrics
        )
        
        # Deploy
        deployed = self.deployer.deploy_model(version_id)
        
        if deployed:
            logger.info(f"✅ Model deployed: {version_id}")
        else:
            logger.error("❌ Deployment failed")
        
        return deployed
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'orchestrator': self.orchestrator.get_orchestration_status(),
            'deployer': self.deployer.generate_deployment_report(),
            'validation': self.validator.generate_validation_report(),
            'regression': self.regression_tester.generate_regression_report(),
            'workflows': {
                'total': len(self.workflows),
                'recent': self.workflows[-5:] if self.workflows else []
            }
        }
        
        return status
    
    def print_status(self):
        """Print system status."""
        print("\n" + "="*80)
        print("FINE-TUNING ORCHESTRATION SYSTEM STATUS")
        print("="*80)
        
        # Orchestrator status
        trigger_result = self.orchestrator.check_trigger_conditions()
        print(f"\n📊 Fine-Tuning Trigger Status:")
        print(f"   Should trigger: {'Yes ✅' if trigger_result['should_trigger'] else 'No ⏸️'}")
        print(f"   Error cases: {trigger_result['metrics']['total_error_cases']}")
        print(f"   Corrected cases: {trigger_result['metrics']['corrected_cases']}")
        
        # Deployment status
        active = self.deployer.get_active_version()
        print(f"\n🚀 Deployment Status:")
        if active:
            print(f"   Active model: {active.version_id}")
            print(f"   WER: {active.wer:.4f}" if active.wer else "   WER: N/A")
        else:
            print("   No active model")
        
        print(f"   Total versions: {len(self.deployer.versions)}")
        
        # Validation status
        val_report = self.validator.generate_validation_report()
        print(f"\n✅ Validation Status:")
        print(f"   Total validations: {val_report.get('total_validations', 0)}")
        print(f"   Pass rate: {val_report.get('pass_rate', 0):.1%}")
        
        # Regression testing status
        reg_report = self.regression_tester.generate_regression_report()
        if 'error' not in reg_report:
            print(f"\n🔬 Regression Testing:")
            print(f"   Registered tests: {reg_report['registered_tests']}")
            print(f"   Total runs: {reg_report['total_test_runs']}")
            print(f"   Pass rate: {reg_report['pass_rate']:.1%}")
        
        # Workflows
        print(f"\n🔄 Workflows:")
        print(f"   Total workflows: {len(self.workflows)}")
        if self.workflows:
            recent = self.workflows[-1]
            print(f"   Latest: {recent['workflow_id']} ({recent['status']})")
        
        print("="*80 + "\n")


