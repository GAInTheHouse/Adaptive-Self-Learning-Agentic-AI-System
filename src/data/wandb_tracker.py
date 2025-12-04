"""
Weights & Biases Integration for Fine-Tuning Orchestration

Tracks experiments, metrics, and generates visualizations for fine-tuning runs.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available. Install with: pip install wandb")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WandbTracker:
    """
    Weights & Biases integration for tracking fine-tuning experiments.
    
    Features:
    - Track fine-tuning metrics
    - Log validation results
    - Track regression tests
    - Generate plots automatically
    - Compare model versions
    - Visualize performance over time
    """
    
    def __init__(
        self,
        project_name: str = "stt-finetuning",
        entity: Optional[str] = None,
        enabled: bool = True,
        config: Optional[Dict] = None
    ):
        """
        Initialize W&B tracker.
        
        Args:
            project_name: W&B project name
            entity: W&B entity (username or team)
            enabled: Whether to enable W&B logging
            config: Initial configuration to log
        """
        self.project_name = project_name
        self.entity = entity
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        self.config = config or {}
        
        if not WANDB_AVAILABLE and enabled:
            logger.warning("W&B not available. Tracking disabled.")
            self.enabled = False
        
        if self.enabled:
            logger.info(f"W&B Tracker initialized for project: {project_name}")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        job_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ) -> bool:
        """
        Start a new W&B run.
        
        Args:
            run_name: Name for the run
            job_id: Fine-tuning job ID
            tags: Tags for the run
            config: Configuration to log
            
        Returns:
            True if run started successfully
        """
        if not self.enabled:
            return False
        
        try:
            # Merge configs
            run_config = {**self.config, **(config or {})}
            
            # Generate run name if not provided
            if run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"finetuning_{timestamp}"
            
            # Add job_id to tags
            run_tags = tags or []
            if job_id:
                run_tags.append(f"job_{job_id}")
            
            # Initialize run
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=run_config,
                tags=run_tags,
                reinit=True
            )
            
            logger.info(f"Started W&B run: {run_name}")
            logger.info(f"View at: {self.run.url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")
            return False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled or not self.run:
            return
        
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        additional_metrics: Optional[Dict] = None
    ):
        """
        Log training metrics.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            learning_rate: Current learning rate
            additional_metrics: Additional metrics to log
        """
        metrics = {
            'epoch': epoch,
            'train/loss': train_loss
        }
        
        if val_loss is not None:
            metrics['val/loss'] = val_loss
        
        if learning_rate is not None:
            metrics['train/learning_rate'] = learning_rate
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.log_metrics(metrics, step=epoch)
        logger.info(f"Logged training metrics for epoch {epoch}")
    
    def log_validation_results(
        self,
        validation_result: Dict,
        model_id: str
    ):
        """
        Log model validation results.
        
        Args:
            validation_result: Validation result dictionary
            model_id: Model identifier
        """
        if not self.enabled or not self.run:
            return
        
        try:
            metrics = {
                'validation/model_wer': validation_result.get('model_wer', 0),
                'validation/model_cer': validation_result.get('model_cer', 0),
                'validation/baseline_wer': validation_result.get('baseline_wer', 0),
                'validation/baseline_cer': validation_result.get('baseline_cer', 0),
                'validation/wer_improvement': validation_result.get('wer_improvement', 0),
                'validation/cer_improvement': validation_result.get('cer_improvement', 0),
                'validation/passed': int(validation_result.get('passed', False)),
                'validation/num_samples': validation_result.get('num_samples', 0)
            }
            
            self.log_metrics(metrics)
            
            # Create comparison chart
            self._create_validation_chart(validation_result, model_id)
            
            logger.info(f"Logged validation results for {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to log validation results: {e}")
    
    def log_regression_results(
        self,
        test_results: Dict,
        model_version: str
    ):
        """
        Log regression test results.
        
        Args:
            test_results: Regression test results
            model_version: Model version being tested
        """
        if not self.enabled or not self.run:
            return
        
        try:
            metrics = {
                'regression/total_tests': test_results.get('total_tests', 0),
                'regression/passed': test_results.get('passed', 0),
                'regression/failed': test_results.get('failed', 0),
                'regression/pass_rate': test_results.get('pass_rate', 0),
                'regression/avg_wer_degradation': test_results.get('avg_wer_degradation', 0),
                'regression/avg_cer_degradation': test_results.get('avg_cer_degradation', 0)
            }
            
            self.log_metrics(metrics)
            
            # Create regression test chart
            self._create_regression_chart(test_results, model_version)
            
            logger.info(f"Logged regression results for {model_version}")
            
        except Exception as e:
            logger.error(f"Failed to log regression results: {e}")
    
    def log_dataset_info(
        self,
        dataset_id: str,
        split_sizes: Dict[str, int],
        error_type_distribution: Dict[str, int]
    ):
        """
        Log dataset information.
        
        Args:
            dataset_id: Dataset identifier
            split_sizes: Dictionary with train/val/test sizes
            error_type_distribution: Distribution of error types
        """
        if not self.enabled or not self.run:
            return
        
        try:
            metrics = {
                'dataset/train_size': split_sizes.get('train', 0),
                'dataset/val_size': split_sizes.get('val', 0),
                'dataset/test_size': split_sizes.get('test', 0),
                'dataset/total_size': sum(split_sizes.values())
            }
            
            self.log_metrics(metrics)
            
            # Log error type distribution as table
            error_data = [
                [error_type, count] 
                for error_type, count in error_type_distribution.items()
            ]
            
            table = wandb.Table(
                data=error_data,
                columns=["Error Type", "Count"]
            )
            
            wandb.log({
                "dataset/error_distribution": wandb.plot.bar(
                    table,
                    "Error Type",
                    "Count",
                    title="Error Type Distribution"
                )
            })
            
            logger.info(f"Logged dataset info for {dataset_id}")
            
        except Exception as e:
            logger.error(f"Failed to log dataset info: {e}")
    
    def log_model_artifact(
        self,
        model_path: str,
        model_name: str,
        metadata: Optional[Dict] = None
    ):
        """
        Log model as W&B artifact.
        
        Args:
            model_path: Path to model files
            model_name: Name for the model artifact
            metadata: Additional metadata
        """
        if not self.enabled or not self.run:
            return
        
        try:
            artifact = wandb.Artifact(
                name=model_name,
                type='model',
                metadata=metadata or {}
            )
            
            # Add model files
            model_path = Path(model_path)
            if model_path.is_dir():
                artifact.add_dir(str(model_path))
            elif model_path.is_file():
                artifact.add_file(str(model_path))
            
            wandb.log_artifact(artifact)
            logger.info(f"Logged model artifact: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model artifact: {e}")
    
    def _create_validation_chart(self, validation_result: Dict, model_id: str):
        """Create validation comparison chart."""
        try:
            data = [
                ["Baseline", validation_result.get('baseline_wer', 0), validation_result.get('baseline_cer', 0)],
                [model_id, validation_result.get('model_wer', 0), validation_result.get('model_cer', 0)]
            ]
            
            table = wandb.Table(
                data=data,
                columns=["Model", "WER", "CER"]
            )
            
            wandb.log({
                "validation/wer_comparison": wandb.plot.bar(
                    table,
                    "Model",
                    "WER",
                    title="WER Comparison"
                ),
                "validation/cer_comparison": wandb.plot.bar(
                    table,
                    "Model",
                    "CER",
                    title="CER Comparison"
                )
            })
            
        except Exception as e:
            logger.error(f"Failed to create validation chart: {e}")
    
    def _create_regression_chart(self, test_results: Dict, model_version: str):
        """Create regression test results chart."""
        try:
            # Test pass/fail chart
            data = [
                ["Passed", test_results.get('passed', 0)],
                ["Failed", test_results.get('failed', 0)]
            ]
            
            table = wandb.Table(
                data=data,
                columns=["Status", "Count"]
            )
            
            wandb.log({
                "regression/test_results": wandb.plot.bar(
                    table,
                    "Status",
                    "Count",
                    title=f"Regression Test Results - {model_version}"
                )
            })
            
            # Individual test results if available
            if 'results' in test_results:
                test_data = [
                    [r['test_id'], r['wer_degradation'], int(r['passed'])]
                    for r in test_results['results']
                ]
                
                test_table = wandb.Table(
                    data=test_data,
                    columns=["Test ID", "WER Degradation", "Passed"]
                )
                
                wandb.log({
                    "regression/degradation_by_test": wandb.plot.scatter(
                        test_table,
                        "Test ID",
                        "WER Degradation",
                        title="WER Degradation by Test"
                    )
                })
            
        except Exception as e:
            logger.error(f"Failed to create regression chart: {e}")
    
    def log_system_metrics(
        self,
        error_cases: int,
        corrected_cases: int,
        correction_rate: float,
        models_deployed: int
    ):
        """
        Log system-level metrics.
        
        Args:
            error_cases: Total error cases
            corrected_cases: Corrected cases
            correction_rate: Correction rate
            models_deployed: Number of models deployed
        """
        if not self.enabled or not self.run:
            return
        
        try:
            metrics = {
                'system/error_cases': error_cases,
                'system/corrected_cases': corrected_cases,
                'system/correction_rate': correction_rate,
                'system/models_deployed': models_deployed
            }
            
            self.log_metrics(metrics)
            logger.info("Logged system metrics")
            
        except Exception as e:
            logger.error(f"Failed to log system metrics: {e}")
    
    def log_performance_history(
        self,
        history: List[Dict],
        metric_name: str = 'wer'
    ):
        """
        Log performance history over time.
        
        Args:
            history: List of historical performance records
            metric_name: Name of metric to plot
        """
        if not self.enabled or not self.run:
            return
        
        try:
            # Extract data
            data = [
                [i, record.get(metric_name, 0), record.get('timestamp', '')]
                for i, record in enumerate(history)
            ]
            
            table = wandb.Table(
                data=data,
                columns=["Index", metric_name.upper(), "Timestamp"]
            )
            
            wandb.log({
                f"history/{metric_name}_trend": wandb.plot.line(
                    table,
                    "Index",
                    metric_name.upper(),
                    title=f"{metric_name.upper()} Over Time"
                )
            })
            
            logger.info(f"Logged performance history for {metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to log performance history: {e}")
    
    def log_confusion_matrix(
        self,
        predictions: List[str],
        references: List[str],
        class_names: Optional[List[str]] = None
    ):
        """
        Log confusion matrix for error types.
        
        Args:
            predictions: Predicted error types
            references: Reference error types
            class_names: Names of error type classes
        """
        if not self.enabled or not self.run:
            return
        
        try:
            wandb.log({
                "evaluation/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=references,
                    preds=predictions,
                    class_names=class_names
                )
            })
            
            logger.info("Logged confusion matrix")
            
        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")
    
    def log_audio_samples(
        self,
        audio_paths: List[str],
        transcripts: List[str],
        references: Optional[List[str]] = None,
        max_samples: int = 10
    ):
        """
        Log audio samples with transcriptions.
        
        Args:
            audio_paths: Paths to audio files
            transcripts: Model transcriptions
            references: Reference transcriptions
            max_samples: Maximum number of samples to log
        """
        if not self.enabled or not self.run:
            return
        
        try:
            samples = []
            for i, (audio_path, transcript) in enumerate(zip(audio_paths[:max_samples], transcripts[:max_samples])):
                ref = references[i] if references and i < len(references) else ""
                
                audio_path = Path(audio_path)
                if audio_path.exists():
                    samples.append(wandb.Audio(
                        str(audio_path),
                        caption=f"Transcript: {transcript}\nReference: {ref}",
                        sample_rate=16000
                    ))
            
            if samples:
                wandb.log({"evaluation/audio_samples": samples})
                logger.info(f"Logged {len(samples)} audio samples")
            
        except Exception as e:
            logger.error(f"Failed to log audio samples: {e}")
    
    def log_hyperparameters(self, hyperparameters: Dict):
        """
        Log hyperparameters.
        
        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        if not self.enabled or not self.run:
            return
        
        try:
            wandb.config.update(hyperparameters)
            logger.info("Logged hyperparameters")
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")
    
    def log_text(self, key: str, value: str):
        """
        Log text data.
        
        Args:
            key: Key for the text
            value: Text value
        """
        if not self.enabled or not self.run:
            return
        
        try:
            wandb.log({key: wandb.Html(value)})
        except Exception as e:
            logger.error(f"Failed to log text: {e}")
    
    def log_summary(self, summary: Dict):
        """
        Log run summary.
        
        Args:
            summary: Summary dictionary
        """
        if not self.enabled or not self.run:
            return
        
        try:
            for key, value in summary.items():
                wandb.run.summary[key] = value
            logger.info("Logged run summary")
        except Exception as e:
            logger.error(f"Failed to log summary: {e}")
    
    def finish_run(self):
        """Finish the current W&B run."""
        if not self.enabled or not self.run:
            return
        
        try:
            wandb.finish()
            logger.info("Finished W&B run")
            self.run = None
        except Exception as e:
            logger.error(f"Failed to finish run: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish_run()


def create_tracker(
    project_name: str = "stt-finetuning",
    enabled: bool = True,
    config: Optional[Dict] = None
) -> WandbTracker:
    """
    Factory function to create W&B tracker.
    
    Args:
        project_name: W&B project name
        enabled: Whether to enable tracking
        config: Initial configuration
        
    Returns:
        WandbTracker instance
    """
    return WandbTracker(
        project_name=project_name,
        enabled=enabled,
        config=config
    )

