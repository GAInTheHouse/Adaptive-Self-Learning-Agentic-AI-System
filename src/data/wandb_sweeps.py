"""
W&B Sweeps Integration for Hyperparameter Optimization

Automatically finds the best hyperparameters for fine-tuning using various search strategies.
"""

import logging
from typing import Dict, List, Optional, Callable, Any
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


class SweepConfig:
    """Configuration for W&B hyperparameter sweeps."""
    
    # Search strategies
    RANDOM = "random"
    BAYES = "bayes"
    GRID = "grid"
    
    # Optimization goals
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    
    @staticmethod
    def create_finetuning_sweep(
        metric_name: str = "validation/model_wer",
        goal: str = "minimize",
        method: str = "random",
        num_trials: int = 20
    ) -> Dict:
        """
        Create sweep configuration for STT fine-tuning.
        
        Args:
            metric_name: Metric to optimize (e.g., 'validation/model_wer')
            goal: 'minimize' or 'maximize'
            method: 'random', 'bayes', or 'grid'
            num_trials: Number of trials to run
            
        Returns:
            Sweep configuration dictionary
        """
        config = {
            'method': method,
            'metric': {
                'name': metric_name,
                'goal': goal
            },
            'parameters': {
                # Learning rate - most critical parameter
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-6,
                    'max': 1e-4
                },
                
                # Batch size
                'batch_size': {
                    'values': [4, 8, 16, 32]
                },
                
                # Training epochs
                'epochs': {
                    'values': [3, 5, 10, 15]
                },
                
                # Warmup steps
                'warmup_steps': {
                    'values': [100, 500, 1000]
                },
                
                # Weight decay
                'weight_decay': {
                    'distribution': 'uniform',
                    'min': 0.0,
                    'max': 0.1
                },
                
                # Gradient accumulation
                'gradient_accumulation_steps': {
                    'values': [1, 2, 4]
                },
                
                # Dropout
                'dropout': {
                    'distribution': 'uniform',
                    'min': 0.0,
                    'max': 0.3
                }
            }
        }
        
        # Add early termination for Bayesian optimization
        if method == 'bayes':
            config['early_terminate'] = {
                'type': 'hyperband',
                'min_iter': 3
            }
        
        return config
    
    @staticmethod
    def create_minimal_sweep(
        metric_name: str = "validation/model_wer",
        num_trials: int = 10
    ) -> Dict:
        """
        Create minimal sweep config for quick testing.
        
        Args:
            metric_name: Metric to optimize
            num_trials: Number of trials
            
        Returns:
            Minimal sweep configuration
        """
        return {
            'method': 'random',
            'metric': {
                'name': metric_name,
                'goal': 'minimize'
            },
            'parameters': {
                'learning_rate': {
                    'values': [1e-5, 5e-5, 1e-4]
                },
                'batch_size': {
                    'values': [8, 16]
                },
                'epochs': {
                    'values': [5, 10]
                }
            }
        }
    
    @staticmethod
    def create_custom_sweep(
        parameters: Dict[str, Any],
        metric_name: str = "validation/model_wer",
        goal: str = "minimize",
        method: str = "random"
    ) -> Dict:
        """
        Create custom sweep configuration.
        
        Args:
            parameters: Custom parameter definitions
            metric_name: Metric to optimize
            goal: 'minimize' or 'maximize'
            method: Search method
            
        Returns:
            Custom sweep configuration
        """
        return {
            'method': method,
            'metric': {
                'name': metric_name,
                'goal': goal
            },
            'parameters': parameters
        }


class WandbSweepOrchestrator:
    """
    Orchestrates hyperparameter optimization sweeps for fine-tuning.
    
    Features:
    - Multiple search strategies (random, Bayesian, grid)
    - Automatic optimization of WER/CER
    - Integration with fine-tuning pipeline
    - Best hyperparameters selection
    - Parallel sweep execution support
    """
    
    def __init__(
        self,
        project_name: str = "stt-finetuning-sweeps",
        entity: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize sweep orchestrator.
        
        Args:
            project_name: W&B project name for sweeps
            entity: W&B entity (username/team)
            enabled: Whether sweeps are enabled
        """
        self.project_name = project_name
        self.entity = entity
        self.enabled = enabled and WANDB_AVAILABLE
        self.sweep_id = None
        
        if not WANDB_AVAILABLE and enabled:
            logger.warning("W&B not available. Sweeps disabled.")
            self.enabled = False
        
        if self.enabled:
            logger.info(f"W&B Sweep Orchestrator initialized for project: {project_name}")
    
    def create_sweep(
        self,
        sweep_config: Dict,
        sweep_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new hyperparameter sweep.
        
        Args:
            sweep_config: Sweep configuration dictionary
            sweep_name: Optional name for the sweep
            
        Returns:
            Sweep ID if successful, None otherwise
        """
        if not self.enabled:
            logger.warning("Sweeps not enabled")
            return None
        
        try:
            # Add sweep name if provided
            if sweep_name:
                sweep_config['name'] = sweep_name
            
            # Create sweep
            self.sweep_id = wandb.sweep(
                sweep_config,
                project=self.project_name,
                entity=self.entity
            )
            
            logger.info(f"Created sweep: {self.sweep_id}")
            logger.info(f"View at: https://wandb.ai/{self.entity or 'your-username'}/{self.project_name}/sweeps/{self.sweep_id}")
            
            return self.sweep_id
            
        except Exception as e:
            logger.error(f"Failed to create sweep: {e}")
            return None
    
    def run_sweep_agent(
        self,
        train_function: Callable,
        sweep_id: Optional[str] = None,
        count: Optional[int] = None
    ):
        """
        Run sweep agent to execute hyperparameter trials.
        
        Args:
            train_function: Training function that uses wandb.config for hyperparameters
            sweep_id: Sweep ID to run (uses self.sweep_id if not provided)
            count: Number of trials to run (None for unlimited)
        """
        if not self.enabled:
            logger.warning("Sweeps not enabled")
            return
        
        sweep_id = sweep_id or self.sweep_id
        if not sweep_id:
            logger.error("No sweep ID provided")
            return
        
        try:
            logger.info(f"Starting sweep agent for sweep: {sweep_id}")
            
            wandb.agent(
                sweep_id,
                function=train_function,
                count=count,
                project=self.project_name,
                entity=self.entity
            )
            
            logger.info("Sweep agent completed")
            
        except Exception as e:
            logger.error(f"Sweep agent failed: {e}")
    
    def get_best_run(
        self,
        sweep_id: Optional[str] = None,
        metric_name: str = "validation/model_wer",
        minimize: bool = True
    ) -> Optional[Dict]:
        """
        Get best run from a completed sweep.
        
        Args:
            sweep_id: Sweep ID to analyze
            metric_name: Metric to use for comparison
            minimize: Whether to minimize (True) or maximize (False) the metric
            
        Returns:
            Dictionary with best run information
        """
        if not self.enabled:
            return None
        
        sweep_id = sweep_id or self.sweep_id
        if not sweep_id:
            logger.error("No sweep ID provided")
            return None
        
        try:
            api = wandb.Api()
            sweep = api.sweep(f"{self.entity or api.viewer()['entity']}/{self.project_name}/{sweep_id}")
            
            # Get all runs from sweep
            runs = sweep.runs
            
            if not runs:
                logger.warning("No runs found in sweep")
                return None
            
            # Find best run
            best_run = None
            best_metric = float('inf') if minimize else float('-inf')
            
            for run in runs:
                if metric_name in run.summary:
                    metric_value = run.summary[metric_name]
                    
                    if minimize and metric_value < best_metric:
                        best_metric = metric_value
                        best_run = run
                    elif not minimize and metric_value > best_metric:
                        best_metric = metric_value
                        best_run = run
            
            if not best_run:
                logger.warning(f"No runs with metric {metric_name} found")
                return None
            
            # Extract best hyperparameters
            best_config = {
                'run_id': best_run.id,
                'run_name': best_run.name,
                'metric_value': best_metric,
                'hyperparameters': dict(best_run.config),
                'summary': dict(best_run.summary)
            }
            
            logger.info(f"Best run: {best_run.name}")
            logger.info(f"Best {metric_name}: {best_metric:.4f}")
            logger.info(f"Best hyperparameters: {best_config['hyperparameters']}")
            
            return best_config
            
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None
    
    def save_best_config(
        self,
        output_path: str,
        sweep_id: Optional[str] = None,
        metric_name: str = "validation/model_wer"
    ) -> bool:
        """
        Save best hyperparameters to file.
        
        Args:
            output_path: Path to save configuration
            sweep_id: Sweep ID to analyze
            metric_name: Metric to optimize
            
        Returns:
            True if successful
        """
        best_config = self.get_best_run(sweep_id, metric_name)
        
        if not best_config:
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            
            logger.info(f"Saved best configuration to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def create_and_run_sweep(
        self,
        train_function: Callable,
        sweep_config: Optional[Dict] = None,
        sweep_name: Optional[str] = None,
        num_trials: int = 20
    ) -> Optional[Dict]:
        """
        Create and run a complete sweep, returning best hyperparameters.
        
        Args:
            train_function: Training function
            sweep_config: Sweep configuration (uses default if not provided)
            sweep_name: Sweep name
            num_trials: Number of trials to run
            
        Returns:
            Best hyperparameters dictionary
        """
        # Use default config if not provided
        if sweep_config is None:
            sweep_config = SweepConfig.create_finetuning_sweep(
                method='random',
                num_trials=num_trials
            )
        
        # Create sweep
        sweep_id = self.create_sweep(sweep_config, sweep_name)
        
        if not sweep_id:
            return None
        
        # Run sweep
        self.run_sweep_agent(train_function, sweep_id, count=num_trials)
        
        # Get best configuration
        best_config = self.get_best_run(sweep_id)
        
        return best_config


def create_sweep_training_wrapper(
    base_train_function: Callable,
    data_manager,
    orchestrator
) -> Callable:
    """
    Create a training wrapper function compatible with W&B sweeps.
    
    Args:
        base_train_function: Your actual training function
        data_manager: DataManager instance
        orchestrator: FinetuningOrchestrator instance
        
    Returns:
        Wrapped function that uses wandb.config for hyperparameters
    """
    def train():
        """Training function for sweep."""
        # Initialize W&B run (sweep agent does this automatically)
        config = wandb.config
        
        logger.info(f"Running trial with config: {dict(config)}")
        
        # Trigger fine-tuning with sweep hyperparameters
        job = orchestrator.trigger_finetuning(force=True)
        
        if not job:
            logger.error("Failed to trigger fine-tuning")
            return
        
        # Train with hyperparameters from sweep
        training_params = {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'warmup_steps': config.get('warmup_steps', 500),
            'weight_decay': config.get('weight_decay', 0.01),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1)
        }
        
        # Call your actual training function
        result = base_train_function(job, training_params)
        
        # Log final metrics
        if result and 'validation' in result:
            wandb.log({
                'validation/model_wer': result['validation']['wer'],
                'validation/model_cer': result['validation']['cer'],
                'validation/wer_improvement': result['validation']['wer_improvement']
            })
    
    return train


# Example usage
def example_sweep():
    """Example of running a hyperparameter sweep."""
    
    # 1. Create sweep orchestrator
    sweep_orch = WandbSweepOrchestrator(
        project_name="stt-finetuning-sweeps"
    )
    
    # 2. Create sweep configuration
    sweep_config = SweepConfig.create_finetuning_sweep(
        metric_name="validation/model_wer",
        goal="minimize",
        method="random",
        num_trials=20
    )
    
    # 3. Define your training function
    def train():
        # Your training code here
        # Use wandb.config for hyperparameters
        config = wandb.config
        
        # ... training logic ...
        
        # Log metrics
        wandb.log({
            'validation/model_wer': 0.15,  # Your actual WER
            'validation/model_cer': 0.08   # Your actual CER
        })
    
    # 4. Create and run sweep
    sweep_id = sweep_orch.create_sweep(sweep_config, "my_sweep")
    sweep_orch.run_sweep_agent(train, count=20)
    
    # 5. Get best hyperparameters
    best_config = sweep_orch.get_best_run(sweep_id)
    
    # 6. Save best config
    sweep_orch.save_best_config("best_hyperparameters.json", sweep_id)
    
    return best_config

