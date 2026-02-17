# Weights & Biases Integration

Experiment tracking and hyperparameter optimization for fine-tuning.

## Quick Start

```bash
pip install wandb
wandb login
```

```python
from src.data.finetuning_orchestrator import FinetuningConfig

config = FinetuningConfig(
    use_wandb=True,
    wandb_project="my-project"
)

coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    finetuning_config=config
)
workflow = coordinator.run_complete_workflow()
# Check W&B dashboard for visualizations
```

## What Gets Tracked

- **Training**: Loss per epoch, learning rate, duration
- **Validation**: WER/CER comparison, improvement metrics, pass/fail
- **Regression tests**: Test pass rates, degradation metrics
- **Dataset**: Split sizes, error type distribution
- **Model artifacts**: Trained models, metadata, version tracking

## Hyperparameter Sweeps

**Recommendation**: Use Random Search (20 trials) for initial optimization.

```python
from src.data.wandb_sweeps import WandbSweepOrchestrator, SweepConfig

sweep_orch = WandbSweepOrchestrator(project_name="stt-optimization")
config = SweepConfig.create_finetuning_sweep(method='random', num_trials=20)
sweep_id = sweep_orch.create_sweep(config)
sweep_orch.run_sweep_agent(your_train_fn, sweep_id, count=20)

best = sweep_orch.get_best_run(sweep_id)
print(f"Optimal config: {best['hyperparameters']}")
```

### Search Strategies

| Strategy | Best For | Trials | Expected Improvement |
|----------|----------|--------|---------------------|
| Random | Initial exploration | 20-50 | 20-30% WER |
| Bayesian | Refinement | 30-100 | 30-40% WER |
| Grid | Final validation | Limited | Exhaustive |

### Recommended Approach

1. **Phase 1**: Random search (20 trials) - find good config (~1-2 days)
2. **Phase 2**: Use best config for all future fine-tuning
3. **Phase 3**: Periodic mini-sweeps (5 trials) every 6 months as data evolves

## Demo

```bash
python experiments/demo_wandb_tracking.py
python experiments/demo_wandb_sweeps.py
```

## Configuration

```python
config = FinetuningConfig(
    use_wandb=True,
    wandb_project="stt-finetuning",
    wandb_entity="my-team"
)
```
