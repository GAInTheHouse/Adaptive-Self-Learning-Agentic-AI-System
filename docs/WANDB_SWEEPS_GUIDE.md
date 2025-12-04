# W&B Sweeps for Hyperparameter Optimization

## Should You Use W&B Random Search for This Project?

**YES! This project will significantly benefit from W&B Sweeps. Here's why:**

---

## 🎯 Benefits for Your STT Fine-Tuning Project

### 1. **Optimize Multiple Hyperparameters Simultaneously**

When fine-tuning speech-to-text models, performance depends on:

- **Learning Rate** - Most critical parameter
- **Batch Size** - Affects convergence and memory
- **Training Epochs** - Balance between underfitting and overfitting
- **Warmup Steps** - Critical for transformer models
- **Weight Decay** - Regularization strength
- **Dropout** - Prevent overfitting
- **Gradient Accumulation** - Effective batch size

**Manual tuning = weeks of experimentation**  
**W&B Sweeps = automated optimization in hours**

### 2. **Automatic Discovery of Best Configuration**

Different datasets and error types may need different hyperparameters:

- **Word Substitution Errors** → May need lower learning rate
- **Missing Word Errors** → May benefit from more epochs
- **Pronunciation Errors** → May need different dropout

W&B Sweeps finds the optimal configuration for YOUR specific data.

### 3. **Cost Efficiency**

- Avoid wasting GPU time on suboptimal configurations
- Early termination of poor-performing trials
- Parallel execution on multiple GPUs
- Smart search strategies reduce total trials needed

### 4. **Reproducibility**

- All hyperparameters logged automatically
- Easy to reproduce best results
- Track what worked and what didn't
- Share configurations with team

---

## 📊 Search Strategies Comparison

### Random Search (RECOMMENDED START)

**Best For:** Initial exploration, limited compute budget

**Pros:**
- ✅ Fast and efficient
- ✅ Good coverage of search space
- ✅ No assumptions about parameter space
- ✅ Parallelizes perfectly
- ✅ Often finds good solutions quickly

**Cons:**
- ❌ May miss optimal configuration
- ❌ Doesn't learn from previous trials

**Recommended Trials:** 20-50

**When to Use:**
- First time optimizing
- Limited GPU hours
- Need quick results
- Exploring new dataset

```python
sweep_config = SweepConfig.create_finetuning_sweep(
    method='random',
    num_trials=20
)
```

---

### Bayesian Optimization (RECOMMENDED FOR REFINEMENT)

**Best For:** Expensive training, refined optimization

**Pros:**
- ✅ Learns from previous trials
- ✅ More efficient than random
- ✅ Focuses on promising regions
- ✅ Better for expensive computations
- ✅ Can find better solutions with fewer trials

**Cons:**
- ❌ Sequential (harder to parallelize)
- ❌ Needs more trials to warm up
- ❌ Can get stuck in local optima

**Recommended Trials:** 30-100

**When to Use:**
- Refining after random search
- Long training times
- Want best possible results
- Have computational budget

```python
sweep_config = SweepConfig.create_finetuning_sweep(
    method='bayes',
    num_trials=50
)
```

---

### Grid Search

**Best For:** Final tuning, specific parameter ranges

**Pros:**
- ✅ Exhaustive search
- ✅ No configurations missed
- ✅ Guaranteed to find best in search space
- ✅ Good for final validation

**Cons:**
- ❌ Exponentially grows with parameters
- ❌ Very expensive
- ❌ Overkill for most cases

**Recommended Trials:** Limited parameter space only

**When to Use:**
- Final tuning with narrow ranges
- Few parameters to tune
- Need absolute certainty
- Publication/production validation

```python
sweep_config = SweepConfig.create_custom_sweep(
    parameters={
        'learning_rate': {'values': [1e-5, 5e-5, 1e-4]},
        'batch_size': {'values': [16, 32]}
    },
    method='grid'
)
# This creates 3 × 2 = 6 trials
```

---

## 🎓 Recommended Strategy for This Project

### Phase 1: Random Search (Quick Exploration)

```python
# Start with random search - 20 trials
sweep_config = SweepConfig.create_finetuning_sweep(
    method='random',
    num_trials=20
)

# Focus on critical parameters
sweep_config['parameters'] = {
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-6,
        'max': 1e-4
    },
    'batch_size': {'values': [8, 16, 32]},
    'epochs': {'values': [5, 10, 15]}
}
```

**Expected Time:** 4-8 hours (depending on GPU)  
**Expected Benefit:** Find configurations within 80-90% of optimal

---

### Phase 2: Bayesian Optimization (Refinement)

```python
# Use best parameters from Phase 1 as starting point
# Narrow down the search space

sweep_config = SweepConfig.create_finetuning_sweep(
    method='bayes',
    num_trials=30
)

# Refine learning rate around best value from Phase 1
best_lr = 3e-5  # From Phase 1
sweep_config['parameters']['learning_rate'] = {
    'distribution': 'log_uniform_values',
    'min': best_lr * 0.5,
    'max': best_lr * 2.0
}
```

**Expected Time:** 6-12 hours  
**Expected Benefit:** Achieve 95-99% of optimal performance

---

### Phase 3: Production (Use Best Config)

```python
# Use best hyperparameters for all future fine-tuning
best_config = sweep_orch.get_best_run(sweep_id)

production_params = {
    'learning_rate': best_config['hyperparameters']['learning_rate'],
    'batch_size': best_config['hyperparameters']['batch_size'],
    'epochs': best_config['hyperparameters']['epochs']
}

# Use in orchestrator
orchestrator.start_training(job_id, training_params=production_params)
```

---

## 💡 Practical Implementation

### Option 1: One-Time Optimization (Recommended)

```python
from src.data.wandb_sweeps import WandbSweepOrchestrator, SweepConfig

# Create sweep
sweep_orch = WandbSweepOrchestrator(project_name="stt-finetuning")
sweep_config = SweepConfig.create_finetuning_sweep(
    method='random',
    num_trials=20
)

sweep_id = sweep_orch.create_sweep(sweep_config)

# Run sweep
sweep_orch.run_sweep_agent(your_training_function, sweep_id, count=20)

# Get best hyperparameters
best = sweep_orch.get_best_run(sweep_id)

# Save for future use
sweep_orch.save_best_config('optimal_hyperparameters.json', sweep_id)

# Use best config for all future fine-tuning
with open('optimal_hyperparameters.json') as f:
    optimal_params = json.load(f)['hyperparameters']
```

**Cost:** 20 training runs  
**Time:** 1-2 days  
**Benefit:** Optimal hyperparameters for lifetime of project

---

### Option 2: Continuous Optimization (Advanced)

```python
# Run mini-sweeps periodically as data evolves
# Every 1000 new error cases, run a small sweep

if new_error_cases > 1000:
    # Quick 5-trial sweep
    sweep_config = SweepConfig.create_minimal_sweep(num_trials=5)
    sweep_id = sweep_orch.create_sweep(sweep_config)
    sweep_orch.run_sweep_agent(train, sweep_id, count=5)
    
    # Update production config if significantly better
    new_best = sweep_orch.get_best_run(sweep_id)
    if new_best['metric_value'] < current_best_wer * 0.95:
        update_production_config(new_best)
```

**Cost:** 5 runs per 1000 error cases  
**Benefit:** Adapt to evolving data distribution

---

## 📈 Expected Improvements

Based on typical hyperparameter optimization results:

### Baseline (No Optimization)
- Default hyperparameters
- WER: 0.20
- Training time: 2 hours

### After Random Search (20 trials)
- Optimized hyperparameters
- **WER: 0.14** (30% improvement)
- Training time: 1.5 hours (faster convergence)

### After Bayesian Refinement (30 more trials)
- Fine-tuned hyperparameters
- **WER: 0.12** (40% improvement)
- Training time: 1.2 hours

**ROI:** 50 trials × 2 hours = 100 GPU hours
**Result:** Permanent 40% performance boost for all future fine-tuning

---

## 🔧 Implementation Guide

### Step 1: Create Sweep Configuration

```python
from src.data.wandb_sweeps import SweepConfig

# For this STT project, start with:
sweep_config = SweepConfig.create_finetuning_sweep(
    metric_name="validation/model_wer",  # Optimize WER
    goal="minimize",                      # Lower is better
    method="random",                      # Start with random
    num_trials=20                         # 20 trials
)

# Customize for your needs
sweep_config['parameters']['learning_rate'] = {
    'distribution': 'log_uniform_values',
    'min': 5e-6,  # Lower bound based on your model
    'max': 5e-5   # Upper bound
}
```

### Step 2: Integrate with Training

```python
from src.data.finetuning_orchestrator import FinetuningOrchestrator
from src.data.data_manager import DataManager
import wandb

def train_with_sweep():
    """Training function compatible with W&B sweep."""
    
    # Get hyperparameters from sweep
    config = wandb.config
    
    # Initialize orchestrator
    data_manager = DataManager(use_gcs=True)
    orchestrator = FinetuningOrchestrator(data_manager)
    
    # Trigger fine-tuning
    job = orchestrator.trigger_finetuning(force=True)
    
    # Train with sweep hyperparameters
    training_params = {
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'warmup_steps': config.warmup_steps,
        'weight_decay': config.weight_decay
    }
    
    # Your actual training code here
    result = run_actual_training(job, training_params)
    
    # Log final metrics (W&B uses these to find best run)
    wandb.log({
        'validation/model_wer': result['wer'],
        'validation/model_cer': result['cer'],
        'validation/wer_improvement': result['wer_improvement']
    })
```

### Step 3: Launch Sweep

```python
from src.data.wandb_sweeps import WandbSweepOrchestrator

sweep_orch = WandbSweepOrchestrator(project_name="stt-optimization")

# Create sweep
sweep_id = sweep_orch.create_sweep(
    sweep_config,
    sweep_name="initial_hyperparameter_optimization"
)

# Run sweep (can run on multiple GPUs in parallel)
sweep_orch.run_sweep_agent(train_with_sweep, sweep_id, count=20)
```

### Step 4: Get and Use Best Config

```python
# Get best hyperparameters
best = sweep_orch.get_best_run(sweep_id)

print(f"Best WER: {best['metric_value']:.4f}")
print("Optimal hyperparameters:")
for param, value in best['hyperparameters'].items():
    print(f"  {param}: {value}")

# Save for production
sweep_orch.save_best_config('config/optimal_hyperparameters.json', sweep_id)

# Use in all future fine-tuning
optimal_params = best['hyperparameters']
orchestrator.start_training(job_id, training_params=optimal_params)
```

---

## 🚀 Quick Start

### Minimal Example (5 trials, ~2 hours)

```python
from src.data.wandb_sweeps import WandbSweepOrchestrator, SweepConfig

# 1. Create minimal sweep
sweep_orch = WandbSweepOrchestrator(project_name="quick-test")
config = SweepConfig.create_minimal_sweep(num_trials=5)
sweep_id = sweep_orch.create_sweep(config)

# 2. Run sweep
sweep_orch.run_sweep_agent(your_train_function, sweep_id, count=5)

# 3. Get best
best = sweep_orch.get_best_run(sweep_id)
print(f"Best config: {best['hyperparameters']}")
```

---

## 📊 What You'll See in W&B Dashboard

### Sweeps Overview Page

```
┌────────────────────────────────────────────────────────┐
│  Sweep: initial_hyperparameter_optimization           │
├────────────────────────────────────────────────────────┤
│  Status: ✅ Completed (20/20 runs)                    │
│  Best WER: 0.12 (Run: helpful-surf-42)                │
│  Improvement: 40% vs worst run                         │
├────────────────────────────────────────────────────────┤
│  📊 Visualizations                                     │
│  • Parallel Coordinates Plot                          │
│    └─ Shows relationship between hyperparameters      │
│       and performance                                  │
│  • Parameter Importance                                │
│    └─ Which parameters matter most                    │
│  • Optimization History                                │
│    └─ Performance improving over trials               │
├────────────────────────────────────────────────────────┤
│  🏆 Best Configuration                                │
│  • learning_rate: 3.2e-5                              │
│  • batch_size: 16                                      │
│  • epochs: 10                                          │
│  • warmup_steps: 500                                   │
│  • weight_decay: 0.01                                  │
└────────────────────────────────────────────────────────┘
```

### Parallel Coordinates Plot

Shows how each hyperparameter affects WER:
- Lines colored by performance
- Best runs highlighted
- Easy to see patterns
- Interactive exploration

### Parameter Importance

Bar chart showing:
- Which parameters affect performance most
- Focus future optimization on these
- Ignore parameters with low impact

---

## 🎯 Recommended Approach

### For This STT Project:

**1. Initial Optimization (One-Time, ~1-2 days)**

```python
# Random search with 20-30 trials
sweep_config = SweepConfig.create_finetuning_sweep(
    method='random',
    num_trials=25
)

sweep_id = sweep_orch.create_sweep(sweep_config, "initial_optimization")
sweep_orch.run_sweep_agent(train_fn, sweep_id, count=25)

# Find best hyperparameters
best = sweep_orch.get_best_run(sweep_id)
```

**Result:** Optimal hyperparameters for your specific error cases and data

---

**2. Use Best Config in Production**

```python
# Load optimized hyperparameters
with open('optimal_hyperparameters.json') as f:
    optimal = json.load(f)['hyperparameters']

# Use for all automated fine-tuning
config = FinetuningConfig(
    use_wandb=True,
    # Use optimal hyperparameters
)

# All future fine-tuning uses optimal config
coordinator = FinetuningCoordinator(
    data_manager=data_manager,
    finetuning_config=config
)
```

---

**3. Periodic Re-Optimization (Optional)**

```python
# Every 6 months or after major data changes
# Run a small sweep to check if optimal config still works

mini_sweep = SweepConfig.create_minimal_sweep(num_trials=5)
# Re-validate optimal hyperparameters
```

---

## 💰 Cost-Benefit Analysis

### Without Sweeps (Manual Tuning)

- **Time:** 2-4 weeks of experimentation
- **GPU Hours:** 100-200 hours (trial and error)
- **Result:** Suboptimal configuration
- **Confidence:** Low (only tested a few combinations)

### With Random Search (20 trials)

- **Time:** 1-2 days automated
- **GPU Hours:** 40-60 hours (systematic)
- **Result:** Near-optimal configuration
- **Confidence:** High (tested 20 combinations)

### ROI

**Investment:** 40-60 GPU hours  
**Return:** 20-40% better WER for EVERY future fine-tuning  
**Lifetime Benefit:** If you fine-tune 10 times → 10x better performance

**Verdict: Excellent ROI! 🎯**

---

## 🔥 Quick Implementation

### Add to Your Workflow

```python
# In your fine-tuning coordinator

from src.data.wandb_sweeps import WandbSweepOrchestrator, SweepConfig

class FinetuningCoordinator:
    
    def optimize_hyperparameters(
        self,
        num_trials: int = 20,
        method: str = 'random'
    ) -> Dict:
        """
        Run hyperparameter optimization sweep.
        
        Returns best configuration found.
        """
        # Create sweep orchestrator
        sweep_orch = WandbSweepOrchestrator(
            project_name=f"{self.wandb_project}_sweeps"
        )
        
        # Create sweep config
        sweep_config = SweepConfig.create_finetuning_sweep(
            method=method,
            num_trials=num_trials
        )
        
        # Create sweep
        sweep_id = sweep_orch.create_sweep(sweep_config)
        
        # Define training wrapper
        def train():
            config = wandb.config
            job = self.orchestrator.trigger_finetuning(force=True)
            result = self.train_with_params(job, dict(config))
            wandb.log({'validation/model_wer': result['wer']})
        
        # Run sweep
        sweep_orch.run_sweep_agent(train, sweep_id, count=num_trials)
        
        # Get best
        best = sweep_orch.get_best_run(sweep_id)
        
        # Save best config
        self.optimal_hyperparameters = best['hyperparameters']
        
        return best
```

---

## 🎯 Answer to Your Question

### **Should you use W&B Random Search?**

**YES! Here's the recommendation:**

1. **Use Random Search (20 trials) for initial optimization**
   - Quick and efficient
   - Good results
   - Low cost

2. **Then use best hyperparameters for all automated fine-tuning**
   - 20-40% better performance
   - Faster convergence
   - More stable training

3. **Optionally refine with Bayesian (30 trials) if:**
   - You have the GPU budget
   - You want absolute best performance
   - You're preparing for production deployment

### Cost vs Benefit

| Strategy | Trials | Time | GPU Cost | WER Improvement | Recommended |
|----------|--------|------|----------|-----------------|-------------|
| Manual | ~10 | 2 weeks | Low | Unknown | ❌ |
| Random Search | 20 | 1-2 days | Medium | 20-30% | ✅ YES |
| Bayesian | 50 | 3-4 days | High | 30-40% | ✅ If budget allows |
| Grid Search | 100+ | 1 week+ | Very High | 40%+ | ❌ Overkill |

### For Your Project

**Start with:**
- ✅ Random Search with 20 trials
- ✅ Optimize for `validation/model_wer`
- ✅ Focus on learning_rate, batch_size, epochs

**Then:**
- ✅ Use best config for all future fine-tuning
- ✅ Re-optimize every 6 months or when data changes significantly

---

## 🚀 Get Started Now

```bash
# 1. Run the demo
python experiments/demo_wandb_sweeps.py

# 2. Try a quick 5-trial sweep
python -c "
from src.data.wandb_sweeps import WandbSweepOrchestrator, SweepConfig
sweep_orch = WandbSweepOrchestrator(project_name='quick-test')
config = SweepConfig.create_minimal_sweep(num_trials=5)
sweep_id = sweep_orch.create_sweep(config)
print(f'Sweep created: {sweep_id}')
print('Run: wandb agent {sweep_id}')
"

# 3. View results
# Go to https://wandb.ai/ → Your Project → Sweeps
```

---

## 📚 Additional Resources

- **W&B Sweeps Docs:** https://docs.wandb.ai/guides/sweeps
- **Best Practices:** https://wandb.ai/site/articles/bayesian-hyperparameter-optimization
- **Examples:** https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion

---

## ✅ Summary

**Will this project benefit from W&B Random Search?**

**ABSOLUTELY YES! 🎯**

**Benefits:**
- ✅ 20-40% better WER/CER
- ✅ Automated optimization
- ✅ Save weeks of manual tuning
- ✅ Reproducible results
- ✅ Confidence in hyperparameters

**Recommendation:**
1. Run ONE random search sweep (20 trials) 
2. Use best config for ALL future fine-tuning
3. ROI is excellent - pays for itself quickly

**Get started:**
```bash
python experiments/demo_wandb_sweeps.py
```

**The system is already integrated and ready to use!** 🚀

