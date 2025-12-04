#!/usr/bin/env python3
"""
Demo: W&B Sweeps for Hyperparameter Optimization

Shows how to use W&B Sweeps to automatically find the best hyperparameters
for fine-tuning your STT model.

Usage:
    pip install wandb
    wandb login
    python experiments/demo_wandb_sweeps.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
import random

from src.data.wandb_sweeps import (
    WandbSweepOrchestrator,
    SweepConfig,
    create_sweep_training_wrapper
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_mock_training():
    """Mock training function for demonstration."""
    # Get hyperparameters from W&B sweep
    config = wandb.config if WANDB_AVAILABLE else {}
    
    learning_rate = config.get('learning_rate', 1e-5)
    batch_size = config.get('batch_size', 16)
    epochs = config.get('epochs', 5)
    
    logger.info(f"Training with lr={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    
    # Simulate training
    for epoch in range(1, min(epochs, 3) + 1):  # Quick demo
        # Simulate metrics (better with lower learning rate in this mock)
        train_loss = 0.5 / epoch + (learning_rate * 1000)
        val_loss = train_loss * 1.2
        
        # Mock WER (better with optimal hyperparameters)
        wer = 0.20 - (0.02 * epoch) + abs(learning_rate - 3e-5) * 1000
        cer = wer * 0.6
        
        if WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'validation/model_wer': wer,
                'validation/model_cer': cer
            })
        
        time.sleep(0.5)  # Simulate training time
    
    # Final metrics
    final_wer = wer
    final_cer = cer
    
    if WANDB_AVAILABLE:
        wandb.log({
            'validation/model_wer': final_wer,
            'validation/model_cer': final_cer,
            'validation/wer_improvement': 0.20 - final_wer
        })
    
    logger.info(f"Final WER: {final_wer:.4f}, CER: {final_cer:.4f}")
    
    return {'wer': final_wer, 'cer': final_cer}


def demo_1_create_sweep_config():
    """Demo 1: Creating different sweep configurations."""
    print_section("DEMO 1: Creating Sweep Configurations")
    
    print("📝 Creating default fine-tuning sweep config...")
    config1 = SweepConfig.create_finetuning_sweep(
        metric_name="validation/model_wer",
        goal="minimize",
        method="random",
        num_trials=10
    )
    
    print(f"   Method: {config1['method']}")
    print(f"   Metric: {config1['metric']['name']} (goal: {config1['metric']['goal']})")
    print(f"   Parameters optimized: {len(config1['parameters'])}")
    print(f"   - Learning rate: log-uniform")
    print(f"   - Batch size: {config1['parameters']['batch_size']['values']}")
    print(f"   - Epochs: {config1['parameters']['epochs']['values']}")
    
    print("\n📝 Creating minimal sweep config (for quick testing)...")
    config2 = SweepConfig.create_minimal_sweep(num_trials=5)
    
    print(f"   Parameters optimized: {len(config2['parameters'])}")
    print(f"   - Learning rate: {config2['parameters']['learning_rate']['values']}")
    print(f"   - Batch size: {config2['parameters']['batch_size']['values']}")
    print(f"   - Epochs: {config2['parameters']['epochs']['values']}")
    
    print("\n📝 Creating custom sweep config...")
    custom_params = {
        'learning_rate': {'values': [1e-5, 5e-5]},
        'batch_size': {'values': [16, 32]},
        'model_type': {'values': ['whisper-tiny', 'whisper-base']}
    }
    config3 = SweepConfig.create_custom_sweep(
        parameters=custom_params,
        metric_name="validation/model_wer",
        goal="minimize",
        method="grid"  # Grid search for exhaustive testing
    )
    
    print(f"   Method: {config3['method']} (exhaustive search)")
    print(f"   Custom parameters: {list(config3['parameters'].keys())}")
    
    print("\n✅ Sweep configurations demo completed")
    return config2  # Return minimal config for next demo


def demo_2_create_sweep(sweep_config):
    """Demo 2: Creating a sweep on W&B."""
    print_section("DEMO 2: Creating a W&B Sweep")
    
    if not WANDB_AVAILABLE:
        print("⚠️  W&B not available. Skipping this demo.")
        print("   Install with: pip install wandb && wandb login")
        return None
    
    print("🚀 Initializing sweep orchestrator...")
    orchestrator = WandbSweepOrchestrator(
        project_name="stt-sweeps-demo",
        enabled=True
    )
    
    print("\n📊 Creating sweep on W&B...")
    sweep_id = orchestrator.create_sweep(
        sweep_config=sweep_config,
        sweep_name="demo_hyperparameter_optimization"
    )
    
    if sweep_id:
        print(f"   ✅ Sweep created: {sweep_id}")
        print(f"   View at: https://wandb.ai/")
        return sweep_id
    else:
        print("   ❌ Failed to create sweep")
        return None


def demo_3_run_sweep(sweep_id):
    """Demo 3: Running sweep trials."""
    print_section("DEMO 3: Running Sweep Trials")
    
    if not WANDB_AVAILABLE or not sweep_id:
        print("⚠️  Skipping - W&B not available or no sweep ID")
        return
    
    print("🔄 Running sweep agent (3 trials for demo)...")
    print("   Each trial tests different hyperparameter combinations")
    print("   W&B automatically tracks all metrics and compares runs")
    
    orchestrator = WandbSweepOrchestrator(
        project_name="stt-sweeps-demo",
        enabled=True
    )
    
    # Run limited trials for demo
    orchestrator.run_sweep_agent(
        train_function=demo_mock_training,
        sweep_id=sweep_id,
        count=3  # Just 3 trials for demo
    )
    
    print("\n✅ Sweep trials completed")
    print("   Check W&B dashboard to see:")
    print("   - Parallel coordinate plot of hyperparameters")
    print("   - Metric comparison across runs")
    print("   - Best performing configuration")


def demo_4_get_best_config(sweep_id):
    """Demo 4: Getting best hyperparameters."""
    print_section("DEMO 4: Getting Best Hyperparameters")
    
    if not WANDB_AVAILABLE or not sweep_id:
        print("⚠️  Skipping - W&B not available or no sweep ID")
        return None
    
    print("🔍 Analyzing sweep results...")
    
    orchestrator = WandbSweepOrchestrator(
        project_name="stt-sweeps-demo",
        enabled=True
    )
    
    # Give W&B a moment to process results
    time.sleep(2)
    
    best_config = orchestrator.get_best_run(
        sweep_id=sweep_id,
        metric_name="validation/model_wer",
        minimize=True
    )
    
    if best_config:
        print(f"\n✅ Best configuration found:")
        print(f"   Run: {best_config['run_name']}")
        print(f"   Best WER: {best_config['metric_value']:.4f}")
        print(f"\n   Optimal Hyperparameters:")
        for param, value in best_config['hyperparameters'].items():
            print(f"   - {param}: {value}")
        
        # Save to file
        output_path = "experiments/best_hyperparameters.json"
        orchestrator.save_best_config(output_path, sweep_id)
        print(f"\n   💾 Saved to: {output_path}")
        
        return best_config
    else:
        print("   ⚠️  Could not retrieve best configuration")
        return None


def demo_5_sweep_strategies():
    """Demo 5: Different sweep strategies."""
    print_section("DEMO 5: Sweep Strategy Comparison")
    
    print("📊 Available sweep strategies:\n")
    
    print("1️⃣  RANDOM SEARCH")
    print("   - Fast and efficient")
    print("   - Good for initial exploration")
    print("   - Recommended starting point")
    print("   - Works well with 20-50 trials")
    
    print("\n2️⃣  BAYESIAN OPTIMIZATION")
    print("   - Smart, adaptive search")
    print("   - Learns from previous trials")
    print("   - More efficient than random")
    print("   - Best for expensive training")
    print("   - Works well with 30-100 trials")
    
    print("\n3️⃣  GRID SEARCH")
    print("   - Exhaustive search")
    print("   - Tests all combinations")
    print("   - Most thorough but slowest")
    print("   - Good for final tuning")
    print("   - Use when you have specific values to test")
    
    print("\n💡 Recommendations for this project:")
    print("   - Start with RANDOM (10-20 trials)")
    print("   - Refine with BAYESIAN (20-30 trials)")
    print("   - Final tune with GRID (if needed)")
    
    # Show config examples
    print("\n📝 Example configurations:")
    
    print("\n   Random Search (recommended):")
    print("   ```python")
    print("   config = SweepConfig.create_finetuning_sweep(")
    print("       method='random',")
    print("       num_trials=20")
    print("   )")
    print("   ```")
    
    print("\n   Bayesian Optimization:")
    print("   ```python")
    print("   config = SweepConfig.create_finetuning_sweep(")
    print("       method='bayes',")
    print("       num_trials=30")
    print("   )")
    print("   ```")


def demo_6_integration_example():
    """Demo 6: Integration with fine-tuning pipeline."""
    print_section("DEMO 6: Integration with Fine-Tuning Pipeline")
    
    print("🔗 How to integrate sweeps with your fine-tuning:")
    
    print("\n1️⃣  Setup (one-time):")
    print("```python")
    print("from src.data.wandb_sweeps import WandbSweepOrchestrator, SweepConfig")
    print("")
    print("# Create sweep")
    print("sweep_orch = WandbSweepOrchestrator(project_name='my-project')")
    print("sweep_config = SweepConfig.create_finetuning_sweep(method='random')")
    print("sweep_id = sweep_orch.create_sweep(sweep_config)")
    print("```")
    
    print("\n2️⃣  Define training function:")
    print("```python")
    print("def train():")
    print("    # Get hyperparameters from sweep")
    print("    config = wandb.config")
    print("    ")
    print("    # Use them in training")
    print("    job = orchestrator.trigger_finetuning()")
    print("    result = train_model(")
    print("        learning_rate=config.learning_rate,")
    print("        batch_size=config.batch_size")
    print("    )")
    print("    ")
    print("    # Log metrics")
    print("    wandb.log({'validation/model_wer': result['wer']})")
    print("```")
    
    print("\n3️⃣  Run sweep:")
    print("```python")
    print("# Run 20 trials")
    print("sweep_orch.run_sweep_agent(train, sweep_id, count=20)")
    print("")
    print("# Get best hyperparameters")
    print("best = sweep_orch.get_best_run(sweep_id)")
    print("")
    print("# Use best config for production")
    print("production_config = best['hyperparameters']")
    print("```")
    
    print("\n💡 Benefits:")
    print("   ✅ Automatically finds optimal hyperparameters")
    print("   ✅ Saves hours of manual tuning")
    print("   ✅ Improves model performance")
    print("   ✅ Tracks all experiments")
    print("   ✅ Reproducible results")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("  W&B SWEEPS - HYPERPARAMETER OPTIMIZATION DEMO")
    print("="*80)
    print("\nAutomatic hyperparameter tuning for fine-tuning!")
    print("\nTopics covered:")
    print("  1. Creating sweep configurations")
    print("  2. Launching sweeps on W&B")
    print("  3. Running sweep trials")
    print("  4. Getting best hyperparameters")
    print("  5. Sweep strategy comparison")
    print("  6. Integration with fine-tuning")
    
    try:
        # Demo 1: Create configs
        sweep_config = demo_1_create_sweep_config()
        
        # Demo 2-4: Run actual sweep (if W&B available)
        sweep_id = demo_2_create_sweep(sweep_config)
        if sweep_id:
            demo_3_run_sweep(sweep_id)
            best_config = demo_4_get_best_config(sweep_id)
        
        # Demo 5-6: Educational content
        demo_5_sweep_strategies()
        demo_6_integration_example()
        
        # Summary
        print_section("DEMO COMPLETE - Summary")
        
        print("✅ Successfully demonstrated:")
        print("  ✓ Sweep configuration creation")
        print("  ✓ Different optimization strategies")
        print("  ✓ Best hyperparameter extraction")
        print("  ✓ Integration patterns")
        
        print("\n🎯 Next Steps:")
        print("  1. Create your sweep config")
        print("  2. Integrate with your training function")
        print("  3. Run sweep with 10-20 trials")
        print("  4. Use best hyperparameters in production")
        
        print("\n📊 Expected Improvements:")
        print("  • 10-30% better WER/CER")
        print("  • Faster convergence")
        print("  • More stable training")
        print("  • Optimal resource usage")
        
        print("\n💡 View Results:")
        print("   https://wandb.ai/ → Your Project → Sweeps Tab")
        
        print("\n" + "="*80)
        print("  Demo completed successfully!")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("   Make sure wandb is installed: pip install wandb")
        print("   And you're logged in: wandb login")


if __name__ == "__main__":
    main()

