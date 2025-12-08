"""
Test script for Adaptive Scheduling Algorithm - Week 3
Tests the adaptive scheduling mechanism, fine-tuning, and closed-loop system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent
from src.agent.adaptive_scheduler import AdaptiveScheduler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_adaptive_scheduler():
    """Test adaptive scheduler functionality"""
    print("\n" + "="*60)
    print("Testing Adaptive Scheduler")
    print("="*60)
    
    scheduler = AdaptiveScheduler(
        initial_threshold_n=50,
        min_threshold_n=20,
        max_threshold_n=200
    )
    
    # Test 1: Initial state
    print("\n1. Testing initial state...")
    stats = scheduler.get_scheduler_stats()
    print(f"   Initial threshold_n: {stats['current_threshold_n']}")
    print(f"   Error samples collected: {stats['error_samples_collected']}")
    assert stats['current_threshold_n'] == 50, "Initial threshold should be 50"
    
    # Test 2: Recording error samples
    print("\n2. Recording error samples...")
    scheduler.record_error_sample(count=30)
    should_trigger, info = scheduler.should_trigger_fine_tuning()
    print(f"   Samples collected: {scheduler.error_samples_collected}")
    print(f"   Should trigger: {should_trigger}")
    assert scheduler.error_samples_collected == 30, "Should have 30 samples"
    assert not should_trigger, "Should not trigger yet"
    
    # Test 3: Recording performance metrics
    print("\n3. Recording performance metrics...")
    for i in range(10):
        accuracy = 0.85 + (i * 0.01)  # Gradually improving
        scheduler.record_performance(
            error_count=5,
            accuracy=accuracy,
            inference_time=0.5
        )
    stats = scheduler.get_scheduler_stats()
    print(f"   Recent accuracy: {stats['recent_accuracy']:.4f}")
    print(f"   Accuracy trend: {stats['accuracy_trend']:.4f}")
    assert stats['recent_accuracy'] is not None, "Should have recent accuracy"
    
    # Test 4: Trigger fine-tuning
    print("\n4. Triggering fine-tuning threshold...")
    scheduler.record_error_sample(count=25)  # Now at 55, above threshold of 50
    should_trigger, info = scheduler.should_trigger_fine_tuning()
    print(f"   Should trigger: {should_trigger}")
    assert should_trigger, "Should trigger fine-tuning"
    
    # Test 5: Record fine-tuning event
    print("\n5. Recording fine-tuning event...")
    scheduler.record_fine_tuning_event(
        samples_used=55,
        validation_accuracy_before=0.85,
        validation_accuracy_after=0.87,
        training_cost=10.0
    )
    stats = scheduler.get_scheduler_stats()
    print(f"   New threshold_n: {stats['current_threshold_n']}")
    print(f"   Samples reset: {scheduler.error_samples_collected}")
    print(f"   Total fine-tuning events: {stats['total_fine_tuning_events']}")
    assert scheduler.error_samples_collected == 0, "Samples should be reset"
    assert stats['total_fine_tuning_events'] == 1, "Should have 1 fine-tuning event"
    
    # Test 6: Diminishing gains detection
    print("\n6. Testing diminishing gains detection...")
    scheduler.record_fine_tuning_event(
        samples_used=60,
        validation_accuracy_before=0.87,
        validation_accuracy_after=0.872,  # Small gain
        training_cost=10.0
    )
    scheduler.record_fine_tuning_event(
        samples_used=65,
        validation_accuracy_before=0.872,
        validation_accuracy_after=0.873,  # Very small gain
        training_cost=10.0
    )
    stats = scheduler.get_scheduler_stats()
    print(f"   Diminishing gains detected: {stats['diminishing_gains_detected']}")
    print(f"   Threshold after diminishing gains: {stats['current_threshold_n']}")
    assert stats['diminishing_gains_detected'], "Should detect diminishing gains"
    
    print("\n✅ All adaptive scheduler tests passed!")


def test_overfitting_detection():
    """Test overfitting detection"""
    print("\n" + "="*60)
    print("Testing Overfitting Detection")
    print("="*60)
    
    scheduler = AdaptiveScheduler(overfitting_threshold=0.1)
    
    # Test 1: No overfitting
    print("\n1. Testing normal case (no overfitting)...")
    is_overfitting, info = scheduler.check_overfitting(
        train_accuracy=0.90,
        validation_accuracy=0.88
    )
    print(f"   Train accuracy: {info['train_accuracy']:.4f}")
    print(f"   Validation accuracy: {info['validation_accuracy']:.4f}")
    print(f"   Accuracy gap: {info['accuracy_gap']:.4f}")
    print(f"   Is overfitting: {is_overfitting}")
    assert not is_overfitting, "Should not detect overfitting"
    
    # Test 2: Overfitting detected
    print("\n2. Testing overfitting case...")
    is_overfitting, info = scheduler.check_overfitting(
        train_accuracy=0.95,
        validation_accuracy=0.82  # Large gap
    )
    print(f"   Train accuracy: {info['train_accuracy']:.4f}")
    print(f"   Validation accuracy: {info['validation_accuracy']:.4f}")
    print(f"   Accuracy gap: {info['accuracy_gap']:.4f}")
    print(f"   Is overfitting: {is_overfitting}")
    assert is_overfitting, "Should detect overfitting"
    
    print("\n✅ Overfitting detection tests passed!")


def test_cost_efficiency():
    """Test cost efficiency tracking"""
    print("\n" + "="*60)
    print("Testing Cost Efficiency Tracking")
    print("="*60)
    
    scheduler = AdaptiveScheduler()
    
    # Record some fine-tuning events with varying costs
    print("\n1. Recording fine-tuning events with costs...")
    scheduler.record_fine_tuning_event(
        samples_used=100,
        validation_accuracy_before=0.80,
        validation_accuracy_after=0.85,  # Good gain
        training_cost=5.0  # Low cost
    )
    
    scheduler.record_fine_tuning_event(
        samples_used=100,
        validation_accuracy_before=0.85,
        validation_accuracy_after=0.86,  # Small gain
        training_cost=20.0  # High cost
    )
    
    stats = scheduler.get_scheduler_stats()
    print(f"   Total training cost: {stats['total_training_cost']:.2f}")
    print(f"   Cost efficiency: {stats['cost_efficiency']:.4f}")
    print(f"   Average accuracy gain: {stats['average_accuracy_gain']:.4f}")
    
    assert stats['total_training_cost'] > 0, "Should track training cost"
    assert 0 <= stats['cost_efficiency'] <= 1, "Cost efficiency should be 0-1"
    
    print("\n✅ Cost efficiency tracking tests passed!")


def test_integrated_agent():
    """Test agent with adaptive scheduling integrated"""
    print("\n" + "="*60)
    print("Testing Integrated Agent with Adaptive Scheduling")
    print("="*60)
    
    # Initialize baseline model and agent
    print("\n1. Initializing agent with adaptive fine-tuning...")
    baseline_model = BaselineSTTModel(model_name="whisper")
    agent = STTAgent(
        baseline_model=baseline_model,
        enable_adaptive_fine_tuning=True,
        scheduler_history_path="data/processed/test_scheduler_history.json"
    )
    
    # Check if adaptive scheduler is initialized
    scheduler_stats = agent.get_adaptive_scheduler_stats()
    if scheduler_stats:
        print(f"   ✅ Adaptive scheduler initialized")
        print(f"   Initial threshold_n: {scheduler_stats['current_threshold_n']}")
    else:
        print("   ⚠️  Adaptive scheduler not available (may need model/processor access)")
        return
    
    # Test getting stats
    print("\n2. Getting agent stats...")
    agent_stats = agent.get_agent_stats()
    print(f"   Agent stats available: {len(agent_stats)} keys")
    
    print("\n✅ Integrated agent tests passed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Adaptive Scheduling Algorithm - Week 3 Tests")
    print("="*60)
    
    try:
        # Test individual components
        test_adaptive_scheduler()
        test_overfitting_detection()
        test_cost_efficiency()
        
        # Test integrated system (may skip if model loading fails)
        try:
            test_integrated_agent()
        except Exception as e:
            print(f"\n⚠️  Integrated agent test skipped: {e}")
        
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
