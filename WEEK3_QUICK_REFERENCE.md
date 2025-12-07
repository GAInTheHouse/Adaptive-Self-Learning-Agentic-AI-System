# Week 3 Quick Reference Guide
## Adaptive Scheduling Algorithm

---

## 🚀 Quick Start

### Initialize Agent with Adaptive Fine-Tuning

```python
from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent

# Initialize model
baseline_model = BaselineSTTModel(model_name="whisper")

# Initialize agent with adaptive fine-tuning enabled
agent = STTAgent(
    baseline_model=baseline_model,
    enable_adaptive_fine_tuning=True,
    scheduler_history_path="data/processed/scheduler_history.json"
)
```

### Use Agent (Fine-Tuning Triggers Automatically)

```python
# Transcribe audio - automatically tracks performance and triggers fine-tuning
result = agent.transcribe_with_agent("audio.wav")

# Check if fine-tuning was triggered
if result['agent_metadata']['fine_tuning_triggered']:
    print("Fine-tuning was triggered!")
```

---

## 📊 Monitoring Scheduler

### Get Scheduler Statistics

```python
stats = agent.get_adaptive_scheduler_stats()

print(f"Current threshold_n: {stats['current_threshold_n']}")
print(f"Error samples collected: {stats['error_samples_collected']}")
print(f"Samples until fine-tuning: {stats['samples_until_fine_tuning']}")
print(f"Recent accuracy: {stats['recent_accuracy']:.4f}")
print(f"Accuracy trend: {stats['accuracy_trend']:.4f}")
print(f"Diminishing gains detected: {stats['diminishing_gains_detected']}")
print(f"Cost efficiency: {stats['cost_efficiency']:.4f}")
print(f"Total fine-tuning events: {stats['total_fine_tuning_events']}")
```

---

## ⚙️ Configuration

### Adaptive Scheduler Configuration

```python
from src.agent import AdaptiveScheduler

scheduler = AdaptiveScheduler(
    initial_threshold_n=100,           # Starting threshold
    min_threshold_n=50,                # Minimum threshold
    max_threshold_n=1000,              # Maximum threshold
    performance_window_size=20,        # Metrics window size
    accuracy_gain_threshold=0.01,      # Min worthwhile gain
    diminishing_gain_threshold=0.005,  # Diminishing threshold
    cost_efficiency_weight=0.3,        # Cost weight (0-1)
    validation_split=0.2,              # Validation fraction
    overfitting_threshold=0.1,         # Max accuracy gap
    history_path="data/processed/scheduler_history.json"
)
```

### Fine-Tuner Configuration

```python
from src.agent import FineTuner

fine_tuner = FineTuner(
    model=model,
    processor=processor,
    device="cuda",
    validation_split=0.2,              # Validation fraction
    overfitting_threshold=0.1,         # Max accuracy gap
    early_stopping_patience=3,         # Early stopping patience
    min_accuracy_gain=0.01            # Min successful gain
)
```

---

## 🔧 Key Features

### 1. Dynamic Threshold Adjustment

The scheduler automatically adjusts threshold `n` based on:
- **Diminishing gains**: Increases `n` when accuracy gains are small
- **Cost efficiency**: Increases `n` when cost per gain is high
- **Performance trends**: Adjusts based on accuracy trends
- **Fine-tuning results**: Adapts based on recent fine-tuning outcomes

### 2. Performance-Aware Logic

- Tracks accuracy over time
- Detects diminishing returns
- Monitors cost per accuracy gain
- Adjusts threshold accordingly

### 3. Cost-Efficiency Tracking

- Tracks training costs
- Tracks inference costs
- Calculates cost efficiency
- Optimizes resource usage

### 4. Overfitting Prevention

- Automatic validation split
- Real-time overfitting detection
- Early stopping mechanism
- Best model checkpointing

---

## 📈 How It Works

### Closed-Loop Flow

```
1. Transcription → Error Detection
2. Record Performance Metrics
3. Collect Error Samples
4. Check Threshold n
5. Trigger Fine-Tuning? (if samples >= n)
6. Fine-Tune with Validation
7. Check Overfitting
8. Record Results
9. Adjust Threshold n
10. Continue Monitoring
```

### Threshold Adjustment Factors

| Factor | Condition | Adjustment |
|--------|-----------|------------|
| Diminishing Gains | Small accuracy gains | Increase by 20% |
| Poor Cost Efficiency | Cost efficiency < 0.5 | Increase by 15% |
| Small Gain | Gain < diminishing threshold | Increase by 30% |
| Good Gain | Gain > accuracy threshold | Decrease by 10% |
| Accuracy Declining | Trend < -0.01 | Decrease by 5% |
| Accuracy Improving | Trend > 0.01 | Increase by 10% |

---

## 🧪 Testing

### Run Tests

```bash
cd experiments
python test_adaptive_scheduler.py
```

### Manual Fine-Tuning Trigger

```python
# Manually trigger fine-tuning (for testing)
result = agent.manually_trigger_fine_tuning()
print(f"Success: {result['success']}")
print(f"Scheduler stats: {result['scheduler_stats']}")
```

---

## 📝 Key Methods

### AdaptiveScheduler

- `should_trigger_fine_tuning()` - Check if fine-tuning should trigger
- `record_performance()` - Record performance metrics
- `record_error_sample()` - Track error samples
- `adjust_threshold_n()` - Adjust threshold dynamically
- `check_overfitting()` - Detect overfitting
- `record_fine_tuning_event()` - Record fine-tuning results
- `get_scheduler_stats()` - Get comprehensive statistics

### FineTuner

- `fine_tune()` - Perform fine-tuning with validation
- `_evaluate()` - Evaluate model on dataset
- `_estimate_training_cost()` - Estimate training cost

### STTAgent (New Methods)

- `_trigger_adaptive_fine_tuning()` - Trigger fine-tuning
- `get_adaptive_scheduler_stats()` - Get scheduler stats
- `manually_trigger_fine_tuning()` - Manual trigger

---

## 🎯 Use Cases

### 1. Production Deployment

```python
# Enable adaptive fine-tuning for production
agent = STTAgent(
    baseline_model=baseline_model,
    enable_adaptive_fine_tuning=True
)

# Agent automatically adapts and fine-tunes
```

### 2. Monitoring and Debugging

```python
# Check scheduler status
stats = agent.get_adaptive_scheduler_stats()

# Monitor threshold adjustments
print(f"Threshold: {stats['current_threshold_n']}")
print(f"Diminishing gains: {stats['diminishing_gains_detected']}")
```

### 3. Custom Configuration

```python
# Create custom scheduler
scheduler = AdaptiveScheduler(
    initial_threshold_n=200,  # Higher threshold
    min_threshold_n=100,
    max_threshold_n=500
)

# Use with agent (requires custom integration)
```

---

## ⚠️ Important Notes

1. **Model Access**: Fine-tuning requires access to `model` and `processor` from baseline model
2. **GPU Recommended**: Fine-tuning is faster on GPU
3. **History Persistence**: Scheduler history is saved automatically
4. **Validation Split**: Default 20% for validation
5. **Overfitting Threshold**: Default 0.1 (10% accuracy gap)

---

## 🔍 Troubleshooting

### Fine-Tuning Not Triggering

- Check `error_samples_collected` vs `current_threshold_n`
- Verify error samples have corrections
- Check scheduler statistics

### Overfitting Detected

- Reduce learning rate
- Increase validation split
- Add regularization
- Increase early stopping patience

### High Costs

- Increase `initial_threshold_n`
- Adjust `cost_efficiency_weight`
- Monitor cost efficiency score

---

## 📚 Related Files

- `src/agent/adaptive_scheduler.py` - Scheduler implementation
- `src/agent/fine_tuner.py` - Fine-tuning module
- `src/agent/agent.py` - Integrated agent
- `experiments/test_adaptive_scheduler.py` - Test suite
- `WEEK3_DELIVERABLES_REPORT.md` - Full documentation

---

**Status**: ✅ Week 3 features complete and tested!
