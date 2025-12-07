# Week 4 Quick Reference Guide
## System Integration & Testing

---

## 🚀 Quick Start

### Unified System

```python
from src.integration import UnifiedSTTSystem

# Initialize unified system
system = UnifiedSTTSystem(
    model_name="whisper",
    enable_error_detection=True,
    enable_llm_correction=True,
    enable_adaptive_fine_tuning=True
)

# Transcribe with full pipeline
result = system.transcribe("audio.wav", reference_transcript="...")

# Batch evaluation
results = system.evaluate_batch(audio_files, reference_transcripts)
```

---

## 🧪 Testing Frameworks

### End-to-End Testing

```python
from src.integration import UnifiedSTTSystem, EndToEndTester

system = UnifiedSTTSystem()
tester = EndToEndTester(system)

# Run full test suite
results = tester.run_full_test_suite(audio_files, reference_transcripts)

# Individual tests
feedback_results = tester.test_feedback_loop(audio_files, references, num_iterations=3)
error_results = tester.test_error_detection_accuracy(audio_files, references)
correction_results = tester.test_correction_effectiveness(audio_files, references)
finetuning_results = tester.test_fine_tuning_impact(audio_files, references)
```

### Statistical Analysis

```python
from src.integration import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Compare two systems
comparison = analyzer.compare_systems(
    baseline_scores=[0.25, 0.30, 0.28],
    treatment_scores=[0.20, 0.22, 0.21],
    system_a_name="Baseline",
    system_b_name="Full System"
)

# Paired t-test
result = analyzer.paired_t_test(
    baseline_scores=[0.25, 0.30, 0.28],
    treatment_scores=[0.20, 0.22, 0.21],
    alpha=0.05
)

print(f"p-value: {result['p_value']:.4f}")
print(f"Significant: {result['is_significant']}")
print(f"Effect size: {result['cohens_d']:.4f}")
```

### Ablation Studies

```python
from src.integration import AblationStudy

study = AblationStudy()

# Run ablation study
results = study.run_ablation_study(
    audio_files=["audio1.wav", "audio2.wav"],
    reference_transcripts=["ref1", "ref2"],
    model_name="whisper"
)

# Generate report
report = study.generate_ablation_report(results, output_path="ablation_report.txt")
```

### Comprehensive Test Suite

```python
from experiments.comprehensive_test_suite import ComprehensiveTestSuite

# Run all tests
suite = ComprehensiveTestSuite(output_dir="experiments/test_outputs")
results = suite.run_all_tests(
    audio_files=audio_files,
    reference_transcripts=references,
    model_name="whisper"
)
```

---

## 📊 Key Methods

### UnifiedSTTSystem

- `transcribe()` - Transcribe single audio file
- `evaluate_batch()` - Evaluate on batch of files
- `get_system_status()` - Get system status
- `get_component_contributions()` - Get component info

### EndToEndTester

- `test_feedback_loop()` - Test feedback loop
- `test_error_detection_accuracy()` - Test error detection
- `test_correction_effectiveness()` - Test corrections
- `test_fine_tuning_impact()` - Test fine-tuning
- `run_full_test_suite()` - Run all tests

### StatisticalAnalyzer

- `paired_t_test()` - Perform paired t-test
- `compare_systems()` - Compare two systems
- `analyze_component_contributions()` - Analyze components
- `analyze_trajectory()` - Analyze performance trajectory
- `generate_report()` - Generate report

### AblationStudy

- `run_ablation_study()` - Run ablation study
- `generate_ablation_report()` - Generate report

---

## 📈 Test Configurations

### Ablation Study Configurations

1. **baseline_only** - Baseline STT model only
2. **baseline_error_detection** - + Error Detection
3. **baseline_error_self_learning** - + Self-Learning
4. **baseline_error_llm** - + LLM Correction
5. **full_no_finetuning** - Full system without fine-tuning
6. **full_system** - Full system with all components

---

## 🔍 Statistical Interpretation

### p-value:
- **< 0.05**: Statistically significant
- **≥ 0.05**: Not statistically significant

### Effect Size (Cohen's d):
- **< 0.2**: Small effect
- **0.2 - 0.5**: Medium effect
- **≥ 0.5**: Large effect

### Example Output:
```python
{
    'p_value': 0.001,  # Significant!
    'is_significant': True,
    'cohens_d': 0.6,   # Large effect
    'mean_difference': -0.05,  # Improvement
    'interpretation': 'Treatment is significantly better'
}
```

---

## 📝 Running Tests

### Command Line:

```bash
python experiments/comprehensive_test_suite.py \
    --audio-dir data/test_audio \
    --references data/test_references.json \
    --output-dir experiments/test_outputs
```

### Python:

```python
from experiments.comprehensive_test_suite import ComprehensiveTestSuite

suite = ComprehensiveTestSuite()
results = suite.run_all_tests(audio_files, references)
```

---

## 📁 Output Files

### Test Results:
- `test_results_YYYYMMDD_HHMMSS.json` - Detailed results
- `test_report_YYYYMMDD_HHMMSS.txt` - Text report
- `ablation_report_YYYYMMDD_HHMMSS.txt` - Ablation report

---

## 🎯 Common Use Cases

### 1. Compare Baseline vs Full System

```python
baseline_system = UnifiedSTTSystem(
    enable_error_detection=False,
    enable_llm_correction=False,
    enable_adaptive_fine_tuning=False
)

full_system = UnifiedSTTSystem(
    enable_error_detection=True,
    enable_llm_correction=True,
    enable_adaptive_fine_tuning=True
)

# Evaluate both
baseline_scores = [baseline_system.transcribe(f, r)['evaluation']['wer'] 
                   for f, r in zip(files, refs)]
full_scores = [full_system.transcribe(f, r)['evaluation']['wer'] 
               for f, r in zip(files, refs)]

# Compare statistically
analyzer = StatisticalAnalyzer()
comparison = analyzer.compare_systems(baseline_scores, full_scores)
```

### 2. Test Component Contributions

```python
study = AblationStudy()
results = study.run_ablation_study(audio_files, references)

# Check which components are significant
contributions = results['contribution_analysis']['component_contributions']
for component, contrib in contributions.items():
    if contrib['is_significant']:
        print(f"{component}: {contrib['improvement']:.4f} improvement")
```

### 3. Test Feedback Loop

```python
tester = EndToEndTester(system)
results = tester.test_feedback_loop(
    audio_files, references,
    num_iterations=5
)

# Check if performance improves
analysis = results['feedback_analysis']
if analysis['improvement_trend']:
    print(f"Performance improved by {analysis['total_improvement']:.4f}")
```

---

## ⚠️ Important Notes

1. **Paired Data**: Paired t-tests require same number of samples for baseline and treatment
2. **Statistical Significance**: p < 0.05 indicates statistical significance
3. **Effect Size**: Consider both p-value and effect size for practical significance
4. **Ablation Studies**: Test systematically by removing one component at a time
5. **Multiple Comparisons**: Consider Bonferroni correction for multiple tests

---

## 📚 Related Files

- `src/integration/unified_system.py` - Unified system
- `src/integration/end_to_end_testing.py` - E2E testing
- `src/integration/statistical_analysis.py` - Statistical analysis
- `src/integration/ablation_studies.py` - Ablation studies
- `experiments/comprehensive_test_suite.py` - Test suite
- `WEEK4_DELIVERABLES_REPORT.md` - Full documentation

---

**Status**: ✅ Week 4 features complete and ready to use!
