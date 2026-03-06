# Evaluation Metrics Upgrade

## Overview

The unified evaluator (`src/evaluation/metrics.py`) has been upgraded with additional metrics for comprehensive speech-to-text evaluation.

## New Metrics

### 1. Diarization Error Rate (DER)

**Purpose**: Measures accuracy of speaker diarization (who spoke when).

**Formula**: `DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Reference Duration`

**Usage**:
```python
from src.evaluation.metrics import STTEvaluator

evaluator = STTEvaluator()
der_score = evaluator.calculate_der(
    reference_segments=[
        {'start': 0.0, 'end': 5.0, 'speaker': 'A'},
        {'start': 5.0, 'end': 10.0, 'speaker': 'B'}
    ],
    hypothesis_segments=[
        {'start': 0.1, 'end': 5.1, 'speaker': 'A'},
        {'start': 5.1, 'end': 10.1, 'speaker': 'B'}
    ],
    tolerance=0.25  # 250ms tolerance collar
)
```

**Note**: Requires speaker segment information (start, end, speaker ID) in RTTM-like format.

### 2. Verb Error Rate

**Purpose**: Measures accuracy of verb transcription specifically, as verbs are critical for meaning.

**Calculation**: Compares verbs extracted from reference vs hypothesis using NLTK POS tagging.

**Usage**:
```python
evaluator = STTEvaluator()
verb_rate = evaluator.calculate_verb_error_rate(
    reference="The patient was diagnosed with pneumonia.",
    hypothesis="The patient was diagnose with pneumonia."
)
```

**Returns**: Error rate (0-1), where 0 is perfect and 1 is complete failure.

### 3. Domain Error Rate

**Purpose**: Measures accuracy within specific domains (medical, legal, technical, business).

**Calculation**: Groups transcripts by detected domain and calculates domain-specific WER.

**Usage**:
```python
evaluator = STTEvaluator()
domain_rates = evaluator.calculate_domain_error_rate(
    references=["Patient shows symptoms of fever.", "The court ruled in favor."],
    hypotheses=["Patient show symptom of fever.", "The court rule in favor."]
)
# Returns: {'medical': 0.33, 'legal': 0.25}
```

**Customization**: Domain keywords can be customized via `evaluator.domain_keywords`.

## Complete Evaluation Example

```python
from src.evaluation.metrics import STTEvaluator

evaluator = STTEvaluator()

results = evaluator.evaluate_batch(
    references=["Reference transcript 1", "Reference transcript 2"],
    hypotheses=["Hypothesis transcript 1", "Hypothesis transcript 2"],
    include_verb_rate=True,
    include_domain_rate=True,
    include_der=False,  # Requires segments
    reference_segments=None,
    hypothesis_segments=None
)

print(f"WER: {results['wer']:.4f}")
print(f"CER: {results['cer']:.4f}")
print(f"Verb Error Rate: {results['verb_error_rate']:.4f}")
print(f"Domain Error Rates: {results['domain_error_rates']}")
```

## Dependencies

New dependencies added to `requirements.txt`:
- `nltk>=3.8.0` - For POS tagging (verb extraction)
- `pyannote.metrics>=4.0.0` - For advanced DER calculation (optional)

## Model Investigation Scripts

### AReal (RealtimeSTT) Investigation

**Script**: `experiments/investigate_areal.py`

**Purpose**: Evaluate AReal/RealtimeSTT model performance on available data.

**Usage**:
```bash
python experiments/investigate_areal.py
```

**Output**: 
- Latency metrics
- WER, CER, Verb Error Rate, Domain Error Rate
- Results saved to `experiments/evaluation_outputs/areal_evaluation_results.json`

### Miles/Moonshine Investigation

**Script**: `experiments/investigate_miles.py`

**Purpose**: Evaluate Miles/Moonshine model performance on available data.

**Usage**:
```bash
python experiments/investigate_miles.py
```

**Output**:
- Latency metrics
- WER, CER, Verb Error Rate, Domain Error Rate
- Results saved to `experiments/evaluation_outputs/miles_moonshine_evaluation_results.json`

## Oracle Teacher Script

**Script**: `experiments/oracle_teacher.py`

**Purpose**: Generate synthetic "gold" transcripts using GPT-4o/Llama 3 API for cases without ground truth.

**Usage**:
```bash
# Using OpenAI GPT-4o
export OPENAI_API_KEY='your-key'
python experiments/oracle_teacher.py \
    --audio-dir data \
    --output experiments/evaluation_outputs/oracle_gold_transcripts.json \
    --api-type openai \
    --model gpt-4o \
    --limit 10

# Using Ollama (Llama 3)
python experiments/oracle_teacher.py \
    --audio-dir data \
    --output experiments/evaluation_outputs/oracle_gold_transcripts.json \
    --api-type llama \
    --model llama3 \
    --limit 10
```

**How it works**:
1. Gets baseline transcript from Whisper
2. Refines transcript using LLM (GPT-4o/Llama 3)
3. LLM fixes errors, adds punctuation, corrects grammar
4. Outputs high-quality "gold" transcript

**Output Format**:
```json
[
  {
    "audio_file": "data/test.wav",
    "baseline_transcript": "the patient was diagnose with pneumonia",
    "gold_transcript": "The patient was diagnosed with pneumonia.",
    "refinement_time": 2.5,
    "model": "gpt-4o",
    "api_type": "openai"
  }
]
```

## Benefits

1. **Comprehensive Evaluation**: Multiple metrics provide different perspectives on model performance
2. **Domain-Specific Analysis**: Domain Error Rate helps identify domain-specific weaknesses
3. **Linguistic Accuracy**: Verb Error Rate focuses on critical grammatical elements
4. **Speaker Analysis**: DER enables multi-speaker evaluation
5. **Gold Transcript Generation**: Oracle Teacher creates high-quality references for evaluation

## Future Enhancements

- [ ] Add semantic similarity metrics (BERTScore, BLEU)
- [ ] Add confidence score analysis
- [ ] Add temporal alignment metrics
- [ ] Support for more domain keywords
- [ ] Batch processing optimization for Oracle Teacher
