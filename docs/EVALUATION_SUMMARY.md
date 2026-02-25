# Evaluation Summary

Single place for how to run evaluation, input options (including LLM as gold), outputs, verification, and the evaluation module.

---

## Script: `experiments/run_evaluation.py`

**Purpose:** Run any dataset on the **baseline (Whisper)** and on **improved (fine-tuned) models** if they exist. Supports three ways to get references (gold standard): from an eval set file, from a refs file, or **from the LLM** (Ollama).

**Outputs:** WER/CER (and optional benchmark) in `experiments/evaluation_outputs/`.

---

## Input: How to provide the dataset

### 1. Evaluation set file (recommended when you have ground truth)

A single file listing each audio path and its reference text.

```bash
python experiments/run_evaluation.py --eval-set path/to/eval_set.json
```

**Formats:** JSON, JSONL, CSV.

**Fields (any of these names):** Audio: `audio_path`, `audio`, or `path`. Reference: `reference`, `text`, or `target_text`.

**Example JSON:**
```json
[
  {"audio_path": "data/test_audio/sample1.wav", "reference": "hello world"},
  {"audio_path": "data/test_audio/sample2.wav", "reference": "goodbye"}
]
```

### 2. Audio directory + refs file

Discover WAV/MP3 in a directory and pair with references from a file.

```bash
python experiments/run_evaluation.py --audio-dir data/recordings_for_test --refs path/to/refs.json
```

- **`--audio-dir`:** Directory of `.wav` / `.mp3` (sorted by name).
- **`--refs`:** Same format as `--eval-set`. Matched by path or filename. Without `--refs`, WER/CER cannot be computed.

### 3. Audio directory with LLM as gold standard

Point to a directory of audio files; the script uses the **LLM (Ollama)** as gold: it checks that Ollama is running, gets a baseline transcript for each file, then asks the LLM to correct it and uses that corrected text as the reference for WER/CER.

```bash
python experiments/run_evaluation.py --audio-dir data/recordings_for_test --gold-from-llm
```

- **Requires:** Ollama running (e.g. `ollama serve`) and the model pulled (e.g. `ollama pull llama3.2:3b`).
- **Behavior:** Same LLM-gold logic as `scripts/finetune_wav2vec2.py`: baseline transcribes each file, then `LlamaLLMCorrector.correct_transcript(transcript, errors=[], context={})` gives the gold reference.
- **Optional:** `--llm-model llama3.2:3b` (default) to choose the Ollama model.

If the LLM is not available, the script exits with a clear message to start Ollama and pull the model.

---

## Output

- **`experiments/evaluation_outputs/evaluation_report.json`** – Full report: baseline and improved-model WER/CER, `gold_source` (eval_set / refs / llm), optional benchmark.
- **`experiments/evaluation_outputs/evaluation_report.txt`** – Short summary.
- With **`--benchmark`**: **`experiments/evaluation_outputs/benchmark_report.json`** – Latency/throughput from baseline.

---

## Options

| Option | Description |
|--------|-------------|
| `--eval-set PATH` | Eval set file (JSON/JSONL/CSV) with audio_path + reference. |
| `--audio-dir DIR` | Directory of WAV/MP3. Use with `--refs` or `--gold-from-llm`. |
| `--refs PATH` | Refs file (same format as --eval-set). Ignored if --gold-from-llm. |
| `--gold-from-llm` | Use LLM (Ollama) as gold; requires --audio-dir. |
| `--llm-model NAME` | Ollama model for --gold-from-llm (default: llama3.2:3b). |
| `--output-dir DIR` | Output directory (default: experiments/evaluation_outputs). |
| `--baseline-only` | Run only baseline, skip improved models. |
| `--benchmark` | Include latency/throughput benchmark. |

---

## Quick examples

```bash
# Eval set file
python experiments/run_evaluation.py --eval-set data/eval_set.json

# Audio dir + refs file
python experiments/run_evaluation.py --audio-dir data/recordings_for_test --refs data/refs.json

# Audio dir + LLM as gold (Ollama must be running)
python experiments/run_evaluation.py --audio-dir data/recordings_for_test --gold-from-llm

# Baseline only, with benchmark
python experiments/run_evaluation.py --eval-set data/eval_set.json --baseline-only --benchmark
```

---

## Verifying results

There is **no separate verification script**. To verify or report evaluation numbers:

1. **Run evaluation** with your chosen input (eval set, refs, or `--gold-from-llm`).
2. **Inspect the report:** Open `experiments/evaluation_outputs/evaluation_report.json` and check:
   - `baseline.wer`, `baseline.cer` for baseline metrics
   - `improved_models[*].wer`, `improved_models[*].cer` for fine-tuned models (if any)
   - `gold_source` to see how references were obtained (`eval_set` / `refs` / `llm`)
   - With `--benchmark`: latency/throughput in `benchmark_report.json` or inside the main report

**For reports/papers:** Cite values from `evaluation_report.json` as the single source of truth.

### What gets verified

When you run `run_evaluation.py` with a proper eval set (audio + reference), the following are **measured**:

- **Baseline WER/CER** – `baseline.wer`, `baseline.cer`
- **Improved model WER/CER** – `improved_models[*].wer`, `.cer` (if any improved models exist)
- **Model info** – `baseline.model_info`
- **Num samples** – `num_samples`
- **Latency/throughput** – If `--benchmark` was used

### Numbers that require ground truth

The following metrics require actual ground-truth reference transcripts (not LLM-generated) to verify:

- **Full system performance** (if you have a full system with error detection/correction)
- **Ablation study results** (component-specific contributions)
- **Statistical significance** (p-values, Cohen's d, confidence intervals)
- **Error detection precision/recall** (requires known errors)

**Note:** When using `--gold-from-llm`, the gold standard comes from LLM correction of baseline transcripts, which is useful for comparing model versions but not a substitute for human-verified ground truth for absolute accuracy claims.

---

## Unified evaluation module (`src.evaluation.metrics`)

Evaluation logic lives in one module: **streaming (inference)** and **batch/offline** test sets.

### Components

- **STTEvaluator** – Low-level WER/CER (single pair and batch).
- **EvaluationModule** – Unified API:
  - **Streaming:** `add_prediction(reference, hypothesis)` then `get_metrics()`; per-sample in `.results`.
  - **Batch:** `evaluate_batch(references, hypotheses)` or `evaluate_from_file(path)` for JSON/JSONL/CSV.

### Example

```python
from src.evaluation.metrics import EvaluationModule

eval_mod = EvaluationModule()
for ref, hyp in stream_of_predictions:
    eval_mod.add_prediction(ref, hyp)
metrics = eval_mod.get_metrics()  # {"wer", "cer", "num_samples"}
```

---

## Other components

- **BaselineSTTModel** – Loads Whisper or fine-tuned Wav2Vec2; used by `run_evaluation.py`.
- **LlamaLLMCorrector** (`src.agent.llm_corrector`) – Used for `--gold-from-llm`; same pattern as `scripts/finetune_wav2vec2.py`.
- **BaselineBenchmark** – Latency/throughput when you pass `--benchmark`.
- **Model versioning** – `get_all_model_versions()` / `get_current_model_path()` to discover improved models under `models/`.
- **scripts/finetune_wav2vec2.py** – Fine-tuning script; also runs baseline vs fine-tuned evaluation with LLM gold on its own test set.
- **src/data/model_validator.py** – Library to compare a model vs baseline on an evaluation set.
- **experiments/test_baseline.py** – Smoke test (load baseline, one inference).

---

## File structure (after a run)

```
experiments/evaluation_outputs/
├── evaluation_report.json   # Full report (baseline + improved models, gold_source)
├── evaluation_report.txt   # Short summary
└── benchmark_report.json   # If --benchmark was used
```

---

## Report metrics reference

| Metric | Source in report |
|--------|------------------|
| Baseline WER / CER | `evaluation_report.json` → `baseline.wer`, `baseline.cer` |
| Improved model WER / CER | `evaluation_report.json` → `improved_models[*].wer`, `.cer` |
| Latency / throughput | `benchmark_report.json` (when using `--benchmark`) |
| Num samples | `evaluation_report.json` → `num_samples` |
| Gold source | `evaluation_report.json` → `gold_source` |

**Conclusion:** Run `experiments/run_evaluation.py` with a proper eval set (audio + reference) to obtain verified metrics; use the generated report as the single source of truth for baseline and improved models.
