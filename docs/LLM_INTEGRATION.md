# LLM Integration (Gemma)

Gemma LLM integration for intelligent transcript error correction.

## Usage

```python
agent = STTAgent(
    baseline_model=baseline_model,
    use_llm_correction=True,
    llm_model_name="google/gemma-2b-it",
    use_quantization=False  # True to save memory
)
result = agent.transcribe_with_agent("audio.wav", enable_auto_correction=True)
```

## Architecture

```
STT Transcription → Error Detection → Gemma LLM Correction → Final Transcript
                                      ↓ (if unavailable)
                                   Rule-based Correction
```

## Configuration

- **Default model**: `google/gemma-2b-it`
- **Quantization**: `use_quantization=True` reduces memory ~50% (requires bitsandbytes + CUDA)
- **Disable LLM**: `use_llm_correction=False` for rule-based only

## Files

- `src/agent/llm_corrector.py` - LLM corrector
- `src/agent/agent.py` - Integration
