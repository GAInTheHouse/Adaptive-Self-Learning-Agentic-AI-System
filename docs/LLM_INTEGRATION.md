# LLM Integration (Ollama + Llama)

Ollama-based Llama integration for intelligent transcript error correction.

## Prerequisites

- **Ollama** installed: [https://ollama.ai/download](https://ollama.ai/download)
- **Llama model** pulled (e.g., `ollama pull llama3.2:3b`)

## Usage

```python
from src.agent import STTAgent
from src.baseline_model import BaselineSTTModel

agent = STTAgent(
    baseline_model=BaselineSTTModel(model_name="whisper"),
    use_llm_correction=True,
    llm_model_name="llama3.2:3b"  # Default; use llama3.1:8b, llama2:7b, etc.
)
result = agent.transcribe_with_agent("audio.wav", enable_auto_correction=True)
```

If Ollama is unavailable, the agent automatically falls back to rule-based correction without raising.

## Architecture

```
STT Transcription → Error Detection → Ollama (Llama) Correction → Final Transcript
                                      ↓ (if unavailable)
                                   Rule-based Correction
```

## Configuration

- **Default model**: `llama3.2:3b`
- **Supported models**: `llama3.2:3b`, `llama3.1:8b`, `llama2:7b` (or any model pulled in Ollama)
- **Ollama URL**: `http://localhost:11434` (hardcoded; customize via `LlamaLLMCorrector` if needed)
- **Disable LLM**: `use_llm_correction=False` for rule-based only

## Verification

```bash
# Check available Ollama models
python scripts/check_ollama_models.py

# Test LLM connection and correction
python scripts/test_llm_connection.py
```

## Files

- `src/agent/ollama_llm.py` - Ollama client (OllamaLLM)
- `src/agent/llm_corrector.py` - LlamaLLMCorrector integration
- `src/agent/agent.py` - Agent integration
