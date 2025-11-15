from src.baseline_model import BaselineSTTModel

model = BaselineSTTModel(model_name="whisper")
result = model.transcribe("test_audio.wav")
print(result)
