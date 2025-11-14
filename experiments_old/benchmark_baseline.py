from src.benchmark import BaselineBenchmark

benchmark = BaselineBenchmark(model_name="whisper")
report = benchmark.generate_report(["test_audio_1.wav", "test_audio_2.wav"])
benchmark.save_report(report)