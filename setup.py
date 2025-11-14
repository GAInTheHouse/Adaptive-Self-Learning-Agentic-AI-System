from setuptools import setup, find_packages

setup(
    name="stt-agentic-ai",
    version="0.1.0",
    description="Adaptive Self-Learning Agentic AI System for Speech-to-Text",
    author="Your Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "jiwer>=3.0.0",
        "google-cloud-storage>=2.10.0",
        "gcsfs>=2023.6.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)
