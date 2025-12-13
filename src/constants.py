"""
Constants used across the Adaptive Self-Learning Agentic AI System.
Centralizes minimum sample counts and thresholds for consistency.
"""

# Fine-tuning sample requirements
MIN_SAMPLES_FOR_FINETUNING = 2  # Absolute minimum samples required for fine-tuning
RECOMMENDED_SAMPLES_FOR_FINETUNING = 10  # Recommended minimum for better results
SMALL_DATASET_THRESHOLD = 10  # Threshold below which dataset is considered "small"

# Fine-tuning orchestration triggers
MIN_ERROR_CASES_FOR_TRIGGER = 100  # Minimum error cases before triggering fine-tuning
MIN_CORRECTED_CASES_FOR_TRIGGER = 50  # Minimum corrected cases before triggering

# Dataset validation
MIN_VAL_SAMPLES_FOR_SMALL_DATASET = 1  # Minimum validation samples for small datasets (< 10 samples)

