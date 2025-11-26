"""
Data Management Module for Self-Learning STT Agent

This module provides comprehensive data management, versioning,
and fine-tuning pipeline capabilities.
"""

from .data_manager import DataManager, FailedCase
from .metadata_tracker import MetadataTracker, PerformanceMetrics
from .finetuning_pipeline import FinetuningDatasetPipeline, DatasetSplit
from .version_control import DataVersionControl, DataVersion, QualityMetrics

__all__ = [
    'DataManager',
    'FailedCase',
    'MetadataTracker',
    'PerformanceMetrics',
    'FinetuningDatasetPipeline',
    'DatasetSplit',
    'DataVersionControl',
    'DataVersion',
    'QualityMetrics'
]

