"""
Fine-tuning Dataset Preparation Pipeline
Prepares datasets for model fine-tuning from failed cases and corrections.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from .data_manager import DataManager, FailedCase
from ..utils.gcs_utils import GCSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetSplit:
    """Represents a dataset split (train/val/test)."""
    
    def __init__(self, name: str):
        self.name = name
        self.samples: List[Dict] = []
    
    def add_sample(self, sample: Dict):
        """Add a sample to this split."""
        self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'size': len(self.samples),
            'samples': self.samples
        }


class FinetuningDatasetPipeline:
    """
    Comprehensive pipeline for preparing fine-tuning datasets from
    failed cases and corrections.
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        output_dir: str = "data/finetuning",
        use_gcs: bool = True,
        gcs_bucket_name: str = "stt-project-datasets",
        gcs_prefix: str = "finetuning_datasets",
        project_id: str = "stt-agentic-ai-2025"
    ):
        """
        Initialize fine-tuning pipeline.
        
        Args:
            data_manager: DataManager instance with failed cases
            output_dir: Output directory for prepared datasets
            use_gcs: Whether to use Google Cloud Storage
            gcs_bucket_name: GCS bucket name
            gcs_prefix: Prefix for GCS paths
            project_id: GCP project ID
        """
        self.data_manager = data_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gcs = use_gcs
        self.gcs_prefix = gcs_prefix
        
        # Initialize GCS manager
        self.gcs_manager = None
        if use_gcs:
            try:
                self.gcs_manager = GCSManager(project_id, gcs_bucket_name)
                logger.info("GCS integration enabled for fine-tuning pipeline")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS: {e}. Using local storage only.")
                self.use_gcs = False
        
        logger.info(f"Fine-tuning Pipeline initialized (output: {self.output_dir})")
    
    def prepare_dataset(
        self,
        min_error_score: float = 0.5,
        include_uncorrected: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        max_samples: Optional[int] = None,
        balance_error_types: bool = True
    ) -> Dict:
        """
        Prepare fine-tuning dataset from failed cases.
        
        Args:
            min_error_score: Minimum error score to include
            include_uncorrected: Include cases without corrections
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            test_ratio: Ratio for test split
            max_samples: Maximum number of samples to include
            balance_error_types: Balance samples across error types
        
        Returns:
            Dictionary with dataset information
        """
        logger.info("Preparing fine-tuning dataset...")
        
        # Validate split ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Split ratios must sum to 1.0"
        
        # Get eligible cases
        eligible_cases = self._get_eligible_cases(
            min_error_score=min_error_score,
            include_uncorrected=include_uncorrected
        )
        
        logger.info(f"Found {len(eligible_cases)} eligible cases")
        
        if not eligible_cases:
            logger.warning("No eligible cases found for dataset preparation")
            return {'error': 'No eligible cases'}
        
        # Balance error types if requested
        if balance_error_types:
            eligible_cases = self._balance_error_types(eligible_cases, max_samples)
        elif max_samples:
            eligible_cases = eligible_cases[:max_samples]
        
        logger.info(f"Using {len(eligible_cases)} cases for dataset")
        
        # Convert to samples
        samples = self._convert_to_samples(eligible_cases)
        
        # Split dataset
        splits = self._split_dataset(samples, train_ratio, val_ratio, test_ratio)
        
        # Generate dataset metadata
        dataset_id = self._generate_dataset_id()
        metadata = self._generate_metadata(dataset_id, splits, eligible_cases)
        
        # Save dataset
        dataset_info = self._save_dataset(dataset_id, splits, metadata)
        
        logger.info(f"Dataset prepared: {dataset_id}")
        logger.info(f"  Train: {len(splits['train'])} samples")
        logger.info(f"  Val: {len(splits['val'])} samples")
        logger.info(f"  Test: {len(splits['test'])} samples")
        
        return dataset_info
    
    def _get_eligible_cases(
        self,
        min_error_score: float,
        include_uncorrected: bool
    ) -> List[FailedCase]:
        """Get cases eligible for fine-tuning."""
        if include_uncorrected:
            cases = list(self.data_manager.failed_cases_cache.values())
        else:
            cases = self.data_manager.get_corrected_cases()
        
        # Filter by error score
        eligible = [
            case for case in cases
            if case.error_score >= min_error_score
        ]
        
        return eligible
    
    def _balance_error_types(
        self,
        cases: List[FailedCase],
        max_samples: Optional[int]
    ) -> List[FailedCase]:
        """Balance cases across different error types."""
        # Group by primary error type
        by_error_type = defaultdict(list)
        for case in cases:
            # Get primary error type, handling both list and string cases
            if case.error_types:
                primary_error = case.error_types[0] if isinstance(case.error_types, list) else case.error_types
                # If it's still a list (nested), get first element
                if isinstance(primary_error, list):
                    primary_error = primary_error[0] if primary_error else 'unknown'
            else:
                primary_error = 'unknown'
            by_error_type[primary_error].append(case)
        
        # Calculate samples per error type
        num_error_types = len(by_error_type)
        if max_samples:
            samples_per_type = max_samples // num_error_types
        else:
            # Use the minimum count across all types
            samples_per_type = min(len(cases) for cases in by_error_type.values())
        
        # Sample equally from each error type
        balanced = []
        for error_type, type_cases in by_error_type.items():
            sampled = type_cases[:samples_per_type]
            balanced.extend(sampled)
            logger.info(f"  {error_type}: {len(sampled)} samples")
        
        return balanced
    
    def _convert_to_samples(self, cases: List[FailedCase]) -> List[Dict]:
        """Convert failed cases to training samples."""
        samples = []
        
        for case in cases:
            # Normalize error_types to always be a flat list
            error_types = case.error_types if isinstance(case.error_types, list) else [case.error_types]
            # Flatten nested lists
            flat_error_types = []
            for et in error_types:
                if isinstance(et, list):
                    flat_error_types.extend(et)
                else:
                    flat_error_types.append(et)
            
            sample = {
                'case_id': case.case_id,
                'audio_path': case.audio_path,
                'input_text': case.original_transcript,
                'target_text': case.corrected_transcript or case.original_transcript,
                'error_types': flat_error_types,
                'error_score': case.error_score,
                'metadata': case.metadata,
                'has_correction': case.corrected_transcript is not None
            }
            samples.append(sample)
        
        return samples
    
    def _split_dataset(
        self,
        samples: List[Dict],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Dict[str, DatasetSplit]:
        """Split samples into train/val/test sets."""
        # Shuffle samples
        np.random.shuffle(samples)
        
        # Calculate split sizes
        total = len(samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Create splits
        splits = {
            'train': DatasetSplit('train'),
            'val': DatasetSplit('val'),
            'test': DatasetSplit('test')
        }
        
        # Assign samples to splits
        for i, sample in enumerate(samples):
            if i < train_size:
                splits['train'].add_sample(sample)
            elif i < train_size + val_size:
                splits['val'].add_sample(sample)
            else:
                splits['test'].add_sample(sample)
        
        return splits
    
    def _generate_dataset_id(self) -> str:
        """Generate unique dataset ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"finetuning_dataset_{timestamp}"
    
    def _generate_metadata(
        self,
        dataset_id: str,
        splits: Dict[str, DatasetSplit],
        cases: List[FailedCase]
    ) -> Dict:
        """Generate dataset metadata."""
        # Analyze error types
        error_type_counts = Counter()
        for case in cases:
            # Handle error_types being a list or nested list
            error_types = case.error_types if isinstance(case.error_types, list) else [case.error_types]
            # Flatten nested lists
            for et in error_types:
                if isinstance(et, list):
                    error_type_counts.update(et)
                else:
                    error_type_counts[et] += 1
        
        # Calculate statistics
        error_scores = [case.error_score for case in cases]
        
        metadata = {
            'dataset_id': dataset_id,
            'created_at': datetime.now().isoformat(),
            'total_samples': sum(len(split) for split in splits.values()),
            'split_sizes': {
                name: len(split) for name, split in splits.items()
            },
            'error_type_distribution': dict(error_type_counts),
            'error_score_stats': {
                'mean': float(np.mean(error_scores)),
                'std': float(np.std(error_scores)),
                'min': float(np.min(error_scores)),
                'max': float(np.max(error_scores))
            },
            'corrections_included': sum(
                1 for case in cases if case.corrected_transcript is not None
            ),
            'data_sources': {
                'failed_cases': len(cases),
                'data_manager_version': '1.0'
            }
        }
        
        return metadata
    
    def _save_dataset(
        self,
        dataset_id: str,
        splits: Dict[str, DatasetSplit],
        metadata: Dict
    ) -> Dict:
        """Save dataset to disk and optionally GCS."""
        dataset_dir = self.output_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        for split_name, split in splits.items():
            split_file = dataset_dir / f"{split_name}.jsonl"
            with open(split_file, 'w') as f:
                for sample in split.samples:
                    f.write(json.dumps(sample) + '\n')
            logger.info(f"Saved {split_name} split: {len(split)} samples")
        
        # Save metadata
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create dataset manifest with metadata included
        manifest = {
            'dataset_id': dataset_id,
            'created_at': metadata['created_at'],
            'splits': {
                name: f"{split_name}.jsonl" for name, split_name in 
                [('train', 'train'), ('val', 'val'), ('test', 'test')]
            },
            'metadata_file': 'metadata.json',
            'local_path': str(dataset_dir),
            # Include key metadata fields for easy access
            'total_samples': metadata['total_samples'],
            'split_sizes': metadata['split_sizes'],
            'error_type_distribution': metadata['error_type_distribution']
        }
        
        manifest_file = dataset_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            self._sync_dataset_to_gcs(dataset_id, dataset_dir)
            manifest['gcs_path'] = f"gs://{self.gcs_manager.bucket_name}/{self.gcs_prefix}/{dataset_id}"
        
        return manifest
    
    def _sync_dataset_to_gcs(self, dataset_id: str, dataset_dir: Path):
        """Sync dataset to Google Cloud Storage."""
        if not self.gcs_manager:
            return
        
        try:
            gcs_prefix = f"{self.gcs_prefix}/{dataset_id}"
            
            # Upload all files in dataset directory
            for file_path in dataset_dir.glob("*"):
                if file_path.is_file():
                    gcs_path = f"{gcs_prefix}/{file_path.name}"
                    self.gcs_manager.upload_file(str(file_path), gcs_path)
            
            logger.info(f"Synced dataset to GCS: gs://{self.gcs_manager.bucket_name}/{gcs_prefix}")
        except Exception as e:
            logger.error(f"Failed to sync dataset to GCS: {e}")
    
    def prepare_huggingface_dataset(
        self,
        dataset_id: str,
        output_format: str = "json"
    ) -> str:
        """
        Prepare dataset in HuggingFace format for training.
        
        Args:
            dataset_id: ID of the prepared dataset
            output_format: Output format ('json', 'csv', 'parquet')
        
        Returns:
            Path to HuggingFace-formatted dataset
        """
        dataset_dir = self.output_dir / dataset_id
        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        hf_dir = dataset_dir / "huggingface_format"
        hf_dir.mkdir(exist_ok=True)
        
        # Convert each split to HuggingFace format
        for split_name in ['train', 'val', 'test']:
            split_file = dataset_dir / f"{split_name}.jsonl"
            if not split_file.exists():
                continue
            
            # Load samples
            samples = []
            with open(split_file, 'r') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            
            # Convert to HuggingFace format
            hf_samples = []
            for sample in samples:
                hf_sample = {
                    'audio': sample['audio_path'],
                    'text': sample['target_text'],
                    'original_text': sample['input_text'],
                    'case_id': sample['case_id']
                }
                hf_samples.append(hf_sample)
            
            # Save in requested format
            df = pd.DataFrame(hf_samples)
            output_file = hf_dir / f"{split_name}.{output_format}"
            
            if output_format == "json":
                df.to_json(output_file, orient='records', lines=True)
            elif output_format == "csv":
                df.to_csv(output_file, index=False)
            elif output_format == "parquet":
                df.to_parquet(output_file, index=False)
            else:
                raise ValueError(f"Unsupported format: {output_format}")
            
            logger.info(f"Converted {split_name} to HuggingFace format: {output_file}")
        
        return str(hf_dir)
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """Get information about a prepared dataset."""
        dataset_dir = self.output_dir / dataset_id
        manifest_file = dataset_dir / "manifest.json"
        
        if not manifest_file.exists():
            return None
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        # Load metadata
        metadata_file = dataset_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                manifest['metadata'] = json.load(f)
        
        return manifest
    
    def list_datasets(self) -> List[Dict]:
        """List all prepared datasets."""
        datasets = []
        
        for dataset_dir in self.output_dir.iterdir():
            if dataset_dir.is_dir():
                info = self.get_dataset_info(dataset_dir.name)
                if info:
                    datasets.append(info)
        
        return sorted(datasets, key=lambda x: x['created_at'], reverse=True)
    
    def augment_dataset(
        self,
        dataset_id: str,
        augmentation_factor: int = 2,
        augmentation_methods: Optional[List[str]] = None
    ) -> str:
        """
        Apply data augmentation to increase dataset size.
        
        Args:
            dataset_id: ID of dataset to augment
            augmentation_factor: Factor to increase dataset size
            augmentation_methods: List of augmentation methods to use
        
        Returns:
            ID of augmented dataset
        """
        logger.info(f"Augmenting dataset {dataset_id}...")
        
        # Load original dataset
        dataset_dir = self.output_dir / dataset_id
        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        augmented_id = f"{dataset_id}_augmented_{augmentation_factor}x"
        augmented_dir = self.output_dir / augmented_id
        augmented_dir.mkdir(parents=True, exist_ok=True)
        
        # Augmentation methods (placeholder for actual implementation)
        if augmentation_methods is None:
            augmentation_methods = ['synonym_replacement', 'random_insertion']
        
        # Process each split
        for split_name in ['train', 'val', 'test']:
            split_file = dataset_dir / f"{split_name}.jsonl"
            if not split_file.exists():
                continue
            
            # Load samples
            samples = []
            with open(split_file, 'r') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            
            # For training data, apply augmentation
            if split_name == 'train':
                augmented_samples = samples.copy()
                for _ in range(augmentation_factor - 1):
                    for sample in samples:
                        # Create augmented version
                        aug_sample = sample.copy()
                        aug_sample['case_id'] = f"{sample['case_id']}_aug_{len(augmented_samples)}"
                        aug_sample['metadata'] = aug_sample.get('metadata', {})
                        aug_sample['metadata']['augmented'] = True
                        aug_sample['metadata']['augmentation_methods'] = augmentation_methods
                        augmented_samples.append(aug_sample)
                samples = augmented_samples
            
            # Save augmented split
            aug_split_file = augmented_dir / f"{split_name}.jsonl"
            with open(aug_split_file, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            
            logger.info(f"Augmented {split_name}: {len(samples)} samples")
        
        # Copy and update metadata
        original_metadata_file = dataset_dir / "metadata.json"
        if original_metadata_file.exists():
            with open(original_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['dataset_id'] = augmented_id
            metadata['created_at'] = datetime.now().isoformat()
            metadata['augmented'] = True
            metadata['augmentation_factor'] = augmentation_factor
            metadata['augmentation_methods'] = augmentation_methods
            metadata['original_dataset'] = dataset_id
            
            aug_metadata_file = augmented_dir / "metadata.json"
            with open(aug_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Created augmented dataset: {augmented_id}")
        
        return augmented_id
    
    def validate_dataset(self, dataset_id: str) -> Dict:
        """
        Validate a prepared dataset.
        
        Args:
            dataset_id: ID of dataset to validate
        
        Returns:
            Validation report
        """
        logger.info(f"Validating dataset {dataset_id}...")
        
        dataset_dir = self.output_dir / dataset_id
        if not dataset_dir.exists():
            return {'error': f'Dataset not found: {dataset_id}'}
        
        validation_report = {
            'dataset_id': dataset_id,
            'validated_at': datetime.now().isoformat(),
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for required files
        required_files = ['train.jsonl', 'val.jsonl', 'test.jsonl', 'metadata.json']
        for filename in required_files:
            if not (dataset_dir / filename).exists():
                validation_report['issues'].append(f"Missing required file: {filename}")
        
        # Validate each split
        for split_name in ['train', 'val', 'test']:
            split_file = dataset_dir / f"{split_name}.jsonl"
            if not split_file.exists():
                continue
            
            samples = []
            with open(split_file, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            samples.append(sample)
                            
                            # Validate sample structure
                            required_fields = ['audio_path', 'input_text', 'target_text']
                            for field in required_fields:
                                if field not in sample:
                                    validation_report['issues'].append(
                                        f"{split_name} line {i}: Missing field '{field}'"
                                    )
                        except json.JSONDecodeError:
                            validation_report['issues'].append(
                                f"{split_name} line {i}: Invalid JSON"
                            )
            
            # Calculate statistics
            validation_report['statistics'][split_name] = {
                'num_samples': len(samples),
                'avg_input_length': np.mean([len(s.get('input_text', '')) for s in samples]),
                'avg_target_length': np.mean([len(s.get('target_text', '')) for s in samples])
            }
        
        # Check for data leakage
        train_ids = set()
        val_ids = set()
        test_ids = set()
        
        for split_name, id_set in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
            split_file = dataset_dir / f"{split_name}.jsonl"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            id_set.add(sample.get('case_id'))
        
        # Check overlaps
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        if train_val_overlap:
            validation_report['issues'].append(
                f"Data leakage: {len(train_val_overlap)} samples overlap between train and val"
            )
        if train_test_overlap:
            validation_report['issues'].append(
                f"Data leakage: {len(train_test_overlap)} samples overlap between train and test"
            )
        if val_test_overlap:
            validation_report['issues'].append(
                f"Data leakage: {len(val_test_overlap)} samples overlap between val and test"
            )
        
        validation_report['is_valid'] = len(validation_report['issues']) == 0
        
        logger.info(f"Validation complete: {'PASSED' if validation_report['is_valid'] else 'FAILED'}")
        
        return validation_report

