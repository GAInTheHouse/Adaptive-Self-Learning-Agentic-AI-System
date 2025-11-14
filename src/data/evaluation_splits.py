"""
Create standardized train/dev/test splits for evaluation.
"""

from datasets import load_from_disk, DatasetDict, Dataset
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationSplitter:
    """Create reproducible dataset splits"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize splitter.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
    
    def create_splits(
        self,
        dataset_path: str,
        output_path: str,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> DatasetDict:
        """
        Create train/dev/test splits.
        
        Args:
            dataset_path: Path to dataset
            output_path: Output path for splits
            train_ratio: Training set ratio
            dev_ratio: Development set ratio
            test_ratio: Test set ratio
        
        Returns:
            DatasetDict with splits
        """
        # Validate ratios
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)
        
        # Handle different dataset types
        if isinstance(dataset, DatasetDict):
            # If already split, concatenate all splits
            dataset = Dataset.from_dict({
                key: sum([split[key] for split in dataset.values()], [])
                for key in dataset[list(dataset.keys())[0]].features
            })
        
        # Shuffle
        dataset = dataset.shuffle(seed=self.seed)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        dev_size = int(total_size * dev_ratio)
        
        # Create splits
        train_dataset = dataset.select(range(train_size))
        dev_dataset = dataset.select(range(train_size, train_size + dev_size))
        test_dataset = dataset.select(range(train_size + dev_size, total_size))
        
        # Create dataset dict
        splits = DatasetDict({
            'train': train_dataset,
            'dev': dev_dataset,
            'test': test_dataset
        })
        
        # Save
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        splits.save_to_disk(str(output_path))
        
        # Save statistics
        stats = {
            'total_samples': total_size,
            'train_samples': len(train_dataset),
            'dev_samples': len(dev_dataset),
            'test_samples': len(test_dataset),
            'train_ratio': train_ratio,
            'dev_ratio': dev_ratio,
            'test_ratio': test_ratio,
            'seed': self.seed
        }
        
        with open(output_path / 'split_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Created splits: Train={len(train_dataset)}, "
                   f"Dev={len(dev_dataset)}, Test={len(test_dataset)}")
        
        return splits
