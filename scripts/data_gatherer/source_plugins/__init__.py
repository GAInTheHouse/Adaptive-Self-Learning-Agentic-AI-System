"""
Data source plugins for downloading from different platforms.

Each plugin implements the DataSourcePlugin interface to handle
downloading and manifest generation for a specific source type.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional


class DataSourcePlugin(ABC):
    """Base class for data source plugins."""
    
    @abstractmethod
    def download(self, config: Dict, output_dir: Path, force: bool) -> Optional[Path]:
        """
        Download dataset from source.
        
        Args:
            config: Dataset configuration from registry
            output_dir: Target directory for downloaded data
            force: If True, re-download even if data exists
            
        Returns:
            Path to downloaded data directory, or None if download failed
        """
        pass
    
    @abstractmethod
    def generate_manifest(
        self, 
        data_dir: Path, 
        manifest_dir: Path, 
        dataset_name: str,
        force: bool
    ) -> List[Path]:
        """
        Generate manifest CSV for downloaded data.
        
        Args:
            data_dir: Directory containing downloaded data
            manifest_dir: Directory to write manifest CSV files
            dataset_name: Name of dataset from registry
            force: If True, overwrite existing manifests
            
        Returns:
            List of paths to generated manifest CSV files
        """
        pass
    
    @abstractmethod
    def get_source_type(self) -> str:
        """Return source type identifier (huggingface, openslr, git)."""
        pass
