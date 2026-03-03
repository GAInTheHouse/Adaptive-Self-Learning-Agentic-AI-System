#!/usr/bin/env python3
"""
Git repository plugin with LFS support.

Handles downloading datasets stored in Git repositories, particularly those
using Git LFS for large audio files (e.g., PriMock57).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from source_plugins import DataSourcePlugin


LOGGER = logging.getLogger("GitPlugin")


class GitPlugin(DataSourcePlugin):
    """Plugin for downloading Git repositories with optional LFS support."""
    
    def __init__(self):
        self.logger = LOGGER
    
    def get_source_type(self) -> str:
        return "git"
    
    def download(self, config: Dict, output_dir: Path, force: bool) -> Optional[Path]:
        """
        Clone Git repository with LFS support.
        
        Checks for git-lfs availability if requires_lfs is True.
        """
        repo_url = config["url"]
        requires_lfs = config.get("requires_lfs", False)
        
        self.logger.info("Cloning Git repository: %s", repo_url)
        
        # Check if already cloned
        if output_dir.exists():
            if not force:
                self.logger.info("Repository already exists at: %s", output_dir)
                return output_dir
            self.logger.info("Removing existing repository (force=True)")
            shutil.rmtree(output_dir)
        
        # Check for git-lfs if required
        if requires_lfs and not self._check_git_lfs():
            self.logger.error(
                "Git LFS required but not available. Install:\n"
                "  macOS: brew install git-lfs && git lfs install\n"
                "  Ubuntu: sudo apt install git-lfs && git lfs install"
            )
            return None
        
        # Clone repository
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            result = subprocess.run(
                ["git", "clone", repo_url, str(output_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode != 0:
                self.logger.error("Git clone failed:\n%s", result.stderr)
                return None
            
            self.logger.info("Repository cloned successfully to: %s", output_dir)
            
            # Verify LFS files if applicable
            if requires_lfs:
                lfs_check = subprocess.run(
                    ["git", "lfs", "ls-files"],
                    cwd=str(output_dir),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                
                if lfs_check.returncode == 0 and lfs_check.stdout.strip():
                    lfs_count = len(lfs_check.stdout.strip().split('\n'))
                    self.logger.info("Git LFS files downloaded: %d files", lfs_count)
            
            return output_dir
            
        except Exception as exc:
            self.logger.error("Failed to clone repository: %s", exc)
            return None
    
    def generate_manifest(
        self,
        data_dir: Path,
        manifest_dir: Path,
        dataset_name: str,
        force: bool
    ) -> List[Path]:
        """
        Generate manifest for Git repository dataset.
        
        Routes to appropriate generator based on dataset_type.
        """
        sys.path.insert(0, str(Path(__file__).parent.parent / "manifest_generators"))
        import primock57
        
        dataset_type = dataset_name.lower()
        
        if "primock" in dataset_type:
            return primock57.generate(data_dir, manifest_dir, force)
        else:
            self.logger.warning("No manifest generator for dataset: %s", dataset_name)
            return []
    
    def _check_git_lfs(self) -> bool:
        """Check if git-lfs is installed and available."""
        try:
            result = subprocess.run(
                ["git", "lfs", "version"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self.logger.info("Git LFS detected: %s", result.stdout.strip().split('\n')[0])
                return True
        except FileNotFoundError:
            pass
        
        return False
