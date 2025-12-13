"""
Model Versioning Utilities
Handles versioned model naming and management
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import logging

logger = logging.getLogger(__name__)


def get_next_model_version(models_dir: str = "models") -> int:
    """
    Get the next version number for a fine-tuned model.
    
    Looks for existing models matching pattern: finetuned_wav2vec2_v{N}
    Returns the next available version number.
    
    Args:
        models_dir: Directory containing model folders
        
    Returns:
        Next version number (e.g., 1, 2, 3, ...)
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return 1
    
    version_pattern = re.compile(r'^finetuned_wav2vec2_v(\d+)$')
    existing_versions = []
    
    # Check all directories in models folder
    for item in models_path.iterdir():
        if item.is_dir():
            match = version_pattern.match(item.name)
            if match:
                version_num = int(match.group(1))
                existing_versions.append(version_num)
    
    # Also check legacy names and convert if needed
    legacy_names = ["finetuned_wav2vec2", "finetuned"]
    for legacy_name in legacy_names:
        legacy_path = models_path / legacy_name
        if legacy_path.exists():
            # This is v1 if no v1 exists yet
            if 1 not in existing_versions:
                existing_versions.append(0)  # Mark for v1 assignment
    
    if not existing_versions:
        return 1
    
    max_version = max(existing_versions)
    return max_version + 1


def get_model_version_name(version_num: int) -> str:
    """Get the folder name for a version number."""
    return f"finetuned_wav2vec2_v{version_num}"


def migrate_legacy_models(models_dir: str = "models") -> Dict[str, str]:
    """
    Migrate legacy model names to versioned names.
    
    Renames:
    - finetuned_wav2vec2 -> finetuned_wav2vec2_v1
    - finetuned -> finetuned_wav2vec2_v2 (or next available)
    
    Args:
        models_dir: Directory containing model folders
        
    Returns:
        Dictionary mapping old names to new names
    """
    models_path = Path(models_dir)
    migrations = {}
    
    # Migrate finetuned_wav2vec2 to v1
    old_v1_path = models_path / "finetuned_wav2vec2"
    new_v1_path = models_path / "finetuned_wav2vec2_v1"
    
    if old_v1_path.exists() and not new_v1_path.exists():
        try:
            old_v1_path.rename(new_v1_path)
            migrations["finetuned_wav2vec2"] = "finetuned_wav2vec2_v1"
            logger.info(f"Migrated finetuned_wav2vec2 -> finetuned_wav2vec2_v1")
        except Exception as e:
            logger.error(f"Failed to migrate finetuned_wav2vec2: {e}")
    
    # Migrate finetuned to v2 (or next available)
    old_v2_path = models_path / "finetuned"
    if old_v2_path.exists():
        next_version = get_next_model_version(models_dir)
        new_v2_name = get_model_version_name(next_version)
        new_v2_path = models_path / new_v2_name
        
        if not new_v2_path.exists():
            try:
                old_v2_path.rename(new_v2_path)
                migrations["finetuned"] = new_v2_name
                logger.info(f"Migrated finetuned -> {new_v2_name}")
            except Exception as e:
                logger.error(f"Failed to migrate finetuned: {e}")
    
    return migrations


def get_all_model_versions(models_dir: str = "models") -> List[Dict[str, any]]:
    """
    Get all fine-tuned model versions with their metadata.
    
    Returns:
        List of dictionaries with version info: {
            'version_name': 'finetuned_wav2vec2_v1',
            'version_num': 1,
            'path': Path(...),
            'wer': float or None,
            'cer': float or None,
            'created_at': str or None,
            'is_current': bool
        }
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    versions = []
    version_pattern = re.compile(r'^finetuned_wav2vec2_v(\d+)$')
    
    # Find all versioned models
    for item in models_path.iterdir():
        if item.is_dir():
            match = version_pattern.match(item.name)
            if match:
                version_num = int(match.group(1))
                
                # Load evaluation results if available
                eval_file = item / "evaluation_results.json"
                wer = None
                cer = None
                if eval_file.exists():
                    try:
                        with open(eval_file, 'r') as f:
                            eval_data = json.load(f)
                            fine_tuned_metrics = eval_data.get("fine_tuned_metrics", {})
                            wer = fine_tuned_metrics.get("wer")
                            cer = fine_tuned_metrics.get("cer")
                    except Exception as e:
                        logger.warning(f"Could not load evaluation results for {item.name}: {e}")
                
                # Load metadata
                metadata_file = item / "model_metadata.json"
                created_at = None
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            created_at = metadata.get("saved_at")
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {item.name}: {e}")
                
                versions.append({
                    'version_name': item.name,
                    'version_num': version_num,
                    'path': str(item),
                    'wer': wer,
                    'cer': cer,
                    'created_at': created_at,
                    'is_current': False  # Will be set by caller
                })
    
    # Sort by version number (descending - newest first)
    versions.sort(key=lambda x: x['version_num'], reverse=True)
    
    return versions


def get_best_model_version(models_dir: str = "models", current_model_path: Optional[str] = None) -> Optional[str]:
    """
    Find the model version with the best (lowest) WER.
    
    Args:
        models_dir: Directory containing model folders
        current_model_path: Optional path to current model (will be marked as current)
        
    Returns:
        Path to the best model, or None if no models found
    """
    versions = get_all_model_versions(models_dir)
    
    if not versions:
        return None
    
    # Filter versions that have WER values
    versions_with_wer = [v for v in versions if v['wer'] is not None]
    
    if not versions_with_wer:
        # If no WER available, return the latest version
        return versions[0]['path']
    
    # Find version with lowest WER
    best_version = min(versions_with_wer, key=lambda x: x['wer'])
    
    # Mark as current
    if current_model_path:
        for v in versions:
            v['is_current'] = (v['path'] == current_model_path)
    else:
        best_version['is_current'] = True
    
    return best_version['path']


def set_current_model(models_dir: str = "models", model_path: str = None):
    """
    Set the current model by creating a symlink or marker file.
    
    Args:
        models_dir: Directory containing model folders
        model_path: Path to the model to set as current
    """
    models_path = Path(models_dir)
    current_marker = models_path / "current_model.txt"
    
    if model_path:
        try:
            with open(current_marker, 'w') as f:
                f.write(str(model_path))
            logger.info(f"Set current model to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to set current model: {e}")
    else:
        # Clear current model
        if current_marker.exists():
            current_marker.unlink()


def get_current_model_path(models_dir: str = "models") -> Optional[str]:
    """
    Get the path to the current model.
    
    Args:
        models_dir: Directory containing model folders
        
    Returns:
        Path to current model, or None if not set
    """
    models_path = Path(models_dir)
    current_marker = models_path / "current_model.txt"
    
    if current_marker.exists():
        try:
            with open(current_marker, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Could not read current model marker: {e}")
    
    # Fallback: find best model by WER
    return get_best_model_version(models_dir)

