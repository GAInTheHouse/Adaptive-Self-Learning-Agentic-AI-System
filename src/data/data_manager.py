"""
Data Management System for Self-Learning STT Agent
Stores failed cases, corrections, and learning data with GCS integration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import pandas as pd
from collections import defaultdict

from ..utils.gcs_utils import GCSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailedCase:
    """Represents a failed transcription case."""
    
    def __init__(
        self,
        case_id: str,
        audio_path: str,
        original_transcript: str,
        corrected_transcript: Optional[str],
        error_types: List[str],
        error_score: float,
        metadata: Dict[str, Any],
        timestamp: Optional[str] = None
    ):
        self.case_id = case_id
        self.audio_path = audio_path
        self.original_transcript = original_transcript
        self.corrected_transcript = corrected_transcript
        self.error_types = error_types
        self.error_score = error_score
        self.metadata = metadata
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'case_id': self.case_id,
            'audio_path': self.audio_path,
            'original_transcript': self.original_transcript,
            'corrected_transcript': self.corrected_transcript,
            'error_types': self.error_types,
            'error_score': self.error_score,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FailedCase':
        """Create from dictionary."""
        return cls(
            case_id=data['case_id'],
            audio_path=data['audio_path'],
            original_transcript=data['original_transcript'],
            corrected_transcript=data.get('corrected_transcript'),
            error_types=data['error_types'],
            error_score=data['error_score'],
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp')
        )


class DataManager:
    """
    Comprehensive data management system for self-learning STT agent.
    Handles storage, retrieval, and versioning of failed cases and corrections.
    """
    
    def __init__(
        self,
        local_storage_dir: str = "data/failed_cases",
        use_gcs: bool = True,
        gcs_bucket_name: str = "stt-project-datasets",
        gcs_prefix: str = "learning_data",
        project_id: str = "stt-agentic-ai-2025"
    ):
        """
        Initialize data manager.
        
        Args:
            local_storage_dir: Local directory for data storage
            use_gcs: Whether to use Google Cloud Storage
            gcs_bucket_name: GCS bucket name
            gcs_prefix: Prefix for GCS paths
            project_id: GCP project ID
        """
        self.local_storage_dir = Path(local_storage_dir)
        self.local_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create audio storage directory for permanent audio files
        self.audio_storage_dir = self.local_storage_dir / "audio_files"
        self.audio_storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gcs = use_gcs
        self.gcs_prefix = gcs_prefix
        
        # Initialize GCS manager if enabled
        self.gcs_manager = None
        if use_gcs:
            try:
                self.gcs_manager = GCSManager(project_id, gcs_bucket_name)
                logger.info("GCS integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS: {e}. Using local storage only.")
                self.use_gcs = False
        
        # In-memory cache for fast access
        self.failed_cases_cache: Dict[str, FailedCase] = {}
        self.corrections_cache: List[Dict] = []
        
        # Storage paths
        self.failed_cases_file = self.local_storage_dir / "failed_cases.jsonl"
        self.corrections_file = self.local_storage_dir / "corrections.jsonl"
        self.metadata_file = self.local_storage_dir / "metadata.json"
        
        # Load existing data
        self._load_local_data()
        
        logger.info(f"Data Manager initialized (local: {self.local_storage_dir}, GCS: {use_gcs})")
    
    def _generate_case_id(self, audio_path: str, transcript: str) -> str:
        """Generate unique case ID from audio path and transcript."""
        content = f"{audio_path}_{transcript}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _load_local_data(self):
        """Load existing data from local storage."""
        # Load failed cases
        if self.failed_cases_file.exists():
            with open(self.failed_cases_file, 'r') as f:
                for line in f:
                    if line.strip():
                        case_data = json.loads(line)
                        case = FailedCase.from_dict(case_data)
                        self.failed_cases_cache[case.case_id] = case
            logger.info(f"Loaded {len(self.failed_cases_cache)} failed cases from local storage")
        
        # Load corrections
        if self.corrections_file.exists():
            with open(self.corrections_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.corrections_cache.append(json.loads(line))
            logger.info(f"Loaded {len(self.corrections_cache)} corrections from local storage")
    
    def store_failed_case(
        self,
        audio_path: str,
        original_transcript: str,
        corrected_transcript: Optional[str],
        error_types: List[str],
        error_score: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a failed transcription case.
        Automatically copies temporary audio files to permanent storage.
        
        Args:
            audio_path: Path to audio file (temporary or permanent)
            original_transcript: Original (failed) transcription
            corrected_transcript: Corrected transcription (if available)
            error_types: List of error types detected
            error_score: Error confidence score
            metadata: Additional metadata
        
        Returns:
            Case ID
        """
        case_id = self._generate_case_id(audio_path, original_transcript)
        
        # Check if audio file is temporary (in /tmp, /var/folders, or tempfile pattern)
        audio_path_obj = Path(audio_path)
        is_temporary = (
            '/tmp' in str(audio_path_obj) or
            '/var/folders' in str(audio_path_obj) or
            str(audio_path_obj.parent).startswith('/var/folders') or
            'tmp' in audio_path_obj.name.lower()
        )
        
        # Copy to permanent storage if temporary
        permanent_audio_path = audio_path
        if is_temporary and audio_path_obj.exists():
            try:
                import shutil
                # Create permanent filename using case_id to avoid conflicts
                permanent_filename = f"{case_id}_{audio_path_obj.name}"
                permanent_audio_path_obj = self.audio_storage_dir / permanent_filename
                
                # Copy the file
                shutil.copy2(audio_path_obj, permanent_audio_path_obj)
                permanent_audio_path = str(permanent_audio_path_obj)
                logger.info(f"Copied temporary audio file to permanent storage: {permanent_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to copy temporary audio file to permanent storage: {e}. Using original path.")
                permanent_audio_path = audio_path
        elif not audio_path_obj.exists():
            logger.warning(f"Audio file does not exist: {audio_path}. Storing path as-is, but fine-tuning may fail.")
        
        case = FailedCase(
            case_id=case_id,
            audio_path=permanent_audio_path,  # Use permanent path
            original_transcript=original_transcript,
            corrected_transcript=corrected_transcript,
            error_types=error_types,
            error_score=error_score,
            metadata=metadata or {}
        )
        
        # Store in cache
        self.failed_cases_cache[case_id] = case
        
        # Append to local storage
        with open(self.failed_cases_file, 'a') as f:
            f.write(json.dumps(case.to_dict()) + '\n')
        
        logger.info(f"Stored failed case: {case_id} (audio: {permanent_audio_path})")
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            self._sync_to_gcs()
        
        return case_id
    
    def store_correction(
        self,
        case_id: str,
        corrected_transcript: str,
        correction_method: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store a correction for a failed case.
        
        Args:
            case_id: ID of the failed case
            corrected_transcript: Corrected transcription
            correction_method: Method used for correction (e.g., 'manual', 'auto', 'feedback')
            metadata: Additional metadata
        
        Returns:
            True if successful
        """
        if case_id not in self.failed_cases_cache:
            logger.warning(f"Case ID {case_id} not found")
            return False
        
        correction = {
            'case_id': case_id,
            'corrected_transcript': corrected_transcript,
            'correction_method': correction_method,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Update the failed case
        self.failed_cases_cache[case_id].corrected_transcript = corrected_transcript
        
        # Store correction
        self.corrections_cache.append(correction)
        
        # Append to local storage
        with open(self.corrections_file, 'a') as f:
            f.write(json.dumps(correction) + '\n')
        
        logger.info(f"Stored correction for case: {case_id}")
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            self._sync_to_gcs()
        
        return True
    
    def get_failed_case(self, case_id: str) -> Optional[FailedCase]:
        """Retrieve a failed case by ID."""
        return self.failed_cases_cache.get(case_id)
    
    def get_failed_cases_by_error_type(self, error_type: str) -> List[FailedCase]:
        """Get all failed cases with a specific error type."""
        result = []
        for case in self.failed_cases_cache.values():
            # Handle error_types being a list or nested list
            error_types = case.error_types if isinstance(case.error_types, list) else [case.error_types]
            # Flatten nested lists
            flat_errors = []
            for et in error_types:
                if isinstance(et, list):
                    flat_errors.extend(et)
                else:
                    flat_errors.append(et)
            
            if error_type in flat_errors:
                result.append(case)
        return result
    
    def get_corrected_cases(self) -> List[FailedCase]:
        """Get all cases that have been corrected."""
        return [
            case for case in self.failed_cases_cache.values()
            if case.corrected_transcript is not None
        ]
    
    def get_uncorrected_cases(self) -> List[FailedCase]:
        """Get all cases that haven't been corrected yet."""
        return [
            case for case in self.failed_cases_cache.values()
            if case.corrected_transcript is None
        ]
    
    def get_statistics(self) -> Dict:
        """Get statistics about stored data."""
        # Reload data from file to ensure we have the latest count
        # This ensures statistics are always up-to-date even if the file was modified
        self._reload_failed_cases()
        
        total_cases = len(self.failed_cases_cache)
        corrected_cases = len(self.get_corrected_cases())
        
        error_type_counts = defaultdict(int)
        for case in self.failed_cases_cache.values():
            # Handle error_types being a list
            error_types = case.error_types if isinstance(case.error_types, list) else [case.error_types]
            for error_type in error_types:
                # Skip if error_type is still a list (nested)
                if isinstance(error_type, list):
                    for et in error_type:
                        error_type_counts[et] += 1
                else:
                    error_type_counts[error_type] += 1
        
        return {
            'total_failed_cases': total_cases,
            'corrected_cases': corrected_cases,
            'uncorrected_cases': total_cases - corrected_cases,
            'correction_rate': corrected_cases / total_cases if total_cases > 0 else 0.0,
            'error_type_distribution': dict(error_type_counts),
            'total_corrections': len(self.corrections_cache),
            'last_updated': datetime.now().isoformat()
        }
    
    def _reload_failed_cases(self):
        """Reload failed cases from file to refresh the cache."""
        if self.failed_cases_file.exists():
            # Clear existing cache
            self.failed_cases_cache.clear()
            # Reload from file
            with open(self.failed_cases_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            case_data = json.loads(line)
                            case = FailedCase.from_dict(case_data)
                            self.failed_cases_cache[case.case_id] = case
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to parse failed case line: {e}. Line: {line[:100]}")
                            continue
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export failed cases to pandas DataFrame for analysis."""
        data = [case.to_dict() for case in self.failed_cases_cache.values()]
        return pd.DataFrame(data)
    
    def export_corrections_to_dataframe(self) -> pd.DataFrame:
        """Export corrections to pandas DataFrame."""
        return pd.DataFrame(self.corrections_cache)
    
    def _sync_to_gcs(self):
        """Sync local data to Google Cloud Storage."""
        if not self.gcs_manager:
            return
        
        try:
            # Upload failed cases
            gcs_path = f"{self.gcs_prefix}/failed_cases.jsonl"
            self.gcs_manager.upload_file(str(self.failed_cases_file), gcs_path)
            
            # Upload corrections
            gcs_path = f"{self.gcs_prefix}/corrections.jsonl"
            self.gcs_manager.upload_file(str(self.corrections_file), gcs_path)
            
            # Upload metadata
            metadata = self.get_statistics()
            metadata_local = self.local_storage_dir / "metadata.json"
            with open(metadata_local, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            gcs_path = f"{self.gcs_prefix}/metadata.json"
            self.gcs_manager.upload_file(str(metadata_local), gcs_path)
            
            logger.info("Synced data to GCS")
        except Exception as e:
            logger.error(f"Failed to sync to GCS: {e}")
    
    def sync_from_gcs(self):
        """Sync data from Google Cloud Storage to local."""
        if not self.gcs_manager:
            logger.warning("GCS not enabled")
            return
        
        try:
            # Download failed cases
            gcs_path = f"{self.gcs_prefix}/failed_cases.jsonl"
            self.gcs_manager.download_file(gcs_path, str(self.failed_cases_file))
            
            # Download corrections
            gcs_path = f"{self.gcs_prefix}/corrections.jsonl"
            self.gcs_manager.download_file(gcs_path, str(self.corrections_file))
            
            # Reload local data
            self.failed_cases_cache.clear()
            self.corrections_cache.clear()
            self._load_local_data()
            
            logger.info("Synced data from GCS")
        except Exception as e:
            logger.error(f"Failed to sync from GCS: {e}")
    
    def clear_all_data(self, confirm: bool = False):
        """
        Clear all stored data (USE WITH CAUTION).
        
        Args:
            confirm: Must be True to actually clear data
        """
        if not confirm:
            logger.warning("Clear operation not confirmed. Set confirm=True to proceed.")
            return
        
        self.failed_cases_cache.clear()
        self.corrections_cache.clear()
        
        if self.failed_cases_file.exists():
            self.failed_cases_file.unlink()
        if self.corrections_file.exists():
            self.corrections_file.unlink()
        
        logger.info("Cleared all local data")
    
    def clear_failed_cases(self):
        """
        Clear all failed cases (error cases) after they've been used for fine-tuning.
        This resets the error case count to zero.
        """
        self.failed_cases_cache.clear()
        
        if self.failed_cases_file.exists():
            self.failed_cases_file.unlink()
            logger.info("Cleared all failed cases (error cases reset to zero)")
        
        # Also clear from GCS if enabled
        if self.use_gcs and self.gcs_manager:
            try:
                gcs_path = f"{self.gcs_prefix}/failed_cases.jsonl"
                # Delete from GCS (upload empty file or delete)
                # For now, we'll just log - GCS deletion would require additional API calls
                logger.info(f"Failed cases cleared locally. GCS path: {gcs_path}")
            except Exception as e:
                logger.warning(f"Could not sync failed cases clearing to GCS: {e}")

