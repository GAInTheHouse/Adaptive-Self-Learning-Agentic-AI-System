"""
Data Versioning and Quality Control System
Manages dataset versions and ensures data quality standards.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
import pandas as pd
import numpy as np
from collections import defaultdict

from ..utils.gcs_utils import GCSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataVersion:
    """Represents a versioned dataset."""
    
    def __init__(
        self,
        version_id: str,
        dataset_path: str,
        metadata: Dict,
        parent_version: Optional[str] = None,
        checksum: Optional[str] = None
    ):
        self.version_id = version_id
        self.dataset_path = dataset_path
        self.metadata = metadata
        self.parent_version = parent_version
        self.checksum = checksum or self._calculate_checksum()
        self.created_at = datetime.now().isoformat()
    
    def _calculate_checksum(self) -> str:
        """Calculate dataset checksum."""
        hasher = hashlib.sha256()
        dataset_path = Path(self.dataset_path)
        
        # Hash all files in dataset directory
        if dataset_path.is_dir():
            for file_path in sorted(dataset_path.glob("**/*")):
                if file_path.is_file():
                    hasher.update(file_path.read_bytes())
        elif dataset_path.is_file():
            hasher.update(dataset_path.read_bytes())
        
        return hasher.hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'version_id': self.version_id,
            'dataset_path': self.dataset_path,
            'metadata': self.metadata,
            'parent_version': self.parent_version,
            'checksum': self.checksum,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataVersion':
        """Create from dictionary."""
        version = cls(
            version_id=data['version_id'],
            dataset_path=data['dataset_path'],
            metadata=data['metadata'],
            parent_version=data.get('parent_version'),
            checksum=data.get('checksum')
        )
        version.created_at = data.get('created_at', version.created_at)
        return version


class QualityMetrics:
    """Quality metrics for dataset validation."""
    
    def __init__(self):
        self.metrics = {
            'completeness': 0.0,
            'consistency': 0.0,
            'validity': 0.0,
            'accuracy': 0.0,
            'uniqueness': 0.0
        }
        self.issues: List[Dict] = []
        self.warnings: List[str] = []
    
    def calculate_score(self) -> float:
        """Calculate overall quality score."""
        return float(np.mean(list(self.metrics.values())))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        # Ensure all values are JSON serializable
        return {
            'metrics': {k: float(v) for k, v in self.metrics.items()},
            'overall_score': float(self.calculate_score()),
            'issues': self.issues,
            'warnings': self.warnings
        }


class DataVersionControl:
    """
    Comprehensive data versioning system with quality control.
    """
    
    def __init__(
        self,
        local_storage_dir: str = "data/versions",
        use_gcs: bool = True,
        gcs_bucket_name: str = "stt-project-datasets",
        gcs_prefix: str = "data_versions",
        project_id: str = "stt-agentic-ai-2025"
    ):
        """
        Initialize version control system.
        
        Args:
            local_storage_dir: Local directory for version storage
            use_gcs: Whether to use Google Cloud Storage
            gcs_bucket_name: GCS bucket name
            gcs_prefix: Prefix for GCS paths
            project_id: GCP project ID
        """
        self.local_storage_dir = Path(local_storage_dir)
        self.local_storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gcs = use_gcs
        self.gcs_prefix = gcs_prefix
        
        # Initialize GCS manager
        self.gcs_manager = None
        if use_gcs:
            try:
                self.gcs_manager = GCSManager(project_id, gcs_bucket_name)
                logger.info("GCS integration enabled for version control")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS: {e}. Using local storage only.")
                self.use_gcs = False
        
        # Version registry
        self.registry_file = self.local_storage_dir / "version_registry.json"
        self.versions: Dict[str, DataVersion] = {}
        
        # Quality control settings
        self.quality_thresholds = {
            'min_completeness': 0.95,
            'min_consistency': 0.90,
            'min_validity': 0.95,
            'min_overall_score': 0.90
        }
        
        # Load existing versions
        self._load_registry()
        
        logger.info(f"Data Version Control initialized (local: {self.local_storage_dir})")
    
    def _load_registry(self):
        """Load version registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                    for version_id, version_data in registry_data.items():
                        self.versions[version_id] = DataVersion.from_dict(version_data)
                logger.info(f"Loaded {len(self.versions)} versions from registry")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to load registry (corrupted JSON): {e}. Starting with empty registry.")
                # Backup corrupted file and start fresh
                backup_path = self.registry_file.with_suffix('.json.corrupted')
                self.registry_file.rename(backup_path)
                logger.info(f"Backed up corrupted registry to {backup_path}")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}. Starting with empty registry.")
    
    def _save_registry(self):
        """Save version registry to disk."""
        registry_data = {
            version_id: version.to_dict()
            for version_id, version in self.versions.items()
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            try:
                gcs_path = f"{self.gcs_prefix}/version_registry.json"
                self.gcs_manager.upload_file(str(self.registry_file), gcs_path)
            except Exception as e:
                logger.error(f"Failed to sync registry to GCS: {e}")
    
    def create_version(
        self,
        dataset_path: str,
        version_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        parent_version: Optional[str] = None,
        run_quality_check: bool = True
    ) -> str:
        """
        Create a new dataset version.
        
        Args:
            dataset_path: Path to dataset
            version_name: Optional version name (auto-generated if not provided)
            metadata: Version metadata
            parent_version: ID of parent version (for tracking lineage)
            run_quality_check: Whether to run quality checks
        
        Returns:
            Version ID
        """
        # Generate version ID
        if version_name:
            version_id = f"v_{version_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Creating version: {version_id}")
        
        # Run quality check if requested
        quality_report = None
        if run_quality_check:
            quality_report = self.check_quality(dataset_path)
            if not quality_report['passed']:
                logger.warning(f"Quality check failed for {version_id}")
                logger.warning(f"Issues: {quality_report['quality_metrics']['issues']}")
        
        # Create version
        version_metadata = metadata or {}
        version_metadata['quality_report'] = quality_report
        
        version = DataVersion(
            version_id=version_id,
            dataset_path=dataset_path,
            metadata=version_metadata,
            parent_version=parent_version
        )
        
        # Register version
        self.versions[version_id] = version
        self._save_registry()
        
        logger.info(f"Version created: {version_id} (checksum: {version.checksum})")
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get a specific version."""
        return self.versions.get(version_id)
    
    def list_versions(
        self,
        parent_version: Optional[str] = None,
        min_quality_score: Optional[float] = None
    ) -> List[DataVersion]:
        """
        List all versions with optional filters.
        
        Args:
            parent_version: Filter by parent version
            min_quality_score: Filter by minimum quality score
        
        Returns:
            List of versions
        """
        versions = list(self.versions.values())
        
        # Filter by parent version
        if parent_version:
            versions = [v for v in versions if v.parent_version == parent_version]
        
        # Filter by quality score
        if min_quality_score is not None:
            versions = [
                v for v in versions
                if v.metadata.get('quality_report', {}).get('quality_metrics', {}).get('overall_score', 0) >= min_quality_score
            ]
        
        # Sort by creation time
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        return versions
    
    def get_version_lineage(self, version_id: str) -> List[DataVersion]:
        """
        Get the lineage (ancestry) of a version.
        
        Args:
            version_id: Version ID
        
        Returns:
            List of versions from oldest ancestor to current
        """
        lineage = []
        current_id = version_id
        
        while current_id:
            version = self.versions.get(current_id)
            if not version:
                break
            lineage.append(version)
            current_id = version.parent_version
        
        return list(reversed(lineage))
    
    def compare_versions(
        self,
        version_id1: str,
        version_id2: str
    ) -> Dict:
        """
        Compare two versions.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
        
        Returns:
            Comparison report
        """
        v1 = self.versions.get(version_id1)
        v2 = self.versions.get(version_id2)
        
        if not v1 or not v2:
            return {'error': 'One or both versions not found'}
        
        comparison = {
            'version1': version_id1,
            'version2': version_id2,
            'checksum_match': v1.checksum == v2.checksum,
            'metadata_diff': self._compare_metadata(v1.metadata, v2.metadata),
            'quality_comparison': self._compare_quality(v1, v2),
            'lineage_related': self._check_lineage_relation(v1, v2)
        }
        
        return comparison
    
    def _compare_metadata(self, meta1: Dict, meta2: Dict) -> Dict:
        """Compare metadata between two versions."""
        keys1 = set(meta1.keys())
        keys2 = set(meta2.keys())
        
        return {
            'keys_only_in_v1': list(keys1 - keys2),
            'keys_only_in_v2': list(keys2 - keys1),
            'common_keys': list(keys1 & keys2),
            'differing_values': [
                key for key in (keys1 & keys2)
                if meta1.get(key) != meta2.get(key)
            ]
        }
    
    def _compare_quality(self, v1: DataVersion, v2: DataVersion) -> Dict:
        """Compare quality metrics between two versions."""
        q1 = v1.metadata.get('quality_report', {}).get('quality_metrics', {})
        q2 = v2.metadata.get('quality_report', {}).get('quality_metrics', {})
        
        return {
            'v1_score': q1.get('overall_score', 0),
            'v2_score': q2.get('overall_score', 0),
            'improvement': q2.get('overall_score', 0) - q1.get('overall_score', 0)
        }
    
    def _check_lineage_relation(self, v1: DataVersion, v2: DataVersion) -> str:
        """Check if two versions are related in lineage."""
        # Check if v2 is descendant of v1
        lineage2 = self.get_version_lineage(v2.version_id)
        if v1.version_id in [v.version_id for v in lineage2]:
            return f"{v2.version_id} is descendant of {v1.version_id}"
        
        # Check if v1 is descendant of v2
        lineage1 = self.get_version_lineage(v1.version_id)
        if v2.version_id in [v.version_id for v in lineage1]:
            return f"{v1.version_id} is descendant of {v2.version_id}"
        
        # Check if they share a common ancestor
        common_ancestors = set(v.version_id for v in lineage1) & set(v.version_id for v in lineage2)
        if common_ancestors:
            return f"Share common ancestor(s): {list(common_ancestors)}"
        
        return "Not related in lineage"
    
    def check_quality(self, dataset_path: str) -> Dict:
        """
        Run comprehensive quality checks on a dataset.
        
        Args:
            dataset_path: Path to dataset
        
        Returns:
            Quality report
        """
        logger.info(f"Running quality checks on: {dataset_path}")
        
        quality_metrics = QualityMetrics()
        dataset_path = Path(dataset_path)
        
        # Check if dataset exists
        if not dataset_path.exists():
            quality_metrics.issues.append({
                'severity': 'critical',
                'category': 'existence',
                'message': 'Dataset path does not exist'
            })
            return {
                'passed': False,
                'quality_metrics': quality_metrics.to_dict()
            }
        
        # Load dataset samples
        samples = []
        if dataset_path.is_dir():
            # Look for data files
            for data_file in dataset_path.glob("*.jsonl"):
                with open(data_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError:
                                quality_metrics.issues.append({
                                    'severity': 'error',
                                    'category': 'format',
                                    'message': f'Invalid JSON in {data_file.name}'
                                })
        
        if not samples:
            quality_metrics.warnings.append("No samples found in dataset")
            return {
                'passed': False,
                'quality_metrics': quality_metrics.to_dict()
            }
        
        # Check completeness
        quality_metrics.metrics['completeness'] = self._check_completeness(samples, quality_metrics)
        
        # Check consistency
        quality_metrics.metrics['consistency'] = self._check_consistency(samples, quality_metrics)
        
        # Check validity
        quality_metrics.metrics['validity'] = self._check_validity(samples, quality_metrics)
        
        # Check uniqueness
        quality_metrics.metrics['uniqueness'] = self._check_uniqueness(samples, quality_metrics)
        
        # Calculate overall score
        overall_score = quality_metrics.calculate_score()
        
        # Determine if passed
        passed = (
            overall_score >= self.quality_thresholds['min_overall_score'] and
            quality_metrics.metrics['completeness'] >= self.quality_thresholds['min_completeness'] and
            quality_metrics.metrics['consistency'] >= self.quality_thresholds['min_consistency'] and
            quality_metrics.metrics['validity'] >= self.quality_thresholds['min_validity'] and
            len([i for i in quality_metrics.issues if i['severity'] == 'critical']) == 0
        )
        
        logger.info(f"Quality check {'PASSED' if passed else 'FAILED'} (score: {overall_score:.2f})")
        
        # Convert to JSON-serializable format (handle numpy types)
        def make_json_serializable(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            return obj
        
        result = {
            'passed': bool(passed),
            'quality_metrics': quality_metrics.to_dict(),
            'num_samples': int(len(samples)),
            'timestamp': datetime.now().isoformat()
        }
        
        return make_json_serializable(result)
    
    def _check_completeness(self, samples: List[Dict], quality_metrics: QualityMetrics) -> float:
        """Check data completeness."""
        required_fields = ['audio_path', 'input_text', 'target_text']
        
        complete_samples = 0
        for sample in samples:
            if all(field in sample and sample[field] for field in required_fields):
                complete_samples += 1
            else:
                missing = [f for f in required_fields if f not in sample or not sample[f]]
                if missing:
                    quality_metrics.issues.append({
                        'severity': 'warning',
                        'category': 'completeness',
                        'message': f'Sample missing fields: {missing}'
                    })
        
        return complete_samples / len(samples) if samples else 0.0
    
    def _check_consistency(self, samples: List[Dict], quality_metrics: QualityMetrics) -> float:
        """Check data consistency."""
        consistent_samples = 0
        
        for sample in samples:
            is_consistent = True
            
            # Check that input and target are not identical (should have corrections)
            if sample.get('input_text') == sample.get('target_text'):
                if sample.get('has_correction', False):
                    quality_metrics.warnings.append(
                        "Sample marked as corrected but input equals target"
                    )
                    is_consistent = False
            
            # Check audio path format
            audio_path = sample.get('audio_path', '')
            if audio_path and not any(audio_path.endswith(ext) for ext in ['.wav', '.mp3', '.flac', '.ogg']):
                quality_metrics.warnings.append(
                    f"Unusual audio file extension: {audio_path}"
                )
                is_consistent = False
            
            # Check text lengths
            input_len = len(sample.get('input_text', ''))
            target_len = len(sample.get('target_text', ''))
            if abs(input_len - target_len) > max(input_len, target_len) * 0.5:
                quality_metrics.warnings.append(
                    "Large discrepancy between input and target lengths"
                )
                is_consistent = False
            
            if is_consistent:
                consistent_samples += 1
        
        return consistent_samples / len(samples) if samples else 0.0
    
    def _check_validity(self, samples: List[Dict], quality_metrics: QualityMetrics) -> float:
        """Check data validity."""
        valid_samples = 0
        
        for sample in samples:
            is_valid = True
            
            # Check text validity (not empty, not too short, contains words)
            for field in ['input_text', 'target_text']:
                text = sample.get(field, '')
                if len(text) < 3:
                    quality_metrics.issues.append({
                        'severity': 'error',
                        'category': 'validity',
                        'message': f'{field} is too short: "{text}"'
                    })
                    is_valid = False
                elif not any(c.isalpha() for c in text):
                    quality_metrics.issues.append({
                        'severity': 'error',
                        'category': 'validity',
                        'message': f'{field} contains no letters: "{text}"'
                    })
                    is_valid = False
            
            # Check error score validity
            error_score = sample.get('error_score')
            if error_score is not None and not (0.0 <= error_score <= 1.0):
                quality_metrics.issues.append({
                    'severity': 'warning',
                    'category': 'validity',
                    'message': f'Invalid error_score: {error_score}'
                })
                is_valid = False
            
            if is_valid:
                valid_samples += 1
        
        return valid_samples / len(samples) if samples else 0.0
    
    def _check_uniqueness(self, samples: List[Dict], quality_metrics: QualityMetrics) -> float:
        """Check data uniqueness (detect duplicates)."""
        # Create fingerprints for deduplication
        fingerprints = set()
        unique_samples = 0
        
        for sample in samples:
            # Create fingerprint from audio path and text
            fingerprint = hashlib.md5(
                f"{sample.get('audio_path', '')}_{sample.get('input_text', '')}".encode()
            ).hexdigest()
            
            if fingerprint not in fingerprints:
                fingerprints.add(fingerprint)
                unique_samples += 1
            else:
                quality_metrics.warnings.append(
                    f"Duplicate sample detected: {sample.get('audio_path', 'unknown')}"
                )
        
        return unique_samples / len(samples) if samples else 0.0
    
    def rollback_to_version(self, version_id: str, target_path: str) -> bool:
        """
        Rollback to a specific version.
        
        Args:
            version_id: Version to rollback to
            target_path: Path where to restore the version
        
        Returns:
            True if successful
        """
        version = self.versions.get(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False
        
        source_path = Path(version.dataset_path)
        target_path = Path(target_path)
        
        if not source_path.exists():
            logger.error(f"Source version path does not exist: {source_path}")
            return False
        
        try:
            # Copy version to target path
            import shutil
            if source_path.is_dir():
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
            
            logger.info(f"Rolled back to version {version_id} at {target_path}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def generate_version_report(self) -> Dict:
        """Generate comprehensive version control report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_versions': len(self.versions),
            'versions_by_quality': defaultdict(int),
            'recent_versions': [],
            'quality_trends': []
        }
        
        # Categorize by quality
        for version in self.versions.values():
            quality_score = version.metadata.get('quality_report', {}).get('quality_metrics', {}).get('overall_score', 0)
            if quality_score >= 0.95:
                report['versions_by_quality']['excellent'] += 1
            elif quality_score >= 0.90:
                report['versions_by_quality']['good'] += 1
            elif quality_score >= 0.80:
                report['versions_by_quality']['acceptable'] += 1
            else:
                report['versions_by_quality']['poor'] += 1
        
        # Get recent versions
        recent = sorted(self.versions.values(), key=lambda v: v.created_at, reverse=True)[:10]
        report['recent_versions'] = [
            {
                'version_id': v.version_id,
                'created_at': v.created_at,
                'checksum': v.checksum,
                'quality_score': v.metadata.get('quality_report', {}).get('quality_metrics', {}).get('overall_score', 0)
            }
            for v in recent
        ]
        
        return report

