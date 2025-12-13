"""
Metadata Tracking System for Performance Monitoring
Tracks model performance, improvements, and learning progress over time.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

from ..utils.gcs_utils import GCSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(
        self,
        timestamp: str,
        wer: Optional[float] = None,
        cer: Optional[float] = None,
        error_rate: Optional[float] = None,
        correction_rate: Optional[float] = None,
        inference_time: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        self.timestamp = timestamp
        self.wer = wer
        self.cer = cer
        self.error_rate = error_rate
        self.correction_rate = correction_rate
        self.inference_time = inference_time
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'wer': self.wer,
            'cer': self.cer,
            'error_rate': self.error_rate,
            'correction_rate': self.correction_rate,
            'inference_time': self.inference_time,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PerformanceMetrics':
        """Create from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            wer=data.get('wer'),
            cer=data.get('cer'),
            error_rate=data.get('error_rate'),
            correction_rate=data.get('correction_rate'),
            inference_time=data.get('inference_time'),
            metadata=data.get('metadata', {})
        )


class MetadataTracker:
    """
    Comprehensive metadata tracking system for monitoring performance
    improvements and learning progress over time.
    """
    
    def __init__(
        self,
        local_storage_dir: str = "data/metadata",
        use_gcs: bool = True,
        gcs_bucket_name: str = "stt-project-datasets",
        gcs_prefix: str = "metadata",
        project_id: str = "stt-agentic-ai-2025"
    ):
        """
        Initialize metadata tracker.
        
        Args:
            local_storage_dir: Local directory for metadata storage
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
                logger.info("GCS integration enabled for metadata tracking")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS: {e}. Using local storage only.")
                self.use_gcs = False
        
        # Storage files
        self.performance_file = self.local_storage_dir / "performance_history.jsonl"
        self.model_versions_file = self.local_storage_dir / "model_versions.json"
        self.learning_progress_file = self.local_storage_dir / "learning_progress.jsonl"
        self.inference_stats_file = self.local_storage_dir / "inference_stats.jsonl"
        
        # In-memory caches
        self.performance_history: List[PerformanceMetrics] = []
        self.model_versions: Dict[str, Any] = {}
        self.learning_progress: List[Dict] = []
        self.inference_stats: List[Dict] = []
        
        # Load existing data
        self._load_local_data()
        
        logger.info(f"Metadata Tracker initialized (local: {self.local_storage_dir})")
    
    def _load_local_data(self):
        """Load existing metadata from local storage."""
        # Load performance history
        if self.performance_file.exists() and self.performance_file.stat().st_size > 0:
            try:
                with open(self.performance_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                self.performance_history.append(
                                    PerformanceMetrics.from_dict(json.loads(line))
                                )
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.warning(f"Invalid JSON line in performance file: {e}")
                                continue
                logger.info(f"Loaded {len(self.performance_history)} performance records")
            except Exception as e:
                logger.warning(f"Error loading performance history: {e}")
                self.performance_history = []
        
        # Load model versions
        if self.model_versions_file.exists() and self.model_versions_file.stat().st_size > 0:
            try:
                with open(self.model_versions_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.model_versions = json.loads(content)
                        logger.info(f"Loaded {len(self.model_versions)} model versions")
                    else:
                        logger.warning(f"Model versions file {self.model_versions_file} is empty, initializing empty dict")
                        self.model_versions = {}
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in model versions file {self.model_versions_file}: {e}, initializing empty dict")
                self.model_versions = {}
            except Exception as e:
                logger.warning(f"Error loading model versions from {self.model_versions_file}: {e}, initializing empty dict")
                self.model_versions = {}
        else:
            if self.model_versions_file.exists():
                logger.info(f"Model versions file {self.model_versions_file} exists but is empty, initializing empty dict")
            self.model_versions = {}
        
        # Load learning progress
        if self.learning_progress_file.exists() and self.learning_progress_file.stat().st_size > 0:
            try:
                with open(self.learning_progress_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                self.learning_progress.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON line in learning progress file: {e}")
                                continue
                logger.info(f"Loaded {len(self.learning_progress)} learning progress records")
            except Exception as e:
                logger.warning(f"Error loading learning progress: {e}")
                self.learning_progress = []
        
        # Load inference stats
        if self.inference_stats_file.exists() and self.inference_stats_file.stat().st_size > 0:
            try:
                with open(self.inference_stats_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                self.inference_stats.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON line in inference stats file: {e}")
                                continue
                logger.info(f"Loaded {len(self.inference_stats)} inference stats records")
            except Exception as e:
                logger.warning(f"Error loading inference stats: {e}")
                self.inference_stats = []
    
    def record_performance(
        self,
        wer: Optional[float] = None,
        cer: Optional[float] = None,
        error_rate: Optional[float] = None,
        correction_rate: Optional[float] = None,
        inference_time: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record performance metrics.
        
        Args:
            wer: Word Error Rate
            cer: Character Error Rate
            error_rate: Error detection rate
            correction_rate: Correction success rate
            inference_time: Inference time in seconds
            metadata: Additional metadata
        
        Returns:
            Timestamp of the record
        """
        timestamp = datetime.now().isoformat()
        
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            wer=wer,
            cer=cer,
            error_rate=error_rate,
            correction_rate=correction_rate,
            inference_time=inference_time,
            metadata=metadata
        )
        
        self.performance_history.append(metrics)
        
        # Append to local storage
        with open(self.performance_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
        
        logger.info(f"Recorded performance metrics at {timestamp}")
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            self._sync_to_gcs()
        
        return timestamp
    
    def record_model_version(
        self,
        version_id: str,
        model_name: str,
        training_data_size: int,
        training_metadata: Dict,
        performance_metrics: Optional[Dict] = None
    ):
        """
        Record a new model version.
        
        Args:
            version_id: Unique version identifier
            model_name: Model name/type
            training_data_size: Number of training samples
            training_metadata: Training configuration and metadata
            performance_metrics: Initial performance metrics
        """
        self.model_versions[version_id] = {
            'version_id': version_id,
            'model_name': model_name,
            'training_data_size': training_data_size,
            'training_metadata': training_metadata,
            'performance_metrics': performance_metrics or {},
            'created_at': datetime.now().isoformat()
        }
        
        # Save to local storage
        with open(self.model_versions_file, 'w') as f:
            json.dump(self.model_versions, f, indent=2)
        
        logger.info(f"Recorded model version: {version_id}")
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            self._sync_to_gcs()
    
    def record_learning_progress(
        self,
        stage: str,
        metrics: Dict,
        metadata: Optional[Dict] = None
    ):
        """
        Record learning progress at a specific stage.
        
        Args:
            stage: Learning stage (e.g., 'initial', 'week1', 'week2')
            metrics: Performance metrics at this stage
            metadata: Additional metadata
        """
        progress = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.learning_progress.append(progress)
        
        # Append to local storage
        with open(self.learning_progress_file, 'a') as f:
            f.write(json.dumps(progress) + '\n')
        
        logger.info(f"Recorded learning progress at stage: {stage}")
        
        # Sync to GCS if enabled
        if self.use_gcs and self.gcs_manager:
            self._sync_to_gcs()
    
    def record_inference_stats(
        self,
        audio_path: str,
        inference_time: float,
        model_confidence: Optional[float] = None,
        error_detected: bool = False,
        corrected: bool = False,
        metadata: Optional[Dict] = None
    ):
        """
        Record inference statistics for individual transcriptions.
        
        Args:
            audio_path: Path to audio file
            inference_time: Time taken for inference
            model_confidence: Model confidence score
            error_detected: Whether errors were detected
            corrected: Whether corrections were applied
            metadata: Additional metadata
        """
        stats = {
            'timestamp': datetime.now().isoformat(),
            'audio_path': audio_path,
            'inference_time': inference_time,
            'model_confidence': model_confidence,
            'error_detected': error_detected,
            'corrected': corrected,
            'metadata': metadata or {}
        }
        
        self.inference_stats.append(stats)
        
        # Append to local storage
        with open(self.inference_stats_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
        
        # Sync periodically (every 100 records)
        if len(self.inference_stats) % 100 == 0 and self.use_gcs and self.gcs_manager:
            self._sync_to_gcs()
    
    def get_performance_trend(
        self,
        metric: str = 'wer',
        time_window_days: Optional[int] = None
    ) -> Dict:
        """
        Get performance trend for a specific metric.
        
        Args:
            metric: Metric to analyze ('wer', 'cer', 'error_rate', etc.)
            time_window_days: Limit to last N days (None for all)
        
        Returns:
            Dictionary with trend analysis
        """
        if not self.performance_history:
            return {'error': 'No performance history available'}
        
        # Filter by time window if specified
        records = self.performance_history
        if time_window_days:
            cutoff = datetime.now() - timedelta(days=time_window_days)
            records = [
                r for r in records
                if datetime.fromisoformat(r.timestamp) > cutoff
            ]
        
        # Extract metric values
        values = [getattr(r, metric) for r in records if getattr(r, metric) is not None]
        
        if not values:
            return {'error': f'No data for metric: {metric}'}
        
        # Calculate trend
        timestamps = [r.timestamp for r in records if getattr(r, metric) is not None]
        
        return {
            'metric': metric,
            'count': len(values),
            'latest': values[-1],
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'improvement': values[0] - values[-1] if len(values) > 1 else 0.0,
            'improvement_percent': ((values[0] - values[-1]) / values[0] * 100) if len(values) > 1 and values[0] != 0 else 0.0,
            'timestamps': timestamps,
            'values': values
        }
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress."""
        if not self.learning_progress:
            return {'error': 'No learning progress recorded'}
        
        stages = [p['stage'] for p in self.learning_progress]
        latest = self.learning_progress[-1]
        
        return {
            'total_stages': len(set(stages)),
            'total_records': len(self.learning_progress),
            'latest_stage': latest['stage'],
            'latest_metrics': latest['metrics'],
            'latest_timestamp': latest['timestamp'],
            'all_stages': list(set(stages))
        }
    
    def get_inference_statistics(self, time_window_hours: int = 24) -> Dict:
        """
        Get inference statistics for recent time window.
        
        Args:
            time_window_hours: Time window in hours
        
        Returns:
            Dictionary with inference statistics
        """
        if not self.inference_stats:
            return {'error': 'No inference statistics available'}
        
        # Filter by time window
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        recent_stats = [
            s for s in self.inference_stats
            if datetime.fromisoformat(s['timestamp']) > cutoff
        ]
        
        if not recent_stats:
            return {'error': 'No recent inference statistics'}
        
        inference_times = [s['inference_time'] for s in recent_stats]
        errors_detected = sum(1 for s in recent_stats if s['error_detected'])
        corrections_applied = sum(1 for s in recent_stats if s['corrected'])
        
        return {
            'time_window_hours': time_window_hours,
            'total_inferences': len(recent_stats),
            'avg_inference_time': np.mean(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'error_detection_rate': errors_detected / len(recent_stats),
            'correction_rate': corrections_applied / len(recent_stats),
            'errors_detected': errors_detected,
            'corrections_applied': corrections_applied
        }
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all model versions."""
        if not self.model_versions:
            return pd.DataFrame()
        
        data = []
        for version_id, info in self.model_versions.items():
            row = {
                'version_id': version_id,
                'model_name': info['model_name'],
                'training_data_size': info['training_data_size'],
                'created_at': info['created_at']
            }
            # Add performance metrics
            if info.get('performance_metrics'):
                row.update(info['performance_metrics'])
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_performance_history(self) -> pd.DataFrame:
        """Export performance history as DataFrame."""
        data = [m.to_dict() for m in self.performance_history]
        return pd.DataFrame(data)
    
    def export_learning_progress(self) -> pd.DataFrame:
        """Export learning progress as DataFrame."""
        return pd.DataFrame(self.learning_progress)
    
    def export_inference_stats(self) -> pd.DataFrame:
        """Export inference statistics as DataFrame."""
        return pd.DataFrame(self.inference_stats)
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'performance_trends': {},
            'learning_summary': self.get_learning_summary(),
            'recent_inference_stats': self.get_inference_statistics(),
            'model_versions': len(self.model_versions),
            'total_records': {
                'performance': len(self.performance_history),
                'learning_progress': len(self.learning_progress),
                'inference_stats': len(self.inference_stats)
            }
        }
        
        # Add trends for each metric
        for metric in ['wer', 'cer', 'error_rate', 'correction_rate', 'inference_time']:
            trend = self.get_performance_trend(metric)
            if 'error' not in trend:
                report['performance_trends'][metric] = trend
        
        return report
    
    def _sync_to_gcs(self):
        """Sync metadata to Google Cloud Storage."""
        if not self.gcs_manager:
            return
        
        try:
            # Upload performance history
            gcs_path = f"{self.gcs_prefix}/performance_history.jsonl"
            self.gcs_manager.upload_file(str(self.performance_file), gcs_path)
            
            # Upload model versions
            gcs_path = f"{self.gcs_prefix}/model_versions.json"
            self.gcs_manager.upload_file(str(self.model_versions_file), gcs_path)
            
            # Upload learning progress
            gcs_path = f"{self.gcs_prefix}/learning_progress.jsonl"
            self.gcs_manager.upload_file(str(self.learning_progress_file), gcs_path)
            
            # Upload inference stats
            gcs_path = f"{self.gcs_prefix}/inference_stats.jsonl"
            self.gcs_manager.upload_file(str(self.inference_stats_file), gcs_path)
            
            logger.debug("Synced metadata to GCS")
        except Exception as e:
            logger.error(f"Failed to sync metadata to GCS: {e}")
    
    def sync_from_gcs(self):
        """Sync metadata from Google Cloud Storage."""
        if not self.gcs_manager:
            logger.warning("GCS not enabled")
            return
        
        try:
            # Download all metadata files
            files = [
                ('performance_history.jsonl', self.performance_file),
                ('model_versions.json', self.model_versions_file),
                ('learning_progress.jsonl', self.learning_progress_file),
                ('inference_stats.jsonl', self.inference_stats_file)
            ]
            
            for gcs_file, local_file in files:
                gcs_path = f"{self.gcs_prefix}/{gcs_file}"
                self.gcs_manager.download_file(gcs_path, str(local_file))
            
            # Reload local data
            self.performance_history.clear()
            self.model_versions.clear()
            self.learning_progress.clear()
            self.inference_stats.clear()
            self._load_local_data()
            
            logger.info("Synced metadata from GCS")
        except Exception as e:
            logger.error(f"Failed to sync metadata from GCS: {e}")

