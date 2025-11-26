"""
Integration Module for Data Management System
Provides unified interface for all data management components.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .data_manager import DataManager
from .metadata_tracker import MetadataTracker
from .finetuning_pipeline import FinetuningDatasetPipeline
from .version_control import DataVersionControl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedDataManagementSystem:
    """
    Unified interface for the complete data management system.
    Integrates data storage, metadata tracking, fine-tuning pipeline,
    and version control.
    """
    
    def __init__(
        self,
        base_dir: str = "data",
        use_gcs: bool = True,
        gcs_bucket_name: str = "stt-project-datasets",
        project_id: str = "stt-agentic-ai-2025"
    ):
        """
        Initialize integrated data management system.
        
        Args:
            base_dir: Base directory for all data storage
            use_gcs: Whether to use Google Cloud Storage
            gcs_bucket_name: GCS bucket name
            project_id: GCP project ID
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all components
        logger.info("Initializing Integrated Data Management System...")
        
        self.data_manager = DataManager(
            local_storage_dir=str(self.base_dir / "failed_cases"),
            use_gcs=use_gcs,
            gcs_bucket_name=gcs_bucket_name,
            gcs_prefix="learning_data",
            project_id=project_id
        )
        
        self.metadata_tracker = MetadataTracker(
            local_storage_dir=str(self.base_dir / "metadata"),
            use_gcs=use_gcs,
            gcs_bucket_name=gcs_bucket_name,
            gcs_prefix="metadata",
            project_id=project_id
        )
        
        self.finetuning_pipeline = FinetuningDatasetPipeline(
            data_manager=self.data_manager,
            output_dir=str(self.base_dir / "finetuning"),
            use_gcs=use_gcs,
            gcs_bucket_name=gcs_bucket_name,
            gcs_prefix="finetuning_datasets",
            project_id=project_id
        )
        
        self.version_control = DataVersionControl(
            local_storage_dir=str(self.base_dir / "versions"),
            use_gcs=use_gcs,
            gcs_bucket_name=gcs_bucket_name,
            gcs_prefix="data_versions",
            project_id=project_id
        )
        
        logger.info("Integrated Data Management System initialized successfully")
    
    def record_failed_transcription(
        self,
        audio_path: str,
        original_transcript: str,
        corrected_transcript: Optional[str],
        error_types: List[str],
        error_score: float,
        inference_time: float,
        model_confidence: Optional[float] = None,
        additional_metadata: Optional[Dict] = None
    ) -> str:
        """
        Record a failed transcription with full tracking.
        
        Args:
            audio_path: Path to audio file
            original_transcript: Original (failed) transcription
            corrected_transcript: Corrected transcription (if available)
            error_types: List of error types detected
            error_score: Error confidence score
            inference_time: Inference time in seconds
            model_confidence: Model confidence score
            additional_metadata: Additional metadata
        
        Returns:
            Case ID
        """
        # Store failed case
        metadata = additional_metadata or {}
        metadata['inference_time'] = inference_time
        metadata['model_confidence'] = model_confidence
        
        case_id = self.data_manager.store_failed_case(
            audio_path=audio_path,
            original_transcript=original_transcript,
            corrected_transcript=corrected_transcript,
            error_types=error_types,
            error_score=error_score,
            metadata=metadata
        )
        
        # Track inference statistics
        self.metadata_tracker.record_inference_stats(
            audio_path=audio_path,
            inference_time=inference_time,
            model_confidence=model_confidence,
            error_detected=True,
            corrected=corrected_transcript is not None,
            metadata={'case_id': case_id, 'error_score': error_score}
        )
        
        logger.info(f"Recorded failed transcription: {case_id}")
        
        return case_id
    
    def add_correction(
        self,
        case_id: str,
        corrected_transcript: str,
        correction_method: str = 'manual'
    ) -> bool:
        """
        Add a correction to a failed case.
        
        Args:
            case_id: ID of the failed case
            corrected_transcript: Corrected transcription
            correction_method: Method used for correction
        
        Returns:
            True if successful
        """
        success = self.data_manager.store_correction(
            case_id=case_id,
            corrected_transcript=corrected_transcript,
            correction_method=correction_method
        )
        
        if success:
            logger.info(f"Added correction for case: {case_id}")
        
        return success
    
    def prepare_finetuning_dataset(
        self,
        min_error_score: float = 0.5,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        max_samples: Optional[int] = None,
        balance_error_types: bool = True,
        create_version: bool = True
    ) -> Dict:
        """
        Prepare a fine-tuning dataset with versioning.
        
        Args:
            min_error_score: Minimum error score to include
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            test_ratio: Ratio for test split
            max_samples: Maximum number of samples
            balance_error_types: Balance samples across error types
            create_version: Whether to create a dataset version
        
        Returns:
            Dataset information including version ID
        """
        logger.info("Preparing fine-tuning dataset...")
        
        # Prepare dataset
        dataset_info = self.finetuning_pipeline.prepare_dataset(
            min_error_score=min_error_score,
            include_uncorrected=False,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            max_samples=max_samples,
            balance_error_types=balance_error_types
        )
        
        if 'error' in dataset_info:
            return dataset_info
        
        dataset_id = dataset_info['dataset_id']
        dataset_path = dataset_info['local_path']
        
        # Create version if requested
        if create_version:
            version_id = self.version_control.create_version(
                dataset_path=dataset_path,
                version_name=dataset_id,
                metadata={
                    'dataset_info': dataset_info,
                    'preparation_params': {
                        'min_error_score': min_error_score,
                        'max_samples': max_samples,
                        'balance_error_types': balance_error_types
                    }
                },
                run_quality_check=True
            )
            dataset_info['version_id'] = version_id
            logger.info(f"Created dataset version: {version_id}")
        
        return dataset_info
    
    def record_training_performance(
        self,
        model_version: str,
        wer: float,
        cer: float,
        training_metadata: Dict
    ):
        """
        Record model training performance.
        
        Args:
            model_version: Model version identifier
            wer: Word Error Rate
            cer: Character Error Rate
            training_metadata: Training configuration and metadata
        """
        # Record performance metrics
        self.metadata_tracker.record_performance(
            wer=wer,
            cer=cer,
            metadata={'model_version': model_version}
        )
        
        # Record model version
        self.metadata_tracker.record_model_version(
            version_id=model_version,
            model_name=training_metadata.get('model_name', 'unknown'),
            training_data_size=training_metadata.get('training_data_size', 0),
            training_metadata=training_metadata,
            performance_metrics={'wer': wer, 'cer': cer}
        )
        
        logger.info(f"Recorded training performance for {model_version}")
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics."""
        return {
            'data_management': self.data_manager.get_statistics(),
            'performance_trends': {
                'wer': self.metadata_tracker.get_performance_trend('wer'),
                'cer': self.metadata_tracker.get_performance_trend('cer')
            },
            'learning_summary': self.metadata_tracker.get_learning_summary(),
            'inference_stats': self.metadata_tracker.get_inference_statistics(),
            'version_control': self.version_control.generate_version_report(),
            'available_datasets': len(self.finetuning_pipeline.list_datasets())
        }
    
    def generate_comprehensive_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive system report.
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            Complete system report
        """
        logger.info("Generating comprehensive report...")
        
        report = {
            'generated_at': self.metadata_tracker.generate_performance_report()['generated_at'],
            'system_statistics': self.get_system_statistics(),
            'performance_report': self.metadata_tracker.generate_performance_report(),
            'data_quality': self._assess_overall_quality(),
            'recommendations': self._generate_recommendations()
        }
        
        if output_path:
            import json
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _assess_overall_quality(self) -> Dict:
        """Assess overall data quality."""
        stats = self.data_manager.get_statistics()
        
        return {
            'total_failed_cases': stats['total_failed_cases'],
            'correction_rate': stats['correction_rate'],
            'quality_status': 'good' if stats['correction_rate'] > 0.7 else 'needs_improvement',
            'error_type_diversity': len(stats['error_type_distribution'])
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on system state."""
        recommendations = []
        
        stats = self.data_manager.get_statistics()
        
        # Check correction rate
        if stats['correction_rate'] < 0.5:
            recommendations.append(
                "Low correction rate detected. Consider adding more manual corrections "
                "to improve fine-tuning dataset quality."
            )
        
        # Check dataset size
        if stats['total_failed_cases'] < 100:
            recommendations.append(
                "Limited failed cases collected. Continue monitoring to build a larger "
                "dataset before fine-tuning."
            )
        elif stats['total_failed_cases'] >= 500:
            recommendations.append(
                "Sufficient failed cases collected. Consider preparing a fine-tuning dataset."
            )
        
        # Check error type distribution
        error_types = len(stats['error_type_distribution'])
        if error_types < 3:
            recommendations.append(
                "Limited error type diversity. The model may benefit from more varied error cases."
            )
        
        # Check versions
        versions = len(self.version_control.versions)
        if versions == 0:
            recommendations.append(
                "No dataset versions created yet. Consider versioning your prepared datasets "
                "for better tracking and reproducibility."
            )
        
        return recommendations
    
    def sync_all_to_gcs(self):
        """Sync all data to Google Cloud Storage."""
        logger.info("Syncing all data to GCS...")
        
        self.data_manager._sync_to_gcs()
        self.metadata_tracker._sync_to_gcs()
        self.version_control._save_registry()
        
        logger.info("All data synced to GCS")
    
    def sync_all_from_gcs(self):
        """Sync all data from Google Cloud Storage."""
        logger.info("Syncing all data from GCS...")
        
        self.data_manager.sync_from_gcs()
        self.metadata_tracker.sync_from_gcs()
        
        logger.info("All data synced from GCS")

