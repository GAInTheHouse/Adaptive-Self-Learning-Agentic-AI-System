"""
Google Cloud Storage utility functions for data management.
"""

import os
from pathlib import Path
from typing import Optional, List
from google.cloud import storage
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCSManager:
    """Manager for Google Cloud Storage operations"""
    
    def __init__(self, project_id: str, bucket_name: str):
        """
        Initialize GCS Manager.
        
        Args:
            project_id: GCP project ID
            bucket_name: GCS bucket name (without gs:// prefix)
        """
        self.project_id = project_id
        self.bucket_name = bucket_name.replace("gs://", "")
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(self.bucket_name)
        logger.info(f"GCS Manager initialized for bucket: {self.bucket_name}")
    
    def upload_file(self, local_path: str, gcs_path: str) -> bool:
        """
        Upload a single file to GCS.
        
        Args:
            local_path: Local file path
            gcs_path: Destination path in GCS (relative to bucket)
        
        Returns:
            True if successful
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded: {local_path} -> gs://{self.bucket_name}/{gcs_path}")
            return True
        except Exception as e:
            logger.error(f"Upload failed for {local_path}: {e}")
            return False
    
    def download_file(self, gcs_path: str, local_path: str) -> bool:
        """
        Download a single file from GCS.
        
        Args:
            gcs_path: Source path in GCS (relative to bucket)
            local_path: Destination local path
        
        Returns:
            True if successful
        """
        try:
            # Create parent directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded: gs://{self.bucket_name}/{gcs_path} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed for {gcs_path}: {e}")
            return False
    
    def upload_directory(self, local_dir: str, gcs_prefix: str, 
                        extensions: Optional[List[str]] = None) -> int:
        """
        Upload entire directory to GCS.
        
        Args:
            local_dir: Local directory path
            gcs_prefix: Prefix for GCS paths
            extensions: File extensions to include (e.g., ['.wav', '.mp3'])
        
        Returns:
            Number of files uploaded
        """
        local_path = Path(local_dir)
        if not local_path.exists():
            logger.error(f"Directory does not exist: {local_dir}")
            return 0
        
        files = []
        if extensions:
            for ext in extensions:
                files.extend(local_path.rglob(f"*{ext}"))
        else:
            files = list(local_path.rglob("*"))
        
        files = [f for f in files if f.is_file()]
        
        uploaded = 0
        for file_path in tqdm(files, desc="Uploading to GCS"):
            relative_path = file_path.relative_to(local_path)
            gcs_path = f"{gcs_prefix}/{relative_path}"
            if self.upload_file(str(file_path), gcs_path):
                uploaded += 1
        
        logger.info(f"Uploaded {uploaded}/{len(files)} files to gs://{self.bucket_name}/{gcs_prefix}")
        return uploaded
    
    def download_directory(self, gcs_prefix: str, local_dir: str,
                          extensions: Optional[List[str]] = None) -> int:
        """
        Download entire directory from GCS.
        
        Args:
            gcs_prefix: Prefix for GCS paths
            local_dir: Local destination directory
            extensions: File extensions to include
        
        Returns:
            Number of files downloaded
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix=gcs_prefix)
        
        downloaded = 0
        for blob in tqdm(list(blobs), desc="Downloading from GCS"):
            # Skip if extension filter is set
            if extensions and not any(blob.name.endswith(ext) for ext in extensions):
                continue
            
            # Calculate local path
            relative_path = blob.name[len(gcs_prefix):].lstrip("/")
            local_path = Path(local_dir) / relative_path
            
            # Skip directories
            if blob.name.endswith("/"):
                continue
            
            if self.download_file(blob.name, str(local_path)):
                downloaded += 1
        
        logger.info(f"Downloaded {downloaded} files to {local_dir}")
        return downloaded
    
    def list_files(self, prefix: str = "") -> List[str]:
        """
        List files in GCS bucket with given prefix.
        
        Args:
            prefix: Path prefix to filter
        
        Returns:
            List of file paths
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs if not blob.name.endswith("/")]
    
    def delete_file(self, gcs_path: str) -> bool:
        """
        Delete a file from GCS.
        
        Args:
            gcs_path: Path in GCS to delete
        
        Returns:
            True if successful
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.delete()
            logger.info(f"Deleted: gs://{self.bucket_name}/{gcs_path}")
            return True
        except Exception as e:
            logger.error(f"Delete failed for {gcs_path}: {e}")
            return False

def get_gcs_manager(bucket_type: str = "datasets") -> GCSManager:
    """
    Factory function to get GCS manager for specific bucket.
    
    Args:
        bucket_type: One of 'datasets', 'models', 'logs'
    
    Returns:
        GCSManager instance
    """
    project_id = "stt-agentic-ai-2025"
    bucket_map = {
        "datasets": "stt-project-datasets",
        "models": "stt-project-models",
        "logs": "stt-project-logs"
    }
    
    if bucket_type not in bucket_map:
        raise ValueError(f"Invalid bucket_type. Choose from: {list(bucket_map.keys())}")
    
    return GCSManager(project_id, bucket_map[bucket_type])
