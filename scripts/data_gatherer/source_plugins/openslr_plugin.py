#!/usr/bin/env python3
"""
OpenSLR dataset plugin.

Handles downloading and manifest generation for OpenSLR datasets:
- LibriSpeech (dev-clean, dev-other, test-clean, test-other)
- MUSAN noise corpus
- RIRS_NOISES
- TED-LIUM Release 3
- ST-AEDS
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from source_plugins import DataSourcePlugin
from dataset_utils import configure_logging


LOGGER = logging.getLogger("OpenSLRPlugin")


class OpenSLRPlugin(DataSourcePlugin):
    """Plugin for downloading OpenSLR datasets."""
    
    def __init__(self):
        self.logger = LOGGER
    
    def get_source_type(self) -> str:
        return "openslr"
    
    def download(self, config: Dict, output_dir: Path, force: bool) -> Optional[Path]:
        """
        Download and extract OpenSLR dataset.
        
        Uses resumable HTTP downloads with checksum verification when available.
        """
        url = config["url"]
        extract_path = config.get("extract_path", "")
        
        self.logger.info("Downloading OpenSLR from: %s", url)
        
        # Check if already extracted
        final_dir = output_dir / extract_path if extract_path else output_dir
        if final_dir.exists() and not force:
            self.logger.info("Dataset already exists at: %s", final_dir)
            return final_dir
        
        # Download and extract
        try:
            archive_path = self._download_and_extract(url, output_dir)
            self.logger.info("Download complete: %s", archive_path.name)
            return final_dir
        except Exception as exc:
            self.logger.error("Failed to download %s: %s", url, exc)
            return None
    
    def generate_manifest(
        self,
        data_dir: Path,
        manifest_dir: Path,
        dataset_name: str,
        force: bool
    ) -> List[Path]:
        """
        Generate manifest for OpenSLR dataset.
        
        Routes to appropriate generator based on dataset_type from config.
        """
        sys.path.insert(0, str(Path(__file__).parent.parent / "manifest_generators"))
        import librispeech
        import musan
        import rirs
        import st_aeds
        
        dataset_type = self._infer_dataset_type(dataset_name, data_dir)
        
        if dataset_type == "librispeech":
            return librispeech.generate(data_dir, manifest_dir, force)
        elif dataset_type == "musan":
            return musan.generate(data_dir, manifest_dir, force)
        elif dataset_type == "rirs":
            return rirs.generate(data_dir, manifest_dir, force)
        elif dataset_type == "st_aeds":
            return st_aeds.generate(data_dir, manifest_dir, force)
        else:
            self.logger.warning(
                "No manifest generator for dataset type: %s", dataset_type
            )
            return []
    
    def _infer_dataset_type(self, dataset_name: str, data_dir: Path) -> str:
        """Infer dataset type from name or directory structure."""
        name_lower = dataset_name.lower()
        
        # region agent log
        import json
        inferred = "unknown"
        if "librispeech" in name_lower:
            inferred = "librispeech"
        elif "musan" in name_lower:
            inferred = "musan"
        elif "rirs" in name_lower or "noises" in name_lower:
            inferred = "rirs"
        elif "tedlium" in name_lower:
            inferred = "tedlium"
        elif "aeds" in name_lower:
            inferred = "st_aeds"
        
        log_data = {"hypothesisId": "F", "runId": "debug1", "location": "openslr_plugin.py:115", "message": "Dataset type inference", "data": {"dataset_name": dataset_name, "inferred_type": inferred, "data_dir": str(data_dir)}, "timestamp": int(__import__('time').time() * 1000)}
        try:
            with open('/Users/gainthehouse/Desktop/Code/Adaptive-Self-Learning-Agentic-AI-System/.cursor/debug-3abd3e.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # endregion
        
        if "librispeech" in name_lower:
            return "librispeech"
        elif "musan" in name_lower:
            return "musan"
        elif "rirs" in name_lower or "noises" in name_lower:
            return "rirs"
        elif "tedlium" in name_lower:
            return "tedlium"
        elif "aeds" in name_lower:
            return "st_aeds"
        
        return "unknown"
    
    def _download_and_extract(self, url: str, out_dir: Path) -> Path:
        """Download archive with resume support and extract."""
        out_dir.mkdir(parents=True, exist_ok=True)
        
        filename = self._resolve_filename(url)
        archive_path = out_dir / filename
        
        # Download with resume support
        self._stream_download(url, archive_path)
        
        # Verify checksum if available
        self._verify_checksum_if_available(url, archive_path, filename)
        
        # Extract
        self._extract_archive(archive_path, out_dir)
        
        return archive_path
    
    def _resolve_filename(self, url: str) -> str:
        """Extract filename from URL."""
        parsed = urllib.parse.urlparse(url)
        name = Path(parsed.path).name
        if not name:
            raise RuntimeError(f"Could not infer filename from URL: {url}")
        return name
    
    def _head(self, url: str) -> Tuple[Optional[int], bool]:
        """Send HEAD request to get content length and range support."""
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            content_length_raw = response.headers.get("Content-Length")
            accept_ranges = (response.headers.get("Accept-Ranges") or "").lower()
            content_length = (
                int(content_length_raw) 
                if content_length_raw and content_length_raw.isdigit() 
                else None
            )
            supports_range = "bytes" in accept_ranges
            return content_length, supports_range
    
    def _stream_download(self, url: str, destination: Path) -> None:
        """Download file with resume support."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        existing_bytes = destination.stat().st_size if destination.exists() else 0
        content_length = None
        supports_range = False
        
        try:
            content_length, supports_range = self._head(url)
        except Exception as head_exc:
            # region agent log
            import json
            log_data = {"hypothesisId": "D", "runId": "debug1", "location": "openslr_plugin.py:169", "message": "HEAD request failed", "data": {"url": url, "error_type": type(head_exc).__name__, "error_msg": str(head_exc)}, "timestamp": int(__import__('time').time() * 1000)}
            try:
                with open('/Users/gainthehouse/Desktop/Code/Adaptive-Self-Learning-Agentic-AI-System/.cursor/debug-3abd3e.log', 'a') as f:
                    f.write(json.dumps(log_data) + '\n')
            except: pass
            # endregion
            
            self.logger.warning("HEAD request failed for %s", url)
        
        # Check if already complete
        if content_length and existing_bytes == content_length and content_length > 0:
            self.logger.info("Download already complete: %s", destination.name)
            return
        
        # Resume from existing bytes if supported
        range_start = existing_bytes if (existing_bytes > 0 and supports_range) else 0
        request = urllib.request.Request(url)
        
        if range_start > 0:
            request.add_header("Range", f"bytes={range_start}-")
            self.logger.info("Resuming download at byte %d: %s", range_start, destination.name)
        else:
            self.logger.info("Downloading: %s", destination.name)
        
        with urllib.request.urlopen(request) as response:
            status = getattr(response, "status", 200)
            append_mode = range_start > 0 and status == 206
            mode = "ab" if append_mode else "wb"
            
            if range_start > 0 and not append_mode:
                self.logger.info("Server ignored Range; restarting from 0")
            
            with destination.open(mode) as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        
        final_size = destination.stat().st_size if destination.exists() else 0
        if final_size <= 0:
            raise RuntimeError(f"Download failed or empty file: {destination}")
        
        if content_length and final_size != content_length:
            raise RuntimeError(
                f"Size mismatch for {destination.name}: "
                f"expected {content_length}, got {final_size}"
            )
    
    def _verify_checksum_if_available(self, url: str, file_path: Path, filename: str) -> None:
        """Verify file checksum if available from server."""
        checksum = self._maybe_get_checksum(url, filename)
        
        if not checksum:
            size = file_path.stat().st_size if file_path.exists() else 0
            if size <= 0:
                raise RuntimeError(f"Downloaded file is empty: {file_path}")
            self.logger.info("No checksum available; validated non-empty: %s", filename)
            return
        
        algorithm, expected_hash, checksum_url = checksum
        actual_hash = self._hash_file(file_path, algorithm)
        
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{algorithm.upper()} mismatch for {filename}.\n"
                f"Expected: {expected_hash}\nActual:   {actual_hash}\n"
                f"Source:   {checksum_url}"
            )
        
        self.logger.info("%s verified: %s", algorithm.upper(), filename)
    
    def _hash_file(self, path: Path, algorithm: str) -> str:
        """Compute file hash."""
        hasher = hashlib.new(algorithm)
        with path.open("rb") as f:
            while True:
                block = f.read(1024 * 1024)
                if not block:
                    break
                hasher.update(block)
        return hasher.hexdigest().lower()
    
    def _maybe_get_checksum(self, url: str, filename: str) -> Optional[Tuple[str, str, str]]:
        """Try to fetch checksum file from common locations."""
        base = url.rsplit("/", 1)[0]
        candidates = [
            (f"{url}.sha256", "sha256"),
            (f"{url}.md5", "md5"),
            (f"{base}/sha256sum.txt", None),
            (f"{base}/md5sum.txt", None),
            (f"{base}/checksums.txt", None),
        ]
        
        for checksum_url, forced_algo in candidates:
            try:
                with urllib.request.urlopen(checksum_url) as response:
                    body = response.read().decode("utf-8", errors="replace")
            except (urllib.error.HTTPError, urllib.error.URLError):
                continue
            
            for line in body.splitlines():
                parsed = self._parse_checksum_line(line, filename)
                if not parsed:
                    continue
                
                algorithm, expected_hash = parsed
                if forced_algo and algorithm != forced_algo:
                    continue
                
                return algorithm, expected_hash, checksum_url
        
        return None
    
    def _parse_checksum_line(self, line: str, filename: str) -> Optional[Tuple[str, str]]:
        """Parse checksum line in various formats."""
        stripped = line.strip()
        if not stripped:
            return None
        
        # Format: <hash>  <filename> OR <hash> *<filename>
        match = re.match(r"^([A-Fa-f0-9]{32,128})\s+\*?(.+)$", stripped)
        if match:
            digest = match.group(1).lower()
            referenced_name = Path(match.group(2).strip()).name
            if referenced_name == filename:
                algorithm = "md5" if len(digest) == 32 else "sha256" if len(digest) == 64 else None
                if algorithm:
                    return algorithm, digest
        
        # Format: MD5 (filename) = <hash>
        match = re.match(
            r"^(MD5|SHA256)\s+\(([^)]+)\)\s*=\s*([A-Fa-f0-9]{32,128})$",
            stripped,
            re.IGNORECASE
        )
        if match:
            referenced_name = Path(match.group(2).strip()).name
            if referenced_name == filename:
                algo = match.group(1).lower()
                digest = match.group(3).lower()
                algorithm = "md5" if algo == "md5" else "sha256"
                return algorithm, digest
        
        # Format: only hash
        match = re.match(r"^([A-Fa-f0-9]{32}|[A-Fa-f0-9]{64})$", stripped)
        if match:
            digest = match.group(1).lower()
            algorithm = "md5" if len(digest) == 32 else "sha256"
            return algorithm, digest
        
        return None
    
    def _extract_archive(self, archive_path: Path, out_dir: Path) -> None:
        """Extract tar.gz, tgz, or zip archive."""
        archive_name = archive_path.name.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_count = 0
        
        if archive_name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, mode="r:gz") as tf:
                members = tf.getmembers()
                extracted_count = len(members)
                tf.extractall(out_dir)
        elif archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, mode="r") as zf:
                members = zf.infolist()
                extracted_count = len(members)
                zf.extractall(out_dir)
        else:
            raise RuntimeError(
                f"Unsupported archive format: {archive_path}. "
                "Expected .tar.gz/.tgz or .zip"
            )
        
        if extracted_count <= 0:
            raise RuntimeError(f"Archive extraction produced no entries: {archive_path}")
        
        self.logger.info("Extracted %d items from %s", extracted_count, archive_path.name)
