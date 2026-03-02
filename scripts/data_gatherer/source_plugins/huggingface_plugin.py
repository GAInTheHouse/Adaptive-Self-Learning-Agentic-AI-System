#!/usr/bin/env python3
"""
HuggingFace dataset plugin.

Consolidates download and manifest generation logic for all Hugging Face datasets:
- Common Voice (16.1, 17.0)
- LibriSpeech ASR
- Speech Commands
- VoxPopuli
- AfriMed-QA (text-only)
"""

from __future__ import annotations

import datetime
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from source_plugins import DataSourcePlugin
from dataset_utils import (
    CSV_FIELDS,
    get_audio_metadata_soundfile,
    pick_field,
    require_package,
    safe_name,
    write_manifest,
)


LOGGER = logging.getLogger("HuggingFacePlugin")

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / 'logs'
_SESSION_TS = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')


def _debug_log_path(operation: str) -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR / f'{_SESSION_TS}-{operation}.log'

# Different CSV fields for text-only datasets
AFRIMEDQA_FIELDS = [
    "dataset",
    "split",
    "utt_id",
    "question_id",
    "question_type",
    "question",
    "answer",
    "specialty",
    "country",
    "difficulty",
    "options",
    "rationale",
]


class HuggingFacePlugin(DataSourcePlugin):
    """Plugin for downloading Hugging Face datasets."""
    
    def __init__(self):
        self.logger = LOGGER
    
    def get_source_type(self) -> str:
        return "huggingface"
    
    def download(self, config: Dict, output_dir: Path, force: bool) -> Optional[Path]:
        """
        Download dataset from Hugging Face Hub.
        
        Handles both audio and text-only datasets. For audio datasets,
        materializes decoded audio to WAV files for consistent processing.
        """
        require_package("datasets")
        require_package("numpy")
        require_package("soundfile")
        
        from datasets import Audio, load_dataset
        
        dataset_name = config["dataset"]
        dataset_config = config.get("config")
        splits = config.get("splits", ["train"])
        text_only = config.get("text_only", False)
        quality_filter = config.get("quality_filter", False)
        
        self.logger.info(
            "Downloading HF dataset: %s (config=%s, splits=%s)",
            dataset_name, dataset_config, splits
        )
        
        # Check if already downloaded
        if output_dir.exists() and not force:
            self.logger.info("Dataset already exists at: %s", output_dir)
            return output_dir
        
        # Load dataset
        try:
            # region agent log
            import json
            from pathlib import Path as LogPath
            log_data = {"hypothesisId": "A", "runId": "debug1", "location": "huggingface_plugin.py:92", "message": "Attempting HF load", "data": {"dataset_name": dataset_name, "config": dataset_config, "splits": splits}, "timestamp": int(__import__('time').time() * 1000)}
            try:
                with open(_debug_log_path('download'), 'a') as f:
                    f.write(json.dumps(log_data) + '\n')
            except: pass
            # endregion
            
            dataset = load_dataset(
                dataset_name,
                name=dataset_config,
                cache_dir=str(output_dir.parent / ".hf_cache"),
            )
            
            # region agent log
            log_data2 = {"hypothesisId": "A,B", "runId": "debug1", "location": "huggingface_plugin.py:110", "message": "HF load success", "data": {"dataset_name": dataset_name, "available_splits": list(dataset.keys()) if hasattr(dataset, 'keys') else []}, "timestamp": int(__import__('time').time() * 1000)}
            try:
                with open(_debug_log_path('download'), 'a') as f:
                    f.write(json.dumps(log_data2) + '\n')
            except: pass
            # endregion
            
        except Exception as exc:
            # region agent log
            log_data3 = {"hypothesisId": "A,B,C", "runId": "debug1", "location": "huggingface_plugin.py:121", "message": "HF load failed", "data": {"dataset_name": dataset_name, "error_type": type(exc).__name__, "error_msg": str(exc)}, "timestamp": int(__import__('time').time() * 1000)}
            try:
                with open(_debug_log_path('download'), 'a') as f:
                    f.write(json.dumps(log_data3) + '\n')
            except: pass
            # endregion
            
            self.logger.error("Failed to load dataset %s: %s", dataset_name, exc)
            return None
        
        # Process each split
        for split_name in splits:
            # region agent log
            import json
            log_data = {"hypothesisId": "B", "runId": "debug1", "location": "huggingface_plugin.py:137", "message": "Checking split", "data": {"split_name": split_name, "available_splits": list(dataset.keys()), "split_exists": split_name in dataset}, "timestamp": int(__import__('time').time() * 1000)}
            try:
                with open(_debug_log_path('download'), 'a') as f:
                    f.write(json.dumps(log_data) + '\n')
            except: pass
            # endregion
            
            if split_name not in dataset:
                self.logger.warning("Split '%s' not found in dataset", split_name)
                continue
            
            split_ds = dataset[split_name]
            
            # Apply quality filtering (Common Voice)
            if quality_filter and "up_votes" in split_ds.column_names:
                original_count = len(split_ds)
                split_ds = split_ds.filter(
                    lambda ex: ex.get("up_votes", 0) > ex.get("down_votes", 0)
                )
                self.logger.info(
                    "Quality filter: %d -> %d samples", original_count, len(split_ds)
                )
            
            # Handle audio datasets
            if not text_only:
                audio_col = self._detect_audio_column(split_ds)
                
                # region agent log
                import json
                log_data = {"hypothesisId": "E", "runId": "debug1", "location": "huggingface_plugin.py:157", "message": "Audio column detection", "data": {"dataset_name": dataset_name, "split_name": split_name, "audio_col": audio_col, "columns": list(split_ds.column_names)[:10]}, "timestamp": int(__import__('time').time() * 1000)}
                try:
                    with open(_debug_log_path('download'), 'a') as f:
                        f.write(json.dumps(log_data) + '\n')
                except: pass
                # endregion
                
                if audio_col:
                    try:
                        split_ds = split_ds.cast_column(audio_col, Audio(decode=True))
                    except Exception as exc:
                        self.logger.warning(
                            "Failed to cast audio column for split %s: %s",
                            split_name, exc
                        )
            
            # Save split to disk
            split_slug = safe_name(split_name)
            split_output = output_dir / split_slug
            
            # region agent log
            import json
            log_data = {"hypothesisId": "E", "runId": "debug1", "location": "huggingface_plugin.py:181", "message": "Before save", "data": {"split_name": split_name, "split_output": str(split_output), "exists": split_output.exists(), "force": force, "num_examples": len(split_ds)}, "timestamp": int(__import__('time').time() * 1000)}
            try:
                with open(_debug_log_path('download'), 'a') as f:
                    f.write(json.dumps(log_data) + '\n')
            except: pass
            # endregion
            
            if split_output.exists() and not force:
                self.logger.info("Split already saved: %s", split_output)
                continue
            
            if split_output.exists():
                shutil.rmtree(split_output)
            
            split_output.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                split_ds.save_to_disk(str(split_output))
                self.logger.info("Saved split '%s' to: %s", split_name, split_output)
                
                # region agent log
                import os
                saved_files = list(split_output.glob("*")) if split_output.exists() else []
                log_data2 = {"hypothesisId": "E", "runId": "debug1", "location": "huggingface_plugin.py:203", "message": "After save", "data": {"split_name": split_name, "split_output": str(split_output), "saved_files_count": len(saved_files), "has_arrow_files": any(f.suffix == '.arrow' for f in saved_files)}, "timestamp": int(__import__('time').time() * 1000)}
                try:
                    with open(_debug_log_path('download'), 'a') as f:
                        f.write(json.dumps(log_data2) + '\n')
                except: pass
                # endregion
                
            except Exception as save_exc:
                # region agent log
                log_data3 = {"hypothesisId": "E", "runId": "debug1", "location": "huggingface_plugin.py:214", "message": "Save failed", "data": {"split_name": split_name, "error_type": type(save_exc).__name__, "error_msg": str(save_exc)}, "timestamp": int(__import__('time').time() * 1000)}
                try:
                    with open(_debug_log_path('download'), 'a') as f:
                        f.write(json.dumps(log_data3) + '\n')
                except: pass
                # endregion
                raise
        
        return output_dir
    
    def generate_manifest(
        self,
        data_dir: Path,
        manifest_dir: Path,
        dataset_name: str,
        force: bool
    ) -> List[Path]:
        """Generate manifests for all saved splits."""
        require_package("datasets")
        require_package("tqdm")
        
        from datasets import Dataset, load_from_disk
        from tqdm import tqdm
        
        manifest_paths = []
        
        # Find all saved splits
        split_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        
        # region agent log
        import json
        log_data = {"hypothesisId": "E", "runId": "debug1", "location": "huggingface_plugin.py:244", "message": "Manifest gen start", "data": {"data_dir": str(data_dir), "split_dirs_found": [str(d) for d in split_dirs]}, "timestamp": int(__import__('time').time() * 1000)}
        try:
            with open(_debug_log_path('manifest'), 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # endregion
        
        for split_dir in split_dirs:
            split_name = split_dir.name
            
            # region agent log
            import os
            files_in_split = list(split_dir.glob("*")) if split_dir.exists() else []
            log_data2 = {"hypothesisId": "E", "runId": "debug1", "location": "huggingface_plugin.py:257", "message": "Attempting load", "data": {"split_name": split_name, "split_dir": str(split_dir), "exists": split_dir.exists(), "files_count": len(files_in_split), "has_arrow": any(f.suffix == '.arrow' for f in files_in_split)}, "timestamp": int(__import__('time').time() * 1000)}
            try:
                with open(_debug_log_path('manifest'), 'a') as f:
                    f.write(json.dumps(log_data2) + '\n')
            except: pass
            # endregion
            
            try:
                split_ds = load_from_disk(str(split_dir))
                
                # region agent log
                log_data3 = {"hypothesisId": "E", "runId": "debug1", "location": "huggingface_plugin.py:270", "message": "Load success", "data": {"split_name": split_name, "num_examples": len(split_ds)}, "timestamp": int(__import__('time').time() * 1000)}
                try:
                    with open(_debug_log_path('manifest'), 'a') as f:
                        f.write(json.dumps(log_data3) + '\n')
                except: pass
                # endregion
                
            except Exception as exc:
                # region agent log
                log_data4 = {"hypothesisId": "E", "runId": "debug1", "location": "huggingface_plugin.py:280", "message": "Load failed", "data": {"split_name": split_name, "error_type": type(exc).__name__, "error_msg": str(exc)[:200]}, "timestamp": int(__import__('time').time() * 1000)}
                try:
                    with open(_debug_log_path('manifest'), 'a') as f:
                        f.write(json.dumps(log_data4) + '\n')
                except: pass
                # endregion
                
                self.logger.warning("Failed to load split %s: %s", split_name, exc)
                continue
            
            # Determine if text-only (AfriMed-QA)
            is_afrimedqa = "afrimedqa" in dataset_name.lower()
            
            if is_afrimedqa:
                manifest_path = self._generate_afrimedqa_manifest(
                    split_ds, split_name, dataset_name, manifest_dir, force
                )
            else:
                manifest_path = self._generate_audio_manifest(
                    split_ds, split_name, dataset_name, data_dir, manifest_dir, force
                )
            
            if manifest_path:
                manifest_paths.append(manifest_path)
        
        return manifest_paths
    
    def _detect_audio_column(self, dataset) -> Optional[str]:
        """Detect audio column in dataset."""
        from datasets import Audio
        
        for col, feature in dataset.features.items():
            if isinstance(feature, Audio):
                return col
        
        if "audio" in dataset.column_names:
            return "audio"
        
        return None
    
    def _generate_audio_manifest(
        self,
        split_ds,
        split_name: str,
        dataset_name: str,
        data_dir: Path,
        manifest_dir: Path,
        force: bool,
    ) -> Optional[Path]:
        """Generate manifest for audio dataset."""
        from tqdm import tqdm
        import numpy as np
        import soundfile as sf
        
        dataset_slug = safe_name(dataset_name)
        split_slug = safe_name(split_name)
        manifest_path = manifest_dir / f"{dataset_slug}__{split_slug}.csv"
        
        if manifest_path.exists() and not force:
            self.logger.info("Manifest exists: %s", manifest_path)
            return manifest_path
        
        audio_col = self._detect_audio_column(split_ds)
        audio_out_dir = data_dir / "audio" / split_slug
        
        rows: List[Dict[str, str]] = []
        
        for idx, example in enumerate(tqdm(split_ds, desc=f"{dataset_slug}:{split_slug}")):
            utt_src = pick_field(
                example, 
                ["id", "utterance_id", "path", "client_id"]
            ) or f"{split_slug}_{idx}"
            utt_id = safe_name(utt_src)
            
            # Handle audio
            path_str = ""
            duration_str = ""
            sampling_rate_str = ""
            
            if audio_col:
                path_str, duration_str, sampling_rate_str = self._resolve_audio(
                    example, audio_col, audio_out_dir, utt_id, force
                )
            elif "path" in example:
                p = Path(str(example["path"]))
                path_str = str(p)
                if p.exists():
                    duration, sr = get_audio_metadata_soundfile(p)
                    duration_str = f"{duration:.6f}"
                    sampling_rate_str = str(sr)
            
            rows.append({
                "dataset": dataset_slug,
                "split": split_name,
                "utt_id": utt_id,
                "path": path_str,
                "duration_seconds": duration_str,
                "sampling_rate": sampling_rate_str,
                "text": pick_field(
                    example, 
                    ["sentence", "text", "normalized_text", "transcription"]
                ),
                "speaker": pick_field(
                    example, 
                    ["speaker_id", "speaker", "client_id"]
                ),
                "accent": pick_field(example, ["accent", "variant"]),
            })
        
        write_manifest(rows, manifest_path, fieldnames=CSV_FIELDS, force=force)
        return manifest_path
    
    def _generate_afrimedqa_manifest(
        self,
        split_ds,
        split_name: str,
        dataset_name: str,
        manifest_dir: Path,
        force: bool,
    ) -> Optional[Path]:
        """Generate manifest for AfriMed-QA text dataset."""
        from tqdm import tqdm
        
        manifest_path = manifest_dir / f"afrimedqa__{split_name}.csv"
        
        if manifest_path.exists() and not force:
            self.logger.info("Manifest exists: %s", manifest_path)
            return manifest_path
        
        rows: List[Dict[str, str]] = []
        
        for idx, example in enumerate(tqdm(split_ds, desc=f"afrimedqa:{split_name}")):
            question_id = pick_field(
                example,
                ["id", "question_id", "ID", "Question_ID"],
                default=f"{split_name}_{idx}"
            )
            
            # Extract options for multiple-choice
            options = ""
            for opt_field in ["options", "Options", "choices", "Choices"]:
                if opt_field in example and example[opt_field] is not None:
                    opts = example[opt_field]
                    if isinstance(opts, (list, tuple)):
                        options = " | ".join(str(o) for o in opts)
                    else:
                        options = str(opts)
                    break
            
            rows.append({
                "dataset": "afrimedqa",
                "split": split_name,
                "utt_id": safe_name(question_id),
                "question_id": question_id,
                "question_type": pick_field(
                    example,
                    ["type", "question_type", "Type", "Question_Type"],
                    default="unknown"
                ),
                "question": pick_field(
                    example,
                    ["question", "Question", "query", "Query"]
                ),
                "answer": pick_field(
                    example,
                    ["answer", "Answer", "correct_answer", "Correct_Answer"]
                ),
                "specialty": pick_field(
                    example,
                    ["specialty", "Specialty", "subject", "Subject", "category"]
                ),
                "country": pick_field(
                    example,
                    ["country", "Country", "region", "Region"]
                ),
                "difficulty": pick_field(
                    example,
                    ["difficulty", "Difficulty", "level", "Level"]
                ),
                "options": options,
                "rationale": pick_field(
                    example,
                    ["rationale", "Rationale", "explanation", "Explanation"]
                ),
            })
        
        write_manifest(rows, manifest_path, fieldnames=AFRIMEDQA_FIELDS, force=force)
        return manifest_path
    
    def _resolve_audio(
        self,
        example: Dict,
        audio_column: str,
        audio_out_dir: Path,
        utt_id: str,
        force: bool,
    ) -> tuple[str, str, str]:
        """Resolve audio path and metadata, materializing if needed."""
        import numpy as np
        import soundfile as sf
        
        audio_value = example.get(audio_column)
        if audio_value is None:
            return "", "", ""
        
        # Handle string paths
        if isinstance(audio_value, str):
            local_path = Path(audio_value)
            if local_path.exists():
                duration, sr = get_audio_metadata_soundfile(local_path)
                return str(local_path), f"{duration:.6f}", str(sr)
            return audio_value, "", ""
        
        if not isinstance(audio_value, dict):
            return "", "", ""
        
        # Check if local file path already exists
        candidate_path = audio_value.get("path")
        if isinstance(candidate_path, str):
            local_path = Path(candidate_path)
            if local_path.exists():
                duration, sr = get_audio_metadata_soundfile(local_path)
                return str(local_path), f"{duration:.6f}", str(sr)
        
        # Materialize decoded audio to WAV
        out_path = audio_out_dir / f"{utt_id}.wav"
        
        if out_path.exists() and not force:
            duration, sr = get_audio_metadata_soundfile(out_path)
            return str(out_path), f"{duration:.6f}", str(sr)
        
        # Extract decoded audio array
        data = audio_value.get("array")
        sr = audio_value.get("sampling_rate")
        
        if data is None or sr is None:
            self.logger.warning("Audio missing 'array' or 'sampling_rate': %s", utt_id)
            return "", "", ""
        
        audio = np.asarray(data, dtype=np.float32)
        
        # Convert stereo to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # Write WAV file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, int(sr), subtype="PCM_16")
        
        duration = float(len(audio)) / float(sr) if sr else 0.0
        return str(out_path), f"{duration:.6f}", str(int(sr))
