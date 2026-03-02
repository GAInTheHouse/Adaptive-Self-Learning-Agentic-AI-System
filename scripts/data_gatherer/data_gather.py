#!/usr/bin/env python3
"""
Unified modular data gathering system.

Single entry point for downloading ALL datasets from the registry.
Supports 12+ unique data sources across Hugging Face, OpenSLR, and Git.

Usage:
    # Download all datasets
    python data_gather.py --sources all
    
    # Download specific source types
    python data_gather.py --sources huggingface openslr
    
    # Download specific datasets by name
    python data_gather.py --datasets common_voice_17_0 tedlium3 primock57
    
    # Download with augmentation
    python data_gather.py --sources all --augment
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_utils import configure_logging, require_package
from source_plugins.huggingface_plugin import HuggingFacePlugin
from source_plugins.openslr_plugin import OpenSLRPlugin
from source_plugins.git_plugin import GitPlugin


LOGGER = configure_logging("data_gather")

# Datasets excluded from "all" - only downloaded when explicitly requested
# (VoxPopuli is 122GB and requires special torchcodec/FFmpeg setup)
DATASETS_EXCLUDED_FROM_ALL = ["voxpopuli"]

# Noise/impulse-response corpora: skip as augmentation sources; they are the noise itself
AUGMENTATION_SKIP = {"musan", "rirs_noises"}


def load_registry(registry_path: Path) -> dict:
    """
    Load dataset registry YAML configuration.
    
    Args:
        registry_path: Path to dataset_registry.yaml
        
    Returns:
        Parsed registry dictionary
    """
    import yaml
    
    if not registry_path.exists():
        raise RuntimeError(f"Registry file not found: {registry_path}")
    
    with registry_path.open("r") as f:
        registry = yaml.safe_load(f)
    
    LOGGER.info(
        "Loaded registry v%s with %d source types",
        registry.get("version", "unknown"),
        len([k for k in registry.keys() if k not in ["version", "last_updated", "description"]])
    )
    
    return registry


def get_plugin(source_type: str):
    """
    Get appropriate plugin instance for source type.
    
    Args:
        source_type: One of 'huggingface', 'openslr', 'git'
        
    Returns:
        Plugin instance
    """
    plugins = {
        'huggingface': HuggingFacePlugin(),
        'openslr': OpenSLRPlugin(),
        'git': GitPlugin(),
    }
    
    plugin = plugins.get(source_type)
    if plugin is None:
        raise ValueError(f"Unknown source type: {source_type}")
    
    return plugin


def download_datasets(
    source_types: List[str],
    dataset_names: Optional[List[str]],
    registry: dict,
    output_base: Path,
    manifest_dir: Path,
    force: bool,
) -> Tuple[Dict[str, List[Path]], Dict[str, Path]]:
    """
    Download datasets from registry and generate manifests.

    Args:
        source_types: List of source types to download from
        dataset_names: Optional list of specific dataset names to download
        registry: Loaded registry dictionary
        output_base: Base directory for downloaded data
        manifest_dir: Directory for manifest CSV files
        force: If True, re-download existing datasets

    Returns:
        Tuple of:
          - manifests: dict mapping source_type -> list of manifest paths
          - data_dirs: dict mapping dataset_name -> downloaded data directory
    """
    results: Dict[str, List[Path]] = {}
    data_dirs: Dict[str, Path] = {}

    for source_type in source_types:
        if source_type not in registry:
            LOGGER.warning("Source type '%s' not in registry", source_type)
            continue

        plugin = get_plugin(source_type)
        datasets = registry[source_type]

        manifests_for_source: List[Path] = []

        for dataset_name, config in datasets.items():
            # Filter if specific datasets requested
            if dataset_names and dataset_name not in dataset_names:
                continue
            # Skip datasets excluded from "all" (only download when explicitly requested)
            if dataset_names is None and dataset_name in DATASETS_EXCLUDED_FROM_ALL:
                LOGGER.info(
                    "Skipping %s (excluded from --sources all; use --datasets %s to download)",
                    dataset_name, dataset_name,
                )
                continue

            LOGGER.info("=" * 60)
            LOGGER.info("Processing: %s/%s", source_type, dataset_name)
            LOGGER.info("Description: %s", config.get("description", "N/A"))
            LOGGER.info("=" * 60)

            # Download
            output_dir = output_base / source_type / dataset_name
            data_dir = plugin.download(config, output_dir, force)

            if data_dir is None:
                LOGGER.error("Download failed for %s/%s", source_type, dataset_name)
                continue

            data_dirs[dataset_name] = data_dir

            # Generate manifests
            try:
                manifest_paths = plugin.generate_manifest(
                    data_dir, manifest_dir, dataset_name, force
                )
                manifests_for_source.extend(manifest_paths)
                LOGGER.info("Generated %d manifests for %s", len(manifest_paths), dataset_name)
            except Exception as exc:
                LOGGER.error("Manifest generation failed for %s: %s", dataset_name, exc)

        results[source_type] = manifests_for_source

    return results, data_dirs


def run_augmentation(
    data_dirs: Dict[str, Path],
    output_base: Path,
    force: bool,
) -> None:
    """
    Run noise augmentation over all downloaded speech datasets.

    Imports internal helpers from augment_audio.py to avoid the argparse
    dependency in that script's main() entry point.

    Requires MUSAN to have been downloaded (openslr/musan entry in registry).
    RIRS_NOISES is optional; augmentation proceeds without reverb if missing.

    Args:
        data_dirs: Mapping of dataset_name -> data directory from download phase.
        output_base: Base output directory (augmented files go under
                     output_base/derived/augmented/<dataset_name>/).
        force: If True, overwrite existing augmented files.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from augment_audio import (  # type: ignore[import]
            _augment_one,
            _list_audio_files,
            _require_package,
            DEFAULT_SNRS,
        )
    except ImportError as exc:
        LOGGER.error("Could not import augment_audio helpers: %s", exc)
        return

    _require_package("numpy")
    _require_package("librosa")
    _require_package("soundfile")

    musan_dir = output_base / "openslr" / "musan" / "musan" / "noise"
    rirs_dir = output_base / "openslr" / "rirs_noises" / "RIRS_NOISES"

    if not musan_dir.exists():
        LOGGER.error(
            "MUSAN noise directory not found at %s; cannot run augmentation. "
            "Download MUSAN first (it is included in the openslr registry entries).",
            musan_dir,
        )
        return

    noise_files = _list_audio_files(musan_dir)
    if not noise_files:
        LOGGER.error("No audio files found in MUSAN dir: %s", musan_dir)
        return

    rir_files = _list_audio_files(rirs_dir) if rirs_dir.exists() else []
    if not rir_files:
        LOGGER.warning("RIRS directory missing or empty; reverb will be skipped: %s", rirs_dir)

    rng = random.Random(1337)

    LOGGER.info("MUSAN noise files: %d", len(noise_files))
    LOGGER.info("RIRS files: %d", len(rir_files))

    for dataset_name, data_dir in sorted(data_dirs.items()):
        if dataset_name in AUGMENTATION_SKIP:
            LOGGER.info("Skipping augmentation for noise corpus: %s", dataset_name)
            continue

        source_files = _list_audio_files(data_dir)
        if not source_files:
            LOGGER.info("No audio files to augment in %s; skipping", dataset_name)
            continue

        out_dir = output_base / "derived" / "augmented" / dataset_name
        LOGGER.info(
            "Augmenting %s: %d files -> %s", dataset_name, len(source_files), out_dir
        )

        try:
            from tqdm import tqdm  # type: ignore[import]
            iterable = tqdm(source_files, desc=f"augment {dataset_name}")
        except ImportError:
            iterable = source_files  # type: ignore[assignment]

        for src in iterable:
            try:
                rel = src.relative_to(data_dir)
            except ValueError:
                rel = Path(src.name)
            out_path = out_dir / rel.with_name(f"{src.stem}_aug.wav")
            try:
                _augment_one(
                    src_audio=src,
                    noise_files=noise_files,
                    rir_files=rir_files,
                    out_path=out_path,
                    rng=rng,
                    snr_choices=DEFAULT_SNRS,
                    reverb_probability=0.5,
                    force=force,
                )
            except Exception as exc:
                LOGGER.warning("Augmentation failed for %s: %s", src.name, exc)

    LOGGER.info("Augmentation phase complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified data gathering system - download from all sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets from registry
  %(prog)s --sources all
  
  # Download only Hugging Face datasets
  %(prog)s --sources huggingface
  
  # Download specific datasets by name
  %(prog)s --datasets common_voice_17_0 tedlium3 primock57
  
  # Force re-download with noise augmentation (requires MUSAN in registry)
  %(prog)s --sources all --force --augment
        """
    )
    
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).parent / "dataset_registry.yaml",
        help="Path to dataset registry YAML",
    )
    
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["all", "huggingface", "openslr", "git"],
        default=["all"],
        help="Source types to download from (default: all)",
    )
    
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific dataset names from registry (if omitted, download all from selected sources)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base output directory (default: data/)",
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory for unified manifest CSVs (default: data/manifests/)",
    )
    
    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Run noise augmentation after download using MUSAN and optionally RIRS. "
            "MUSAN must be present in the registry and already downloaded. "
            "Augmented files are written to <output-dir>/derived/augmented/<dataset>/"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing data",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main orchestrator function."""
    args = parse_args()
    
    # Check for pyyaml
    require_package("yaml", "pyyaml")
    
    # Load registry
    try:
        registry = load_registry(args.registry)
    except Exception as exc:
        LOGGER.error("Failed to load registry: %s", exc)
        return 1
    
    # Determine source types to process
    if "all" in args.sources:
        sources = [k for k in registry.keys() if k not in ["version", "last_updated", "description"]]
    else:
        sources = args.sources
    
    LOGGER.info("Target sources: %s", ", ".join(sources))
    if args.datasets:
        LOGGER.info("Target datasets: %s", ", ".join(args.datasets))
    
    # Download datasets and generate manifests
    try:
        results, data_dirs = download_datasets(
            source_types=sources,
            dataset_names=args.datasets,
            registry=registry,
            output_base=args.output_dir,
            manifest_dir=args.manifest_dir,
            force=args.force,
        )
    except Exception as exc:
        LOGGER.error("Data gathering failed: %s", exc)
        return 1

    # Summary
    total_manifests = sum(len(manifests) for manifests in results.values())
    LOGGER.info("=" * 60)
    LOGGER.info("DATA GATHERING COMPLETE")
    LOGGER.info("=" * 60)
    LOGGER.info("Total manifests generated: %d", total_manifests)
    for source_type, manifests in results.items():
        LOGGER.info("  %s: %d manifests", source_type, len(manifests))

    # Optional: noise augmentation using MUSAN / RIRS
    if args.augment:
        LOGGER.info("=" * 60)
        LOGGER.info("AUGMENTATION PHASE")
        LOGGER.info("=" * 60)
        run_augmentation(data_dirs, args.output_dir, args.force)
    
    LOGGER.info("All operations completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        LOGGER.error("Unexpected error: %s", exc)
        sys.exit(1)
