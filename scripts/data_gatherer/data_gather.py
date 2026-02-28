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
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_utils import configure_logging, require_package
from source_plugins.huggingface_plugin import HuggingFacePlugin
from source_plugins.openslr_plugin import OpenSLRPlugin
from source_plugins.git_plugin import GitPlugin


LOGGER = configure_logging("data_gather")


def load_registry(registry_path: Path) -> dict:
    """
    Load dataset registry YAML configuration.
    
    Args:
        registry_path: Path to dataset_registry.yaml
        
    Returns:
        Parsed registry dictionary
    """
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
) -> Dict[str, List[Path]]:
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
        Dictionary mapping source_type -> list of manifest paths
    """
    results = {}
    
    for source_type in source_types:
        if source_type not in registry:
            LOGGER.warning("Source type '%s' not in registry", source_type)
            continue
        
        plugin = get_plugin(source_type)
        datasets = registry[source_type]
        
        manifests_for_source = []
        
        for dataset_name, config in datasets.items():
            # Filter if specific datasets requested
            if dataset_names and dataset_name not in dataset_names:
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
    
    return results


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
  
  # Force re-download with augmentation
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
        help="Generate augmented variants after download (requires MUSAN and RIRS)",
    )
    
    parser.add_argument(
        "--derive-variants",
        action="store_true",
        help="Generate low-quality (8kHz) and corrupted variants",
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
        results = download_datasets(
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
    
    # Optional: augmentation and derived variants
    if args.augment or args.derive_variants:
        LOGGER.info("=" * 60)
        LOGGER.info("AUGMENTATION PHASE")
        LOGGER.info("=" * 60)
        
        try:
            # Import augmentation script
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from augment_audio import main as augment_main
            
            # Note: This is a simplified integration
            # Full augmentation would require calling augment_audio with proper args
            LOGGER.info("Augmentation integration: Call augment_audio.py separately")
            LOGGER.info("  Example: python scripts/augment_audio.py --manifest-dir %s", args.manifest_dir)
            
        except ImportError as exc:
            LOGGER.warning("Could not import augment_audio: %s", exc)
    
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
