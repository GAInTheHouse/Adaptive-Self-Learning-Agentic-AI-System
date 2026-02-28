#!/usr/bin/env python3
"""
AfriMed-QA dataset downloader from Hugging Face.

Downloads the AfriMed-QA medical question-answering dataset, which contains:
- 15,000 medical questions (multiple-choice and open-ended)
- Questions from 60+ medical schools across 16 African countries
- Coverage of 32 medical specialties

Source: https://huggingface.co/datasets/afrimedqa/afrimedqa_v2
Paper: https://aclanthology.org/2025.acl-long.96/
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional


LOGGER = logging.getLogger("afrimedqa_download")

DATASET_NAME = "afrimedqa/afrimedqa_v2"

CSV_FIELDS = [
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
    "options",  # For multiple-choice questions
    "rationale",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def check_dependencies() -> None:
    """Check if required packages are installed."""
    try:
        import datasets
        import pandas
    except ImportError as exc:
        missing = str(exc).split("'")[1]
        raise RuntimeError(
            f"Missing required package: {missing}\n"
            f"Install with: pip install datasets pandas"
        ) from exc


def safe_id(value: str) -> str:
    """Convert string to safe identifier."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", value).strip("_") or "item"


def extract_field(row: Dict, field_names: List[str], default: str = "") -> str:
    """Extract field from row, trying multiple possible field names."""
    for field in field_names:
        if field in row and row[field] is not None:
            value = str(row[field]).strip()
            if value:
                return value
    return default


def download_dataset(
    output_dir: Path,
    manifest_dir: Path,
    max_samples: Optional[int] = None,
    force: bool = False,
) -> None:
    """Download AfriMed-QA dataset from Hugging Face and generate manifests."""
    from datasets import load_dataset
    from tqdm import tqdm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Loading AfriMed-QA dataset from Hugging Face: %s", DATASET_NAME)
    
    try:
        # Load dataset
        dataset = load_dataset(DATASET_NAME)
        
        LOGGER.info("Dataset loaded successfully. Available splits: %s", list(dataset.keys()))
        
    except Exception as exc:
        LOGGER.error("Failed to load dataset from Hugging Face: %s", exc)
        raise RuntimeError(
            f"Could not load {DATASET_NAME}. "
            "Ensure you have internet connection and the dataset is publicly accessible."
        ) from exc
    
    # Process each split
    for split_name, split_data in dataset.items():
        LOGGER.info("Processing split: %s (%d samples)", split_name, len(split_data))
        
        # Apply max_samples limit if specified
        if max_samples is not None and len(split_data) > max_samples:
            split_data = split_data.select(range(max_samples))
            LOGGER.info("Limited to %d samples for testing", max_samples)
        
        # Convert to pandas for easier CSV export
        df = split_data.to_pandas()
        
        # Save raw data
        csv_path = output_dir / f"afrimedqa_{split_name}.csv"
        if csv_path.exists() and not force:
            LOGGER.info("Data already exists: %s", csv_path)
        else:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            LOGGER.info("Saved raw data: %s", csv_path)
        
        # Generate standardized manifest
        manifest_rows = []
        
        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Creating manifest for {split_name}",
        ):
            # Extract question ID or generate one
            question_id = extract_field(
                row,
                ["id", "question_id", "ID", "Question_ID"],
                default=f"{split_name}_{idx}"
            )
            
            # Determine question type
            question_type = extract_field(
                row,
                ["type", "question_type", "Type", "Question_Type"],
                default="unknown"
            )
            
            # Extract question text
            question = extract_field(
                row,
                ["question", "Question", "query", "Query"],
                default=""
            )
            
            # Extract answer
            answer = extract_field(
                row,
                ["answer", "Answer", "correct_answer", "Correct_Answer"],
                default=""
            )
            
            # Extract specialty
            specialty = extract_field(
                row,
                ["specialty", "Specialty", "subject", "Subject", "category", "Category"],
                default=""
            )
            
            # Extract country
            country = extract_field(
                row,
                ["country", "Country", "region", "Region"],
                default=""
            )
            
            # Extract difficulty
            difficulty = extract_field(
                row,
                ["difficulty", "Difficulty", "level", "Level"],
                default=""
            )
            
            # Extract options for MCQ
            options = ""
            for opt_field in ["options", "Options", "choices", "Choices"]:
                if opt_field in row and row[opt_field] is not None:
                    opts = row[opt_field]
                    if isinstance(opts, (list, tuple)):
                        options = " | ".join(str(o) for o in opts)
                    else:
                        options = str(opts)
                    break
            
            # Extract rationale/explanation
            rationale = extract_field(
                row,
                ["rationale", "Rationale", "explanation", "Explanation"],
                default=""
            )
            
            manifest_rows.append({
                "dataset": "afrimedqa",
                "split": split_name,
                "utt_id": safe_id(question_id),
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "specialty": specialty,
                "country": country,
                "difficulty": difficulty,
                "options": options,
                "rationale": rationale,
            })
        
        # Write manifest
        manifest_path = manifest_dir / f"afrimedqa__{split_name}.csv"
        
        if manifest_path.exists() and not force:
            LOGGER.info("Manifest already exists: %s", manifest_path)
        else:
            with manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(manifest_rows)
            
            LOGGER.info("Wrote manifest with %d entries: %s", len(manifest_rows), manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AfriMed-QA medical QA dataset from Hugging Face."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "afrimedqa",
        help="Output directory for dataset files.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data") / "manifests",
        help="Directory for manifest CSV files.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples per split (for testing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing files.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    
    # Check dependencies
    try:
        check_dependencies()
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        return 1
    
    # Download and process dataset
    try:
        download_dataset(
            output_dir=args.out_dir,
            manifest_dir=args.manifest_dir,
            max_samples=args.max_samples,
            force=args.force,
        )
    except Exception as exc:
        LOGGER.error("Failed to download AfriMed-QA dataset: %s", exc)
        return 1
    
    LOGGER.info("AfriMed-QA download completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.info("Download interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("Unexpected error: %s", exc)
        raise SystemExit(1)
