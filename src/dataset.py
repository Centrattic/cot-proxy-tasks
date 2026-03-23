"""
Versioned dataset storage and management.

Creates and loads datasets with the structure:
    datasets/{name}/v{N}/train/*.json
    datasets/{name}/v{N}/val/*.json
    datasets/{name}/v{N}/test/*.json
    datasets/{name}/v{N}/metadata.json
    datasets/{name}/latest -> v{N}
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .data_slice import DataSlice

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_dataset(
    name: str,
    train_samples: list[dict],
    test_samples: list[dict],
    val_samples: list[dict] | None = None,
    version: str | None = None,
    base_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a versioned dataset folder with train/val/test splits.

    Each sample dict is written as an individual JSON file.
    Returns the path to the created version folder.
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT / "datasets"

    dataset_dir = base_dir / name
    if version is None:
        version = _next_version(name, base_dir)

    version_dir = dataset_dir / version
    if version_dir.exists():
        raise FileExistsError(f"Dataset version already exists: {version_dir}")

    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        if samples is None:
            continue
        split_dir = version_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            with open(split_dir / f"{i:06d}.json", "w") as f:
                json.dump(sample, f, indent=2)

    meta: dict[str, Any] = {
        "name": name,
        "version": version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if metadata:
        meta.update(metadata)

    splits_info: dict[str, Any] = {}
    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        if samples is None:
            continue
        info: dict[str, Any] = {"count": len(samples)}
        labels = [s.get("label") for s in samples if "label" in s]
        if labels:
            from collections import Counter
            info["labels"] = dict(Counter(labels))
        splits_info[split_name] = info
    meta["splits"] = splits_info

    with open(version_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    latest = dataset_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(version)

    return version_dir


def create_dataset_from_data_slice(
    name: str,
    data_slice: DataSlice,
    version: str | None = None,
    base_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a versioned dataset from DataSlice train_df/val_df/test_df."""
    def _df_to_records(df: pd.DataFrame | None) -> list[dict] | None:
        if df is None or df.empty:
            return None
        return df.to_dict(orient="records")

    train = _df_to_records(data_slice.train_df)
    val = _df_to_records(data_slice.val_df)
    test = _df_to_records(data_slice.test_df)

    if train is None and test is None:
        raise ValueError("DataSlice has no train_df or test_df to save")

    return create_dataset(
        name=name,
        train_samples=train or [],
        test_samples=test or [],
        val_samples=val,
        version=version,
        base_dir=base_dir,
        metadata=metadata,
    )


def load_dataset(
    name: str,
    version: str = "latest",
    base_dir: Path | None = None,
) -> DataSlice:
    """Load a dataset into a DataSlice with train_df/val_df/test_df populated."""
    if base_dir is None:
        base_dir = PROJECT_ROOT / "datasets"

    version_dir = base_dir / name / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {version_dir}")

    return DataSlice.from_dataset(version_dir)


def list_versions(name: str, base_dir: Path | None = None) -> list[str]:
    """List available versions for a dataset, sorted naturally."""
    if base_dir is None:
        base_dir = PROJECT_ROOT / "datasets"

    dataset_dir = base_dir / name
    if not dataset_dir.is_dir():
        return []

    versions = []
    for child in dataset_dir.iterdir():
        if child.is_dir() and not child.is_symlink() and re.match(r"v\d+$", child.name):
            versions.append(child.name)

    return sorted(versions, key=lambda v: int(v[1:]))


def _next_version(name: str, base_dir: Path) -> str:
    """Compute the next version string (v1, v2, ...)."""
    existing = list_versions(name, base_dir)
    if not existing:
        return "v1"
    last_num = max(int(v[1:]) for v in existing)
    return f"v{last_num + 1}"
