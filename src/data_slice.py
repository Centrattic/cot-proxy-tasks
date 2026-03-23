"""
DataSlice — data selection layer for filtering data points across tasks.

Provides optional filtering by IDs, sentence indices, timestamps, and direct
path overrides. None means "no filter" (include all).

Holds optional train/val/test DataFrames (keyed on `filepath` and `label`
columns) for structured split access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Set

import pandas as pd


@dataclass
class DataSlice:
    ids: Optional[Set[str]] = None
    sentence_indices: Optional[Set[int]] = None

    # Timestamp filters
    timestamps: Optional[List[str]] = None
    latest_n: Optional[int] = None

    # Direct path override
    run_paths: Optional[List[Path]] = None

    # Split DataFrames (expected columns: filepath, label, plus task-specific)
    train_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    val_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    test_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    # Dataset provenance
    dataset_path: Optional[Path] = None
    dataset_version: Optional[str] = None

    EXPECTED_COLS = ("filepath", "label")

    @property
    def df(self) -> pd.DataFrame:
        """Concatenation of all non-None split DataFrames."""
        parts = [x for x in (self.train_df, self.val_df, self.test_df) if x is not None]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    @property
    def train(self) -> DataSlice:
        return DataSlice(train_df=self.train_df)

    @property
    def val(self) -> DataSlice:
        return DataSlice(val_df=self.val_df)

    @property
    def test(self) -> DataSlice:
        return DataSlice(test_df=self.test_df)

    @property
    def filepaths(self) -> List[str]:
        return list(self.df["filepath"]) if "filepath" in self.df.columns else []

    @property
    def label_series(self) -> pd.Series:
        return self.df["label"] if "label" in self.df.columns else pd.Series(dtype=object)

    def labeled(self, label: Any) -> pd.DataFrame:
        return self.df[self.df["label"] == label]

    def matches_id(self, id: str) -> bool:
        return self.ids is None or id in self.ids

    def matches_sentence(self, idx: int) -> bool:
        return self.sentence_indices is None or idx in self.sentence_indices

    def filter_paths(
        self,
        paths: List[Path],
        timestamp_pattern: str = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}|\d{8}_\d{6}",
    ) -> List[Path]:
        """Filter a list of file paths by timestamp directory names."""
        if self.run_paths is not None:
            run_path_strs = {str(p) for p in self.run_paths}
            paths = [
                p for p in paths if any(str(p).startswith(rp) for rp in run_path_strs)
            ]

        if self.timestamps is not None:
            ts_set = set(self.timestamps)
            paths = [
                p for p in paths
                if self._path_has_timestamp(p, ts_set, timestamp_pattern)
            ]

        if self.latest_n is not None:
            all_timestamps = set()
            compiled = re.compile(timestamp_pattern)
            for p in paths:
                for part in p.parts:
                    if compiled.fullmatch(part):
                        all_timestamps.add(part)
            if all_timestamps:
                latest = sorted(all_timestamps, reverse=True)[: self.latest_n]
                latest_set = set(latest)
                paths = [
                    p for p in paths
                    if self._path_has_timestamp(p, latest_set, timestamp_pattern)
                ]

        return paths

    @staticmethod
    def _path_has_timestamp(path: Path, ts_set: Set[str], pattern: str) -> bool:
        for part in path.parts:
            if part in ts_set:
                return True
        return False

    @classmethod
    def all(cls) -> DataSlice:
        return cls()

    @classmethod
    def from_ids(cls, ids) -> DataSlice:
        return cls(ids=set(ids))

    @classmethod
    def latest(cls, n: int = 1) -> DataSlice:
        return cls(latest_n=n)

    @classmethod
    def from_paths(cls, paths: List[Path]) -> DataSlice:
        return cls(run_paths=paths)

    @classmethod
    def from_dataset(cls, path: Path) -> DataSlice:
        """Load train/val/test splits from a versioned dataset folder."""
        path = Path(path)
        if path.name == "latest" or path.is_symlink():
            path = path.resolve()

        version = path.name

        def _load_split(split_dir: Path) -> Optional[pd.DataFrame]:
            if not split_dir.is_dir():
                return None
            records = []
            for f in sorted(split_dir.glob("*.json")):
                with open(f) as fh:
                    records.append(json.load(fh))
            if not records:
                return None
            return pd.DataFrame(records)

        train_df = _load_split(path / "train")
        val_df = _load_split(path / "val")
        test_df = _load_split(path / "test")

        return cls(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            dataset_path=path,
            dataset_version=version,
        )

    def __len__(self) -> int:
        n = len(self.ids) if self.ids is not None else 0
        df_len = len(self.df)
        return max(n, df_len)

    def __contains__(self, id: object) -> bool:
        return self.matches_id(id)

    def dataset_info(self) -> dict:
        info = {}
        if self.dataset_path is not None:
            info["dataset_path"] = str(self.dataset_path)
        if self.dataset_version is not None:
            info["dataset_version"] = self.dataset_version
        return info

    def summary(self) -> str:
        parts: List[str] = []
        n = len(self.ids) if self.ids is not None else "all"
        parts.append(f"DataSlice({n} ids)")
        for name, split_df in [("train", self.train_df), ("val", self.val_df), ("test", self.test_df)]:
            if split_df is not None:
                parts.append(f"  {name}: {len(split_df)} rows")
        if "label" in self.df.columns:
            counts = self.df["label"].value_counts().to_dict()
            label_strs = [f"{lbl}: {cnt}" for lbl, cnt in sorted(counts.items(), key=lambda x: str(x[0]))]
            parts.append(f"  labels: {', '.join(label_strs)}")
        return "\n".join(parts)
