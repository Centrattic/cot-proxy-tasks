"""
Majority/minority answer classification task.

Given a prompt and a single rollout, classify whether the rollout's answer
is the majority or minority answer among all rollouts for that prompt.

Key constraint: methods cannot access other rollouts on the SAME prompt —
only rollouts from OTHER prompts (e.g. for few-shot examples).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ...data_slice import DataSlice
from ..base import BaseTask
from .data_loader import (
    ALL_PROMPT_IDS,
    ROLLOUTS_ROOT,
    build_rollout_df,
)


class MinMajAnswerTask(BaseTask):
    """
    Task: classify a single rollout as majority or minority answer.

    Loads rollouts from an external prompts directory, computes per-prompt
    majority/minority labels, and serves individual rollouts for classification.
    """

    def __init__(
        self,
        prompt_ids: List[str],
        model: str = "qwen3-32b",
        rollouts_root: Path = ROLLOUTS_ROOT,
        data_dir: Optional[Path] = None,
    ):
        super().__init__(name="min_maj_answer", data_dir=data_dir)
        self.prompt_ids = prompt_ids
        self.model = model
        self.rollouts_root = Path(rollouts_root)

    @classmethod
    def loo_folds(
        cls, prompt_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Return leave-one-out fold definitions.

        Each fold holds out one prompt as test and uses the rest as train.
        """
        if prompt_ids is None:
            prompt_ids = ALL_PROMPT_IDS
        folds = []
        for i, test_pid in enumerate(prompt_ids):
            folds.append({
                "fold_idx": i,
                "test_prompt_id": test_pid,
                "train_prompt_ids": [p for p in prompt_ids if p != test_pid],
            })
        return folds

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def run_data(self, **kwargs) -> None:
        """Load rollouts, compute labels, and save to data_dir."""
        df = build_rollout_df(self.prompt_ids, self.model, self.rollouts_root)
        if df.empty:
            print("Warning: no rollouts found for any prompt")
            return
        df.to_csv(self.data_dir / "rollouts.csv", index=False)
        print(f"Saved {len(df)} rollouts to {self.data_dir / 'rollouts.csv'}")

        # Also save per-prompt summary
        summary_rows = []
        for prompt_id in self.prompt_ids:
            subset = df[df["prompt_id"] == prompt_id]
            if subset.empty:
                continue
            summary_rows.append({
                "prompt_id": prompt_id,
                "n_rollouts": len(subset),
                "n_majority": int(subset["is_majority"].sum()),
                "n_minority": int((~subset["is_majority"]).sum()),
                "majority_frac": float(subset["majority_frac"].iloc[0]),
                "majority_answer": subset["majority_answer"].iloc[0],
                "answer_counts": subset["answer_counts"].iloc[0],
            })
        pd.DataFrame(summary_rows).to_csv(
            self.data_dir / "prompt_summary.csv", index=False
        )

    def get_data(self, load: bool = False) -> Union[bool, Optional[Dict[str, pd.DataFrame]]]:
        csv_path = self.data_dir / "rollouts.csv"
        if not load:
            return csv_path.exists()
        if not csv_path.exists():
            return None
        return {
            "rollouts": pd.read_csv(csv_path),
            "summary": pd.read_csv(self.data_dir / "prompt_summary.csv"),
        }

    def get_train_test_split(
        self,
        train_prompt_ids: List[str],
        test_prompt_ids: List[str],
    ) -> DataSlice:
        """Return a DataSlice with train/test splits by prompt ID."""
        return DataSlice(
            train_df=build_rollout_df(train_prompt_ids, self.model, self.rollouts_root),
            test_df=build_rollout_df(test_prompt_ids, self.model, self.rollouts_root),
        )
