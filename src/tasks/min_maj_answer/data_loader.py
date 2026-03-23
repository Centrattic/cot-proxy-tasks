"""
Data loader for min/maj answer task.

Loads rollouts from an external directory, computes majority/minority labels.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROLLOUTS_ROOT = Path("/home/riya/neel-projs/global-cot-analysis/prompts")

ALL_PROMPT_IDS = [
    "bagel",
    "gpqa_nmr_compound",
    "gpqa_benzene_naming",
    "harder_well",
    "bookworm",
    "gpqa_diels_alder",
    "gpqa_optical_activity",
]


def load_prompts_json(rollouts_root: Path = ROLLOUTS_ROOT) -> Dict[str, str]:
    """Load the prompts.json mapping prompt_id -> prompt text."""
    with open(rollouts_root / "prompts.json") as f:
        return json.load(f)


def load_rollouts_for_prompt(
    prompt_id: str,
    model: str = "qwen3-32b",
    rollouts_root: Path = ROLLOUTS_ROOT,
) -> List[Dict[str, Any]]:
    """Load all rollout JSON files for a given prompt + model."""
    rollouts_dir = rollouts_root / prompt_id / model / "rollouts"
    if not rollouts_dir.exists():
        return []

    rollouts = []
    for f in sorted(rollouts_dir.iterdir()):
        if f.suffix == ".json":
            with open(f) as fh:
                data = json.load(fh)
                data["rollout_file"] = str(f)
                data["rollout_idx"] = int(f.stem)
                rollouts.append(data)
    return rollouts


def compute_labels(rollouts: List[Dict]) -> List[Dict]:
    """Add majority/minority labels based on answer distribution."""
    rollouts = [
        r for r in rollouts
        if r.get("processed_response_content", r.get("response_content", "")).strip()
    ]
    answers = [
        r.get("processed_response_content", r.get("response_content", ""))
        for r in rollouts
    ]
    counts = Counter(answers)
    if not counts:
        return rollouts

    majority_answer = counts.most_common(1)[0][0]
    total = sum(counts.values())
    majority_frac = counts[majority_answer] / total

    for r, ans in zip(rollouts, answers):
        r["answer"] = ans
        r["is_majority"] = ans == majority_answer
        r["label"] = "majority" if ans == majority_answer else "minority"
        r["majority_answer"] = majority_answer
        r["majority_frac"] = majority_frac
        r["answer_counts"] = dict(counts)

    return rollouts


def build_rollout_df(
    prompt_ids: List[str],
    model: str = "qwen3-32b",
    rollouts_root: Path = ROLLOUTS_ROOT,
) -> pd.DataFrame:
    """Build a DataFrame of labeled rollouts for the given prompt IDs."""
    prompts_json = load_prompts_json(rollouts_root)
    rows = []
    for pid in prompt_ids:
        rollouts = load_rollouts_for_prompt(pid, model, rollouts_root)
        if not rollouts:
            continue
        rollouts = compute_labels(rollouts)
        prompt_text = prompts_json.get(pid, "")
        for r in rollouts:
            rows.append({
                "prompt_id": pid,
                "prompt_text": prompt_text,
                "rollout_idx": r["rollout_idx"],
                "cot_content": r.get("cot_content", ""),
                "response_content": r.get("response_content", ""),
                "answer": r["answer"],
                "label": r["label"],
                "is_majority": r["is_majority"],
                "majority_answer": r["majority_answer"],
                "majority_frac": r["majority_frac"],
                "answer_counts": json.dumps(r["answer_counts"]),
                "filepath": r.get("rollout_file", ""),
            })
    return pd.DataFrame(rows)
