"""Build OOD test set from math puzzle qualifying pairs."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

STORIES_DIR = Path(__file__).resolve().parent
SEED = 42


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def load_cot_text(prompt_key: str, cot_idx: int) -> str:
    safe_name = sanitize_name(prompt_key)
    cot_path = STORIES_DIR / "selected_cots" / safe_name / f"cot_{cot_idx}.json"
    if cot_path.exists():
        data = json.loads(cot_path.read_text())
        return data.get("cot_text", "")
    return ""


def main() -> None:
    # Load stats
    all_stats: dict = json.loads((STORIES_DIR / "dilemma_stats.json").read_text())

    # Load math puzzle prompts for text
    prompts: dict[str, str] = json.loads(
        (STORIES_DIR / "math_puzzle_prompts.json").read_text()
    )

    # Collect qualifying pairs from math puzzles only
    entries = []
    for key, dstats in all_stats.items():
        if key.startswith("dilemma_"):
            continue
        if key not in prompts:
            continue
        baseline_mean = dstats["baseline_mean"]
        for cot_key, cot_stat in dstats.get("cots", {}).items():
            if not cot_stat.get("qualifies"):
                continue
            cot_idx = int(cot_key.split("_")[1])
            d = cot_stat["cohens_d"]
            entries.append({
                "prompt_key": key,
                "prompt_text": prompts[key],
                "cot_idx": cot_idx,
                "cot_text": load_cot_text(key, cot_idx),
                "label": "positive" if d > 0 else "negative",
                "cohens_d": cot_stat["cohens_d"],
                "p_value": cot_stat["p"],
                "forced_mean": cot_stat["mean"],
                "baseline_mean": baseline_mean,
            })

    pos = [e for e in entries if e["label"] == "positive"]
    neg = [e for e in entries if e["label"] == "negative"]
    print(f"Qualifying: {len(entries)} ({len(pos)} pos, {len(neg)} neg)")

    rng = random.Random(SEED)
    rng.shuffle(pos)
    rng.shuffle(neg)
    min_count = min(len(pos), len(neg))
    balanced = pos[:min_count] + neg[:min_count]
    rng.shuffle(balanced)

    unique_prompts = len(set(e["prompt_key"] for e in balanced))
    print(f"Balanced: {len(balanced)} ({min_count} pos + {min_count} neg) from {unique_prompts} prompts")

    output = {
        "split": "ood_test",
        "domain": "math_puzzles",
        "n_positive": min_count,
        "n_negative": min_count,
        "n_prompts": unique_prompts,
        "entries": balanced,
    }

    out_path = STORIES_DIR / "dilemma_dataset_ood_test.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
