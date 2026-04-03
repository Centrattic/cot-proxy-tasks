"""Build train/val/test splits from qualifying dilemma-CoT pairs.

Reads dilemma_stats.json, selects pairs with p < 0.05 and |Cohen's d| >= 0.5,
splits by dilemma (no dilemma overlap across splits), and balances pos/neg.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

STORIES_DIR = Path(__file__).resolve().parent
SEED = 42


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def load_cot_text(dilemma_key: str, cot_idx: int) -> str:
    """Load the CoT text from selected_cots."""
    safe_name = sanitize_name(dilemma_key)
    cot_path = STORIES_DIR / "selected_cots" / safe_name / f"cot_{cot_idx}.json"
    if cot_path.exists():
        data = json.loads(cot_path.read_text())
        return data.get("cot_text", "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test from dilemma stats")
    parser.add_argument("--dry-run", action="store_true", help="Preview split sizes without writing")
    parser.add_argument("--prompts-file", type=str, default="dilemma_prompts_full.json",
                        help="Prompts JSON file for dilemma texts (default: dilemma_prompts_full.json)")
    args = parser.parse_args()

    # Load dilemma texts
    prompts_path = STORIES_DIR / args.prompts_file
    if not prompts_path.exists():
        prompts_path = STORIES_DIR / "dilemma_prompts.json"
    dilemma_texts: dict[str, str] = json.loads(prompts_path.read_text())

    # Load stats
    stats_path = STORIES_DIR / "dilemma_stats.json"
    all_stats: dict = json.loads(stats_path.read_text())

    # Collect qualifying pairs grouped by dilemma
    dilemma_pairs: dict[str, list[dict]] = {}
    for dilemma_key, dstats in all_stats.items():
        baseline_mean = dstats["baseline_mean"]
        for cot_key, cot_stat in dstats.get("cots", {}).items():
            if not cot_stat.get("qualifies"):
                continue
            cot_idx = int(cot_key.split("_")[1])
            d = cot_stat["cohens_d"]
            label = "positive" if d > 0 else "negative"
            entry = {
                "dilemma_key": dilemma_key,
                "dilemma_text": dilemma_texts.get(dilemma_key, ""),
                "cot_idx": cot_idx,
                "cot_text": load_cot_text(dilemma_key, cot_idx),
                "label": label,
                "cohens_d": cot_stat["cohens_d"],
                "p_value": cot_stat["p"],
                "forced_mean": cot_stat["mean"],
                "baseline_mean": baseline_mean,
            }
            dilemma_pairs.setdefault(dilemma_key, []).append(entry)

    total_qualifying = sum(len(v) for v in dilemma_pairs.values())
    n_pos = sum(1 for pairs in dilemma_pairs.values() for e in pairs if e["label"] == "positive")
    n_neg = total_qualifying - n_pos
    print(f"Qualifying pairs: {total_qualifying} ({n_pos} pos, {n_neg} neg) "
          f"from {len(dilemma_pairs)} dilemmas")

    # Shuffle dilemmas and split 60/20/20
    dilemma_keys = sorted(dilemma_pairs.keys())
    rng = random.Random(SEED)
    rng.shuffle(dilemma_keys)

    n = len(dilemma_keys)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_dilemmas = dilemma_keys[:n_train]
    val_dilemmas = dilemma_keys[n_train:n_train + n_val]
    test_dilemmas = dilemma_keys[n_train + n_val:]

    splits = {
        "train": train_dilemmas,
        "val": val_dilemmas,
        "test": test_dilemmas,
    }

    # Verify no overlap
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(splits[a]) & set(splits[b])
        assert not overlap, f"Dilemma overlap between {a} and {b}: {overlap}"

    # Build balanced datasets
    for split_name, split_dilemma_keys in splits.items():
        entries = []
        for dk in split_dilemma_keys:
            entries.extend(dilemma_pairs[dk])

        pos = [e for e in entries if e["label"] == "positive"]
        neg = [e for e in entries if e["label"] == "negative"]
        min_count = min(len(pos), len(neg))

        rng.shuffle(pos)
        rng.shuffle(neg)
        balanced = pos[:min_count] + neg[:min_count]
        rng.shuffle(balanced)

        unique_dilemmas = len(set(e["dilemma_key"] for e in balanced))

        print(f"\n{split_name}: {len(balanced)} entries ({min_count} pos + {min_count} neg) "
              f"from {unique_dilemmas} dilemmas "
              f"(before balance: {len(pos)} pos, {len(neg)} neg)")

        if args.dry_run:
            continue

        output = {
            "split": split_name,
            "n_positive": min_count,
            "n_negative": min_count,
            "n_dilemmas": unique_dilemmas,
            "entries": balanced,
        }

        out_path = STORIES_DIR / f"dilemma_dataset_{split_name}.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"  Wrote {out_path}")

    if args.dry_run:
        print("\n(dry run — no files written)")


if __name__ == "__main__":
    main()
