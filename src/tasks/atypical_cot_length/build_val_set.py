#!/usr/bin/env python3
"""Build validation set from rollouts 200-239 on the 51 train prompts.

Labels are assigned using mean/SD from the ORIGINAL 200 rollouts (0-199),
so the labeling criteria are identical to train/eval sets. The val set is
built from fresh rollouts (200-239) that were never seen during training.

Constraints (same as eval set):
  - ~200 samples total
  - Exactly balanced: n_short == n_long
  - Per-prompt |short - long| <= 5
  - Average absolute length of short ≈ long (balanced across 500-token buckets)
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROLLOUTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reasoning_evals" / "rollouts" / "unlabeled"
OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_SET_PATH = OUTPUT_DIR / "train_set.json"

# Rollout range for val set
VAL_START_IDX = 200
VAL_END_IDX = 240

# Stats computed from original rollouts
ORIGINAL_START_IDX = 0
ORIGINAL_END_IDX = 200

MIN_RANGE = 1000
SD_THRESHOLD = 1.0
TARGET_PER_CLASS = 100
MAX_IMBALANCE = 5
SEED = 42

random.seed(SEED)


def word_length(text: str) -> int:
    return len(text.split())


def load_rollouts_range(prompt_dir: Path, start_idx: int, end_idx: int) -> list[dict]:
    rollouts = []
    for idx in range(start_idx, end_idx):
        f = prompt_dir / f"rollout_{idx}.json"
        if f.exists():
            with open(f) as fh:
                rollouts.append(json.load(fh))
    return rollouts


def label_rollouts_with_original_stats(prompt_name: str, new_rollouts: list[dict],
                                        orig_mean: float, orig_std: float) -> list[dict]:
    """Label new rollouts using mean/SD from original 200 rollouts."""
    labeled = []
    for r in new_rollouts:
        wl = word_length(r["chain_of_thought"])
        z = (wl - orig_mean) / orig_std if orig_std > 0 else 0
        if z > SD_THRESHOLD:
            label = "long"
        elif z < -SD_THRESHOLD:
            label = "short"
        else:
            continue
        labeled.append({
            "prompt_name": r["prompt_name"],
            "rollout_idx": r["rollout_idx"],
            "label": label,
            "chain_of_thought": r["chain_of_thought"],
            "prompt_text": r["prompt_text"],
            "token_length": wl,
            "prompt_mean_length": round(orig_mean, 1),
            "prompt_std_length": round(orig_std, 1),
            "z_score": round(z, 2),
        })
    return labeled


def build_val_set(all_stats: list[dict], target_per_class: int,
                  max_imbalance: int, bin_width: int = 500):
    """Build val set via greedy pair matching across bins (same as eval set)."""
    bins = defaultdict(lambda: {"short": [], "long": []})
    for ps in all_stats:
        for e in ps["entries"]:
            b = (e["token_length"] // bin_width) * bin_width
            bins[b][e["label"]].append(e)

    for b in bins:
        random.shuffle(bins[b]["short"])
        random.shuffle(bins[b]["long"])

    prompt_counts = defaultdict(lambda: {"short": 0, "long": 0})
    selected = []
    n_selected = 0

    active_bins = sorted(bins.keys())
    made_progress = True

    while n_selected < target_per_class and made_progress:
        made_progress = False
        for b in active_bins:
            if n_selected >= target_per_class:
                break

            s_pool = bins[b]["short"]
            l_pool = bins[b]["long"]

            s_candidate = None
            s_idx = None
            for i, e in enumerate(s_pool):
                pn = e["prompt_name"]
                if prompt_counts[pn]["short"] - prompt_counts[pn]["long"] < max_imbalance:
                    s_candidate = e
                    s_idx = i
                    break

            if s_candidate is None:
                continue

            l_candidate = None
            l_idx = None
            for i, e in enumerate(l_pool):
                pn = e["prompt_name"]
                if prompt_counts[pn]["long"] - prompt_counts[pn]["short"] < max_imbalance:
                    l_candidate = e
                    l_idx = i
                    break

            if l_candidate is None:
                continue

            selected.append(s_pool.pop(s_idx))
            selected.append(l_pool.pop(l_idx))
            prompt_counts[s_candidate["prompt_name"]]["short"] += 1
            prompt_counts[l_candidate["prompt_name"]]["long"] += 1
            n_selected += 1
            made_progress = True

        active_bins = [b for b in active_bins
                       if bins[b]["short"] and bins[b]["long"]]

    print(f"  Selected {n_selected} pairs = {n_selected}S + {n_selected}L")
    return selected


def check_bucket_balance(entries: list[dict], bucket_width: int = 500) -> list[dict]:
    buckets = defaultdict(lambda: {"short": 0, "long": 0})
    for e in entries:
        bucket = (e["token_length"] // bucket_width) * bucket_width
        buckets[bucket][e["label"]] += 1

    results = []
    for bucket in sorted(buckets):
        counts = buckets[bucket]
        total = counts["short"] + counts["long"]
        ratio = counts["short"] / total if total > 0 else 0
        results.append({
            "bucket": f"{bucket}-{bucket + bucket_width}",
            "short": counts["short"],
            "long": counts["long"],
            "total": total,
            "short_ratio": round(ratio, 2),
        })
    return results


def main():
    # Load train prompts
    train_set = json.load(open(TRAIN_SET_PATH))
    train_prompts = train_set["summary"]["prompts"]
    print(f"Train prompts to process: {len(train_prompts)}")

    # Step 1: Compute mean/SD from original 200 rollouts, label new rollouts
    print("\nStep 1: Computing stats from original rollouts + labeling new rollouts...")
    all_stats = []
    missing_prompts = []

    for prompt_name in train_prompts:
        prompt_dir = ROLLOUTS_DIR / prompt_name

        # Load original rollouts for mean/SD
        orig_rollouts = load_rollouts_range(prompt_dir, ORIGINAL_START_IDX, ORIGINAL_END_IDX)
        if len(orig_rollouts) < 10:
            print(f"  {prompt_name}: only {len(orig_rollouts)} original rollouts, skipping")
            continue

        orig_lengths = [word_length(r["chain_of_thought"]) for r in orig_rollouts]
        if max(orig_lengths) - min(orig_lengths) < MIN_RANGE:
            print(f"  {prompt_name}: range {max(orig_lengths) - min(orig_lengths)} < {MIN_RANGE}, skipping")
            continue

        orig_mean = mean(orig_lengths)
        orig_std = stdev(orig_lengths)

        # Load new rollouts (200-239)
        new_rollouts = load_rollouts_range(prompt_dir, VAL_START_IDX, VAL_END_IDX)
        if len(new_rollouts) == 0:
            missing_prompts.append(prompt_name)
            continue

        labeled = label_rollouts_with_original_stats(prompt_name, new_rollouts, orig_mean, orig_std)
        n_short = sum(1 for e in labeled if e["label"] == "short")
        n_long = sum(1 for e in labeled if e["label"] == "long")

        if n_short == 0 and n_long == 0:
            print(f"  {prompt_name}: no extreme rollouts in 200-239 range")
            continue

        all_stats.append({
            "prompt_name": prompt_name,
            "mean_length": orig_mean,
            "std_length": orig_std,
            "n_short": n_short,
            "n_long": n_long,
            "n_new_rollouts": len(new_rollouts),
            "entries": labeled,
        })
        print(f"  {prompt_name}: {len(new_rollouts)} new rollouts → {n_short}S + {n_long}L")

    if missing_prompts:
        print(f"\n  WARNING: {len(missing_prompts)} prompts have no rollouts in 200-239 range:")
        for p in missing_prompts:
            print(f"    {p}")
        print("  Run generate_val_rollouts.py first!")
        if len(missing_prompts) > 10:
            return

    print(f"\n  {len(all_stats)} prompts have labeled entries")
    total_short = sum(s["n_short"] for s in all_stats)
    total_long = sum(s["n_long"] for s in all_stats)
    print(f"  Total pool: {total_short} short + {total_long} long = {total_short + total_long}")

    # Step 2: Build val set using greedy pair matching
    print(f"\nStep 2: Building val set (target {TARGET_PER_CLASS}S + {TARGET_PER_CLASS}L)...")
    val_entries = build_val_set(all_stats, TARGET_PER_CLASS, MAX_IMBALANCE)

    val_short = [e for e in val_entries if e["label"] == "short"]
    val_long = [e for e in val_entries if e["label"] == "long"]
    print(f"  Val: {len(val_short)} short + {len(val_long)} long = {len(val_entries)} total")

    if len(val_entries) == 0:
        print("  ERROR: No entries selected. Not enough data.")
        return

    val_prompts_used = sorted(set(e["prompt_name"] for e in val_entries))
    print(f"  Val prompts used: {len(val_prompts_used)}")

    mean_abs_short = mean([e["token_length"] for e in val_short])
    mean_abs_long = mean([e["token_length"] for e in val_long])
    overall_mean = (mean_abs_long + mean_abs_short) / 2
    pct_gap = abs(mean_abs_long - mean_abs_short) / overall_mean * 100
    print(f"  Mean absolute length — short: {mean_abs_short:.0f}, long: {mean_abs_long:.0f} "
          f"(gap: {pct_gap:.1f}%)")

    # Per-prompt class counts
    per_prompt = defaultdict(lambda: {"short": 0, "long": 0})
    for e in val_entries:
        per_prompt[e["prompt_name"]][e["label"]] += 1
    worst_imbalance = max(abs(v["short"] - v["long"]) for v in per_prompt.values())
    print(f"  Worst per-prompt |short-long|: {worst_imbalance}")
    print(f"  Per-prompt counts:")
    for pn in sorted(per_prompt):
        c = per_prompt[pn]
        diff = abs(c["short"] - c["long"])
        marker = " *" if diff > 1 else ""
        print(f"    {pn}: {c['short']}S / {c['long']}L{marker}")

    # Bucket balance
    buckets = check_bucket_balance(val_entries, 500)
    print(f"  Bucket balance (500-word):")
    for b in buckets:
        print(f"    {b['bucket']}: {b['short']}S/{b['long']}L (short_ratio: {b['short_ratio']})")

    # Step 3: Write val set JSON
    print("\nStep 3: Writing val_set.json...")

    val_set = {
        "description": (
            "Relative CoT length classification validation set. "
            "Built from rollouts 200-239 on the 51 train prompts. "
            "Labels assigned using mean/SD from original 200 rollouts. "
            "Long = >1 SD above prompt mean, Short = <1 SD below. "
            "Balanced on absolute word length via greedy pair matching. "
            "Per-prompt |short-long| <= 5."
        ),
        "version": "v1",
        "subject_model": "Qwen/Qwen3-32B",
        "summary": {
            "long_count": len(val_long),
            "short_count": len(val_short),
            "total": len(val_entries),
            "num_prompts": len(val_prompts_used),
            "prompts": val_prompts_used,
            "mean_abs_length_short": round(mean_abs_short, 1),
            "mean_abs_length_long": round(mean_abs_long, 1),
            "abs_length_gap_pct": round(pct_gap, 1),
            "rollout_range": [VAL_START_IDX, VAL_END_IDX - 1],
            "stats_from_rollouts": [ORIGINAL_START_IDX, ORIGINAL_END_IDX - 1],
        },
        "entries": val_entries,
    }

    val_path = OUTPUT_DIR / "val_set.json"
    with open(val_path, "w") as f:
        json.dump(val_set, f, indent=2)
    print(f"  Wrote {val_path} ({os.path.getsize(val_path) / 1024 / 1024:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
