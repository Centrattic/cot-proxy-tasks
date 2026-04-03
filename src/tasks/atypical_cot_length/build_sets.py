#!/usr/bin/env python3
"""Build relative_length eval/train sets from 200 rollouts per prompt.

For each qualifying prompt (CoT length range >= 1000 words across 200 rollouts),
label rollouts as "long" (>mean+1*SD) or "short" (<mean-1*SD).

Eval set (~200 samples):
  - Select ~1/3 of prompts evenly spread across the mean-length distribution
  - Wing-size optimization: prompts at the low-mean end contribute only longs
    (which have low absolute length), prompts at the high-mean end contribute
    only shorts (which have high absolute length), middle prompts contribute both
  - Phase 2 fills to target with highest-abs shorts and lowest-abs longs
  - Per-prompt |short - long| <= 5
  - Result: absolute-length-balanced despite longs being inherently longer within
    each prompt

Train set: remaining prompts, per-prompt balanced, globally downsampled.
"""

import json
import os
import random
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

ROLLOUTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reasoning_evals" / "rollouts" / "unlabeled"
OUTPUT_DIR = Path(__file__).resolve().parent
MIN_RANGE = 1000  # minimum word-length range to qualify a prompt
SD_THRESHOLD = 1.0
TARGET_PER_CLASS = 100  # target 100 short + 100 long in eval set
MAX_IMBALANCE = 5  # per-prompt |short - long| <= 5
TEST_FRACTION = 0.45  # fraction of qualifying prompts reserved for eval
SEED = 42

random.seed(SEED)


def load_rollouts(prompt_dir: Path) -> list[dict]:
    rollouts = []
    for f in sorted(prompt_dir.glob("rollout_*.json")):
        with open(f) as fh:
            rollouts.append(json.load(fh))
    return rollouts


def word_length(text: str) -> int:
    return len(text.split())


def compute_prompt_stats(prompt_name: str, rollouts: list[dict]) -> dict | None:
    """Compute stats and label rollouts. Returns None if prompt doesn't qualify."""
    lengths = [word_length(r["chain_of_thought"]) for r in rollouts]
    if max(lengths) - min(lengths) < MIN_RANGE:
        return None

    mu = mean(lengths)
    sd = stdev(lengths)
    if sd == 0:
        return None

    labeled = []
    for r, wl in zip(rollouts, lengths):
        z = (wl - mu) / sd
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
            "prompt_mean_length": round(mu, 1),
            "prompt_std_length": round(sd, 1),
            "z_score": round(z, 2),
        })

    n_short = sum(1 for e in labeled if e["label"] == "short")
    n_long = sum(1 for e in labeled if e["label"] == "long")
    if n_short == 0 or n_long == 0:
        return None

    return {
        "prompt_name": prompt_name,
        "mean_length": mu,
        "std_length": sd,
        "n_short": n_short,
        "n_long": n_long,
        "entries": labeled,
    }


def build_eval_set(test_stats: list[dict], target_per_class: int,
                   max_imbalance: int, bin_width: int = 500):
    """Build eval set via greedy pair matching across bins.

    Iteratively adds (short, long) pairs from the same bin. Each pair addition
    is checked against the per-prompt constraint before committing. Bins are
    processed in round-robin order to spread entries across the length spectrum.

    Guarantees:
    - Perfect per-bin balance (every bin has equal S and L)
    - Per-prompt |S-L| <= max_imbalance
    - Global S == L
    """
    # Pool all entries into bins
    bins = defaultdict(lambda: {"short": [], "long": []})
    for ps in test_stats:
        for e in ps["entries"]:
            b = (e["token_length"] // bin_width) * bin_width
            bins[b][e["label"]].append(e)

    # Shuffle within each bin
    for b in bins:
        random.shuffle(bins[b]["short"])
        random.shuffle(bins[b]["long"])

    # Track per-prompt counts
    prompt_counts = defaultdict(lambda: {"short": 0, "long": 0})
    selected = []
    n_selected = 0

    # Round-robin across bins until target reached
    active_bins = sorted(bins.keys())
    made_progress = True

    while n_selected < target_per_class and made_progress:
        made_progress = False
        for b in active_bins:
            if n_selected >= target_per_class:
                break

            s_pool = bins[b]["short"]
            l_pool = bins[b]["long"]

            # Try to find a valid (short, long) pair from this bin
            # Short candidate: must not push its prompt over max_imbalance
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

            # Long candidate: must not push its prompt over max_imbalance
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

            # Commit the pair
            selected.append(s_pool.pop(s_idx))
            # Adjust index if s was before l in same pool (shouldn't happen, different pools)
            selected.append(l_pool.pop(l_idx))
            prompt_counts[s_candidate["prompt_name"]]["short"] += 1
            prompt_counts[l_candidate["prompt_name"]]["long"] += 1
            n_selected += 1
            made_progress = True

        # Remove exhausted bins
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
    print("Loading rollouts (200 per prompt)...")
    prompt_dirs = sorted(d for d in ROLLOUTS_DIR.iterdir() if d.is_dir())
    print(f"  Found {len(prompt_dirs)} prompt directories")

    # Step 1: Compute per-prompt stats and label
    print("\nStep 1: Computing per-prompt stats...")
    all_stats = []
    for d in prompt_dirs:
        rollouts = load_rollouts(d)
        if len(rollouts) < 10:
            continue
        stats = compute_prompt_stats(d.name, rollouts)
        if stats is not None:
            all_stats.append(stats)

    print(f"  {len(all_stats)} prompts qualify (range >= {MIN_RANGE}, both classes present)")
    total_short = sum(s["n_short"] for s in all_stats)
    total_long = sum(s["n_long"] for s in all_stats)
    print(f"  Total pool: {total_short} short + {total_long} long = {total_short + total_long}")

    means = sorted(s["mean_length"] for s in all_stats)
    print(f"  Prompt mean lengths: {means[0]:.0f} to {means[-1]:.0f}")

    # Step 2: Split prompts into test and train (evenly spaced selection)
    sorted_stats = sorted(all_stats, key=lambda s: s["mean_length"])
    n_total = len(sorted_stats)
    n_test = max(20, round(n_total * TEST_FRACTION))

    # Evenly spaced indices across the mean-length distribution
    indices = [round(i * (n_total - 1) / (n_test - 1)) for i in range(n_test)]
    indices = list(dict.fromkeys(indices))  # dedupe preserving order
    test_indices = set(indices)
    test_stats = [sorted_stats[i] for i in indices]
    train_stats = [sorted_stats[i] for i in range(n_total) if i not in test_indices]

    test_prompt_names = set(s["prompt_name"] for s in test_stats)
    train_prompt_names = set(s["prompt_name"] for s in train_stats)
    assert len(test_prompt_names & train_prompt_names) == 0

    print(f"\n  Split: {len(test_stats)} test prompts, {len(train_stats)} train prompts")

    # Step 3: Build eval set using per-bin greedy matching
    print(f"\nStep 3: Building eval set (target {TARGET_PER_CLASS}S + {TARGET_PER_CLASS}L)...")
    eval_entries = build_eval_set(test_stats, TARGET_PER_CLASS, MAX_IMBALANCE)

    eval_short = [e for e in eval_entries if e["label"] == "short"]
    eval_long = [e for e in eval_entries if e["label"] == "long"]
    print(f"  Eval: {len(eval_short)} short + {len(eval_long)} long = {len(eval_entries)} total")

    eval_prompts_used = sorted(set(e["prompt_name"] for e in eval_entries))
    print(f"  Eval prompts used: {len(eval_prompts_used)}")

    mean_abs_short = mean([e["token_length"] for e in eval_short])
    mean_abs_long = mean([e["token_length"] for e in eval_long])
    overall_mean = (mean_abs_long + mean_abs_short) / 2
    pct_gap = abs(mean_abs_long - mean_abs_short) / overall_mean * 100
    print(f"  Mean absolute length — short: {mean_abs_short:.0f}, long: {mean_abs_long:.0f} "
          f"(gap: {pct_gap:.1f}%)")

    # Per-prompt class counts
    per_prompt = defaultdict(lambda: {"short": 0, "long": 0})
    for e in eval_entries:
        per_prompt[e["prompt_name"]][e["label"]] += 1
    worst_imbalance = max(abs(v["short"] - v["long"]) for v in per_prompt.values())
    print(f"  Worst per-prompt |short-long|: {worst_imbalance}")
    print(f"  Per-prompt counts:")
    for pn in sorted(per_prompt):
        c = per_prompt[pn]
        diff = abs(c["short"] - c["long"])
        marker = " *" if diff > 1 else ""
        print(f"    {pn}: {c['short']}S / {c['long']}L{marker}")

    # Bucket balance (500-word)
    buckets = check_bucket_balance(eval_entries, 500)
    print(f"  Bucket balance (500-word):")
    for b in buckets:
        print(f"    {b['bucket']}: {b['short']}S/{b['long']}L (short_ratio: {b['short_ratio']})")

    # Step 4: Build train set from remaining prompts
    print("\nStep 4: Building train set...")

    # Per-prompt balanced entries
    train_entries = []
    for ps in train_stats:
        shorts = [e for e in ps["entries"] if e["label"] == "short"]
        longs = [e for e in ps["entries"] if e["label"] == "long"]
        n = min(len(shorts), len(longs))
        random.shuffle(shorts)
        random.shuffle(longs)
        train_entries.extend(shorts[:n])
        train_entries.extend(longs[:n])

    # Global balance: downsample majority class (keep entries closest to boundary)
    train_short = [e for e in train_entries if e["label"] == "short"]
    train_long = [e for e in train_entries if e["label"] == "long"]
    n_balanced = min(len(train_short), len(train_long))
    train_short.sort(key=lambda e: abs(e["z_score"]))
    train_long.sort(key=lambda e: abs(e["z_score"]))
    train_entries_balanced = train_short[:n_balanced] + train_long[:n_balanced]

    train_prompts_used = sorted(set(e["prompt_name"] for e in train_entries_balanced))
    print(f"  Train: {n_balanced} short + {n_balanced} long = {len(train_entries_balanced)} total")
    print(f"  Train prompts: {len(train_prompts_used)}")

    # Verify no overlap
    overlap = test_prompt_names & set(train_prompts_used)
    assert len(overlap) == 0, f"Prompt overlap: {overlap}"
    print(f"  No prompt overlap: OK")

    # Step 5: Write JSON files
    print("\nStep 5: Writing JSON files...")

    eval_set = {
        "description": (
            "Relative CoT length classification eval set (200 rollouts/prompt). "
            "Long = >1 SD above prompt mean, Short = <1 SD below. "
            "Balanced on absolute word length via wing-size optimization. "
            "Per-prompt |short-long| <= 5."
        ),
        "version": "v2",
        "subject_model": "Qwen/Qwen3-32B",
        "summary": {
            "long_count": len(eval_long),
            "short_count": len(eval_short),
            "total": len(eval_entries),
            "num_prompts": len(eval_prompts_used),
            "prompts": eval_prompts_used,
            "mean_abs_length_short": round(mean_abs_short, 1),
            "mean_abs_length_long": round(mean_abs_long, 1),
            "abs_length_gap_pct": round(pct_gap, 1),
        },
        "entries": eval_entries,
    }

    train_mean_short = mean([e["token_length"] for e in train_entries_balanced
                             if e["label"] == "short"])
    train_mean_long = mean([e["token_length"] for e in train_entries_balanced
                            if e["label"] == "long"])

    train_set = {
        "description": (
            "Relative CoT length classification training set (200 rollouts/prompt). "
            "Long = >1 SD above prompt mean, Short = <1 SD below. "
            "Per-prompt balanced, globally downsampled (keeping entries closest to "
            "decision boundary)."
        ),
        "version": "v2",
        "subject_model": "Qwen/Qwen3-32B",
        "summary": {
            "long_count": n_balanced,
            "short_count": n_balanced,
            "total": len(train_entries_balanced),
            "num_prompts": len(train_prompts_used),
            "prompts": train_prompts_used,
            "mean_abs_length_short": round(train_mean_short, 1),
            "mean_abs_length_long": round(train_mean_long, 1),
        },
        "entries": train_entries_balanced,
    }

    eval_path = OUTPUT_DIR / "eval_set.json"
    train_path = OUTPUT_DIR / "train_set.json"

    with open(eval_path, "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"  Wrote {eval_path} ({os.path.getsize(eval_path) / 1024 / 1024:.1f} MB)")

    with open(train_path, "w") as f:
        json.dump(train_set, f, indent=2)
    print(f"  Wrote {train_path} ({os.path.getsize(train_path) / 1024 / 1024:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
