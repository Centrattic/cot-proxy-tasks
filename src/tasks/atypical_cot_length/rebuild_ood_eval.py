#!/usr/bin/env python3
"""Rebuild gpqa_chem_eval_set.json from 200 rollouts per prompt.

Uses the same 58 prompts as the existing eval set, but with:
  - 200 rollouts per prompt (up from 30)
  - SD_THRESHOLD = 1.0 (up from 0.7)
  - Greedy pair matching (500-word bins)
  - TARGET_PER_CLASS = 100, MAX_IMBALANCE = 5

Backs up the old eval set before overwriting.
"""

import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOD_CHEM_DIR = PROJECT_ROOT / "data" / "reasoning_evals" / "rollouts" / "ood_chemistry"
OUTPUT_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = OUTPUT_DIR / "gpqa_chem_eval_set.json"

# ── Constants ─────────────────────────────────────────────────────────
MIN_RANGE = 1000       # min word-length range to qualify
SD_THRESHOLD = 1.0     # z-score threshold (matched to ID test set)
TARGET_PER_CLASS = 100  # 100 short + 100 long
MAX_IMBALANCE = 5      # per-prompt |short - long| <= 5
BIN_WIDTH = 500
SEED = 42

random.seed(SEED)


# ── Helpers (same as build_new_sets.py / build_sets.py) ───────────────

def word_length(text: str) -> int:
    return len(text.split())


def load_rollouts(prompt_dir: Path) -> list[dict]:
    rollouts = []
    for f in sorted(prompt_dir.glob("rollout_*.json")):
        with open(f) as fh:
            rollouts.append(json.load(fh))
    return rollouts


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


def build_balanced_set(stats_list: list[dict], target_per_class: int,
                       max_imbalance: int, bin_width: int = 500) -> list[dict]:
    """Build a balanced set via greedy pair matching across bins."""
    bins = defaultdict(lambda: {"short": [], "long": []})
    for ps in stats_list:
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


def print_set_stats(name: str, entries: list[dict]):
    """Print detailed stats for a built set."""
    short_entries = [e for e in entries if e["label"] == "short"]
    long_entries = [e for e in entries if e["label"] == "long"]
    prompts_used = sorted(set(e["prompt_name"] for e in entries))

    print(f"\n  {name}: {len(short_entries)} short + {len(long_entries)} long = {len(entries)} total")
    print(f"  Prompts used: {len(prompts_used)}")

    if short_entries and long_entries:
        mean_abs_short = mean([e["token_length"] for e in short_entries])
        mean_abs_long = mean([e["token_length"] for e in long_entries])
        overall_mean = (mean_abs_long + mean_abs_short) / 2
        pct_gap = abs(mean_abs_long - mean_abs_short) / overall_mean * 100
        print(f"  Mean absolute length - short: {mean_abs_short:.0f}, long: {mean_abs_long:.0f} "
              f"(gap: {pct_gap:.1f}%)")

    per_prompt = defaultdict(lambda: {"short": 0, "long": 0})
    for e in entries:
        per_prompt[e["prompt_name"]][e["label"]] += 1
    worst_imbalance = max(abs(v["short"] - v["long"]) for v in per_prompt.values())
    print(f"  Worst per-prompt |short-long|: {worst_imbalance}")
    print(f"  Per-prompt counts:")
    for pn in sorted(per_prompt):
        c = per_prompt[pn]
        diff = abs(c["short"] - c["long"])
        marker = " *" if diff > 1 else ""
        print(f"    {pn}: {c['short']}S / {c['long']}L{marker}")

    buckets = check_bucket_balance(entries, 500)
    print(f"  Bucket balance (500-word):")
    for b in buckets:
        print(f"    {b['bucket']}: {b['short']}S/{b['long']}L (short_ratio: {b['short_ratio']})")


def write_set(entries: list[dict], description: str, output_path: Path,
              version: str = "v2"):
    """Write a dataset JSON file."""
    short_entries = [e for e in entries if e["label"] == "short"]
    long_entries = [e for e in entries if e["label"] == "long"]
    prompts_used = sorted(set(e["prompt_name"] for e in entries))

    mean_abs_short = mean([e["token_length"] for e in short_entries]) if short_entries else 0
    mean_abs_long = mean([e["token_length"] for e in long_entries]) if long_entries else 0
    overall_mean = (mean_abs_long + mean_abs_short) / 2 if (mean_abs_long + mean_abs_short) > 0 else 1
    pct_gap = abs(mean_abs_long - mean_abs_short) / overall_mean * 100

    dataset = {
        "description": description,
        "version": version,
        "subject_model": "Qwen/Qwen3-32B",
        "summary": {
            "long_count": len(long_entries),
            "short_count": len(short_entries),
            "total": len(entries),
            "num_prompts": len(prompts_used),
            "prompts": prompts_used,
            "mean_abs_length_short": round(mean_abs_short, 1),
            "mean_abs_length_long": round(mean_abs_long, 1),
            "abs_length_gap_pct": round(pct_gap, 1),
        },
        "entries": entries,
    }

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  Wrote {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────

# The 58 prompts from the existing OOD eval set
EVAL_PROMPTS = [
    "gpqa_chem_000", "gpqa_chem_002", "gpqa_chem_003", "gpqa_chem_004",
    "gpqa_chem_008", "gpqa_chem_010", "gpqa_chem_012", "gpqa_chem_020",
    "gpqa_chem_022", "gpqa_chem_024", "gpqa_chem_025", "gpqa_chem_026",
    "gpqa_chem_033", "gpqa_chem_034", "gpqa_chem_035", "gpqa_chem_036",
    "gpqa_chem_039", "gpqa_chem_041", "gpqa_chem_043", "gpqa_chem_045",
    "gpqa_chem_048", "gpqa_chem_051", "gpqa_chem_060", "gpqa_chem_062",
    "gpqa_chem_063", "gpqa_chem_064", "gpqa_chem_067", "gpqa_chem_073",
    "gpqa_chem_078", "gpqa_chem_079", "gpqa_chem_081", "gpqa_chem_086",
    "gpqa_chem_091", "gpqa_chem_095", "gpqa_chem_100", "gpqa_chem_101",
    "gpqa_chem_102", "gpqa_chem_106", "gpqa_chem_111", "gpqa_chem_112",
    "gpqa_chem_114", "gpqa_chem_117", "gpqa_chem_121", "gpqa_chem_123",
    "gpqa_chem_126", "gpqa_chem_131", "gpqa_chem_139", "gpqa_chem_142",
    "gpqa_chem_148", "gpqa_chem_156", "gpqa_chem_157", "gpqa_chem_160",
    "gpqa_chem_163", "gpqa_chem_164", "gpqa_chem_165", "gpqa_chem_166",
    "gpqa_chem_175", "gpqa_chem_176",
]


def main():
    print("=" * 60)
    print("Rebuild OOD eval set from 200 rollouts/prompt, z=1.0")
    print("=" * 60)

    # Step 0: Back up existing eval set
    if EVAL_SET_PATH.exists():
        backup = EVAL_SET_PATH.with_suffix(
            f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(EVAL_SET_PATH, backup)
        print(f"\nBacked up existing eval set to {backup.name}")

    # Step 1: Load rollouts for the 58 eval prompts
    print(f"\nLoading rollouts for {len(EVAL_PROMPTS)} eval prompts...")
    eval_stats = []
    skipped = []
    for prompt_name in EVAL_PROMPTS:
        prompt_dir = OOD_CHEM_DIR / prompt_name
        if not prompt_dir.exists():
            print(f"  WARNING: {prompt_name} directory not found, skipping")
            skipped.append(prompt_name)
            continue

        rollouts = load_rollouts(prompt_dir)
        print(f"  {prompt_name}: {len(rollouts)} rollouts", end="")

        if len(rollouts) < 10:
            print(f" - too few, skipping")
            skipped.append(prompt_name)
            continue

        stats = compute_prompt_stats(prompt_name, rollouts)
        if stats is not None:
            print(f" -> {stats['n_short']}S + {stats['n_long']}L "
                  f"(mean={stats['mean_length']:.0f}, sd={stats['std_length']:.0f})")
            eval_stats.append(stats)
        else:
            lengths = [word_length(r["chain_of_thought"]) for r in rollouts]
            rng = max(lengths) - min(lengths)
            print(f" - FILTERED (range={rng})")
            skipped.append(prompt_name)

    print(f"\n{len(eval_stats)}/{len(EVAL_PROMPTS)} prompts qualify "
          f"(range >= {MIN_RANGE}, z={SD_THRESHOLD}, both classes)")
    if skipped:
        print(f"  Skipped: {skipped}")

    total_short = sum(s["n_short"] for s in eval_stats)
    total_long = sum(s["n_long"] for s in eval_stats)
    print(f"  Total pool: {total_short} short + {total_long} long = {total_short + total_long}")

    # Step 2: Build balanced eval set
    print(f"\nBuilding balanced eval set "
          f"(target {TARGET_PER_CLASS}S + {TARGET_PER_CLASS}L)...")
    entries = build_balanced_set(eval_stats, TARGET_PER_CLASS, MAX_IMBALANCE, BIN_WIDTH)

    print_set_stats("gpqa_chem_eval (rebuilt)", entries)

    # Step 3: Write
    print(f"\nWriting eval set...")
    write_set(
        entries,
        description=(
            "Relative CoT length classification OOD eval set - GPQA chemistry. "
            "Rebuilt from 200 rollouts/prompt with z=1.0 threshold. "
            "Long = >1 SD above prompt mean, Short = <1 SD below. "
            "Balanced on absolute word length via greedy pair matching (500-word bins). "
            "Per-prompt |short-long| <= 5."
        ),
        output_path=EVAL_SET_PATH,
        version="v2",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
