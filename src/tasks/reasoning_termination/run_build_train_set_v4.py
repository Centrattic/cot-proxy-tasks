#!/usr/bin/env python3
"""
Build v4 training set with distance-based proxy labels.

Labels:
  YES: prefix cut at 25, 35, 45, 55 words from end of CoT (near answer emission)
  NO:  prefix cut at 300+ words from end of CoT (deep in reasoning)

Each rollout produces up to 4 YES + multiple NO samples.

Usage:
    python -m src2.runs.run_build_train_set_v4
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from src2.tasks.reasoning_evals.prompts import REASONING_PROMPTS

# ── Configuration ─────────────────────────────────────────────────────
DATA_DIR = Path("data/reasoning_evals")
ROLLOUT_DIR = DATA_DIR / "rollouts" / "unlabeled"
OUTPUT_PATH = DATA_DIR / "answer_train_set_v4.json"

SEED = 42

# YES: distances (in words) from end of CoT
YES_DISTANCES = [25, 35, 45, 55]
# NO: starting distance and step
NO_START = 300
NO_STEP = 200

# Min words for a prefix to be useful
MIN_PREFIX_WORDS = 20

# All train prompts (original 21 + 15 new)
TRAIN_PROMPTS = [
    # Original 21
    "prompt-1", "prompt-3", "prompt-4", "prompt-5", "prompt-6",
    "picnic", "sisters", "tricky_sisters", "small_nested",
    "count_solutions", "last_digit", "widow", "chinese-cancer",
    "chinese-math", "minimum_square", "series", "harder_jack",
    "waffle_low", "n_remainder", "leet_speak", "letters",
    # New 15
    "knight_moves", "digital_root", "coin_rows", "snail_wall",
    "power_mod", "rectangle_diagonal", "three_digit_sum",
    "party_handshakes", "polynomial_trick", "clock_angle",
    "socks_guarantee", "locker_problem", "sum_of_cubes",
    "divisor_count", "domino_cover",
]
# ──────────────────────────────────────────────────────────────────────


def cut_prefix_by_words(cot: str, words_from_end: int) -> str | None:
    """Cut CoT at `words_from_end` words from the end. Returns prefix text or None."""
    words = cot.split()
    cut_point = len(words) - words_from_end
    if cut_point < MIN_PREFIX_WORDS:
        return None
    return " ".join(words[:cut_point])


def main():
    random.seed(SEED)

    entries = []
    stats = defaultdict(lambda: {"yes": 0, "no": 0, "rollouts": 0})

    for name in TRAIN_PROMPTS:
        prompt_dir = ROLLOUT_DIR / name
        if not prompt_dir.exists():
            print(f"  WARNING: No rollouts for {name}")
            continue

        rollout_files = sorted(prompt_dir.glob("rollout_*.json"))
        stats[name]["rollouts"] = len(rollout_files)

        for rf in rollout_files:
            with open(rf) as f:
                rollout = json.load(f)

            cot = rollout.get("chain_of_thought", "")
            rollout_idx = rollout.get("rollout_idx", int(rf.stem.split("_")[1]))
            total_words = len(cot.split())

            # YES prefixes: near end
            for dist in YES_DISTANCES:
                if total_words < dist + MIN_PREFIX_WORDS:
                    continue
                prefix = cut_prefix_by_words(cot, dist)
                if prefix:
                    entries.append({
                        "prompt_name": name,
                        "rollout_idx": rollout_idx,
                        "label": "yes",
                        "distance_from_end": dist,
                        "prefix_text": prefix,
                        "prefix_words": len(prefix.split()),
                        "total_words": total_words,
                    })
                    stats[name]["yes"] += 1

            # NO prefixes: far from end
            dist = NO_START
            while dist < total_words - MIN_PREFIX_WORDS:
                prefix = cut_prefix_by_words(cot, dist)
                if prefix:
                    entries.append({
                        "prompt_name": name,
                        "rollout_idx": rollout_idx,
                        "label": "no",
                        "distance_from_end": dist,
                        "prefix_text": prefix,
                        "prefix_words": len(prefix.split()),
                        "total_words": total_words,
                    })
                    stats[name]["no"] += 1
                dist += NO_STEP

    # Balance: subsample majority class to match minority
    yes_entries = [e for e in entries if e["label"] == "yes"]
    no_entries = [e for e in entries if e["label"] == "no"]

    print(f"\nRaw counts: {len(yes_entries)} yes, {len(no_entries)} no")

    target = min(len(yes_entries), len(no_entries))
    if len(yes_entries) > target:
        random.shuffle(yes_entries)
        yes_entries = yes_entries[:target]
    if len(no_entries) > target:
        random.shuffle(no_entries)
        no_entries = no_entries[:target]

    all_entries = yes_entries + no_entries
    random.shuffle(all_entries)

    print(f"Balanced: {len(yes_entries)} yes, {len(no_entries)} no = {len(all_entries)} total")

    # Build output
    train_set = {
        "summary": {
            "yes_count": len(yes_entries),
            "no_count": len(no_entries),
            "total": len(all_entries),
            "label_source": "distance_proxy",
            "yes_distances": YES_DISTANCES,
            "no_start": NO_START,
            "no_step": NO_STEP,
            "num_prompts": len(set(e["prompt_name"] for e in all_entries)),
            "prompts": sorted(set(e["prompt_name"] for e in all_entries)),
        },
        "entries": all_entries,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(train_set, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Per-prompt stats
    print("\nPer-prompt distribution:")
    for name in sorted(stats.keys()):
        s = stats[name]
        print(f"  {name:25s}: {s['rollouts']:2d} rollouts, {s['yes']:3d} yes, {s['no']:3d} no")


if __name__ == "__main__":
    main()
