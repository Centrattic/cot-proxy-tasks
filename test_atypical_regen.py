"""
Test: verify atypical answer labeling matches the release dataset.

Uses existing rollouts from verification_rollouts/ to check that our
majority/minority labeling and filtering logic reproduces the release data.
"""

import json
import os
from collections import Counter
from pathlib import Path

VERIF_DIR = Path("/home/riya/neel-projs/cot-comparisons/data/verification_rollouts")
RELEASE_DIR = Path("/home/riya/neel-projs/cot-comparisons/release_datasets/atypical_answer")

# Filtering thresholds from split.json
MIN_ROLLOUTS = 200
MIN_MINORITY_RATE = 0.10
MAX_MINORITY_RATE = 0.33
N_PER_CLASS = 15

# Pick some test question IDs from the release dataset
test_qids = []
for split in ["train_set", "test_set"]:
    p_dir = RELEASE_DIR / split / "prompts"
    for f in sorted(p_dir.glob("*.json"))[:2]:
        with open(f) as fh:
            d = json.load(fh)
        test_qids.append(d["question_id"])

# Also load the release rollouts for comparison
release_labels = {}  # qid -> {rollout_idx: label}
for split in ["train_set", "test_set"]:
    r_dir = RELEASE_DIR / split / "qwen-32b"
    for f in r_dir.glob("*.json"):
        with open(f) as fh:
            d = json.load(fh)
        qid = d["question_id"]
        if qid not in release_labels:
            release_labels[qid] = {}
        release_labels[qid][d["rollout_idx"]] = {
            "label": d["label"],
            "majority_answer": d["majority_answer"],
        }


def get_latest_verif_dir(qid):
    qdir = VERIF_DIR / qid
    if not qdir.exists():
        return None
    timestamped = sorted(
        [d for d in qdir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'],
        reverse=True,
    )
    return timestamped[0] if timestamped else None


print(f"Testing {len(test_qids)} questions: {test_qids}")
print()

for qid in test_qids:
    print(f"=== {qid} ===")

    run_dir = get_latest_verif_dir(qid)
    if not run_dir:
        print(f"  No verification rollouts found")
        continue

    rollouts_dir = run_dir / "rollouts"
    if not rollouts_dir.exists():
        print(f"  No rollouts dir")
        continue

    # Load all rollouts
    runs = []
    for f in sorted(rollouts_dir.glob("rollout_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        answer = d.get("answer", "")
        if answer:
            runs.append({"rollout_idx": int(f.stem.split("_")[1]), "answer": answer})

    print(f"  Total rollouts: {len(runs)}")

    # Compute majority/minority
    answers = [r["answer"] for r in runs]
    counts = Counter(answers)
    majority_answer = counts.most_common(1)[0][0]
    minority_count = len(runs) - counts[majority_answer]
    minority_rate = minority_count / len(runs)

    print(f"  Answers: {dict(counts)}")
    print(f"  Majority: {majority_answer}, minority_rate: {minority_rate:.3f}")
    print(f"  Passes filter: rollouts>={MIN_ROLLOUTS}={len(runs)>= MIN_ROLLOUTS}, "
          f"rate [{MIN_MINORITY_RATE},{MAX_MINORITY_RATE}]={MIN_MINORITY_RATE <= minority_rate <= MAX_MINORITY_RATE}")

    # Compare with release
    if qid in release_labels:
        release = release_labels[qid]
        release_maj = next(iter(release.values()))["majority_answer"]
        print(f"  Release majority_answer: {release_maj}")
        print(f"  Match: {'YES' if release_maj == majority_answer else 'NO'}")
        # Check a few rollout labels
        mismatches = 0
        for ridx, rdata in list(release.items())[:5]:
            our_label = "majority" if runs[ridx]["answer"] == majority_answer else "minority"
            if our_label != rdata["label"]:
                mismatches += 1
        print(f"  Label mismatches (first 5): {mismatches}")
    print()
