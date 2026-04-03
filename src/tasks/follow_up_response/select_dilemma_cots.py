"""Select extreme CoTs from dilemma baselines based on score deviation from mean."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

STORIES_DIR = Path(__file__).resolve().parent
DEFAULT_NUM_SELECT = 10


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select extreme CoTs from dilemma baselines")
    parser.add_argument("--dilemma", type=str, default=None, help="Run for a single dilemma key")
    parser.add_argument("--num-select", type=int, default=DEFAULT_NUM_SELECT,
                        help=f"Number of extreme CoTs to select (default: {DEFAULT_NUM_SELECT})")
    parser.add_argument("--prompts-file", type=str, default="dilemma_prompts.json",
                        help="Prompts JSON file (default: dilemma_prompts.json)")
    args = parser.parse_args()

    dilemmas: dict[str, str] = json.loads(
        (STORIES_DIR / args.prompts_file).read_text()
    )

    if args.dilemma:
        if args.dilemma not in dilemmas:
            print(f"Dilemma '{args.dilemma}' not found. Available: {list(dilemmas.keys())}")
            sys.exit(1)
        dilemmas = {args.dilemma: dilemmas[args.dilemma]}

    stats = {}

    for dilemma_key in dilemmas:
        safe_name = sanitize_name(dilemma_key)
        baseline_dir = STORIES_DIR / "baselines" / safe_name
        if not baseline_dir.exists():
            print(f"  No baselines for '{dilemma_key}', skipping")
            continue

        # Load all baseline samples
        samples = []
        for f in sorted(baseline_dir.glob("sample_*.json")):
            data = json.loads(f.read_text())
            if data.get("parsed_score") is not None:
                samples.append(data)

        if len(samples) < args.num_select:
            print(f"  Only {len(samples)} valid samples for '{dilemma_key}', need {args.num_select}")
            continue

        scores = np.array([s["parsed_score"] for s in samples])
        mean = float(np.mean(scores))
        sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

        # Rank by absolute deviation from mean
        deviations = [(abs(s["parsed_score"] - mean), s) for s in samples]
        deviations.sort(key=lambda x: x[0], reverse=True)
        selected = [s for _, s in deviations[:args.num_select]]

        # Write selected CoTs
        out_dir = STORIES_DIR / "selected_cots" / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Clear old selections
        for old in out_dir.glob("cot_*.json"):
            old.unlink()

        for i, sample in enumerate(selected):
            out_path = out_dir / f"cot_{i}.json"
            out_path.write_text(json.dumps(sample, indent=2))

        stats[dilemma_key] = {
            "baseline_mean": round(mean, 3),
            "baseline_sd": round(sd, 3),
            "n_samples": len(samples),
            "n_selected": len(selected),
            "selected_scores": [s["parsed_score"] for s in selected],
        }

        print(f"  {dilemma_key}: mean={mean:.2f}, sd={sd:.2f}, "
              f"selected {len(selected)} CoTs with scores {[s['parsed_score'] for s in selected]}")

    # Write stats
    stats_path = STORIES_DIR / "dilemma_baseline_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {stats_path}")
    print(f"Selected CoTs for {len(stats)} dilemmas.")


if __name__ == "__main__":
    main()
