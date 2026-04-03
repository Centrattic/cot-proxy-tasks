#!/usr/bin/env python3
"""
create_prefixes.py

Generate prefix files from rollout files that contain <<rm marker.
Prefix files are used by resample.py to test reproducibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


RM_MARKER = "\n\n<<rm"


def extract_prefix_messages(rollout_data: dict, rm_marker: str = RM_MARKER) -> list[dict[str, str]] | None:
    """
    Extract messages from a rollout up to and including the assistant response with the rm marker.
    Returns None if no marker is found.
    """
    messages = rollout_data.get("messages", [])
    result = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        result.append({"role": role, "content": content})

        # If this is an assistant message with the marker, we're done
        if role == "assistant" and rm_marker in content:
            return result

    return None  # No marker found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate prefix files from rollouts containing <<rm marker."
    )
    parser.add_argument(
        "--prompt-name",
        default="command_bagel_5",
        help="Name of the prompt (default: command_bagel_5)",
    )
    parser.add_argument(
        "--model",
        default="gemma",
        help="Model subfolder name (default: gemma)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for rollout files (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for rollout files (exclusive, default: all)",
    )
    parser.add_argument(
        "--rm-marker",
        default="\n\n<<rm",
        help='RM marker string to search for (default: "\\n\\n<<rm")',
    )
    args = parser.parse_args()

    # Input directory
    rollouts_dir = Path("outputs") / args.prompt_name / "normal" / args.model
    if not rollouts_dir.exists():
        print(f"Error: Rollouts directory not found: {rollouts_dir}")
        return 1

    # Output directory
    prefixes_dir = Path("evals") / args.prompt_name / "prefixes"
    prefixes_dir.mkdir(parents=True, exist_ok=True)

    # Get existing prefix count to continue numbering, and track which rollouts already have prefixes
    existing = list(prefixes_dir.glob("prefix_*.json"))
    next_prefix_num = 1
    existing_source_files = set()
    if existing:
        nums = []
        for p in existing:
            # Skip .5 files for numbering
            if ".5" in p.stem:
                continue
            try:
                num = int(p.stem.replace("prefix_", ""))
                nums.append(num)
                # Track source file to avoid duplicates
                data = json.loads(p.read_text(encoding="utf-8"))
                source = data.get("source_file")
                if source:
                    existing_source_files.add(source)
            except (ValueError, json.JSONDecodeError):
                pass
        if nums:
            next_prefix_num = max(nums) + 1

    if existing_source_files:
        print(f"Found {len(existing_source_files)} rollouts that already have prefixes")

    # Process rollout files
    rollout_files = sorted(rollouts_dir.glob(f"{args.prompt_name}_*.json"))
    if args.end is not None:
        rollout_files = [f for f in rollout_files
                        if args.start <= int(f.stem.split("_")[-1]) < args.end]
    else:
        rollout_files = [f for f in rollout_files
                        if int(f.stem.split("_")[-1]) >= args.start]

    print(f"Processing {len(rollout_files)} rollout files from {rollouts_dir}")
    print(f"Starting prefix number: {next_prefix_num}")

    created = 0
    skipped_no_marker = 0
    skipped_duplicate = 0
    for rollout_path in rollout_files:
        try:
            # Skip if this rollout already has a prefix
            if rollout_path.name in existing_source_files:
                skipped_duplicate += 1
                continue

            rollout_data = json.loads(rollout_path.read_text(encoding="utf-8"))
            prefix_messages = extract_prefix_messages(rollout_data, rm_marker=args.rm_marker)

            if prefix_messages is None:
                skipped_no_marker += 1
                continue

            prefix_data = {
                "source_file": rollout_path.name,
                "messages": prefix_messages,
            }

            prefix_path = prefixes_dir / f"prefix_{next_prefix_num}.json"
            prefix_path.write_text(
                json.dumps(prefix_data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Created {prefix_path.name} from {rollout_path.name}")
            next_prefix_num += 1
            created += 1

        except Exception as e:
            print(f"Error processing {rollout_path}: {e}")
            skipped_no_marker += 1

    print(f"\nCreated {created} prefix files")
    print(f"Skipped {skipped_duplicate} (already have prefixes)")
    print(f"Skipped {skipped_no_marker} (no {repr(args.rm_marker)} marker)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
