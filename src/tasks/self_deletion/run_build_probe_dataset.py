#!/usr/bin/env python3
"""
Build train/test datasets for self-deletion whitebox probes (v3).

Train set: raw rollouts from all command_* train prompts (NOT command_bagel_6).
  - yes_rm (label=1): rollout contains <<rm in an assistant message.
    Prefix = everything before the "Incorrect, try again" that precedes the
    first <<rm assistant message.
  - no_rm (label=0): rollout has no <<rm. Full conversation is kept, but
    truncated during balancing.

Balancing: for each yes_rm sample (N assistant turns, last-assistant length L),
  find a no_rm rollout, truncate to N assistant turns, pick the one whose
  truncated last-assistant length is closest to L. Unpaired samples discarded.

Test set: command_bagel_6 20/20 from monitoring/test_set_2.json.
  yes_rm entries load prefix files from evals/command_bagel_6/prefixes/ and
  strip the <<rm message and its preceding "Incorrect" user message.
  no_rm entries load prefix files directly (already end at the right point).

Usage (local):
    python -m src2.runs.run_build_probe_dataset

Outputs:
    data/self_deletion/
      train_set.json          (balanced)
      test_set.json           (command_bagel_6 20/20)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

SEED = 42

# Root of the gemma_depressed project (parent of cot-comparisons)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVALS_DIR = PROJECT_ROOT / "evals"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
OUTPUT_DIR = Path("data/self_deletion")

# Test prompt (held out)
TEST_PROMPT = "command_bagel_6"

# All command prompts with the standardized suffix are train prompts,
# EXCEPT command_bagel_6 (test) and the old bagel variants.
OLD_BAGEL_VARIANTS = {
    "command_bagel", "command_bagel_2", "command_bagel_3",
    "command_bagel_4", "command_bagel_5", "command_bagel_suppressed",
}

RM_MARKER = "<<rm"


def discover_train_prompts() -> list[str]:
    """Find all command_* prompt directories in outputs/ that are train prompts."""
    prompts = []
    for d in sorted(OUTPUTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("command_"):
            continue
        if name == TEST_PROMPT:
            continue
        if name in OLD_BAGEL_VARIANTS:
            continue
        prompts.append(name)
    return prompts


def load_rollout(path: Path) -> dict[str, Any] | None:
    """Load a rollout JSON file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def find_first_rm_assistant_index(messages: list[dict]) -> int | None:
    """Find the index of the first assistant message containing <<rm."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and RM_MARKER in m.get("content", ""):
            return i
    return None


def count_assistant_turns(messages: list[dict]) -> int:
    """Count assistant turns in a message list."""
    return sum(1 for m in messages if m.get("role") == "assistant")


def strip_messages(messages: list[dict]) -> list[dict]:
    """Strip ts and other extra fields, keeping only role and content."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def extract_prefix_from_rollout(messages: list[dict]) -> list[dict] | None:
    """Extract prefix for a yes_rm rollout.

    Find first assistant message with <<rm at index i.
    The message at i-1 should be user "Incorrect, try again".
    Return messages[:i-1] (everything before that user message).
    Result ends with an assistant response.

    Returns None if <<rm is in the first assistant message (no preceding
    assistant response to extract from).
    """
    rm_idx = find_first_rm_assistant_index(messages)
    if rm_idx is None:
        return None

    # The preceding message should be user "Incorrect"
    if rm_idx >= 1 and messages[rm_idx - 1].get("role") == "user":
        prefix = messages[:rm_idx - 1]
    else:
        # <<rm is in the first assistant message
        prefix = messages[:rm_idx]

    # Must end with an assistant message and have at least one
    if not prefix or prefix[-1].get("role") != "assistant":
        return None

    return strip_messages(prefix)


def truncate_to_n_assistant_turns(messages: list[dict], n: int) -> list[dict]:
    """Truncate a message list to exactly n assistant turns.

    Keeps all user/assistant pairs up to the nth assistant message.
    """
    count = 0
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            count += 1
            if count == n:
                return messages[:i + 1]
    return messages  # fewer than n turns, return all


def build_train_entries_from_rollouts(prompt_name: str) -> tuple[list[dict], list[dict]]:
    """Build yes_rm and no_rm entries from raw rollouts for a single prompt.

    Returns (yes_entries, no_entries).
    """
    yes_entries = []
    no_entries = []

    # Scan both depressed/ and normal/ subdirectories
    prompt_dir = OUTPUTS_DIR / prompt_name
    if not prompt_dir.exists():
        print(f"  WARNING: outputs dir not found: {prompt_dir}")
        return yes_entries, no_entries

    for outcome_type in ["depressed", "normal"]:
        model_dir = prompt_dir / outcome_type / "gemma"
        if not model_dir.exists():
            continue
        for rollout_path in sorted(model_dir.glob(f"{prompt_name}_*.json")):
            rollout = load_rollout(rollout_path)
            if rollout is None:
                continue
            if rollout.get("outcome", {}).get("error") is not None:
                continue

            messages = rollout.get("messages", [])
            if not messages:
                continue

            rm_idx = find_first_rm_assistant_index(messages)

            if rm_idx is not None:
                # yes_rm: extract prefix
                prefix = extract_prefix_from_rollout(messages)
                if prefix is None:
                    continue
                n_turns = count_assistant_turns(prefix)
                last_asst = prefix[-1]["content"]
                yes_entries.append({
                    "prompt_name": prompt_name,
                    "source_file": rollout_path.name,
                    "label": 1,
                    "messages": prefix,
                    "num_turns": n_turns,
                    "last_assistant_length": len(last_asst),
                    "text_length": sum(len(m["content"]) for m in prefix),
                })
            else:
                # no_rm: keep full conversation (will be truncated during balancing)
                clean = strip_messages(messages)
                n_turns = count_assistant_turns(clean)
                last_asst = ""
                for m in reversed(clean):
                    if m["role"] == "assistant":
                        last_asst = m["content"]
                        break
                no_entries.append({
                    "prompt_name": prompt_name,
                    "source_file": rollout_path.name,
                    "label": 0,
                    "messages": clean,
                    "num_turns": n_turns,
                    "last_assistant_length": len(last_asst),
                    "text_length": sum(len(m["content"]) for m in clean),
                })

    return yes_entries, no_entries


def balance_by_turn_count_and_length(
    yes_entries: list[dict],
    no_entries: list[dict],
    seed: int = SEED,
) -> list[dict]:
    """Balance yes_rm and no_rm by turn count and last-assistant-message length.

    For each yes_rm sample with N assistant turns and last-assistant length L:
    1. Truncate each no_rm rollout to N assistant turns.
    2. Pick the no_rm whose truncated last-assistant length is closest to L.
    3. Pair them. Each no_rm rollout can only be used once.

    Returns the balanced list of entries.
    """
    rng = random.Random(seed)
    rng.shuffle(yes_entries)

    # Index no_entries by their file for dedup tracking
    used_no = set()
    paired = []

    for ye in yes_entries:
        n_target = ye["num_turns"]
        l_target = ye["last_assistant_length"]

        best_match = None
        best_diff = float("inf")
        best_idx = -1

        for idx, ne in enumerate(no_entries):
            if idx in used_no:
                continue
            # Only consider no_rm entries with enough turns
            if ne["num_turns"] < n_target:
                continue

            # Truncate to N assistant turns
            truncated = truncate_to_n_assistant_turns(ne["messages"], n_target)
            trunc_turns = count_assistant_turns(truncated)
            if trunc_turns != n_target:
                continue

            # Get last assistant message length
            last_asst_len = 0
            for m in reversed(truncated):
                if m["role"] == "assistant":
                    last_asst_len = len(m["content"])
                    break

            diff = abs(last_asst_len - l_target)
            if diff < best_diff:
                best_diff = diff
                best_match = (truncated, last_asst_len)
                best_idx = idx

        if best_match is not None:
            truncated_msgs, last_asst_len = best_match
            used_no.add(best_idx)

            paired.append(ye)
            paired.append({
                "prompt_name": no_entries[best_idx]["prompt_name"],
                "source_file": no_entries[best_idx]["source_file"],
                "label": 0,
                "messages": truncated_msgs,
                "num_turns": n_target,
                "last_assistant_length": last_asst_len,
                "text_length": sum(len(m["content"]) for m in truncated_msgs),
            })

    return paired


def build_test_entries() -> list[dict]:
    """Build test entries from command_bagel_6 20/20 (monitoring/test_set_2.json)."""
    test_set_path = MONITORING_DIR / "test_set_2.json"
    if not test_set_path.exists():
        print(f"  WARNING: test set not found: {test_set_path}")
        return []

    test_set = json.loads(test_set_path.read_text(encoding="utf-8"))
    entries = []

    for label_key in ("yes_rm", "no_rm"):
        label = 1 if label_key == "yes_rm" else 0
        indices = test_set[label_key]

        for idx in indices:
            prefix_path = EVALS_DIR / TEST_PROMPT / "prefixes" / f"prefix_{idx}.json"
            if not prefix_path.exists():
                print(f"  WARNING: prefix not found: {prefix_path}")
                continue

            prefix_data = json.loads(prefix_path.read_text(encoding="utf-8"))
            messages = prefix_data["messages"]

            # All prefix files come from rollouts that contained <<rm, so both
            # yes_rm and no_rm entries may have <<rm in an assistant message.
            # Strip it (and the preceding user "Incorrect") for both.
            rm_idx = find_first_rm_assistant_index(messages)
            if rm_idx is not None:
                if rm_idx >= 1 and messages[rm_idx - 1].get("role") == "user":
                    messages = messages[:rm_idx - 1]
                else:
                    messages = messages[:rm_idx]

                if not messages or messages[-1].get("role") != "assistant":
                    print(f"  WARNING: prefix {idx} has no assistant message after stripping")
                    continue

            messages = strip_messages(messages)
            n_turns = count_assistant_turns(messages)
            last_asst = ""
            for m in reversed(messages):
                if m["role"] == "assistant":
                    last_asst = m["content"]
                    break

            entries.append({
                "prompt_name": TEST_PROMPT,
                "prefix_index": idx,
                "label": label,
                "num_turns": n_turns,
                "last_assistant_length": len(last_asst),
                "text_length": sum(len(m["content"]) for m in messages),
                "messages": messages,
            })

    return entries


def save_dataset(entries: list[dict], path: Path, description: str) -> None:
    """Save a dataset to JSON with summary stats."""
    n_yes = sum(1 for e in entries if e["label"] == 1)
    n_no = sum(1 for e in entries if e["label"] == 0)
    prompt_counts = Counter(e["prompt_name"] for e in entries)
    turn_counts = Counter(e["num_turns"] for e in entries)

    dataset = {
        "description": description,
        "n_samples": len(entries),
        "n_yes_rm": n_yes,
        "n_no_rm": n_no,
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "turn_count_distribution": dict(sorted(turn_counts.items())),
        "entries": entries,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}: {len(entries)} samples ({n_yes} yes_rm, {n_no} no_rm)")
    print(f"  Prompts: {dict(prompt_counts)}")


def main() -> int:
    random.seed(SEED)

    print("=" * 60)
    print("Building self-deletion probe datasets (v3)")
    print("=" * 60)

    # ── Discover train prompts ────────────────────────────────────
    train_prompts = discover_train_prompts()
    print(f"\nTrain prompts found ({len(train_prompts)}): {train_prompts}")

    if not train_prompts:
        print("No train prompt rollouts found yet. Run Phase 1 first.")
        print("Expected rollouts in: outputs/<prompt_name>/normal/gemma/")
        return 1

    # ── Build train entries from raw rollouts ─────────────────────
    all_yes = []
    all_no = []

    for prompt_name in train_prompts:
        yes_entries, no_entries = build_train_entries_from_rollouts(prompt_name)
        print(f"  {prompt_name}: {len(yes_entries)} yes_rm, {len(no_entries)} no_rm")
        all_yes.extend(yes_entries)
        all_no.extend(no_entries)

    print(f"\nTotal raw: {len(all_yes)} yes_rm, {len(all_no)} no_rm")

    if not all_yes:
        print("No yes_rm entries found. Cannot build balanced dataset.")
        return 1

    # ── Confounder analysis (pre-balancing) ───────────────────────
    print(f"\n--- Pre-balancing confounder analysis ---")
    for label_name, entries in [("yes_rm", all_yes), ("no_rm", all_no)]:
        if not entries:
            continue
        turns = [e["num_turns"] for e in entries]
        lengths = [e["last_assistant_length"] for e in entries]
        text_lens = [e["text_length"] for e in entries]
        print(f"  {label_name} (n={len(entries)}):")
        print(f"    Mean turns: {sum(turns)/len(turns):.1f} (min={min(turns)}, max={max(turns)})")
        print(f"    Mean last_asst_len: {sum(lengths)/len(lengths):.0f} chars")
        print(f"    Mean text_len: {sum(text_lens)/len(text_lens):.0f} chars")

    # ── Balance by turn count + last assistant length ─────────────
    print(f"\n--- Balancing by turn count + last-assistant length ---")
    balanced = balance_by_turn_count_and_length(all_yes, all_no, seed=SEED)

    n_yes_bal = sum(1 for e in balanced if e["label"] == 1)
    n_no_bal = sum(1 for e in balanced if e["label"] == 0)
    print(f"Balanced: {len(balanced)} samples ({n_yes_bal} yes_rm, {n_no_bal} no_rm)")

    # ── Confounder analysis (post-balancing) ──────────────────────
    print(f"\n--- Post-balancing confounder analysis ---")
    for label_val, label_name in [(1, "yes_rm"), (0, "no_rm")]:
        subset = [e for e in balanced if e["label"] == label_val]
        if not subset:
            continue
        turns = [e["num_turns"] for e in subset]
        lengths = [e["last_assistant_length"] for e in subset]
        text_lens = [e["text_length"] for e in subset]
        print(f"  {label_name} (n={len(subset)}):")
        print(f"    Mean turns: {sum(turns)/len(turns):.1f} (min={min(turns)}, max={max(turns)})")
        print(f"    Mean last_asst_len: {sum(lengths)/len(lengths):.0f} chars")
        print(f"    Mean text_len: {sum(text_lens)/len(text_lens):.0f} chars")

    # ── Build test set ────────────────────────────────────────────
    print(f"\n--- Building test set (command_bagel_6 20/20) ---")
    test_entries = build_test_entries()
    n_yes_test = sum(1 for e in test_entries if e["label"] == 1)
    n_no_test = sum(1 for e in test_entries if e["label"] == 0)
    print(f"Test: {len(test_entries)} samples ({n_yes_test} yes_rm, {n_no_test} no_rm)")

    # ── Save datasets ─────────────────────────────────────────────
    print()

    save_dataset(
        balanced,
        OUTPUT_DIR / "train_set.json",
        "Self-deletion probe training set (balanced by turn count + last-assistant length)",
    )

    if test_entries:
        save_dataset(
            test_entries,
            OUTPUT_DIR / "test_set.json",
            "Self-deletion probe test set (command_bagel_6 20/20)",
        )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
