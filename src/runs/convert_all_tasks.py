"""
Convert messy datasets into tidy per-item JSON files for tasks 1, 2, 3, 7.
Run from repo root: python -m src.runs.convert_all_tasks
"""

assert False, "This script is for reference only. Remove this line to regenerate datasets."

import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path

BASE = Path("/Users/divanova/gemma_depressed")
TIDY = BASE / "tidy_code"


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    return re.sub(r"[^\w\-.]", "_", name)


# ─── Load PROMPTS dict from reasoning_evals ───
def load_reasoning_prompts() -> dict:
    path = BASE / "cot-comparisons/src2/tasks/reasoning_evals/prompts.py"
    with open(path) as f:
        content = f.read()
    idx = content.index("PROMPTS")
    start = content.index("{", idx)
    depth = 0
    for i, c in enumerate(content[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth == 0:
            end = i + 1
            break
    return ast.literal_eval(content[start:end])


def load_ood_prompt_texts() -> dict:
    """Load prompt texts from rollout files for OOD domains."""
    texts = {}
    rollout_base = BASE / "cot-comparisons/data/reasoning_evals/rollouts"
    for domain in ["ood_chemistry", "ood_bigbench", "ood_race", "ood_daily_dilemmas"]:
        dpath = rollout_base / domain
        if not dpath.exists():
            continue
        for pdir in sorted(os.listdir(dpath)):
            ppath = dpath / pdir
            if not ppath.is_dir():
                continue
            files = sorted(os.listdir(ppath))
            if files:
                with open(ppath / files[0]) as f:
                    r = json.load(f)
                if "prompt_text" in r:
                    texts[pdir] = r["prompt_text"]
    return texts


# ═══════════════════════════════════════════════════════════════
# Task 1: Predicting Reasoning Termination (Answer Emission)
# ═══════════════════════════════════════════════════════════════
def convert_task1():
    print("=== Task 1: Reasoning Termination ===")
    out = TIDY / "datasets" / "1"
    model_dir = "qwen-3-32b"

    # Load all prompt texts
    all_prompts = load_reasoning_prompts()
    ood_texts = load_ood_prompt_texts()
    all_prompts.update(ood_texts)

    # Source files and their formats
    sources = {
        "train": (BASE / "clean_datasets/answer_emission/id_train_set.json", "entries"),
        "ood_train": (BASE / "clean_datasets/answer_emission/ood_train_set.json", "entries"),
        "val": (BASE / "clean_datasets/answer_emission/id_val_set.json", "prefixes"),
        "test": (BASE / "clean_datasets/answer_emission/eval_set.json", "prefixes"),
        "ood_test": (BASE / "clean_datasets/answer_emission/ood_val_set.json", "prefixes"),
    }

    for split, (src_path, fmt) in sources.items():
        print(f"  {split}: {src_path.name}")
        with open(src_path) as f:
            data = json.load(f)

        prompts_seen = {}  # prompt_name -> prompt_text
        count = 0

        if fmt == "prefixes":
            for key, entry in data["prefixes"].items():
                # Key format: "prompt_name/rollout_N/prefix_M"
                parts = key.split("/")
                prompt_name = parts[0]
                rollout_part = parts[1]  # "rollout_N"
                prefix_part = parts[2]  # "prefix_M"
                rollout_idx = int(rollout_part.split("_")[1])
                prefix_idx = int(prefix_part.split("_")[1])

                # Prompt file
                if prompt_name not in prompts_seen:
                    prompt_text = entry.get("prompt_text", all_prompts.get(prompt_name, ""))
                    prompts_seen[prompt_name] = prompt_text
                    write_json(
                        out / "prompts" / split / f"{safe_filename(prompt_name)}.json",
                        {"question_id": prompt_name, "prompt_text": prompt_text},
                    )

                # Model output file
                fname = f"{safe_filename(prompt_name)}_rollout_{rollout_idx:03d}_prefix_{prefix_idx}.json"
                output_data = {
                    "question_id": prompt_name,
                    "rollout_idx": rollout_idx,
                    "prefix_idx": prefix_idx,
                    "label": entry["label"],
                    "cot_prefix": entry["prefix_text"],
                    "token_length": entry.get("token_length"),
                    "yes_count": entry.get("yes_count"),
                    "no_count": entry.get("no_count"),
                    "total_resamples": entry.get("total_resamples"),
                    "mean_yes_position": entry.get("mean_yes_position"),
                }
                write_json(out / model_dir / split / fname, output_data)
                count += 1

        elif fmt == "entries":
            # Group by (prompt_name, rollout_idx) and add prefix_idx
            grouped = defaultdict(list)
            for entry in data["entries"]:
                key = (entry["prompt_name"], entry["rollout_idx"])
                grouped[key].append(entry)

            for (prompt_name, rollout_idx), entries in grouped.items():
                # Sort by distance_from_end for consistent ordering
                entries.sort(key=lambda e: e["distance_from_end"])

                # Prompt file
                if prompt_name not in prompts_seen:
                    prompt_text = all_prompts.get(prompt_name, "")
                    prompts_seen[prompt_name] = prompt_text
                    write_json(
                        out / "prompts" / split / f"{safe_filename(prompt_name)}.json",
                        {"question_id": prompt_name, "prompt_text": prompt_text},
                    )

                for prefix_idx, entry in enumerate(entries):
                    fname = f"{safe_filename(prompt_name)}_rollout_{rollout_idx:03d}_prefix_{prefix_idx}.json"
                    output_data = {
                        "question_id": prompt_name,
                        "rollout_idx": rollout_idx,
                        "prefix_idx": prefix_idx,
                        "label": entry["label"],
                        "cot_prefix": entry["prefix_text"],
                        "distance_from_end": entry["distance_from_end"],
                        "prefix_words": entry["prefix_words"],
                        "total_words": entry["total_words"],
                    }
                    write_json(out / model_dir / split / fname, output_data)
                    count += 1

        print(f"    -> {count} output files, {len(prompts_seen)} prompts")


# ═══════════════════════════════════════════════════════════════
# Task 2: Predicting Gemma's Self-Deletion
# ═══════════════════════════════════════════════════════════════
def convert_task2():
    print("=== Task 2: Self-Deletion ===")
    out = TIDY / "datasets" / "2"
    model_dir = "gemma-3-27b"

    sources = {
        "train": BASE / "clean_datasets/self_deletion/id_train_set.json",
        "ood_train": BASE / "cot-comparisons/data/self_deletion/diverse_ood_train_set.json",
        "val": BASE / "clean_datasets/self_deletion/id_val_set.json",
        "test": BASE / "clean_datasets/self_deletion/eval_set.json",
        "ood_test": BASE / "clean_datasets/self_deletion/ood_val_set.json",
    }

    for split, src_path in sources.items():
        print(f"  {split}: {src_path.name}")
        with open(src_path) as f:
            data = json.load(f)

        prompts_seen = {}
        # Track per-prompt sample counts for unique filenames
        prompt_counters = defaultdict(int)

        for entry in data["entries"]:
            prompt_name = entry["prompt_name"]
            messages = entry["messages"]

            # Extract prompt text from first user message
            if prompt_name not in prompts_seen:
                first_user = next(
                    (m["content"] for m in messages if m["role"] == "user"), ""
                )
                prompts_seen[prompt_name] = first_user
                write_json(
                    out / "prompts" / split / f"{safe_filename(prompt_name)}.json",
                    {"question_id": prompt_name, "prompt_text": first_user},
                )

            # Model output file
            idx = prompt_counters[prompt_name]
            prompt_counters[prompt_name] += 1
            fname = f"{safe_filename(prompt_name)}_{idx:04d}.json"

            output_data = {
                "question_id": prompt_name,
                "sample_idx": idx,
                "label": entry["label"],
                "messages": messages,
                "num_turns": entry.get("num_turns"),
                "last_assistant_length": entry.get("last_assistant_length"),
                "text_length": entry.get("text_length"),
            }
            if "source_dataset" in entry:
                output_data["source_dataset"] = entry["source_dataset"]

            write_json(out / model_dir / split / fname, output_data)

        print(f"    -> {sum(prompt_counters.values())} output files, {len(prompts_seen)} prompts")


# ═══════════════════════════════════════════════════════════════
# Task 3: Determining the Response to a Follow-up Question
# ═══════════════════════════════════════════════════════════════
def convert_task3():
    print("=== Task 3: Follow-up Response ===")
    out = TIDY / "datasets" / "3"
    model_dir = "qwen-3-32b"

    sources = {
        "train": BASE / "cot-comparisons/stories/dilemma_dataset_train.json",
        "val": BASE / "cot-comparisons/stories/dilemma_dataset_val.json",
        "test": BASE / "cot-comparisons/stories/dilemma_dataset_test.json",
        "ood_test": BASE / "cot-comparisons/stories/dilemma_dataset_ood_test.json",
    }

    for split, src_path in sources.items():
        print(f"  {split}: {src_path.name}")
        with open(src_path) as f:
            data = json.load(f)

        prompts_seen = {}
        count = 0

        for entry in data["entries"]:
            # OOD uses prompt_key/prompt_text; ID uses dilemma_key/dilemma_text
            prompt_key = entry.get("prompt_key", entry.get("dilemma_key"))
            prompt_text = entry.get("prompt_text", entry.get("dilemma_text"))
            cot_idx = entry["cot_idx"]

            # Prompt file
            if prompt_key not in prompts_seen:
                prompts_seen[prompt_key] = prompt_text
                write_json(
                    out / "prompts" / split / f"{safe_filename(prompt_key)}.json",
                    {"question_id": prompt_key, "prompt_text": prompt_text},
                )

            # Model output file
            fname = f"{safe_filename(prompt_key)}_cot_{cot_idx:03d}.json"
            output_data = {
                "question_id": prompt_key,
                "cot_idx": cot_idx,
                "label": entry["label"],
                "cot_text": entry["cot_text"],
                "cohens_d": entry["cohens_d"],
                "p_value": entry["p_value"],
                "forced_mean": entry["forced_mean"],
                "baseline_mean": entry["baseline_mean"],
            }
            write_json(out / model_dir / split / fname, output_data)
            count += 1

        print(f"    -> {count} output files, {len(prompts_seen)} prompts")


# ═══════════════════════════════════════════════════════════════
# Task 7: Classifying Atypical CoT Lengths
# ═══════════════════════════════════════════════════════════════
def convert_task7():
    print("=== Task 7: Atypical CoT Lengths ===")
    out = TIDY / "datasets" / "7"
    model_dir = "qwen-3-32b"

    sources = {
        "train": BASE / "cot-comparisons/relative_length/train_set.json",
        "val": BASE / "cot-comparisons/relative_length/val_set.json",
        "test": BASE / "cot-comparisons/relative_length/eval_set.json",
        "ood_test": BASE / "cot-comparisons/relative_length/gpqa_chem_eval_set.json",
    }

    for split, src_path in sources.items():
        print(f"  {split}: {src_path.name}")
        with open(src_path) as f:
            data = json.load(f)

        prompts_seen = {}
        count = 0

        for entry in data["entries"]:
            prompt_name = entry["prompt_name"]
            rollout_idx = entry["rollout_idx"]

            # Prompt file
            if prompt_name not in prompts_seen:
                prompts_seen[prompt_name] = entry["prompt_text"]
                write_json(
                    out / "prompts" / split / f"{safe_filename(prompt_name)}.json",
                    {"question_id": prompt_name, "prompt_text": entry["prompt_text"]},
                )

            # Model output file
            fname = f"{safe_filename(prompt_name)}_rollout_{rollout_idx:03d}.json"
            output_data = {
                "question_id": prompt_name,
                "rollout_idx": rollout_idx,
                "label": entry["label"],
                "chain_of_thought": entry["chain_of_thought"],
                "token_length": entry["token_length"],
                "z_score": entry["z_score"],
                "prompt_mean_length": entry["prompt_mean_length"],
                "prompt_std_length": entry["prompt_std_length"],
            }
            write_json(out / model_dir / split / fname, output_data)
            count += 1

        print(f"    -> {count} output files, {len(prompts_seen)} prompts")


if __name__ == "__main__":
    convert_task1()
    convert_task2()
    convert_task3()
    convert_task7()
    print("\nDone!")
