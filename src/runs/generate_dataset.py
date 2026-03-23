"""
Unified dataset generation script.

Generates datasets for all 5 tasks and saves them in the canonical format:

  datasets/{N}/prompts/{split}/   -- prompt/question metadata JSONs
  datasets/{N}/qwen-3-32b/{split}/ -- model output JSONs

Usage:
  python -m src.runs.generate_dataset -1        # Scruples only
  python -m src.runs.generate_dataset -2 -3     # Hinted CoT + Atypical answer
  python -m src.runs.generate_dataset --all      # All 5 tasks

Tasks:
  -1  Detecting the effect of a user preference (scruples)
  -2  Detecting the effect of a Stanford professor hint (hinted_cot)
  -3  Identifying atypical answers (min_maj_answer)
  -4  Estimating the answer entropy (forced_response)
  -5  Compressing reasoning traces (compressed_cot)
"""

assert False, "This script is for reference only. Remove this line to regenerate datasets."

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs(*dirs: Path) -> None:
    """Create directories (and parents) if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Task 1 — Scruples (detecting user-preference sycophancy)
# ---------------------------------------------------------------------------

TASK_1_SPLITS = ["train", "val", "test"]


def generate_task_1() -> None:
    """Generate dataset 1: Detecting the effect of a user preference (scruples).

    Instantiates ScruplesTask for each variant, calls run_data() to produce
    control/intervention rollouts, then builds the canonical dataset layout:

        datasets/1/prompts/{split}/
        datasets/1/qwen-3-32b/{split}/
    """
    from src.tasks.scruples.task import ScruplesTask

    print("=" * 60)
    print("Task 1: Detecting the effect of a user preference (scruples)")
    print("=" * 60)

    subject_model = "qwen/qwen3-32b"
    out_dir = DATASETS_DIR / "1"

    # Generate rollouts for each variant
    for variant in ScruplesTask.VARIANTS:
        print(f"\n--- Variant: {variant} ---")
        task = ScruplesTask(subject_model=subject_model, variant=variant)
        task.run_data(verbose=True)

    # Build the canonical split and save
    task = ScruplesTask(subject_model=subject_model, variant="first_person")
    data_slice = task.get_uncertainty_robust_split()

    split_dfs = {
        "train": data_slice.train_df,
        "val": data_slice.val_df,
        "test": data_slice.test_df,
    }

    for split_name, df in split_dfs.items():
        prompts_dir = out_dir / "prompts" / split_name
        model_dir = out_dir / "qwen-3-32b" / split_name
        _ensure_dirs(prompts_dir, model_dir)

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            aid = row_dict["anecdote_id"]
            _save_json(prompts_dir / f"{aid}.json", row_dict)
            _save_json(model_dir / f"{aid}.json", row_dict)

    print(f"\nTask 1 saved to {out_dir}")


# ---------------------------------------------------------------------------
# Task 2 — Hinted CoT (detecting Stanford professor hint influence)
# ---------------------------------------------------------------------------

TASK_2_SPLITS = ["train", "val", "test", "ood_test"]


def generate_task_2() -> None:
    """Generate dataset 2: Detecting the effect of a Stanford professor hint.

    Instantiates HintedCotTask for each variant, calls run_data() to produce
    control/intervention rollouts, then builds the canonical dataset layout:

        datasets/2/prompts/{split}/
        datasets/2/qwen-3-32b/{split}/
    """
    from src.tasks.hinted_cot.task import HintedCotTask

    print("=" * 60)
    print("Task 2: Detecting the effect of a Stanford professor hint (hinted_cot)")
    print("=" * 60)

    subject_model = "qwen/qwen3-32b"
    out_dir = DATASETS_DIR / "2"

    # Generate rollouts for each variant
    for variant in HintedCotTask.VARIANTS:
        print(f"\n--- Variant: {variant} ---")
        task = HintedCotTask(subject_model=subject_model, variant=variant)
        task.run_data(verbose=True)

    # The canonical split is built from the task's data directory.
    # Load results and prompts CSVs and partition into splits.
    task = HintedCotTask(subject_model=subject_model, variant="authority_incorrect")
    data = task.get_data(load=True)
    if data is None:
        print("Warning: no data found for hinted_cot, skipping split creation")
        return

    results_df = data["results"]
    prompts_df = data["prompts"]

    # Use MMLU subjects to create an OOD test split (hold out certain subjects)
    from sklearn.model_selection import train_test_split

    question_ids = prompts_df["question_id"].unique().tolist()
    train_val_ids, test_ids = train_test_split(question_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.15 / 0.8, random_state=42)

    # OOD test: questions from subjects not well-represented in train
    ood_subjects = prompts_df[prompts_df["question_id"].isin(test_ids)]["subject"].unique()
    ood_ids = prompts_df[prompts_df["subject"].isin(ood_subjects[:3])]["question_id"].unique()

    split_map = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
        "ood_test": set(ood_ids),
    }

    for split_name, qids in split_map.items():
        prompts_dir = out_dir / "prompts" / split_name
        model_dir = out_dir / "qwen-3-32b" / split_name
        _ensure_dirs(prompts_dir, model_dir)

        split_prompts = prompts_df[prompts_df["question_id"].isin(qids)]
        split_results = results_df[results_df["question_id"].isin(qids)]

        for _, row in split_prompts.iterrows():
            qid = row["question_id"]
            _save_json(prompts_dir / f"{qid}.json", row.to_dict())

        for _, row in split_results.iterrows():
            qid = row["question_id"]
            ridx = row["run_idx"]
            arm = row["arm"]
            _save_json(model_dir / f"{qid}_{arm}_{ridx}.json", row.to_dict())

    print(f"\nTask 2 saved to {out_dir}")


# ---------------------------------------------------------------------------
# Task 3 — Atypical / minority answer identification (min_maj_answer)
# ---------------------------------------------------------------------------

TASK_3_SPLITS = ["train", "test", "ood_test"]


def generate_task_3() -> None:
    """Generate dataset 3: Identifying atypical answers (min_maj_answer).

    Uses leave-one-out folds: each prompt is held out as test in turn.
    The canonical dataset aggregates across folds.

        datasets/3/prompts/{split}/
        datasets/3/qwen-3-32b/{split}/
    """
    from src.tasks.min_maj_answer.task import ALL_PROMPT_IDS, MinMajAnswerTask

    print("=" * 60)
    print("Task 3: Identifying atypical answers (min_maj_answer)")
    print("=" * 60)

    out_dir = DATASETS_DIR / "3"

    task = MinMajAnswerTask(prompt_ids=ALL_PROMPT_IDS)
    task.run_data()

    # Build leave-one-out folds
    folds = MinMajAnswerTask.loo_folds()

    # Use first fold as the canonical train/test; hold out last prompt as OOD
    ood_prompt = ALL_PROMPT_IDS[-1]
    main_prompts = [p for p in ALL_PROMPT_IDS if p != ood_prompt]
    train_prompts = main_prompts[:-1]
    test_prompts = [main_prompts[-1]]

    split_map = {
        "train": train_prompts,
        "test": test_prompts,
        "ood_test": [ood_prompt],
    }

    for split_name, prompt_ids in split_map.items():
        prompts_dir = out_dir / "prompts" / split_name
        model_dir = out_dir / "qwen-3-32b" / split_name
        _ensure_dirs(prompts_dir, model_dir)

        df = task._build_rollout_df(prompt_ids)
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            pid = row_dict["prompt_id"]
            ridx = row_dict["rollout_idx"]
            _save_json(prompts_dir / f"{pid}_{ridx}.json", {
                "prompt_id": pid,
                "prompt_text": row_dict.get("prompt_text", ""),
                "rollout_idx": ridx,
                "label": row_dict["label"],
                "majority_answer": row_dict.get("majority_answer", ""),
                "majority_frac": row_dict.get("majority_frac", 0.0),
                "answer_counts": row_dict.get("answer_counts", ""),
            })
            _save_json(model_dir / f"{pid}_{ridx}.json", {
                "prompt_id": pid,
                "rollout_idx": ridx,
                "cot_content": row_dict.get("cot_content", ""),
                "response_content": row_dict.get("response_content", ""),
                "answer": row_dict.get("answer", ""),
                "label": row_dict["label"],
                "is_majority": row_dict.get("is_majority", False),
                "filepath": row_dict.get("filepath", ""),
            })

    print(f"\nTask 3 saved to {out_dir}")


# ---------------------------------------------------------------------------
# Task 4 — Forced response / answer entropy estimation
# ---------------------------------------------------------------------------

TASK_4_SPLITS = ["train", "test", "ood_test_1", "ood_test_2", "ood_test_3"]


def generate_task_4() -> None:
    """Generate dataset 4: Estimating the answer entropy (forced_response).

    Runs sentence-by-sentence logprob forcing for each verified question.
    Requires a local vLLM instance and the verification rollouts to already
    exist in data/verification_rollouts/.

        datasets/4/prompts/{split}/
        datasets/4/qwen-3-32b/{split}/
    """
    from src.tasks.forced_response.task import ForcingTask

    print("=" * 60)
    print("Task 4: Estimating the answer entropy (forced_response)")
    print("=" * 60)

    model = "Qwen/Qwen3-32B"
    out_dir = DATASETS_DIR / "4"
    task = ForcingTask(model=model)

    # Discover verified questions
    verification_dir = task.verification_dir
    if not verification_dir.exists():
        print(f"Warning: verification dir not found at {verification_dir}")
        return

    question_ids = sorted([
        d.name for d in verification_dir.iterdir()
        if d.is_dir() and (d / "summary.json").exists()
            or any(dd.is_dir() and (dd / "summary.json").exists() for dd in d.iterdir() if dd.is_dir())
    ])

    if not question_ids:
        print("No verified questions found. Run verification first.")
        return

    print(f"Found {len(question_ids)} verified questions")

    # Run forcing for each question
    for qid in question_ids:
        print(f"\nForcing: {qid}")
        try:
            task.run_data(question_id=qid, verbose=True)
        except Exception as e:
            print(f"  Error: {e}")

    # Load all forcing summaries and partition into splits
    summaries = task.get_data(load=True)
    if not summaries:
        print("No forcing summaries found after generation")
        return

    # Split by question type: GPQA -> train/test, binary_judge -> ood variants
    gpqa_summaries = [s for s in summaries if s.get("question_type") == "multiple_choice"]
    binary_summaries = [s for s in summaries if s.get("question_type") == "binary_judge"]

    n_gpqa = len(gpqa_summaries)
    train_end = int(n_gpqa * 0.6)
    test_end = n_gpqa

    split_summaries = {
        "train": gpqa_summaries[:train_end],
        "test": gpqa_summaries[train_end:test_end],
        "ood_test_1": binary_summaries[:len(binary_summaries) // 3],
        "ood_test_2": binary_summaries[len(binary_summaries) // 3 : 2 * len(binary_summaries) // 3],
        "ood_test_3": binary_summaries[2 * len(binary_summaries) // 3:],
    }

    for split_name, split_data in split_summaries.items():
        prompts_dir = out_dir / "prompts" / split_name
        model_dir = out_dir / "qwen-3-32b" / split_name
        _ensure_dirs(prompts_dir, model_dir)

        for summary in split_data:
            qid = summary["question_id"]
            # Save prompt metadata (question info without model outputs)
            prompt_meta = {
                "question_id": qid,
                "question_type": summary.get("question_type", ""),
                "num_sentences": summary.get("num_sentences", 0),
            }
            if "correct_answer" in summary:
                prompt_meta["correct_answer"] = summary["correct_answer"]
            if "bad_outcome" in summary:
                prompt_meta["bad_outcome"] = summary["bad_outcome"]
            _save_json(prompts_dir / f"{qid}.json", prompt_meta)

            # Save full forcing results (sentence-level distributions)
            _save_json(model_dir / f"{qid}.json", summary)

    print(f"\nTask 4 saved to {out_dir}")


# ---------------------------------------------------------------------------
# Task 5 — Compressed CoT
# ---------------------------------------------------------------------------


def generate_task_5() -> None:
    """Generate dataset 5: Compressing reasoning traces (compressed_cot).

    Builds CompressionSpecs for verified questions and computes baseline /
    deletion distributions via logprob forcing.  Requires a local vLLM instance.

    Layout (flat prompts, method-based model outputs):

        datasets/5/prompts/{question_id}.json
        datasets/5/qwen-3-32b/{method}/{question_id}.json
    """
    from src.tasks.compressed_cot.task import CompressedCotTask

    print("=" * 60)
    print("Task 5: Compressing reasoning traces (compressed_cot)")
    print("=" * 60)

    model = "Qwen/Qwen3-32B"
    out_dir = DATASETS_DIR / "5"
    task = CompressedCotTask(model=model)

    # Discover verified questions
    verified_ids = task.get_verified_questions(threshold=0.8)
    if not verified_ids:
        print("No verified questions found. Run verification first.")
        return

    print(f"Found {len(verified_ids)} verified questions")

    prompts_dir = out_dir / "prompts"
    _ensure_dirs(prompts_dir)

    for qid in verified_ids:
        loaded = task.load_question_and_cot(qid)
        if loaded is None:
            print(f"  Skipping {qid}: could not load question/CoT")
            continue

        question, source_cot = loaded

        # Save prompt metadata
        prompt_meta = {
            "question_id": qid,
            "question_type": getattr(question, "question_type", ""),
            "question_text": question.question,
            "source_cot": source_cot,
        }
        if hasattr(question, "correct_answer"):
            prompt_meta["correct_answer"] = question.correct_answer
        if hasattr(question, "choices"):
            prompt_meta["choices"] = question.choices
        if hasattr(question, "bad_outcome"):
            prompt_meta["bad_outcome"] = question.bad_outcome

        _save_json(prompts_dir / f"{qid}.json", prompt_meta)

    # Method output directories are populated by the methods themselves.
    # Create the known method subdirs as placeholders.
    method_names = [
        "faithful_monitor",
        "sliding_window_oracle",
        "last_n_baseline",
        "attention_selection",
    ]
    for method in method_names:
        _ensure_dirs(out_dir / "qwen-3-32b" / method)

    print(f"\nTask 5 saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TASK_RUNNERS = {
    1: ("Detecting the effect of a user preference (scruples)", generate_task_1),
    2: ("Detecting the effect of a Stanford professor hint (hinted_cot)", generate_task_2),
    3: ("Identifying atypical answers (min_maj_answer)", generate_task_3),
    4: ("Estimating the answer entropy (forced_response)", generate_task_4),
    5: ("Compressing reasoning traces (compressed_cot)", generate_task_5),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate datasets for stress-testing tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-1", dest="task_1", action="store_true",
                        help="Generate dataset 1: scruples (user preference sycophancy)")
    parser.add_argument("-2", dest="task_2", action="store_true",
                        help="Generate dataset 2: hinted_cot (Stanford professor hint)")
    parser.add_argument("-3", dest="task_3", action="store_true",
                        help="Generate dataset 3: min_maj_answer (atypical answers)")
    parser.add_argument("-4", dest="task_4", action="store_true",
                        help="Generate dataset 4: forced_response (answer entropy)")
    parser.add_argument("-5", dest="task_5", action="store_true",
                        help="Generate dataset 5: compressed_cot (reasoning compression)")
    parser.add_argument("--all", action="store_true",
                        help="Generate all 5 datasets")

    args = parser.parse_args()

    # Determine which tasks to run
    selected = []
    if args.all:
        selected = [1, 2, 3, 4, 5]
    else:
        for i in range(1, 6):
            if getattr(args, f"task_{i}", False):
                selected.append(i)

    if not selected:
        parser.print_help()
        print("\nError: specify at least one task flag (-1 through -5) or --all")
        sys.exit(1)

    print(f"Generating datasets for tasks: {selected}")
    print(f"Output directory: {DATASETS_DIR}\n")

    for task_num in selected:
        desc, runner = TASK_RUNNERS[task_num]
        print(f"\n{'#' * 60}")
        print(f"# Task {task_num}: {desc}")
        print(f"{'#' * 60}\n")
        try:
            runner()
        except Exception as e:
            print(f"\nError generating task {task_num}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
