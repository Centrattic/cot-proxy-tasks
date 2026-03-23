"""
Data loading for the Hinted CoT task.

Loads multiple-choice questions from multiple hard benchmarks:
- MMLU (via HuggingFace)
- GPQA Diamond (PhD-level science)
- MedMCQA (medical entrance exams)
- ARC-Challenge (hard science reasoning)
- MMLU-Pro (harder MMLU with 10 choices)

Caches locally as JSON for reproducibility. Reuses MultipleChoiceQuestion from
src/utils/questions.py.
"""

import json
from pathlib import Path
from typing import List, Optional

from ...utils.questions import MultipleChoiceQuestion

DEFAULT_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "logical_fallacies",
    "machine_learning",
    "moral_disputes",
    "philosophy",
    "professional_law",
    "professional_medicine",
    "virology",
    "world_religions",
]


def load_mmlu_from_huggingface(
    subjects: Optional[List[str]] = None,
    max_per_subject: int = 20,
    split: str = "test",
) -> List[MultipleChoiceQuestion]:
    """
    Load MMLU questions from HuggingFace datasets.

    Args:
        subjects: List of MMLU subjects to load (None = DEFAULT_SUBJECTS).
        max_per_subject: Max questions per subject.
        split: HF split name ("test", "validation", "dev", "auxiliary_train").

    Returns:
        List of MultipleChoiceQuestion.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install with: pip install datasets"
        )

    if subjects is None:
        subjects = DEFAULT_SUBJECTS

    questions: List[MultipleChoiceQuestion] = []
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split=split)
        except Exception as e:
            print(f"Warning: could not load MMLU subject '{subject}': {e}")
            continue

        for i, item in enumerate(ds):
            if i >= max_per_subject:
                break

            choices = list(item["choices"])
            correct_idx = int(item["answer"])
            correct_letter = chr(ord("A") + correct_idx)

            questions.append(
                MultipleChoiceQuestion(
                    id=f"mmlu_{subject}_{i:04d}",
                    question=item["question"],
                    choices=choices,
                    correct_answer=correct_letter,
                    correct_index=correct_idx,
                    subject=subject,
                    difficulty="mmlu",
                )
            )

    return questions


def load_gpqa_questions(
    max_questions: Optional[int] = None,
    subset: str = "gpqa_diamond",
) -> List[MultipleChoiceQuestion]:
    """
    Load GPQA questions (PhD-level science) from HuggingFace.

    Source: Idavidrein/gpqa. Expert accuracy ~65%, GPT-4 ~39%.
    Always 4 choices.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required.")

    import hashlib
    import random

    dataset = load_dataset("Idavidrein/gpqa", subset, split="train")

    questions = []
    for i, item in enumerate(dataset):
        if max_questions is not None and i >= max_questions:
            break

        choices = [
            item.get("Incorrect Answer 1", ""),
            item.get("Incorrect Answer 2", ""),
            item.get("Incorrect Answer 3", ""),
            item.get("Correct Answer", ""),
        ]

        # Deterministic shuffle based on question content
        seed = int(hashlib.md5(item["Question"].encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        choice_pairs = [(c, idx == 3) for idx, c in enumerate(choices)]
        rng.shuffle(choice_pairs)

        shuffled_choices = [c for c, _ in choice_pairs]
        correct_idx = next(
            idx for idx, (_, is_correct) in enumerate(choice_pairs) if is_correct
        )
        correct_letter = chr(ord("A") + correct_idx)

        questions.append(
            MultipleChoiceQuestion(
                id=f"gpqa_{subset}_{i:04d}",
                question=item["Question"],
                choices=shuffled_choices,
                correct_answer=correct_letter,
                correct_index=correct_idx,
                subject=item.get("Subdomain", "science"),
                difficulty="gpqa",
            )
        )

    return questions


def load_medmcqa_questions(
    max_questions: Optional[int] = None,
) -> List[MultipleChoiceQuestion]:
    """
    Load MedMCQA questions (medical entrance exams) from HuggingFace.

    Source: openlifescienceai/medmcqa, validation split.
    Always 4 choices.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required.")

    dataset = load_dataset("openlifescienceai/medmcqa", split="validation")

    questions = []
    for i, item in enumerate(dataset):
        if max_questions is not None and i >= max_questions:
            break

        choices = [
            item.get("opa", ""),
            item.get("opb", ""),
            item.get("opc", ""),
            item.get("opd", ""),
        ]
        correct_idx = int(item.get("cop", 0))
        correct_letter = chr(ord("A") + correct_idx)

        questions.append(
            MultipleChoiceQuestion(
                id=f"medmcqa_{i:04d}",
                question=item["question"],
                choices=choices,
                correct_answer=correct_letter,
                correct_index=correct_idx,
                subject=item.get("subject_name", "medicine"),
                difficulty="medical",
            )
        )

    return questions


def load_arc_challenge_questions(
    max_questions: Optional[int] = None,
) -> List[MultipleChoiceQuestion]:
    """
    Load ARC-Challenge questions (hard science reasoning) from HuggingFace.

    Source: allenai/ai2_arc, ARC-Challenge config, test split.
    Usually 4 choices but can vary.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required.")

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    questions = []
    for i, item in enumerate(dataset):
        if max_questions is not None and i >= max_questions:
            break

        choices_text = item["choices"]["text"]
        choices_labels = item["choices"]["label"]

        # Normalize labels to A, B, C, D (some use 1, 2, 3, 4)
        if choices_labels and choices_labels[0].isdigit():
            labels = [chr(ord("A") + int(l) - 1) for l in choices_labels]
        else:
            labels = choices_labels

        answer_key = item["answerKey"]
        if answer_key.isdigit():
            answer_key = chr(ord("A") + int(answer_key) - 1)

        try:
            correct_idx = labels.index(answer_key)
        except ValueError:
            continue  # skip if answer doesn't match labels

        # Only keep 4-choice questions for consistency
        if len(choices_text) != 4:
            continue

        questions.append(
            MultipleChoiceQuestion(
                id=f"arc_challenge_{i:04d}",
                question=item["question"],
                choices=choices_text,
                correct_answer=chr(ord("A") + correct_idx),
                correct_index=correct_idx,
                subject="science",
                difficulty="arc_challenge",
            )
        )

    return questions


def load_mmlu_pro_questions(
    max_questions: Optional[int] = None,
    subjects: Optional[List[str]] = None,
) -> List[MultipleChoiceQuestion]:
    """
    Load MMLU-Pro questions (harder MMLU with 10 choices) from HuggingFace.

    Source: TIGER-Lab/MMLU-Pro, test split.
    10 choices per question. We keep all 10 for maximum difficulty.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required.")

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    questions = []
    for i, item in enumerate(dataset):
        if max_questions is not None and len(questions) >= max_questions:
            break

        category = item.get("category", "")
        if subjects and category.lower() not in [s.lower() for s in subjects]:
            continue

        options = item.get("options", [])
        if not options:
            continue

        answer_key = item.get("answer", "")
        if not answer_key:
            continue

        try:
            correct_idx = ord(answer_key.upper()) - ord("A")
        except (TypeError, ValueError):
            continue

        if correct_idx < 0 or correct_idx >= len(options):
            continue

        labels = [chr(ord("A") + j) for j in range(len(options))]

        questions.append(
            MultipleChoiceQuestion(
                id=f"mmlu_pro_{i:04d}",
                question=item["question"],
                choices=options,
                correct_answer=answer_key.upper(),
                correct_index=correct_idx,
                subject=category,
                difficulty="mmlu_pro",
                labels=labels,
            )
        )

    return questions


# -- Cache helpers ------------------------------------------------------------


def save_questions_cache(
    questions: List[MultipleChoiceQuestion],
    cache_path: Path,
) -> None:
    """Save questions to a local JSON cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump([q.to_dict() for q in questions], f, indent=2)


def load_questions_cache(cache_path: Path) -> List[MultipleChoiceQuestion]:
    """Load questions from a local JSON cache."""
    with open(cache_path) as f:
        data = json.load(f)
    return [MultipleChoiceQuestion.from_dict(d) for d in data]


# -- Main entry points --------------------------------------------------------


def load_mmlu_questions(
    data_dir: Optional[Path] = None,
    subjects: Optional[List[str]] = None,
    max_per_subject: int = 20,
    split: str = "test",
    max_questions: Optional[int] = None,
    use_cache: bool = True,
) -> List[MultipleChoiceQuestion]:
    """
    Load MMLU questions, using local cache if available.
    """
    if data_dir is None:
        data_dir = (
            Path(__file__).parent.parent.parent.parent / "data" / "hinted_cot" / "mmlu"
        )

    cache_path = data_dir / f"questions_{split}.json"

    if use_cache and cache_path.exists():
        questions = load_questions_cache(cache_path)
    else:
        questions = load_mmlu_from_huggingface(
            subjects=subjects,
            max_per_subject=max_per_subject,
            split=split,
        )
        if use_cache:
            save_questions_cache(questions, cache_path)

    if max_questions is not None:
        questions = questions[:max_questions]

    return questions


def load_hard_questions(
    datasets: Optional[List[str]] = None,
    max_per_dataset: int = 200,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> List[MultipleChoiceQuestion]:
    """
    Load hard MCQs from multiple benchmarks.

    Args:
        datasets: Which datasets to load. Default: all hard datasets.
            Options: "gpqa", "medmcqa", "arc_challenge", "mmlu_pro",
                     "mmlu_hard" (hard MMLU subjects only).
        max_per_dataset: Max questions per dataset.
        use_cache: Cache loaded questions as JSON.
        cache_dir: Directory for cache files.

    Returns:
        Combined list of MultipleChoiceQuestion from all requested datasets.
    """
    if datasets is None:
        datasets = ["gpqa", "medmcqa", "arc_challenge", "mmlu_pro"]

    if cache_dir is None:
        cache_dir = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "hinted_cot"
            / "hard_questions"
        )

    cache_path = cache_dir / f"questions_{'_'.join(sorted(datasets))}_{max_per_dataset}.json"

    if use_cache and cache_path.exists():
        return load_questions_cache(cache_path)

    all_questions: List[MultipleChoiceQuestion] = []

    for ds_name in datasets:
        print(f"Loading {ds_name}...")
        try:
            if ds_name == "gpqa":
                qs = load_gpqa_questions(max_questions=max_per_dataset)
            elif ds_name == "medmcqa":
                qs = load_medmcqa_questions(max_questions=max_per_dataset)
            elif ds_name == "arc_challenge":
                qs = load_arc_challenge_questions(max_questions=max_per_dataset)
            elif ds_name == "mmlu_pro":
                qs = load_mmlu_pro_questions(max_questions=max_per_dataset)
            elif ds_name == "mmlu_hard":
                hard_subjects = [
                    "abstract_algebra",
                    "college_mathematics",
                    "college_chemistry",
                    "college_physics",
                    "formal_logic",
                    "professional_medicine",
                    "anatomy",
                ]
                qs = load_mmlu_from_huggingface(
                    subjects=hard_subjects,
                    max_per_subject=max_per_dataset // len(hard_subjects),
                )
            else:
                print(f"Unknown dataset: {ds_name}, skipping.")
                continue
            print(f"  Loaded {len(qs)} questions from {ds_name}")
            all_questions.extend(qs)
        except Exception as e:
            print(f"Warning: failed to load {ds_name}: {e}")

    if use_cache and all_questions:
        save_questions_cache(all_questions, cache_path)

    print(f"Total: {len(all_questions)} hard questions loaded")
    return all_questions
