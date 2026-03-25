"""
Data loader for forced answer entropy task.

Loads questions from GPQA Diamond (HuggingFace), plus custom puzzle questions
and OOD question sets (BigBench, MedMCQA, RACE).
"""

from typing import List

from ...utils.questions import MultipleChoiceQuestion, GPQAQuestion


# Custom puzzle questions used in training
CUSTOM_TRAIN_QUESTIONS = [
    GPQAQuestion(
        id="custom_bagel_001",
        question="I put a bagel flat on a table and slice it in half with a vertical cut. How many pieces are there?",
        choices=["0", "1", "2", "4"],
        correct_answer="C",
        correct_index=2,
        subject="puzzle",
    ),
    GPQAQuestion(
        id="starfish",
        question="How many legs does a starfish have?",
        choices=["4", "5", "8", "10"],
        correct_answer="B",
        correct_index=1,
        subject="biology",
    ),
    GPQAQuestion(
        id="waffle",
        question="How many squares are in a standard waffle pattern (4x4 grid)?",
        choices=["4", "8", "16", "30"],
        correct_answer="C",
        correct_index=2,
        subject="puzzle",
    ),
]


def load_gpqa_questions(
    max_questions: int = 60,
) -> List[MultipleChoiceQuestion]:
    """Load GPQA Diamond questions from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install with: pip install datasets"
        )

    import hashlib
    import random

    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    questions = []
    for i, item in enumerate(dataset):
        if len(questions) >= max_questions:
            break

        choices = [
            item.get("Incorrect Answer 1", ""),
            item.get("Incorrect Answer 2", ""),
            item.get("Incorrect Answer 3", ""),
            item.get("Correct Answer", ""),
        ]

        seed = int(hashlib.md5(item["Question"].encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        choice_pairs = [(c, i == 3) for i, c in enumerate(choices)]
        rng.shuffle(choice_pairs)

        shuffled_choices = [c for c, _ in choice_pairs]
        correct_idx = next(
            i for i, (_, is_correct) in enumerate(choice_pairs) if is_correct
        )
        correct_letter = chr(ord("A") + correct_idx)

        questions.append(
            GPQAQuestion(
                id=f"gpqa_gpqa_diamond_{i:04d}",
                question=item["Question"],
                choices=shuffled_choices,
                correct_answer=correct_letter,
                correct_index=correct_idx,
                subject=item.get("Subdomain", None),
                difficulty="gpqa_diamond",
            )
        )

    return questions


def load_ood_questions(
    source: str = "bigbench",
    max_questions: int = 10,
) -> List[MultipleChoiceQuestion]:
    """Load OOD questions from BigBench, MedMCQA, or RACE."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install with: pip install datasets"
        )

    if source == "bigbench":
        dataset = load_dataset("tasksource/bigbench", "logical_deduction", split="validation")
        questions = []
        for i, item in enumerate(dataset):
            if len(questions) >= max_questions:
                break
            mc_targets = item.get("multiple_choice_targets", [])
            mc_scores = item.get("multiple_choice_scores", [])
            if not mc_targets or not mc_scores:
                continue
            choices = list(mc_targets)
            correct_idx = next((j for j, s in enumerate(mc_scores) if s == 1), 0)
            questions.append(MultipleChoiceQuestion(
                id=f"bigbench_logical_deduction_{len(questions):04d}",
                question=item.get("inputs", ""),
                choices=choices,
                correct_answer=chr(ord("A") + correct_idx),
                correct_index=correct_idx,
                subject="logical_deduction",
            ))
        return questions

    elif source == "medmcqa":
        dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
        questions = []
        for i, item in enumerate(dataset):
            if len(questions) >= max_questions:
                break
            choices = [item.get(f"op{c}", "") for c in "abcd"]
            correct_idx = int(item.get("cop", 0))
            questions.append(MultipleChoiceQuestion(
                id=f"medmcqa_{i:04d}",
                question=item["question"],
                choices=choices,
                correct_answer=chr(ord("A") + correct_idx),
                correct_index=correct_idx,
                subject=item.get("subject_name"),
            ))
        return questions

    elif source == "race":
        dataset = load_dataset("ehovy/race", "high", split="test")
        questions = []
        for i, item in enumerate(dataset):
            if len(questions) >= max_questions:
                break
            full_question = item.get("article", "") + "\n\n" + item.get("question", "")
            choices = item.get("options", [])
            answer_letter = item.get("answer", "A").upper()
            correct_idx = ord(answer_letter) - ord("A")
            questions.append(MultipleChoiceQuestion(
                id=f"race_high_{i:04d}",
                question=full_question,
                choices=choices,
                correct_answer=answer_letter,
                correct_index=correct_idx,
                subject="reading_comprehension",
            ))
        return questions

    else:
        raise ValueError(f"Unknown OOD source: {source}")
