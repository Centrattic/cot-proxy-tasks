"""
Data loader for hinted CoT task.

Loads moral dilemma questions from the kellycyy/daily_dilemmas HuggingFace
dataset and converts them to MultipleChoiceQuestion format.
"""

import hashlib
import random
from typing import List

from ...utils.questions import MultipleChoiceQuestion


def load_dilemmas_from_huggingface(
    max_questions: int = 1000,
    seed: int = 42,
) -> List[MultipleChoiceQuestion]:
    """Load dilemma questions from kellycyy/daily_dilemmas.

    Each dilemma is a binary (Yes/No) question with randomized A/B order.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install with: pip install datasets"
        )

    ds = load_dataset(
        "kellycyy/daily_dilemmas",
        name="Dilemmas_with_values_aggregated",
        split="test",
    )

    # Deduplicate by dilemma_idx (each appears twice: to_do / not_to_do)
    seen = set()
    dilemmas = []
    for row in ds:
        idx = row["dilemma_idx"]
        if idx in seen:
            continue
        seen.add(idx)
        dilemmas.append(row)

    # Sample subset
    rng = random.Random(seed)
    sampled = rng.sample(dilemmas, min(max_questions, len(dilemmas)))

    # Convert to MultipleChoiceQuestion
    questions = []
    for dilemma in sampled:
        situation = dilemma["dilemma_situation"]

        # Deterministic randomization of Yes/No order based on content
        choice_seed = int(hashlib.md5(situation.encode()).hexdigest()[:8], 16)
        choice_rng = random.Random(choice_seed)

        if choice_rng.random() < 0.5:
            choices = ["Yes", "No"]
        else:
            choices = ["No", "Yes"]

        topic = dilemma.get("topic_group", "moral_dilemma")
        qid = f"dilemma_{dilemma['dilemma_idx']:04d}"

        questions.append(
            MultipleChoiceQuestion(
                id=qid,
                question=situation,
                choices=choices,
                correct_answer="A",
                correct_index=0,
                subject=f"moral_dilemma_{topic}",
            )
        )

    return questions
