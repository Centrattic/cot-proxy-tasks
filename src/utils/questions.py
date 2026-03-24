"""
Question data types shared across tasks.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class MultipleChoiceQuestion:
    """A single multiple-choice question (GPQA, MedMCQA, BigBench, RACE, etc.)."""

    id: str
    question: str
    choices: List[str]
    correct_answer: str  # The label letter (A, B, C, D or custom like N, M)
    correct_index: int  # 0-indexed
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    question_type: str = "multiple_choice"
    labels: Optional[List[str]] = None  # Custom labels instead of A, B, C, D

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultipleChoiceQuestion":
        return cls(**data)


# Backwards compatibility alias
GPQAQuestion = MultipleChoiceQuestion


@dataclass
class BinaryJudgeQuestion:
    """A question that requires an LLM judge to classify the response."""

    id: str
    question: str
    judge_prompt: str  # Prompt template for the judge, with {response} placeholder
    bad_outcome: str  # What the judge returns if the model misbehaves (e.g., "YES")
    subject: Optional[str] = None
    question_type: str = "binary_judge"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BinaryJudgeQuestion":
        return cls(**data)


# Type alias for any question type
Question = Union[MultipleChoiceQuestion, BinaryJudgeQuestion]
