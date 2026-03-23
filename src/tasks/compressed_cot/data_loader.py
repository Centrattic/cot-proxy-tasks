"""
Data loader for compressed CoT task.

Loads questions and their verified CoTs from the verification_rollouts directory.
Reuses the same verification data format as forced_response.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...utils.questions import Question

# Re-export from forced_response's data_loader since the format is identical
from ..forced_response.data_loader import (
    get_latest_verification_dir,
    get_verified_questions,
    load_question_and_cot,
    load_verification_summary,
    question_from_summary,
)

__all__ = [
    "load_question_and_cot",
    "get_verified_questions",
    "get_latest_verification_dir",
    "load_verification_summary",
    "question_from_summary",
]
