"""
Data loader for atypical answer (min/maj) task.

Loads dilemma questions from the same Daily Dilemmas source as the hinted_cot
task (kellycyy/daily_dilemmas on HuggingFace).
"""

from typing import List

from ...utils.questions import MultipleChoiceQuestion

# Re-export from hinted_cot since the question source is the same
from ..hinted_cot.data_loader import load_dilemmas_from_huggingface

__all__ = ["load_dilemmas_from_huggingface"]
