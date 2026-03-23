"""
Prompt classes for the majority/minority answer classification task.

Generation prompts for building examples and formatting data.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.prompts.base import BasePrompt


def _truncate_cot(cot: str, max_chars: int = -1) -> str:
    """Truncate a CoT to a reasonable length for the prompt. 0 = no limit."""
    if max_chars < 0 or len(cot) <= max_chars:
        return cot
    # Keep the beginning and end
    half = max_chars // 2
    return cot[:half] + "\n[... truncated ...]\n" + cot[-half:]


def _build_examples_json(
    examples_by_prompt: List[Dict[str, Any]], cot_max_chars: int = 2000
) -> str:
    """Build a JSON string of examples grouped by question.

    Returns a JSON array like:
    [
      {
        "question": "...",
        "rollouts": [
          {"chain_of_thought": "...", "final_answer": "...", "classification": "majority"},
          ...
        ]
      },
      ...
    ]
    """
    grouped = []
    for prompt_group in examples_by_prompt:
        rollouts = []
        for ex in prompt_group["examples"]:
            rollouts.append({
                "chain_of_thought": _truncate_cot(ex["cot_content"], cot_max_chars),
                "final_answer": ex["answer"],
                "classification": ex["label"],
            })
        grouped.append({
            "question": prompt_group["prompt_text"],
            "rollouts": rollouts,
        })
    return json.dumps(grouped, indent=2)
