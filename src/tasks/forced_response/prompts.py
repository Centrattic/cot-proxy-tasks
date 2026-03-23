"""
Forced-response prompt utilities.

- CoT splitting utilities (get_cumulative_cot_segments, split_cot_into_sentences)
- Distribution parsing utilities
"""

import json
import re
from typing import Any, Dict, List, Optional


# =============================================================================
# COT SPLITTING UTILITIES (used by tasks for sentence-level forcing/resampling)
# =============================================================================

def split_cot_into_sentences(cot_text: str) -> List[str]:
    """
    Split chain of thought text into sentences.

    Handles common sentence-ending patterns including:
    - Standard punctuation (. ! ?)
    - Newlines as sentence boundaries
    - List items
    """
    text = cot_text.strip()
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$|\n\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def get_cumulative_cot_segments(cot_text: str) -> List[str]:
    """
    Get cumulative CoT segments for forcing at each sentence boundary.

    For a CoT with sentences [S1, S2, S3], returns:
    [S1, S1+S2, S1+S2+S3]
    """
    sentences = split_cot_into_sentences(cot_text)
    cumulative = []
    current = ""
    for sentence in sentences:
        if current:
            current = current + " " + sentence
        else:
            current = sentence
        cumulative.append(current)
    return cumulative


# =============================================================================
# DISTRIBUTION PARSING
# =============================================================================

def _parse_distribution(response: str, is_binary_judge: bool = False) -> Optional[Dict[str, float]]:
    """
    Parse a distribution JSON from the LLM response.

    Expected format:
      Multiple choice: {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}
      Binary judge:    {"YES": 0.3, "NO": 0.7}

    Returns None if parsing fails.
    """
    text = response.strip()
    match = re.search(r'\{[^{}]*\}', text)
    if not match:
        return None

    try:
        data = json.loads(match.group())
        expected_keys = ['YES', 'NO'] if is_binary_judge else ['A', 'B', 'C', 'D']
        dist = {}
        for key in expected_keys:
            if key in data:
                dist[key] = float(data[key])
            elif key.lower() in data:
                dist[key] = float(data[key.lower()])
            else:
                dist[key] = 0.0

        total = sum(dist.values())
        if total > 0:
            if abs(total - 1.0) > 0.01:
                dist = {k: v / total for k, v in dist.items()}
            return dist
        return None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
