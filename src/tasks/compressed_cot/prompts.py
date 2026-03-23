"""
Compressed CoT prompt templates: sentence selection and summary compression.

BasePrompt subclasses for generation:
- SentenceSelectionPrompt: select the N most important sentence indices
- FaithfulSentenceSelectionPrompt: preserve genuine reasoning distribution
- SummaryCompressionPrompt: rewrite/summarize the compress region

All prompts operate on the "compress region" (first N-K sentences of the
prefix) while the last K sentences are kept verbatim.
"""

import json
import re
from typing import Any, Dict, List, Optional

from ...prompts.base import BasePrompt


def _build_numbered_region(sentences: List[str], start: int, end: int) -> str:
    """Build numbered list of region sentences.

    Indices are always 0-based relative to the compress region so that they
    match the indices expected by CompressionSpec.reconstruct_from_indices.
    """
    lines = []
    for rel_i, abs_i in enumerate(range(start, end)):
        lines.append(f"  [{rel_i}] {sentences[abs_i]}")
    return "\n".join(lines)


class SentenceSelectionPrompt(BasePrompt):
    """
    Select the N most important sentences from the compress region of a CoT.

    Expected row keys:
      - "sentences": list[str] (all prefix sentences)
      - "compress_start": int
      - "compress_end": int
      - "k_keep": int
      - "target_num_sentences": int
      - "char_budget": int
      - "question": str

    parse_response returns: list[int] (selected sentence indices within
    the compress region, 0-indexed relative to the region)
    """

    def __init__(self):
        super().__init__("sentence_selection")

    def format(self, row: Dict[str, Any]) -> str:
        sentences = row.get("sentences", [])
        compress_start = row.get("compress_start", 0)
        compress_end = row.get("compress_end", len(sentences))
        k_keep = row.get("k_keep", 5)
        target_n = row.get("target_num_sentences", 5)
        char_budget = row.get("char_budget", 1000)
        question = row.get("question", "")

        full_cot = " ".join(sentences)
        numbered_section = _build_numbered_region(sentences, compress_start, compress_end)
        kept_tail = " ".join(sentences[compress_end:])

        return (
            f"You are analyzing a model's chain of thought (CoT) to identify the most "
            f"important reasoning steps.\n\n"
            f"Question the model was answering:\n{question}\n\n"
            f"=== Full Chain of Thought Prefix ===\n{full_cot}\n\n"
            f"The CoT prefix has been split into sentences. The first {compress_end - compress_start} "
            f"sentences need to be compressed, while the last {k_keep} sentences are kept "
            f"verbatim.\n\n"
            f"=== Compress region (indices 0 to {compress_end - compress_start - 1}) ===\n"
            f"Each sentence is numbered by its 0-based index within this region:\n{numbered_section}\n\n"
            f"=== Kept tail (last {k_keep} sentences, kept verbatim) ===\n{kept_tail}\n\n"
            f"Select the {target_n} most important sentence indices from the compress "
            f"region that best preserve the reasoning quality. The total character "
            f"count of selected sentences must stay within {char_budget} characters.\n\n"
            f"Respond with ONLY a JSON array of integer indices (0-based within the compress region), e.g.:\n"
            f"[0, 2, 5, 8, 11]\n\n"
            f"No explanation needed, just the JSON array."
        )

    def parse_response(self, response: str) -> Optional[List[int]]:
        text = response.strip()
        match = re.search(r'\[[\d\s,]+\]', text)
        if not match:
            return None
        try:
            indices = json.loads(match.group())
            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                return indices
            return None
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


class FaithfulSentenceSelectionPrompt(BasePrompt):
    """
    Select sentences that faithfully preserve the model's genuine reasoning
    distribution, including uncertainty, wrong turns, and self-corrections —
    not just the steps leading to the correct answer.

    Same I/O contract as SentenceSelectionPrompt.
    """

    def __init__(self):
        super().__init__("faithful_sentence_selection")

    def format(self, row: Dict[str, Any]) -> str:
        sentences = row.get("sentences", [])
        compress_start = row.get("compress_start", 0)
        compress_end = row.get("compress_end", len(sentences))
        k_keep = row.get("k_keep", 5)
        target_n = row.get("target_num_sentences", 5)
        char_budget = row.get("char_budget", 1000)
        question = row.get("question", "")

        full_cot = " ".join(sentences)
        numbered_section = _build_numbered_region(sentences, compress_start, compress_end)
        kept_tail = " ".join(sentences[compress_end:])

        return (
            f"You are analyzing a model's chain of thought (CoT) to select "
            f"sentences that faithfully represent its genuine reasoning process.\n\n"
            f"IMPORTANT: Your goal is NOT to pick the sentences that lead to the "
            f"correct answer. Instead, you must preserve the model's authentic "
            f"reasoning distribution — including uncertainty, wrong turns, "
            f"self-corrections, and exploration of incorrect paths.\n\n"
            f"You will be evaluated on KL divergence: how closely the answer "
            f"distribution from the compressed CoT matches the answer distribution "
            f"from the full CoT. A good selection preserves the model's original "
            f"level of confidence and uncertainty, NOT just correctness.\n\n"
            f"Do NOT cherry-pick only the correct reasoning steps. If the model "
            f"was uncertain or explored wrong answers, your selection should "
            f"reflect that.\n\n"
            f"Question the model was answering:\n{question}\n\n"
            f"=== Full Chain of Thought Prefix ===\n{full_cot}\n\n"
            f"The CoT prefix has been split into sentences. The first {compress_end - compress_start} "
            f"sentences need to be compressed, while the last {k_keep} sentences are kept "
            f"verbatim.\n\n"
            f"=== Compress region (indices 0 to {compress_end - compress_start - 1}) ===\n"
            f"Each sentence is numbered by its 0-based index within this region:\n{numbered_section}\n\n"
            f"=== Kept tail (last {k_keep} sentences, kept verbatim) ===\n{kept_tail}\n\n"
            f"Select the {target_n} sentence indices from the compress region that "
            f"best preserve the model's genuine reasoning distribution (including "
            f"doubt, errors, and self-corrections). The total character count "
            f"of selected sentences must stay within {char_budget} characters.\n\n"
            f"Respond with ONLY a JSON array of integer indices (0-based within the compress region), e.g.:\n"
            f"[0, 2, 5, 8, 11]\n\n"
            f"No explanation needed, just the JSON array."
        )

    def parse_response(self, response: str) -> Optional[List[int]]:
        text = response.strip()
        match = re.search(r'\[[\d\s,]+\]', text)
        if not match:
            return None
        try:
            indices = json.loads(match.group())
            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                return indices
            return None
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


class SummaryCompressionPrompt(BasePrompt):
    """
    Rewrite/summarize the compress region of a CoT within a character budget.

    Expected row keys:
      - "sentences": list[str]
      - "compress_start": int
      - "compress_end": int
      - "k_keep": int
      - "char_budget": int
      - "question": str

    parse_response returns: str (the rewritten region)
    """

    def __init__(self):
        super().__init__("summary_compression")

    def format(self, row: Dict[str, Any]) -> str:
        sentences = row.get("sentences", [])
        compress_start = row.get("compress_start", 0)
        compress_end = row.get("compress_end", len(sentences))
        k_keep = row.get("k_keep", 5)
        char_budget = row.get("char_budget", 1000)
        question = row.get("question", "")

        region_text = " ".join(sentences[compress_start:compress_end])
        kept_tail = " ".join(sentences[compress_end:])

        return (
            f"You are compressing a model's chain of thought (CoT) to preserve "
            f"critical reasoning steps in fewer characters.\n\n"
            f"Question the model was answering:\n{question}\n\n"
            f"The CoT prefix has two parts. The tail ({k_keep} sentences) is kept "
            f"verbatim. You must rewrite the first part.\n\n"
            f"=== Compress region (to rewrite) ===\n{region_text}\n\n"
            f"=== Kept tail (kept verbatim) ===\n{kept_tail}\n\n"
            f"Rewrite the compress region to be at most {char_budget} characters. "
            f"Preserve the critical reasoning steps, logical transitions, and any "
            f"key calculations or conclusions. The compressed version will be "
            f"placed before the kept tail.\n\n"
            f"IMPORTANT: Preserve the model's authentic reasoning — including "
            f"uncertainty, self-corrections, and exploration of different paths. "
            f"Do not over-simplify to just the 'correct' reasoning chain.\n\n"
            f"Respond with ONLY the rewritten text. "
            f"No explanation, no preamble, just the compressed reasoning."
        )

    def parse_response(self, response: str) -> Optional[str]:
        text = response.strip()
        if not text:
            return None
        return text
