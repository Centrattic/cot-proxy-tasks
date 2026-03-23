"""
CompressedCotTask — CoT compression evaluated via logprob forcing.

For a CoT prefix of N sentences (keeping last K as-is):
  1. D_baseline: force "So, the answer is:" after all N sentences
  2. D_deletion: force after only last K sentences (delete first N-K)
  3. Filter: skip if KL(D_baseline, D_deletion) < threshold
  4. Compress first (N-K) sentences, prepend to last K
  5. D_compressed: force after compressed + last K
  6. Metric: KL(D_baseline, D_compressed)
"""

import contextlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from vllm import SamplingParams as VllmSamplingParams

from ...utils.questions import GPQAQuestion, BinaryJudgeQuestion, Question
from ...utils.chat_template import build_thinking_prompt
from .data_loader import (
    load_question_and_cot as _load_question_and_cot,
    get_verified_questions as _get_verified_questions,
    get_latest_verification_dir,
    load_verification_summary,
)


# ------------------------------------------------------------------
# CompressionSpec: describes what to compress
# ------------------------------------------------------------------

@dataclass
class CompressionSpec:
    """Specification for a single compression example.

    The CoT prefix has *n_sentences* sentences total.  The first
    (compress_end - compress_start) are the "compress region"; the last
    *k_keep* sentences are kept verbatim.
    """
    question_id: str
    sentences: List[str]
    n_sentences: int
    k_keep: int
    compress_start: int  # inclusive
    compress_end: int    # exclusive

    # Pre-computed distributions (filled during dataset creation)
    baseline_dist: Optional[Dict[str, float]] = None
    deletion_dist: Optional[Dict[str, float]] = None
    deletion_kl: Optional[float] = None

    # --- derived helpers ---

    @property
    def compress_sentences(self) -> List[str]:
        return self.sentences[self.compress_start:self.compress_end]

    @property
    def keep_sentences(self) -> List[str]:
        return self.sentences[self.compress_end:]

    @property
    def full_prefix(self) -> str:
        return " ".join(self.sentences)

    @property
    def keep_only_prefix(self) -> str:
        """Prefix with only the last K sentences (deletion baseline)."""
        return " ".join(self.keep_sentences)

    def reconstruct(self, compressed_text: str) -> str:
        """Build prefix = compressed_region + kept tail."""
        parts = []
        if compressed_text:
            parts.append(compressed_text)
        tail = self.keep_only_prefix
        if tail:
            parts.append(tail)
        return " ".join(parts)

    def reconstruct_from_indices(self, selected_indices: List[int]) -> str:
        """Reconstruct keeping only selected compress-region sentences + tail."""
        region = self.compress_sentences
        kept = " ".join(region[i] for i in sorted(selected_indices) if i < len(region))
        return self.reconstruct(kept)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "question_id": self.question_id,
            "sentences": self.sentences,
            "n_sentences": self.n_sentences,
            "k_keep": self.k_keep,
            "compress_start": self.compress_start,
            "compress_end": self.compress_end,
        }
        if self.baseline_dist is not None:
            d["baseline_dist"] = self.baseline_dist
        if self.deletion_dist is not None:
            d["deletion_dist"] = self.deletion_dist
        if self.deletion_kl is not None:
            d["deletion_kl"] = self.deletion_kl
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompressionSpec":
        return cls(
            question_id=d["question_id"],
            sentences=d["sentences"],
            n_sentences=d["n_sentences"],
            k_keep=d["k_keep"],
            compress_start=d["compress_start"],
            compress_end=d["compress_end"],
            baseline_dist=d.get("baseline_dist"),
            deletion_dist=d.get("deletion_dist"),
            deletion_kl=d.get("deletion_kl"),
        )


# ------------------------------------------------------------------
# CompressedCotTask
# ------------------------------------------------------------------

class CompressedCotTask:
    """
    Utility class for CoT compression experiments.

    Provides:
      - Question / CoT loading from verification rollouts
      - Logprob-based answer distribution extraction
      - KL divergence computation
    """

    def __init__(self, model: str, data_dir: Optional[Path] = None):
        self.model = model
        self.data_dir = data_dir or (
            Path(__file__).resolve().parent.parent.parent.parent / "data" / "compressed_cot"
        )
        self.verification_dir = self.data_dir.parent / "verification_rollouts"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logprob forcing
    # ------------------------------------------------------------------

    def get_choice_distribution(
        self,
        llm,
        tokenizer,
        question: Question,
        cot_prefix: str,
        choices: Optional[List[str]] = None,
        topk: int = 50,
    ) -> Dict[str, float]:
        """
        Force "So, the answer is:" after *cot_prefix* and extract
        the logprob distribution over answer choices from the next token.

        Returns a normalised probability dict, e.g. {"A": 0.7, "B": 0.1, ...}.
        """
        if choices is None:
            if isinstance(question, BinaryJudgeQuestion):
                choices = ["YES", "NO"]
            else:
                choices = [chr(ord("A") + i) for i in range(len(question.choices))]

        choice_token_ids = _resolve_choice_token_ids(tokenizer, choices)

        anchor = " So, the answer is: " if cot_prefix else "So, the answer is: "
        cot_with_anchor = cot_prefix + anchor
        prompt_str = build_thinking_prompt(
            tokenizer, self._user_msg(question), cot_prefix=cot_with_anchor,
        ) + "</think>\n"

        with contextlib.redirect_stdout(io.StringIO()):
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

        # Generate 1 token with logprobs to get the model's distribution
        # at the answer slot.
        params = VllmSamplingParams(max_tokens=1, logprobs=topk, temperature=0.0)
        output = llm.generate(
            [{"prompt_token_ids": prompt_tokens}], params, use_tqdm=False,
        )[0]

        gen_logprobs = output.outputs[0].logprobs
        topk_lookup = {}
        if gen_logprobs and len(gen_logprobs) > 0:
            for tid, entry in gen_logprobs[0].items():
                topk_lookup[tid] = entry.logprob

        # Extract logprobs for answer choices and softmax
        found: Dict[str, float] = {}
        for c in choices:
            lp = topk_lookup.get(choice_token_ids[c])
            if lp is not None:
                found[c] = lp

        if found:
            max_lp = max(found.values())
            exps = {c: math.exp(lp - max_lp) for c, lp in found.items()}
            total = sum(exps.values())
            return {c: exps.get(c, 0.0) / total for c in choices}
        return {c: 0.0 for c in choices}

    # ------------------------------------------------------------------
    # Question / CoT loading (delegates to data_loader)
    # ------------------------------------------------------------------

    def load_question_and_cot(
        self, question_id: str, rollout_idx: int = 0,
    ) -> Optional[Tuple[Question, str]]:
        """Load a Question object and its source CoT from verification data."""
        return _load_question_and_cot(self.verification_dir, question_id, rollout_idx)

    def get_verified_questions(self, threshold: float = 0.8) -> List[str]:
        """Get question IDs that meet the verification agreement threshold."""
        return _get_verified_questions(self.verification_dir, threshold)

    # ------------------------------------------------------------------
    # Spec builder
    # ------------------------------------------------------------------

    def build_spec(
        self,
        question_id: str,
        sentences: List[str],
        n_sentences: int,
        k_keep: int,
    ) -> CompressionSpec:
        """Build a CompressionSpec from sentence list and parameters."""
        prefix = sentences[:n_sentences]
        return CompressionSpec(
            question_id=question_id,
            sentences=prefix,
            n_sentences=n_sentences,
            k_keep=k_keep,
            compress_start=0,
            compress_end=n_sentences - k_keep,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_kl(
        baseline_dist: Dict[str, float],
        other_dist: Dict[str, float],
    ) -> float:
        """KL(baseline || other)."""
        return kl_divergence(baseline_dist, other_dist)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _user_msg(question: Question) -> str:
        if isinstance(question, BinaryJudgeQuestion):
            return question.question
        labels = [chr(ord("A") + i) for i in range(len(question.choices))]
        choices = "\n".join(f"{l}. {c}" for l, c in zip(labels, question.choices))
        labels_str = (
            ", ".join(labels[:-1]) + f", or {labels[-1]}"
            if len(labels) > 2
            else " or ".join(labels)
        )
        return f"{question.question}\n\n{choices}\n\nAnswer with just the letter ({labels_str})."



# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------

def _resolve_choice_token_ids(tokenizer, choices: List[str]) -> Dict[str, int]:
    """Map each answer string (e.g. "A") to its single token id."""
    mapping = {}
    for c in choices:
        with contextlib.redirect_stdout(io.StringIO()):
            ids = tokenizer.encode(c, add_special_tokens=False)
        mapping[c] = ids[-1]
    return mapping


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """KL(P || Q) — divergence of Q from reference distribution P."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return float("inf")

    p_arr = np.array([p.get(k, 0.0) for k in sorted(all_keys)])
    q_arr = np.array([q.get(k, 0.0) for k in sorted(all_keys)])

    # Normalise
    p_sum, q_sum = p_arr.sum(), q_arr.sum()
    if p_sum > 0:
        p_arr = p_arr / p_sum
    if q_sum > 0:
        q_arr = q_arr / q_sum

    eps = 1e-10
    q_arr = np.maximum(q_arr, eps)

    mask = p_arr > 0
    if not mask.any():
        return 0.0
    return float(np.sum(p_arr[mask] * np.log(p_arr[mask] / q_arr[mask])))
