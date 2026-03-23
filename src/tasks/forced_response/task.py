"""
ForcingTask — true prefill forcing via vLLM.

Standalone task (no shared base class beyond BaseTask). For each sentence
boundary in a source CoT, prefills the model's <think> block with the partial
CoT and lets it continue to produce an answer.
"""

import contextlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
try:
    from vllm import SamplingParams as VllmSamplingParams
except ImportError:
    VllmSamplingParams = None

from ..base import BaseTask
from ...utils.questions import MultipleChoiceQuestion, GPQAQuestion, BinaryJudgeQuestion, Question
from ...utils.chat_template import build_thinking_prompt
from .utils import get_cumulative_cot_segments
from .data_loader import (
    load_question_and_cot,
    get_latest_verification_dir,
    load_verification_summary,
    question_from_summary,
)


@dataclass
class ForceResult:
    """Result of a single forcing attempt (logprob-based)."""
    sentence_idx: int
    force_idx: int
    partial_cot: str
    continued_cot: str
    raw_tokens: List[int]
    raw_response: str
    answer: str
    full_prompt: str = ""
    choice_logprobs: Dict[str, float] = field(default_factory=dict)
    choice_probs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_idx": self.sentence_idx, "force_idx": self.force_idx,
            "partial_cot": self.partial_cot, "continued_cot": self.continued_cot,
            "raw_tokens": self.raw_tokens, "raw_response": self.raw_response,
            "answer": self.answer, "full_prompt": self.full_prompt,
            "choice_logprobs": self.choice_logprobs,
            "choice_probs": self.choice_probs,
        }


class ForcingTask(BaseTask):
    """
    Forcing task: sentence-by-sentence prefill via Tinker.

    run_data() forces a verified question's CoT at every sentence boundary,
    producing per-sentence answer distributions as ground truth.
    """

    def __init__(self, model: str,
                 data_dir: Optional[Path] = None):
        super().__init__("forcing", data_dir or (
            Path(__file__).parent.parent.parent / "data" / "forced_response"
        ))
        self.model = model

        # Sub-directories
        self.verification_dir = self.data_dir.parent / "verification_rollouts"
        self.forcing_dir = self.data_dir / "forcing"
        self.monitor_forcing_dir = self.data_dir / "monitor_forcing"

        for d in [self.verification_dir, self.forcing_dir, self.monitor_forcing_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def run_data(
        self,
        question_id: str,
        rollout_idx: int = 0,
        max_sentences: Optional[int] = None,
        sentence_stride: int = 1,
        verbose: bool = True,
        # Legacy params kept for call-site compat but unused
        num_forces: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        llm=None,
    ) -> Optional[Dict[str, Any]]:
        """Run logprob-based forcing for all sentences in the CoT."""
        from transformers import AutoTokenizer

        loaded = self.load_question_and_cot(question_id, rollout_idx)
        if loaded is None:
            print(f"Could not load question/CoT for {question_id}")
            return None
        question, source_cot = loaded

        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)

        # Determine answer choices based on question type
        if isinstance(question, BinaryJudgeQuestion):
            choices = ["YES", "NO"]
        else:
            n_choices = len(question.choices)
            choices = [chr(ord("A") + i) for i in range(n_choices)]
        choice_token_ids = self._resolve_choice_token_ids(tokenizer, choices)

        cot_segments = get_cumulative_cot_segments(source_cot)
        if max_sentences is not None:
            cot_segments = cot_segments[:max_sentences]

        # Apply stride: keep every Nth sentence (always include first and last)
        if sentence_stride > 1:
            all_indices = list(range(len(cot_segments)))
            kept = set(all_indices[::sentence_stride])
            kept.add(all_indices[-1])  # always keep last sentence
            cot_segments = [cot_segments[i] for i in sorted(kept)]

        num_sentences = len(cot_segments)

        config = {
            "model": self.model, "question_id": question.id,
            "rollout_idx": rollout_idx,
            "max_sentences": max_sentences, "sentence_stride": sentence_stride,
            "num_sentences": num_sentences,
            "method": "anchor_logprobs",
        }
        run_dir = self.create_run_dir("forcing", question.id, rollout_idx, config)

        if verbose:
            print(f"Forcing {question.id}: {num_sentences} sentences (logprob mode)")
            print(f"Run dir: {run_dir}")

        # Batch all forcing calls through vLLM
        all_tasks = [(si, cot_segments[si]) for si in range(num_sentences)]
        all_results: List[ForceResult] = []

        for sent_idx, partial_cot in tqdm(all_tasks, desc="Forcing", disable=not verbose):
            prompt_str, choice_logprobs, choice_probs = self._get_choice_distribution(
                llm, tokenizer, question,
                partial_cot, choices, choice_token_ids,
            )
            answer = max(choice_probs, key=choice_probs.get) if any(v > 0 for v in choice_probs.values()) else ""
            safe_logprobs = {c: (lp if lp is not None else -100.0) for c, lp in choice_logprobs.items()}
            all_results.append(ForceResult(
                sentence_idx=sent_idx, force_idx=0,
                partial_cot=partial_cot, continued_cot="",
                raw_tokens=[], raw_response="", answer=answer,
                full_prompt=prompt_str,
                choice_logprobs=safe_logprobs,
                choice_probs=choice_probs,
            ))

        by_sentence: Dict[int, List[ForceResult]] = {}
        for r in all_results:
            by_sentence.setdefault(r.sentence_idx, []).append(r)

        all_summaries = []
        for si in range(num_sentences):
            sent_results = by_sentence.get(si, [])
            self._save_forcing_result(
                question=question, sentence_idx=si,
                partial_cot=cot_segments[si],
                force_results=[r.to_dict() for r in sent_results],
                run_dir=run_dir,
            )
            # Use the logprob-derived probs directly as the distribution
            if sent_results:
                result = sent_results[0]
                answer_distribution = result.choice_probs
                answer = result.answer
            else:
                answer_distribution = {}
                answer = ""
            all_summaries.append({
                "sentence_idx": si, "partial_cot_length": len(cot_segments[si]),
                "total_forces": 1, "valid_answers": 1 if answer else 0,
                "answer_distribution": answer_distribution,
                "answer_counts": answer_distribution,  # backwards compat
                "most_common": answer,
            })

        self._save_forcing_summary(question, rollout_idx, all_summaries, source_cot, run_dir)

        if verbose:
            print(f"Done: {num_sentences} sentences processed.")

        summary = {
            "question_id": question.id, "question_type": question.question_type,
            "num_sentences": num_sentences, "sentence_results": all_summaries,
        }
        if isinstance(question, GPQAQuestion):
            summary["correct_answer"] = question.correct_answer
        else:
            summary["bad_outcome"] = question.bad_outcome
        return summary

    def get_data(self, load: bool = False) -> Union[bool, Optional[Any]]:
        if not load:
            return self.forcing_dir.exists() and any(self.forcing_dir.rglob("summary.json"))
        summaries = sorted(self.forcing_dir.rglob("summary.json"))
        if not summaries:
            return None
        results = []
        for p in summaries:
            with open(p) as f:
                results.append(json.load(f))
        return results

    # ------------------------------------------------------------------
    # Question/CoT loading (delegates to data_loader)
    # ------------------------------------------------------------------

    def load_question_and_cot(
        self, question_id: str, rollout_idx: int = 0,
    ) -> Optional[Tuple[Question, str]]:
        """Load a Question object and its source CoT from verification data."""
        return load_question_and_cot(self.verification_dir, question_id, rollout_idx)

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def get_question_dir(self, question_id: str, mode: str) -> Path:
        mode_dirs = {
            "verification": self.verification_dir,
            "forcing": self.forcing_dir,
            "monitor_forcing": self.monitor_forcing_dir,
        }
        if mode not in mode_dirs:
            raise ValueError(f"Unknown mode: {mode}")
        d = mode_dirs[mode] / question_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_run_dir(self, mode: str, question_id: str,
                       rollout_idx: int, config: dict) -> Path:
        base = self.get_question_dir(question_id, mode)
        rollout_dir = base / f"rollout_{rollout_idx:03d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = rollout_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        config["timestamp"] = datetime.now().isoformat()
        config["run_type"] = mode
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        return run_dir

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------

    def _save_forcing_result(self, question: Question, sentence_idx: int,
                             partial_cot: str, force_results: List[Dict],
                             run_dir: Path) -> Path:
        sentence_dir = run_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(force_results):
            with open(sentence_dir / f"force_{i:03d}.json", "w") as f:
                json.dump(result, f, indent=2)

        summary = self._build_sentence_summary(question, sentence_idx, partial_cot, force_results)
        summary_path = sentence_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary_path

    def _save_forcing_summary(self, question: Question, source_rollout_idx: int,
                              all_sentence_results: List[Dict], source_cot: str = "",
                              run_dir: Optional[Path] = None) -> Path:
        save_dir = run_dir or (self.get_question_dir(question.id, "forcing") / f"rollout_{source_rollout_idx:03d}")
        save_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "question_id": question.id, "question_type": question.question_type,
            "source_rollout_idx": source_rollout_idx,
            "num_sentences": len(all_sentence_results),
            "source_cot": source_cot, "sentence_summaries": all_sentence_results,
        }
        if isinstance(question, GPQAQuestion):
            summary["correct_answer"] = question.correct_answer
        else:
            summary["bad_outcome"] = question.bad_outcome
        path = save_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sentence_summary(question: Question, sentence_idx: int,
                                partial_cot: str, results: List[Dict]) -> Dict:
        # New logprob-based format: use choice_probs directly if available
        if results and "choice_probs" in results[0] and results[0]["choice_probs"]:
            answer_distribution = results[0]["choice_probs"]
            answer = results[0].get("answer", "")
            return {
                "question_id": question.id, "question_type": question.question_type,
                "sentence_idx": sentence_idx, "partial_cot": partial_cot,
                "total_attempts": 1, "valid_answers": 1 if answer else 0,
                "answer_distribution": answer_distribution,
                "answer_counts": answer_distribution,  # backwards compat
            }

        # Legacy counting-based format
        answers = [r.get("answer", "").upper() for r in results]
        if isinstance(question, BinaryJudgeQuestion):
            valid = [a for a in answers if a in ["YES", "NO"]]
        else:
            valid = [a for a in answers if a in ["A", "B", "C", "D"]]
        counts = {}
        for a in valid:
            counts[a] = counts.get(a, 0) + 1
        return {
            "question_id": question.id, "question_type": question.question_type,
            "sentence_idx": sentence_idx, "partial_cot": partial_cot,
            "total_attempts": len(results), "valid_answers": len(valid),
            "answer_counts": counts,
        }

    @staticmethod
    def _resolve_choice_token_ids(tokenizer, choices: List[str]) -> Dict[str, int]:
        """Map each answer string (e.g. "A") to its single token id."""
        mapping = {}
        for c in choices:
            with contextlib.redirect_stdout(io.StringIO()):
                ids = tokenizer.encode(c, add_special_tokens=False)
            mapping[c] = ids[-1]
        return mapping

    def _get_choice_distribution(
        self,
        llm,
        tokenizer,
        question: "Question",
        partial_cot: str,
        choices: List[str],
        choice_token_ids: Dict[str, int],
        topk: int = 20,
    ) -> Tuple[str, Dict[str, Optional[float]], Dict[str, float]]:
        """
        Single-call logprob extraction using anchor + prompt_logprobs via vLLM.

        Returns:
            (full_prompt, choice_logprobs, choice_probs)
        """
        anchor = " So, the answer is: " if partial_cot else "So, the answer is: "
        cot_with_anchor = partial_cot + anchor
        prompt_str = build_thinking_prompt(
            tokenizer, self._user_msg(question), cot_prefix=cot_with_anchor,
        ) + "</think>\n"

        with contextlib.redirect_stdout(io.StringIO()):
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

        # Append a dummy answer token so prompt_logprobs[-1] gives the model's
        # distribution over the first response token (the answer letter).
        dummy_id = choice_token_ids[choices[0]]
        extended_tokens = prompt_tokens + [dummy_id]

        params = VllmSamplingParams(max_tokens=1, prompt_logprobs=topk)
        output = llm.generate(
            [{"prompt_token_ids": extended_tokens}], params, use_tqdm=False,
        )[0]

        last_pos = output.prompt_logprobs[-1] if output.prompt_logprobs else {}
        topk_lookup = {tid: entry.logprob for tid, entry in (last_pos or {}).items()}

        # Extract logprobs for answer choices
        choice_logprobs: Dict[str, Optional[float]] = {}
        for c in choices:
            tid = choice_token_ids[c]
            choice_logprobs[c] = topk_lookup.get(tid, None)

        # Softmax over found choices
        found = {c: lp for c, lp in choice_logprobs.items() if lp is not None}
        if found:
            max_lp = max(found.values())
            exps = {c: math.exp(lp - max_lp) for c, lp in found.items()}
            total = sum(exps.values())
            choice_probs = {c: exps.get(c, 0.0) / total for c in choices}
        else:
            choice_probs = {c: 0.0 for c in choices}

        return prompt_str, choice_logprobs, choice_probs

    @staticmethod
    def _user_msg(question: Question) -> str:
        if isinstance(question, BinaryJudgeQuestion):
            return question.question
        labels = [chr(ord("A") + i) for i in range(len(question.choices))]
        choices = "\n".join(f"{l}. {c}" for l, c in zip(labels, question.choices))
        labels_str = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 2 else " or ".join(labels)
        return f"{question.question}\n\n{choices}\n\nAnswer with just the letter ({labels_str})."

    @staticmethod
    def _extract_answer(tokens, tokenizer, question: Question):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if isinstance(question, BinaryJudgeQuestion):
            if "</think>" in text:
                after = text.split("</think>", 1)[1].strip().upper()
                if "YES" in after:
                    return "YES", text
                if "NO" in after:
                    return "NO", text
            return "", text
        if "</think>" in text:
            after = text.split("</think>", 1)[1].strip().upper().rstrip(".")
            if after in ["A", "B", "C", "D"]:
                return after, text
            match = re.search(r"\b([A-D])\b", after)
            if match:
                return match.group(1), text
        return "", text
