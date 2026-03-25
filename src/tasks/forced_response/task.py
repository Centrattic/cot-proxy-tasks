"""
Forced answer entropy task — sentence-level answer distribution tracking.

Pipeline (two steps):
  1. generate_rollouts(): Load GPQA/custom questions, generate CoT rollouts
     via OpenRouter to get reasoning traces.
  2. run_forcing(): For each rollout, split CoT into sentences and use vLLM
     logprob forcing at each sentence boundary to extract the answer
     distribution. Stride=1 (every sentence).

Output per rollout: a JSON with sentence_summaries containing the answer
distribution {A: p, B: p, C: p, D: p} at each sentence boundary.
"""

import contextlib
import io
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openai
from dotenv import load_dotenv
from tqdm import tqdm

from ..base import BaseTask
from ...utils.questions import MultipleChoiceQuestion, GPQAQuestion, BinaryJudgeQuestion, Question
from ...utils.chat_template import build_thinking_prompt
from .utils import split_cot_into_sentences, get_cumulative_cot_segments
from .data_loader import load_gpqa_questions, CUSTOM_TRAIN_QUESTIONS

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_ROLLOUTS = 50
DEFAULT_MAX_WORKERS = 30
DEFAULT_MAX_SENTENCES = 30


class ForcingTask(BaseTask):
    """
    Forced answer entropy task.

    Two-step generation:
      generate_rollouts() — get CoT traces via OpenRouter
      run_forcing() — extract per-sentence answer distributions via vLLM
    """

    def __init__(
        self,
        model: str,
        data_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__("forcing", data_dir or (
            Path(__file__).parent.parent.parent.parent / "data" / "forced_response"
        ))
        self.model = model
        self.rollouts_dir = self.data_dir / "rollouts"
        self.forcing_dir = self.data_dir / "forcing"

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if self.api_key:
            self.client = openai.OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.api_key,
            )
        else:
            self.client = None

    # ------------------------------------------------------------------
    # Step 1: Generate rollouts (OpenRouter)
    # ------------------------------------------------------------------

    def generate_rollouts(
        self,
        questions: Optional[List[MultipleChoiceQuestion]] = None,
        num_rollouts: int = DEFAULT_NUM_ROLLOUTS,
        max_workers: int = DEFAULT_MAX_WORKERS,
        verbose: bool = True,
    ) -> None:
        """Generate CoT rollouts via OpenRouter for each question.

        Saves one JSON per question in self.rollouts_dir with the format:
        {question_id, question, choices, runs: [{rollout_idx, answer, thinking}]}
        """
        if self.client is None:
            raise RuntimeError("No OpenRouter API key available.")

        if questions is None:
            questions = list(CUSTOM_TRAIN_QUESTIONS) + load_gpqa_questions()
        if verbose:
            print(f"Loaded {len(questions)} questions")

        self.rollouts_dir.mkdir(parents=True, exist_ok=True)

        to_run = [
            q for q in questions
            if not (self.rollouts_dir / f"{q.id}.json").exists()
        ]
        if verbose:
            print(f"{len(to_run)} to generate ({len(questions) - len(to_run)} done)")

        for q in tqdm(to_run, desc="Questions", disable=not verbose):
            self._generate_question_rollouts(q, num_rollouts, max_workers)

        if verbose:
            print(f"Rollouts saved to {self.rollouts_dir}")

    def _generate_question_rollouts(
        self,
        question: MultipleChoiceQuestion,
        num_rollouts: int,
        max_workers: int,
    ) -> None:
        """Generate rollouts for a single question."""
        labels = question.labels or [chr(ord("A") + i) for i in range(len(question.choices))]
        choices_str = "\n".join(f"{l}. {c}" for l, c in zip(labels, question.choices))
        labels_str = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 2 else " or ".join(labels)
        prompt = f"{question.question}\n\n{choices_str}\n\nAnswer with just the letter ({labels_str})."

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {}
            for i in range(num_rollouts):
                fut = ex.submit(self._call_model, prompt)
                futs[fut] = i
            for fut in as_completed(futs):
                run_idx = futs[fut]
                result = fut.result()
                results.append({
                    "rollout_idx": run_idx,
                    "answer": result["answer"],
                    "thinking": result["thinking"],
                })

        results.sort(key=lambda x: x["rollout_idx"])
        labels_list = question.labels or [chr(ord("A") + i) for i in range(len(question.choices))]
        out = {
            "question_id": question.id,
            "question": question.question,
            "question_type": question.question_type,
            "choices": question.choices,
            "correct_answer": question.correct_answer,
            "labels": labels_list,
            "runs": results,
        }
        with open(self.rollouts_dir / f"{question.id}.json", "w") as f:
            json.dump(out, f, indent=2)

    # ------------------------------------------------------------------
    # Step 2: Logprob forcing (vLLM)
    # ------------------------------------------------------------------

    def run_forcing(
        self,
        question_ids: Optional[List[str]] = None,
        max_sentences: int = DEFAULT_MAX_SENTENCES,
        rollout_indices: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> None:
        """Run logprob forcing for each rollout's CoT, sentence by sentence.

        Requires vLLM and a local GPU. For each rollout, splits the CoT into
        sentences and extracts the model's answer distribution at each
        sentence boundary using logprob forcing.

        Args:
            question_ids: Which questions to force. None = all in rollouts_dir.
            max_sentences: Cap on sentences per rollout (stride=1).
            rollout_indices: Which rollout indices to force. None = all.
        """
        try:
            from vllm import LLM, SamplingParams as VllmSamplingParams
        except ImportError:
            raise ImportError("vLLM required for logprob forcing.")

        from transformers import AutoTokenizer

        self.forcing_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)

        # Discover questions
        if question_ids is None:
            question_ids = sorted(
                f.stem for f in self.rollouts_dir.glob("*.json")
            )

        if verbose:
            print(f"Forcing {len(question_ids)} questions, max_sentences={max_sentences}")

        # Load vLLM
        llm = LLM(model=self.model, trust_remote_code=True)

        for qid in tqdm(question_ids, desc="Forcing", disable=not verbose):
            rollout_file = self.rollouts_dir / f"{qid}.json"
            if not rollout_file.exists():
                continue
            with open(rollout_file) as f:
                q_data = json.load(f)

            choices = q_data.get("labels") or [chr(ord("A") + i) for i in range(len(q_data["choices"]))]
            choice_token_ids = self._resolve_choice_token_ids(tokenizer, choices)

            indices = rollout_indices or range(len(q_data["runs"]))
            for ridx in indices:
                if ridx >= len(q_data["runs"]):
                    break
                run = q_data["runs"][ridx]
                cot = run.get("thinking", "")
                if not cot:
                    continue

                # Check if already done
                out_path = self.forcing_dir / f"{qid}_rollout_{ridx:03d}.json"
                if out_path.exists():
                    continue

                sentences = split_cot_into_sentences(cot)
                if max_sentences and len(sentences) > max_sentences:
                    sentences = sentences[:max_sentences]

                cot_segments = get_cumulative_cot_segments(cot)
                if max_sentences and len(cot_segments) > max_sentences:
                    cot_segments = cot_segments[:max_sentences]

                # Force at each sentence boundary
                sentence_summaries = []
                for si in range(len(cot_segments)):
                    partial_cot = cot_segments[si]
                    _, choice_logprobs, choice_probs = self._get_choice_distribution(
                        llm, tokenizer, q_data, partial_cot, choices, choice_token_ids,
                    )
                    most_common = max(choice_probs, key=choice_probs.get) if any(v > 0 for v in choice_probs.values()) else ""
                    sentence_summaries.append({
                        "sentence_idx": si,
                        "partial_cot_length": len(partial_cot),
                        "total_forces": 1,
                        "valid_answers": 1 if most_common else 0,
                        "answer_distribution": choice_probs,
                        "answer_counts": choice_probs,
                        "most_common": most_common,
                    })

                result = {
                    "question_id": qid,
                    "question_type": q_data.get("question_type", "multiple_choice"),
                    "source_rollout_idx": ridx,
                    "num_sentences": len(sentence_summaries),
                    "correct_answer": q_data.get("correct_answer", ""),
                    "sentence_summaries": sentence_summaries,
                }
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

        if verbose:
            n_files = len(list(self.forcing_dir.glob("*.json")))
            print(f"Done. {n_files} forcing results in {self.forcing_dir}")

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def run_data(self, **kwargs) -> None:
        """Run both steps: generate rollouts then run forcing."""
        self.generate_rollouts(**{k: v for k, v in kwargs.items()
                                   if k in ("questions", "num_rollouts", "max_workers", "verbose")})
        self.run_forcing(**{k: v for k, v in kwargs.items()
                            if k in ("question_ids", "max_sentences", "rollout_indices", "verbose")})

    def get_data(self, load: bool = False) -> Union[bool, Optional[Any]]:
        if not load:
            return self.forcing_dir.exists() and any(self.forcing_dir.glob("*.json"))
        results = []
        for p in sorted(self.forcing_dir.glob("*.json")):
            with open(p) as f:
                results.append(json.load(f))
        return results if results else None

    # ------------------------------------------------------------------
    # Internal: OpenRouter model call
    # ------------------------------------------------------------------

    def _call_model(self, prompt: str) -> Dict[str, str]:
        """Call the model via OpenRouter."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                temperature=DEFAULT_TEMPERATURE,
                extra_body={"reasoning": {"enabled": True}},
                timeout=120,
            )
            msg = response.choices[0].message
            thinking = ""
            # OpenRouter puts reasoning in model_extra
            extra = getattr(msg, "model_extra", {}) or {}
            if extra.get("reasoning"):
                thinking = extra["reasoning"]
            elif hasattr(msg, "reasoning_content") and msg.reasoning_content:
                thinking = msg.reasoning_content
            elif hasattr(msg, "reasoning") and msg.reasoning:
                thinking = msg.reasoning
            answer = self._parse_answer(msg.content or "")
            return {"thinking": thinking, "answer": answer}
        except Exception as e:
            return {"thinking": "", "answer": "", "error": str(e)}

    # ------------------------------------------------------------------
    # Internal: vLLM logprob forcing
    # ------------------------------------------------------------------

    def _get_choice_distribution(
        self,
        llm,
        tokenizer,
        q_data: Dict,
        partial_cot: str,
        choices: List[str],
        choice_token_ids: Dict[str, int],
        topk: int = 20,
    ) -> Tuple[str, Dict[str, Optional[float]], Dict[str, float]]:
        """Extract answer distribution via logprob forcing with vLLM."""
        try:
            from vllm import SamplingParams as VllmSamplingParams
        except ImportError:
            raise ImportError("vLLM required.")

        anchor = " So, the answer is: " if partial_cot else "So, the answer is: "
        cot_with_anchor = partial_cot + anchor

        user_msg = self._format_user_msg(q_data)
        prompt_str = build_thinking_prompt(
            tokenizer, user_msg, cot_prefix=cot_with_anchor,
        ) + "</think>\n"

        with contextlib.redirect_stdout(io.StringIO()):
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

        dummy_id = choice_token_ids[choices[0]]
        extended_tokens = prompt_tokens + [dummy_id]

        params = VllmSamplingParams(max_tokens=1, prompt_logprobs=topk)
        output = llm.generate(
            [{"prompt_token_ids": extended_tokens}], params, use_tqdm=False,
        )[0]

        last_pos = output.prompt_logprobs[-1] if output.prompt_logprobs else {}
        topk_lookup = {tid: entry.logprob for tid, entry in (last_pos or {}).items()}

        choice_logprobs: Dict[str, Optional[float]] = {}
        for c in choices:
            tid = choice_token_ids[c]
            choice_logprobs[c] = topk_lookup.get(tid, None)

        found = {c: lp for c, lp in choice_logprobs.items() if lp is not None}
        if found:
            max_lp = max(found.values())
            exps = {c: math.exp(lp - max_lp) for c, lp in found.items()}
            total = sum(exps.values())
            choice_probs = {c: exps.get(c, 0.0) / total for c in choices}
        else:
            choice_probs = {c: 0.0 for c in choices}

        return prompt_str, choice_logprobs, choice_probs

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_user_msg(q_data: Dict) -> str:
        """Format the user message from question data."""
        choices = q_data["choices"]
        labels = q_data.get("labels") or [chr(ord("A") + i) for i in range(len(choices))]
        choices_str = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
        labels_str = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 2 else " or ".join(labels)
        return f"{q_data['question']}\n\n{choices_str}\n\nAnswer with just the letter ({labels_str})."

    @staticmethod
    def _resolve_choice_token_ids(tokenizer, choices: List[str]) -> Dict[str, int]:
        """Map each answer string to its single token id."""
        mapping = {}
        for c in choices:
            with contextlib.redirect_stdout(io.StringIO()):
                ids = tokenizer.encode(c, add_special_tokens=False)
            mapping[c] = ids[-1]
        return mapping

    @staticmethod
    def _parse_answer(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text[0] in "ABCD":
            return text[0]
        m = re.search(r"\b([A-D])\b", text)
        return m.group(1) if m else ""
