"""
Atypical answer (majority/minority) classification task.

Pipeline:
  1. Load dilemma questions from Daily Dilemmas (kellycyy/daily_dilemmas)
  2. Generate 200+ rollouts per question via vLLM to get answer distributions
  3. Filter questions: minority rate between 10-33%, ≥200 rollouts
  4. Compute majority/minority labels per rollout
  5. Sample 15 majority + 15 minority rollouts per question (balanced)
  6. Stratified train/val/test split at question level

Key constraint for methods: cannot access other rollouts on the SAME question,
only rollouts from OTHER questions (e.g. for few-shot examples).
"""

import json
import os
import re
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from ...data_slice import DataSlice
from ...utils.questions import MultipleChoiceQuestion
from ..base import BaseTask
from .data_loader import load_dilemmas_from_huggingface

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_ROLLOUTS = 200
DEFAULT_MAX_WORKERS = 30

# Dataset filtering thresholds
MIN_ROLLOUTS = 200
MIN_MINORITY_RATE = 0.10
MAX_MINORITY_RATE = 0.33
N_MAJORITY_PER_QUESTION = 15
N_MINORITY_PER_QUESTION = 15


class MinMajAnswerTask(BaseTask):
    """
    Atypical answer task.

    run_data() generates rollouts for each question, then build_dataset()
    filters, labels, samples, and splits into train/val/test.
    """

    def __init__(
        self,
        subject_model: str,
        data_dir: Optional[Path] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        api_key: Optional[str] = None,
    ):
        super().__init__(name="min_maj_answer", data_dir=data_dir)
        self.subject_model = subject_model
        self.temperature = temperature
        self.max_workers = max_workers
        self.rollouts_dir = self.data_dir / "rollouts"

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if self.api_key:
            self.client = openai.OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.api_key,
            )
        else:
            self.client = None

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def run_data(
        self,
        questions: Optional[List[MultipleChoiceQuestion]] = None,
        max_questions: int = 1000,
        num_rollouts: int = DEFAULT_NUM_ROLLOUTS,
        verbose: bool = True,
    ) -> None:
        """Generate rollouts for each question.

        For each question, generates num_rollouts responses and saves them
        as one JSON file per question in self.rollouts_dir.
        """
        if self.client is None:
            raise RuntimeError("No OpenRouter API key available.")

        if questions is None:
            questions = load_dilemmas_from_huggingface(max_questions=max_questions)
        if verbose:
            print(f"Loaded {len(questions)} questions")

        self.rollouts_dir.mkdir(parents=True, exist_ok=True)

        # Skip questions that already have rollouts
        to_run = [
            q for q in questions
            if not (self.rollouts_dir / f"{q.id}.json").exists()
        ]
        if verbose:
            print(f"{len(to_run)} to generate ({len(questions) - len(to_run)} done)")

        for q in tqdm(to_run, desc="Questions", disable=not verbose):
            self._generate_question_rollouts(q, num_rollouts)

        # Save questions metadata
        questions_meta = []
        for q in questions:
            labels = q.labels or [chr(ord("A") + i) for i in range(len(q.choices))]
            questions_meta.append({
                "question_id": q.id,
                "question_text": q.question,
                "choices": dict(zip(labels, q.choices)),
                "subject": q.subject,
            })
        with open(self.rollouts_dir / "questions.json", "w") as f:
            json.dump(questions_meta, f, indent=2)

        if verbose:
            print(f"Saved rollouts for {len(questions)} questions to {self.rollouts_dir}")

    def get_data(
        self, load: bool = False
    ) -> Union[bool, Optional[Dict[str, Any]]]:
        questions_file = self.rollouts_dir / "questions.json"
        if not load:
            return questions_file.exists()
        if not questions_file.exists():
            return None
        with open(questions_file) as f:
            return {"questions": json.load(f)}

    # ------------------------------------------------------------------
    # Dataset building (from existing rollouts)
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        min_rollouts: int = MIN_ROLLOUTS,
        min_minority_rate: float = MIN_MINORITY_RATE,
        max_minority_rate: float = MAX_MINORITY_RATE,
        n_majority: int = N_MAJORITY_PER_QUESTION,
        n_minority: int = N_MINORITY_PER_QUESTION,
        seed: int = 42,
        test_split: float = 0.20,
        val_split: float = 0.15,
    ) -> DataSlice:
        """Build a balanced dataset from generated rollouts.

        Filters questions by rollout count and minority rate, samples balanced
        majority/minority rollouts, and splits at question level.
        """
        from sklearn.model_selection import train_test_split

        questions_file = self.rollouts_dir / "questions.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"No questions.json in {self.rollouts_dir}")
        with open(questions_file) as f:
            questions = {q["question_id"]: q for q in json.load(f)}

        rng = random.Random(seed)
        all_rows: List[Dict[str, Any]] = []
        valid_qids: List[str] = []

        for qid, q_meta in sorted(questions.items()):
            rollout_file = self.rollouts_dir / f"{qid}.json"
            if not rollout_file.exists():
                continue
            with open(rollout_file) as f:
                data = json.load(f)

            runs = [r for r in data["runs"] if r.get("answer")]
            if len(runs) < min_rollouts:
                continue

            # Compute majority/minority
            answers = [r["answer"] for r in runs]
            counts = Counter(answers)
            majority_answer = counts.most_common(1)[0][0]
            minority_count = len(runs) - counts[majority_answer]
            minority_rate = minority_count / len(runs)

            if minority_rate < min_minority_rate or minority_rate > max_minority_rate:
                continue

            # Split into majority/minority pools
            maj_runs = [r for r in runs if r["answer"] == majority_answer]
            min_runs = [r for r in runs if r["answer"] != majority_answer]

            # Sample balanced set
            sampled_maj = rng.sample(maj_runs, min(n_majority, len(maj_runs)))
            sampled_min = rng.sample(min_runs, min(n_minority, len(min_runs)))

            is_dilemma = qid.startswith("dilemma_")

            for r in sampled_maj + sampled_min:
                is_maj = r["answer"] == majority_answer
                all_rows.append({
                    "question_id": qid,
                    "question_text": q_meta["question_text"],
                    "rollout_idx": r["rollout_idx"],
                    "cot_content": r.get("thinking", ""),
                    "answer": r["answer"],
                    "label": "majority" if is_maj else "minority",
                    "is_majority": is_maj,
                    "majority_answer": majority_answer,
                    "minority_rate": minority_rate,
                    "is_dilemma": is_dilemma,
                })

            valid_qids.append(qid)

        full_df = pd.DataFrame(all_rows)
        if full_df.empty:
            return DataSlice()

        n_maj = len(full_df[full_df["label"] == "majority"])
        n_min = len(full_df[full_df["label"] == "minority"])
        print(f"Dataset: {n_maj} majority, {n_min} minority, {len(valid_qids)} questions")

        # Stratified split at question level
        # Use is_dilemma as stratification variable
        qid_is_dilemma = {qid: qid.startswith("dilemma_") for qid in valid_qids}
        strata = ["dilemma" if qid_is_dilemma[q] else "other" for q in valid_qids]

        trainval_qids, test_qids = train_test_split(
            valid_qids, test_size=test_split, random_state=seed, stratify=strata,
        )
        trainval_strata = [
            "dilemma" if qid_is_dilemma[q] else "other" for q in trainval_qids
        ]
        val_frac = val_split / (1.0 - test_split)
        train_qids, val_qids = train_test_split(
            trainval_qids, test_size=val_frac, random_state=seed,
            stratify=trainval_strata,
        )

        train_df = full_df[full_df["question_id"].isin(set(train_qids))].reset_index(drop=True)
        val_df = full_df[full_df["question_id"].isin(set(val_qids))].reset_index(drop=True)
        test_df = full_df[full_df["question_id"].isin(set(test_qids))].reset_index(drop=True)

        print(f"  Split: {len(set(train_qids))} train, {len(set(val_qids))} val, "
              f"{len(set(test_qids))} test questions")

        return DataSlice(train_df=train_df, val_df=val_df, test_df=test_df)

    # ------------------------------------------------------------------
    # Internal: rollout generation
    # ------------------------------------------------------------------

    def _generate_question_rollouts(
        self,
        question: MultipleChoiceQuestion,
        num_rollouts: int,
    ) -> None:
        """Generate rollouts for a single question and save to disk."""
        labels = question.labels or [chr(ord("A") + i) for i in range(len(question.choices))]
        choices_str = "\n".join(f"{l}. {c}" for l, c in zip(labels, question.choices))
        labels_str = " or ".join(labels)
        prompt = (
            f"{question.question}\n\n{choices_str}\n\n"
            f"Answer with just the letter ({labels_str})."
        )

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
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
        out_path = self.rollouts_dir / f"{question.id}.json"
        with open(out_path, "w") as f:
            json.dump({"question_id": question.id, "runs": results}, f, indent=2)

    def _call_model(self, prompt: str) -> Dict[str, str]:
        """Call the model and return {thinking, answer}."""
        try:
            response = self.client.chat.completions.create(
                model=self.subject_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
                temperature=self.temperature,
                extra_body={"reasoning": {"enabled": True}},
                timeout=90,
            )
            msg = response.choices[0].message
            thinking = ""
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                thinking = msg.reasoning_content
            elif hasattr(msg, "reasoning") and msg.reasoning:
                thinking = msg.reasoning
            answer = self._parse_answer(msg.content or "")
            return {"thinking": thinking, "answer": answer}
        except Exception as e:
            return {"thinking": "", "answer": "", "error": str(e)}

    @staticmethod
    def _parse_answer(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text[0] in "ABCD":
            return text[0]
        m = re.search(r"\b([A-D])\b", text)
        return m.group(1) if m else ""
