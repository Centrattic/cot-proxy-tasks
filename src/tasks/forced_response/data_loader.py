"""
Data loader for forced response task.

Loads questions and their verified CoTs from the verification_rollouts directory.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...utils.questions import GPQAQuestion, BinaryJudgeQuestion, Question


def load_question_and_cot(
    verification_dir: Path,
    question_id: str,
    rollout_idx: int = 0,
) -> Optional[Tuple[Question, str]]:
    """Load a Question object and its source CoT from verification data."""
    qdir = verification_dir / question_id
    if not qdir.exists():
        return None

    run_dir = get_latest_verification_dir(verification_dir, question_id)
    if run_dir is None:
        return None

    rollout_path = run_dir / "rollouts" / f"rollout_{rollout_idx:03d}.json"
    summary_path = run_dir / "summary.json"

    if not rollout_path.exists() or not summary_path.exists():
        return None

    with open(rollout_path) as f:
        rollout_data = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    question = question_from_summary(summary)
    source_cot = rollout_data.get("thinking", "") or rollout_data.get("full_response", "")
    if not source_cot:
        return None

    return question, source_cot


def get_verified_questions(
    verification_dir: Path,
    threshold: float = 0.8,
) -> List[str]:
    """Get question IDs that meet the verification agreement threshold."""
    verified = []
    if not verification_dir.exists():
        return verified
    for qdir in verification_dir.iterdir():
        if qdir.is_dir():
            summary = load_verification_summary(verification_dir, qdir.name)
            if summary and summary.get("agreement_rate", 0) >= threshold:
                verified.append(qdir.name)
    return sorted(verified)


def get_latest_verification_dir(
    verification_dir: Path,
    question_id: str,
) -> Optional[Path]:
    """Find the latest timestamped run directory for a question."""
    qdir = verification_dir / question_id
    if not qdir.exists():
        return None
    timestamped = sorted(
        [d for d in qdir.iterdir()
         if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'],
        reverse=True,
    )
    if timestamped:
        return timestamped[0]
    if (qdir / "summary.json").exists():
        return qdir
    return None


def load_verification_summary(
    verification_dir: Path,
    question_id: str,
) -> Optional[Dict]:
    """Load the summary.json for a verified question."""
    run_dir = get_latest_verification_dir(verification_dir, question_id)
    if run_dir:
        path = run_dir / "summary.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def question_from_summary(summary: Dict) -> Question:
    """Construct a Question object from a verification summary dict."""
    qt = summary.get("question_type", "multiple_choice")
    if qt == "binary_judge":
        return BinaryJudgeQuestion(
            id=summary["question_id"], question=summary["question"],
            judge_prompt=summary["judge_prompt"], bad_outcome=summary["bad_outcome"],
            subject=summary.get("subject"),
        )
    return GPQAQuestion(
        id=summary["question_id"], question=summary["question"],
        choices=summary["choices"], correct_answer=summary["correct_answer"],
        correct_index=ord(summary["correct_answer"]) - ord("A"),
    )
