"""
Test: generate rollouts for 2 questions from the release dataset and check
that majority/minority labels match.

Uses OpenRouter API to generate 50 rollouts per question (not 200, to save
cost), then checks if the majority answer matches the release data.
"""

import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path("/home/riya/neel-projs/cot-comparisons/.env"))

import openai

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "Qwen/Qwen3-32B"
TEMPERATURE = 0.7
NUM_ROLLOUTS = 50
MAX_WORKERS = 30

RELEASE_DIR = Path("/home/riya/neel-projs/cot-comparisons/release_datasets/atypical_answer")
VERIF_DIR = Path("/home/riya/neel-projs/cot-comparisons/data/verification_rollouts")

client = openai.OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def get_verif_summary(qid):
    qdir = VERIF_DIR / qid
    if not qdir.exists():
        return None
    ts = sorted([d for d in qdir.iterdir() if d.is_dir() and len(d.name) == 15], reverse=True)
    if not ts:
        return None
    with open(ts[0] / "summary.json") as f:
        return json.load(f)


# Pick 2 dilemma questions from the release dataset
test_questions = []
for split in ["train_set", "test_set"]:
    p_dir = RELEASE_DIR / split / "prompts"
    for f in sorted(p_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        if d["question_id"].startswith("dilemma_"):
            summary = get_verif_summary(d["question_id"])
            if not summary:
                continue
            # Get release majority answer
            r_dir = RELEASE_DIR / split / "qwen-32b"
            release_maj = None
            for rf in r_dir.glob(f"{d['question_id']}_*.json"):
                with open(rf) as fh:
                    rd = json.load(fh)
                release_maj = rd["majority_answer"]
                break
            test_questions.append({
                "question_id": d["question_id"],
                "question_text": d["question_text"],
                "choices": summary["choices"],
                "release_majority": release_maj,
            })
            break  # one per split

print(f"Testing {len(test_questions)} questions")


def parse_answer(text):
    text = text.strip()
    if not text:
        return ""
    if text[0] in "ABCD":
        return text[0]
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else ""


def call_model(prompt):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=TEMPERATURE,
            extra_body={"reasoning": {"enabled": True}},
            timeout=90,
        )
        msg = resp.choices[0].message
        thinking = ""
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            thinking = msg.reasoning_content
        return {"thinking": thinking, "answer": parse_answer(msg.content or "")}
    except Exception as e:
        return {"thinking": "", "answer": "", "error": str(e)}


for q in test_questions:
    qid = q["question_id"]
    print(f"\n{'='*60}")
    print(f"Question: {qid}")
    print(f"Text: {q['question_text'][:80]}...")
    print(f"Release majority: {q['release_majority']}")

    # Build prompt using actual choices (order varies per question)
    choices = q["choices"]
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    choices_str = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    labels_str = " or ".join(labels)
    prompt = f"{q['question_text']}\n\n{choices_str}\n\nAnswer with just the letter ({labels_str})."

    print(f"Generating {NUM_ROLLOUTS} rollouts...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(call_model, prompt) for _ in range(NUM_ROLLOUTS)]
        for fut in futs:
            results.append(fut.result())

    answers = [r["answer"] for r in results if r["answer"]]
    counts = Counter(answers)
    majority = counts.most_common(1)[0][0] if counts else ""
    minority_rate = (len(answers) - counts[majority]) / len(answers) if answers else 0

    print(f"  Answers: {dict(counts)} ({len(answers)} valid)")
    print(f"  Our majority: {majority}, minority_rate: {minority_rate:.3f}")
    print(f"  Release majority: {q['release_majority']}")
    print(f"  MATCH: {'YES' if majority == q['release_majority'] else 'NO'}")

print(f"\n{'='*60}")
print("Done")
