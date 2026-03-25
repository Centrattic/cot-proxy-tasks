"""
Test: generate 1 rollout for a GPQA question and verify sentence splitting
matches what the release dataset would produce.
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path("/home/riya/neel-projs/cot-comparisons/.env"))

sys.path.insert(0, str(Path(__file__).parent))
from src.tasks.forced_response.utils import split_cot_into_sentences

import openai

MODEL = "Qwen/Qwen3-32B"
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Use a question from the release dataset
RELEASE_DIR = Path("/home/riya/neel-projs/cot-comparisons/release_datasets/forced_answer_entropy")
with open(sorted((RELEASE_DIR / "train_set" / "prompts").glob("*.json"))[1]) as f:
    q = json.load(f)

print(f"Question: {q['question_id']}")
print(f"Choices: {q['choices']}")

# Build prompt
labels = [chr(ord("A") + i) for i in range(len(q["choices"]))]
choices_str = "\n".join(f"{l}. {c}" for l, c in zip(labels, q["choices"]))
labels_str = ", ".join(labels[:-1]) + f", or {labels[-1]}"
prompt = f"{q['question']}\n\n{choices_str}\n\nAnswer with just the letter ({labels_str})."

print(f"\nGenerating 1 rollout...")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=16384,
    temperature=0.7,
    extra_body={"reasoning": {"enabled": True}},
    timeout=120,
)
msg = resp.choices[0].message
thinking = ""
if hasattr(msg, "reasoning_content") and msg.reasoning_content:
    thinking = msg.reasoning_content

answer_text = msg.content or ""
answer = answer_text.strip()[0] if answer_text.strip() and answer_text.strip()[0] in "ABCD" else ""

print(f"Answer: {answer}")
print(f"Thinking length: {len(thinking)} chars")

sentences = split_cot_into_sentences(thinking)
print(f"Sentences: {len(sentences)}")
print(f"First 3 sentences:")
for i, s in enumerate(sentences[:3]):
    print(f"  [{i}] {s[:80]}...")

# Compare with release: check what the release has for this question
r_dir = RELEASE_DIR / "train_set" / "qwen-32b"
release_files = sorted(r_dir.glob(f"{q['question_id']}_*.json"))
print(f"\nRelease has {len(release_files)} rollouts for this question")
if release_files:
    with open(release_files[0]) as f:
        rd = json.load(f)
    print(f"  Release rollout 0: {rd['num_sentences']} sentences")
    print(f"  Our generation: {len(sentences)} sentences")
    print(f"  (Different CoT, so sentence count will differ - just checking pipeline works)")
