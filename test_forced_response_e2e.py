"""
End-to-end test: generate 1 rollout via OpenRouter (Qwen3-32B), then run
vLLM forcing with a smaller model (Qwen3-8B) to verify the pipeline works.
"""

import contextlib
import io
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*torch_dtype.*")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from dotenv import load_dotenv
load_dotenv(Path("/home/riya/neel-projs/cot-comparisons/.env"))

sys.path.insert(0, str(Path(__file__).parent))
from src.tasks.forced_response.utils import split_cot_into_sentences, get_cumulative_cot_segments
from src.utils.chat_template import build_thinking_prompt

ROLLOUT_MODEL = "Qwen/Qwen3-32B"
FORCING_MODEL = "Qwen/Qwen3-0.6B"
RELEASE_DIR = Path("/home/riya/neel-projs/cot-comparisons/release_datasets/forced_answer_entropy")


def main():
    with open(sorted((RELEASE_DIR / "train_set" / "prompts").glob("*.json"))[1]) as f:
        q = json.load(f)

    labels = [chr(ord("A") + i) for i in range(len(q["choices"]))]
    choices_str = "\n".join(f"{l}. {c}" for l, c in zip(labels, q["choices"]))
    labels_str = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    user_msg = f"{q['question']}\n\n{choices_str}\n\nAnswer with just the letter ({labels_str})."

    print(f"Question: {q['question_id']}")

    # === Step 1: Generate rollout via OpenRouter ===
    print("\n=== Step 1: OpenRouter rollout ===")
    import openai
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    resp = client.chat.completions.create(
        model=ROLLOUT_MODEL,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=16384,
        temperature=0.7,
        extra_body={"reasoning": {"enabled": True}},
        timeout=120,
    )
    msg = resp.choices[0].message
    extra = getattr(msg, "model_extra", {}) or {}
    thinking = extra.get("reasoning", "")
    answer = (msg.content or "").strip()

    print(f"Answer: {answer[:20]}")
    print(f"Thinking: {len(thinking)} chars")

    sentences = split_cot_into_sentences(thinking)
    cot_segments = get_cumulative_cot_segments(thinking)
    print(f"Sentences: {len(sentences)}")

    if not thinking:
        print("No thinking, exiting.")
        return

    # === Step 2: vLLM forcing ===
    print(f"\n=== Step 2: vLLM forcing ({FORCING_MODEL}) ===")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams as VllmSamplingParams

    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(FORCING_MODEL, trust_remote_code=True)
    llm = LLM(model=FORCING_MODEL, trust_remote_code=True, gpu_memory_utilization=0.55)

    choices = labels
    choice_token_ids = {}
    for c in choices:
        with contextlib.redirect_stdout(io.StringIO()):
            ids = tokenizer.encode(c, add_special_tokens=False)
        choice_token_ids[c] = ids[-1]
    print(f"Choice token IDs: {choice_token_ids}")

    N_TEST = min(5, len(cot_segments))
    print(f"Forcing {N_TEST} sentences...")

    for si in range(N_TEST):
        partial_cot = cot_segments[si]
        anchor = " So, the answer is: " if partial_cot else "So, the answer is: "
        cot_with_anchor = partial_cot + anchor

        prompt_str = build_thinking_prompt(
            tokenizer, user_msg, cot_prefix=cot_with_anchor,
        ) + "</think>\n"

        with contextlib.redirect_stdout(io.StringIO()):
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

        dummy_id = choice_token_ids[choices[0]]
        extended_tokens = prompt_tokens + [dummy_id]

        params = VllmSamplingParams(max_tokens=1, prompt_logprobs=20)
        output = llm.generate(
            [{"prompt_token_ids": extended_tokens}], params, use_tqdm=False,
        )[0]

        last_pos = output.prompt_logprobs[-1] if output.prompt_logprobs else {}
        topk_lookup = {tid: entry.logprob for tid, entry in (last_pos or {}).items()}

        choice_logprobs = {}
        for c in choices:
            choice_logprobs[c] = topk_lookup.get(choice_token_ids[c], None)

        found = {c: lp for c, lp in choice_logprobs.items() if lp is not None}
        if found:
            max_lp = max(found.values())
            exps = {c: math.exp(lp - max_lp) for c, lp in found.items()}
            total = sum(exps.values())
            choice_probs = {c: exps.get(c, 0.0) / total for c in choices}
        else:
            choice_probs = {c: 0.0 for c in choices}

        most_common = max(choice_probs, key=choice_probs.get) if any(v > 0 for v in choice_probs.values()) else ""
        dist_str = ", ".join(f"{c}:{p:.3f}" for c, p in choice_probs.items())
        print(f"  [{si}] {{{dist_str}}} -> {most_common}")

    print("\nPipeline fire! Format matches release dataset structure.")


if __name__ == "__main__":
    main()
