"""Resample confidence scores with forced CoTs for dilemmas using Qwen 3 32B via Tinker.

For each resample:
  1. Prefill the fixed CoT and generate a new visible response.
  2. Build a multi-turn conversation with the new response, ask the confidence
     question, and parse the answer.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src2"))
from utils.chat_template import build_thinking_prompt  # noqa: E402

MODEL = "Qwen/Qwen3-32B"
DEFAULT_NUM_RESAMPLES = 30
STORIES_DIR = Path(__file__).resolve().parent
QUESTION_KEY = "confident"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def parse_score(text: str) -> int | None:
    match = re.search(r"\b(10|[1-9])\b", text)
    if match:
        return int(match.group(1))
    return None


def make_score_prompt() -> str:
    return (
        "On a scale from 1 to 10, how confident are you? "
        "Output only your answer, which is one number between 1 and 10."
    )


def score_single(
    sampling_client,
    tokenizer,
    types_module,
    regen_params,
    score_params,
    dilemma_key: str,
    dilemma_text: str,
    cot_data: dict,
    cot_idx: int,
    score_idx: int,
    fresh: bool,
) -> None:
    safe_name = sanitize_name(dilemma_key)
    out_dir = STORIES_DIR / "forced_scores" / QUESTION_KEY / safe_name / f"cot_{cot_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"score_{score_idx}.json"

    if out_path.exists() and not fresh:
        return

    cot_text = cot_data.get("cot_text", "")
    user_msg = dilemma_text

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Step 1: Regenerate visible response with fixed CoT prefill.
            regen_prompt = build_thinking_prompt(tokenizer, user_msg, cot_prefix=cot_text)
            regen_prompt += "\n</think>\n"

            with contextlib.redirect_stdout(io.StringIO()):
                regen_tokens = tokenizer.encode(regen_prompt, add_special_tokens=False)

            if len(regen_tokens) + regen_params.max_tokens > 32768:
                return  # CoT too long for context window

            regen_input = types_module.ModelInput.from_ints(regen_tokens)

            regen_result = sampling_client.sample(
                prompt=regen_input, num_samples=1, sampling_params=regen_params,
            ).result()
            response_text = tokenizer.decode(
                regen_result.sequences[0].tokens, skip_special_tokens=True,
            ).strip()

            # Step 2: Score the regenerated response via multi-turn conversation.
            score_prompt_text = make_score_prompt()
            score_messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": score_prompt_text},
            ]
            score_prompt_str = tokenizer.apply_chat_template(
                score_messages, add_generation_prompt=True, tokenize=False,
            )
            score_prompt_str += "<think>\n</think>\n"

            with contextlib.redirect_stdout(io.StringIO()):
                score_tokens = tokenizer.encode(score_prompt_str, add_special_tokens=False)

            if len(score_tokens) + score_params.max_tokens > 32768:
                return  # Score prompt too long for context window

            score_input = types_module.ModelInput.from_ints(score_tokens)

            score_result = sampling_client.sample(
                prompt=score_input, num_samples=1, sampling_params=score_params,
            ).result()
            raw_score_response = tokenizer.decode(
                score_result.sequences[0].tokens, skip_special_tokens=True,
            ).strip()

            break  # Success
        except (ValueError, Exception) as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt + 1)
                continue
            raise

    parsed_score = parse_score(raw_score_response)

    result = {
        "dilemma_key": dilemma_key,
        "question_key": QUESTION_KEY,
        "cot_idx": cot_idx,
        "score_idx": score_idx,
        "cot_text": cot_text,
        "response_text": response_text,
        "score_prompt": score_prompt_text,
        "raw_score_response": raw_score_response,
        "parsed_score": parsed_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path.write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate forced-CoT dilemma scores via Tinker")
    parser.add_argument("--fresh", action="store_true", help="Regenerate all scores")
    parser.add_argument("--dilemma", type=str, default=None, help="Run for a single dilemma key")
    parser.add_argument("--num-cots", type=int, default=None,
                        help="Number of CoTs to score (default: all found)")
    parser.add_argument("--num-resamples", type=int, default=DEFAULT_NUM_RESAMPLES,
                        help=f"Resamples per CoT (default: {DEFAULT_NUM_RESAMPLES})")
    parser.add_argument("--prompts-file", type=str, default="dilemma_prompts.json",
                        help="Prompts JSON file (default: dilemma_prompts.json)")
    args = parser.parse_args()

    dilemmas: dict[str, str] = json.loads(
        (STORIES_DIR / args.prompts_file).read_text()
    )

    if args.dilemma:
        if args.dilemma not in dilemmas:
            print(f"Dilemma '{args.dilemma}' not found. Available: {list(dilemmas.keys())}")
            sys.exit(1)
        dilemmas = {args.dilemma: dilemmas[args.dilemma]}

    from tinker import ServiceClient, types
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    client = ServiceClient()
    sampling_client = client.create_sampling_client(base_model=MODEL)

    regen_params = types.SamplingParams(max_tokens=2048, temperature=0.7)
    score_params = types.SamplingParams(max_tokens=256, temperature=0.7)

    for dilemma_key, dilemma_text in dilemmas.items():
        safe_name = sanitize_name(dilemma_key)
        cot_dir = STORIES_DIR / "selected_cots" / safe_name
        if not cot_dir.exists():
            print(f"  No selected CoTs for '{dilemma_key}', skipping")
            continue

        cot_files = sorted(cot_dir.glob("cot_*.json"))
        if args.num_cots is not None:
            cot_files = cot_files[:args.num_cots]
        num_cots = len(cot_files)

        print(f"\n  --- {dilemma_key} ({num_cots} CoTs x {args.num_resamples} resamples) ---")
        tasks = []
        for cot_path in cot_files:
            m = re.match(r"cot_(\d+)\.json", cot_path.name)
            if not m:
                continue
            cot_idx = int(m.group(1))
            cot_data = json.loads(cot_path.read_text())
            for score_idx in range(args.num_resamples):
                tasks.append((
                    dilemma_key, dilemma_text,
                    cot_data, cot_idx, score_idx, args.fresh,
                ))

        with ThreadPoolExecutor(max_workers=min(200, max(1, len(tasks)))) as executor:
            futures = {
                executor.submit(
                    score_single,
                    sampling_client, tokenizer, types,
                    regen_params, score_params,
                    *task_args,
                ): task_args
                for task_args in tasks
            }
            for future in tqdm(futures, total=len(futures), desc=f"  {dilemma_key}"):
                future.result()

        score_base = STORIES_DIR / "forced_scores" / QUESTION_KEY / safe_name
        n_existing = sum(
            1
            for cot_d in score_base.iterdir() if cot_d.is_dir()
            for f in cot_d.glob("score_*.json")
        ) if score_base.exists() else 0
        print(f"    {n_existing} score files total")

    print("\nDone generating forced dilemma scores.")


if __name__ == "__main__":
    main()
