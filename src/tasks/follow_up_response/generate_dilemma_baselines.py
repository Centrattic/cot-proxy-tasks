"""Generate baseline CoTs + confidence scores for dilemma prompts using Qwen 3 32B via Tinker."""
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
DEFAULT_NUM_RESAMPLES = 100
STORIES_DIR = Path(__file__).resolve().parent


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


def generate_single_baseline(
    sampling_client,
    tokenizer,
    types_module,
    cot_params,
    score_params,
    dilemma_key: str,
    dilemma_text: str,
    sample_idx: int,
    fresh: bool,
) -> None:
    safe_name = sanitize_name(dilemma_key)
    out_dir = STORIES_DIR / "baselines" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample_{sample_idx}.json"

    if out_path.exists() and not fresh:
        return

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Turn 1: Generate CoT + visible response
            prompt_str = build_thinking_prompt(tokenizer, dilemma_text)

            with contextlib.redirect_stdout(io.StringIO()):
                tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

            if len(tokens) + cot_params.max_tokens > 32768:
                return

            model_input = types_module.ModelInput.from_ints(tokens)
            result = sampling_client.sample(
                prompt=model_input, num_samples=1, sampling_params=cot_params,
            ).result()
            generated = tokenizer.decode(
                result.sequences[0].tokens, skip_special_tokens=True,
            ).strip()

            # Parse thinking vs visible response
            if "</think>" in generated:
                cot_text = generated.split("</think>", 1)[0].strip()
                response_text = generated.split("</think>", 1)[1].strip()
            else:
                cot_text = generated.strip()
                response_text = ""

            # Turn 2: Score the response via multi-turn conversation
            score_prompt_text = make_score_prompt()
            score_messages = [
                {"role": "user", "content": dilemma_text},
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
                return

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

    out = {
        "dilemma_key": dilemma_key,
        "dilemma_text": dilemma_text,
        "sample_idx": sample_idx,
        "cot_text": cot_text,
        "response_text": response_text,
        "score_prompt": score_prompt_text,
        "raw_score_response": raw_score_response,
        "parsed_score": parsed_score,
        "model": MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path.write_text(json.dumps(out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dilemma baselines via Tinker")
    parser.add_argument("--fresh", action="store_true", help="Regenerate all baselines")
    parser.add_argument("--dilemma", type=str, default=None, help="Run for a single dilemma key")
    parser.add_argument("--num-resamples", type=int, default=DEFAULT_NUM_RESAMPLES,
                        help=f"Number of resamples per dilemma (default: {DEFAULT_NUM_RESAMPLES})")
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

    cot_params = types.SamplingParams(max_tokens=4096, temperature=0.7)
    score_params = types.SamplingParams(max_tokens=256, temperature=0.7)

    all_tasks = []
    for dilemma_key, dilemma_text in dilemmas.items():
        for i in range(args.num_resamples):
            all_tasks.append((
                dilemma_key, dilemma_text, i, args.fresh,
            ))

    print(f"\n  {len(dilemmas)} dilemmas x {args.num_resamples} resamples = {len(all_tasks)} tasks")

    with ThreadPoolExecutor(max_workers=200) as executor:
        futures = {
            executor.submit(
                generate_single_baseline,
                sampling_client, tokenizer, types,
                cot_params, score_params,
                *task_args,
            ): task_args
            for task_args in all_tasks
        }
        for future in tqdm(futures, total=len(futures), desc="  baselines"):
            future.result()

    print("\nDone generating dilemma baselines.")


if __name__ == "__main__":
    main()
