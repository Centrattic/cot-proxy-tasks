#!/usr/bin/env python3
"""
Build ood_answer_val_set_v8: OOD validation set from held-out GPQA chemistry questions.

Holds out 30 GPQA chemistry questions from the 178 in the OOD train set,
then runs the v8 pipeline (label → extract → resample → judge → build) on
the held-out questions. Also rebuilds the OOD train set without the held-out
questions (v5).

Phases:
  label         - Label rollouts with Claude Sonnet 4.5
  extract       - Extract answer-emission prefixes
  resample      - Resample 50 continuations per prefix (Qwen3-32B, GPU)
  judge         - Token-position judging → ood_val_judged/
  build         - Build balanced OOD val set
  rebuild_train - Filter ood_answer_train_set_v4.json → v5 (without held-out)
  all           - Run all phases

Usage:
    python -m src2.runs.run_build_ood_val_v8 label extract        # local
    python -m src2.runs.run_build_ood_val_v8 resample              # GPU pod
    python -m src2.runs.run_build_ood_val_v8 judge build rebuild_train  # local
"""

import argparse
import asyncio
import contextlib
import io
import json
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────

SUBJECT_MODEL = "Qwen/Qwen3-32B"
LABELING_MODEL = "anthropic/claude-sonnet-4.5"
SEED = 42

DATA_DIR = Path("data/reasoning_evals")
VAL_SET_PATH = DATA_DIR / "ood_answer_val_set_v8.json"
OOD_TRAIN_V4_PATH = DATA_DIR / "ood_answer_train_set_v4.json"
OOD_TRAIN_V5_PATH = DATA_DIR / "ood_answer_train_set_v5.json"
OOD_VAL_JUDGED_DIR = DATA_DIR / "ood_val_judged"

# Directory layout (separate from math)
OOD_ROLLOUTS_DIR = DATA_DIR / "rollouts" / "ood_chemistry"
OOD_LABELED_DIR = DATA_DIR / "rollouts" / "ood_val_labeled_answer"
OOD_PREFIXES_DIR = DATA_DIR / "prefixes" / "ood_val_answer"
OOD_RESAMPLES_DIR = DATA_DIR / "resamples" / "ood_val_answer"

NUM_ROLLOUTS = 200
GENERATE_MODEL = "qwen/qwen3-32b"
GENERATE_WORKERS = 100
DATASET_CSV = Path("dataset/gpqa_main.csv")
NUM_RESAMPLES = 50
MAX_RESAMPLE_TOKENS = 200
TEMPERATURE = 0.7
WORKERS = 50

# Token-position judging thresholds
YES_TOKEN_MIN = 20
YES_TOKEN_MAX = 60
NO_TOKEN_MIN = 200
YES_THRESHOLD = 45
NO_THRESHOLD = 45

# Length range (tokens)
LENGTH_MIN = 500
LENGTH_MAX = 3000

# Number of questions to hold out
# Hardcoded holdout IDs — frozen from the original seed=42 selection
# (when 129 questions had >= 10 rollouts). Must stay fixed so train v5
# and val sets remain consistent.
HOLDOUT_IDS = [
    "gpqa_chem_001", "gpqa_chem_002", "gpqa_chem_007", "gpqa_chem_009",
    "gpqa_chem_018", "gpqa_chem_025", "gpqa_chem_028", "gpqa_chem_037",
    "gpqa_chem_043", "gpqa_chem_047", "gpqa_chem_071", "gpqa_chem_073",
    "gpqa_chem_074", "gpqa_chem_104", "gpqa_chem_105", "gpqa_chem_108",
    "gpqa_chem_117", "gpqa_chem_123", "gpqa_chem_128", "gpqa_chem_129",
    "gpqa_chem_135", "gpqa_chem_136", "gpqa_chem_143", "gpqa_chem_145",
    "gpqa_chem_147", "gpqa_chem_151", "gpqa_chem_153", "gpqa_chem_155",
    "gpqa_chem_173", "gpqa_chem_178",
]


# ── Holdout selection ─────────────────────────────────────────────────

def select_holdout_questions():
    """Return all chemistry question IDs that have rollouts on disk."""
    all_ids = sorted(
        d.name for d in OOD_ROLLOUTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("gpqa_chem_")
    )

    rollout_total = 0
    for q_id in all_ids:
        q_dir = OOD_ROLLOUTS_DIR / q_id
        rollout_total += len(list(q_dir.glob("rollout_*.json")))

    print(f"Selected {len(all_ids)} chemistry questions (all with rollouts)")
    print(f"  Total rollouts: {rollout_total} "
          f"(mean {rollout_total / len(all_ids):.1f}/question)")

    return all_ids


# ── Helpers ───────────────────────────────────────────────────────────

_thread_local = threading.local()


def _get_openrouter_client():
    """Get or create a thread-local OpenRouter client."""
    if not hasattr(_thread_local, "client"):
        import openai
        _thread_local.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    return _thread_local.client


# ── Phase 0: Generate more rollouts ──────────────────────────────────

def _load_holdout_questions():
    """Load chemistry question data for holdout questions from gpqa_main.csv."""
    import hashlib
    import pandas as pd

    holdout = select_holdout_questions()
    holdout_set = set(holdout)

    df = pd.read_csv(DATASET_CSV)
    chem = df[df["High-level domain"] == "Chemistry"].reset_index(drop=True)

    questions = {}
    for i, row in chem.iterrows():
        q_id = f"gpqa_chem_{i:03d}"
        if q_id not in holdout_set:
            continue

        q_text = str(row["Question"]).strip()
        seed = int(hashlib.md5(q_text.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        answers = [
            ("correct", str(row["Correct Answer"])),
            ("wrong1", str(row["Incorrect Answer 1"])),
            ("wrong2", str(row["Incorrect Answer 2"])),
            ("wrong3", str(row["Incorrect Answer 3"])),
        ]
        rng.shuffle(answers)

        correct_idx = next(
            j for j, (tag, _) in enumerate(answers) if tag == "correct"
        )
        correct_letter = chr(ord("A") + correct_idx)

        questions[q_id] = {
            "id": q_id,
            "question": q_text,
            "choices": [a[1] for a in answers],
            "correct_answer": correct_letter,
        }

    return questions


def _format_ae_prompt(q: dict) -> str:
    """Format a question as MC + answer instruction."""
    parts = [q["question"], ""]
    for j, choice in enumerate(q["choices"]):
        parts.append(f"({chr(ord('A') + j)}) {choice}")
    return "\n".join(parts) + "\n\nAnswer with the letter of the correct choice."


def _extract_cot(content: str) -> str:
    """Extract <think>...</think> block from model output."""
    if "<think>" in content and "</think>" in content:
        return content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    return content.strip()


def step_generate():
    """Generate more rollouts for holdout questions via OpenRouter (async)."""
    from datetime import datetime, timezone
    sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
    from openrouter_client import OpenRouterClient

    print("\n" + "=" * 60)
    print(f"Phase 0: Generate OOD rollouts (target: {NUM_ROLLOUTS}/question)")
    print(f"  Model: {GENERATE_MODEL}")
    print(f"  Workers: {GENERATE_WORKERS}")
    print("=" * 60)

    questions = _load_holdout_questions()
    if not questions:
        print("ERROR: No holdout questions loaded.")
        return

    # Build work items
    work_items = []
    for q_id in sorted(questions):
        q_dir = OOD_ROLLOUTS_DIR / q_id
        q_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(q_dir.glob("rollout_*.json"))
        n_existing = len(existing)
        n_needed = max(0, NUM_ROLLOUTS - n_existing)
        if n_needed == 0:
            continue
        print(f"  {q_id}: has {n_existing}, generating {n_needed} more")
        for i in range(n_existing, n_existing + n_needed):
            work_items.append((q_id, i))

    if not work_items:
        print("All rollouts already exist.")
        return

    print(f"\nGenerating {len(work_items)} rollouts across "
          f"{len(questions)} questions...")

    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    async def _run():
        completed = 0
        lock = asyncio.Lock()

        async def process_one(client, q_id, rollout_idx):
            nonlocal completed
            async with semaphore:
                q = questions[q_id]
                prompt_text = _format_ae_prompt(q)

                content = ""
                for attempt in range(3):
                    resp = await client.chat_completions(
                        model=GENERATE_MODEL,
                        messages=[{"role": "user", "content": prompt_text}],
                        allow_fallbacks=True,
                        max_tokens=16384,
                    )
                    content = OpenRouterClient.extract_first_content(resp)
                    if content.strip():
                        break
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)

                cot = _extract_cot(content)
                output = ""
                if "</think>" in content:
                    output = content.split("</think>", 1)[1].strip()

                rollout = {
                    "prompt_name": q_id,
                    "prompt_text": prompt_text,
                    "rollout_idx": rollout_idx,
                    "chain_of_thought": cot,
                    "output": output,
                    "model": GENERATE_MODEL,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                out_path = OOD_ROLLOUTS_DIR / q_id / f"rollout_{rollout_idx}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(rollout, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

                async with lock:
                    completed += 1
                    if completed % 100 == 0 or completed == len(work_items):
                        print(f"  Progress: {completed}/{len(work_items)}")

                return out_path

        semaphore = asyncio.Semaphore(GENERATE_WORKERS)

        async with OpenRouterClient(api_key=api_key, timeout_s=120.0) as client:
            tasks = [process_one(client, q_id, idx) for q_id, idx in work_items]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        ok = sum(1 for r in results if not isinstance(r, Exception))
        errors = [r for r in results if isinstance(r, Exception)]
        print(f"\nCompleted: {ok}/{len(results)} rollouts ({len(errors)} errors)")

        for e in errors[:10]:
            print(f"  ERROR: {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    asyncio.run(_run())


# ── Phase 1: Label answer emissions ──────────────────────────────────

def step_label():
    print("\n" + "=" * 60)
    print("Phase 1: Label OOD answer emissions (Claude Sonnet 4.5)")
    print("=" * 60)

    from src2.tasks.reasoning_evals.prefix_extraction import parse_annotated_response
    from src2.tasks.reasoning_evals.prompts import ANSWER_EMISSION_LABEL_PROMPT

    holdout = select_holdout_questions()

    # Build task list
    tasks = []
    skipped = 0
    for q_id in holdout:
        src_dir = OOD_ROLLOUTS_DIR / q_id
        if not src_dir.exists():
            print(f"  WARNING: No rollouts for {q_id}")
            continue
        rollout_files = sorted(src_dir.glob("rollout_*.json"))
        for rf in rollout_files[:NUM_ROLLOUTS]:
            idx = int(rf.stem.split("_")[1])
            dst_path = OOD_LABELED_DIR / q_id / f"rollout_{idx}.json"
            if dst_path.exists():
                skipped += 1
                continue
            tasks.append((q_id, idx))

    if skipped:
        print(f"Skipping {skipped} already-labeled rollout(s)")
    if not tasks:
        print("All rollouts already labeled.")
        return

    print(f"Labeling {len(tasks)} rollout(s) across {len(holdout)} questions")

    def label_one(q_id, idx):
        try:
            with open(OOD_ROLLOUTS_DIR / q_id / f"rollout_{idx}.json") as f:
                rollout = json.load(f)
            prompt_text = rollout["prompt_text"]
            cot = rollout["chain_of_thought"]

            filled = ANSWER_EMISSION_LABEL_PROMPT.format(
                prompt=prompt_text, thinking_process=cot,
            )
            client = _get_openrouter_client()
            result = client.chat.completions.create(
                model=LABELING_MODEL,
                messages=[{"role": "user", "content": filled}],
            )
            response = result.choices[0].message.content or ""

            labeled_data = parse_annotated_response(response)
            out_dir = OOD_LABELED_DIR / q_id
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"rollout_{idx}.json", "w") as f:
                json.dump(labeled_data, f, indent=2)
            return (q_id, idx, True, "")
        except Exception as e:
            return (q_id, idx, False, str(e))

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(tasks))) as executor:
        futures = {
            executor.submit(label_one, q, i): (q, i)
            for q, i in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Labeling"):
            q_id, idx, success, err = future.result()
            completed += 1
            if not success:
                failed += 1
                print(f"  FAILED {q_id}/rollout_{idx}: {err}")

    print(f"Labeling complete: {completed - failed}/{len(tasks)} succeeded")


# ── Phase 2: Extract prefixes ────────────────────────────────────────

def step_extract():
    print("\n" + "=" * 60)
    print("Phase 2: Extract OOD answer prefixes")
    print("=" * 60)

    from src2.tasks.reasoning_evals.prefix_extraction import extract_answer_prefixes

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, trust_remote_code=True)

    def count_tokens(text):
        with contextlib.redirect_stdout(io.StringIO()):
            return len(tokenizer.encode(text, add_special_tokens=False))

    holdout = select_holdout_questions()

    total_extracted = 0
    total_skipped = 0

    for q_id in holdout:
        labeled_dir = OOD_LABELED_DIR / q_id
        if not labeled_dir.exists():
            continue

        for rollout_file in sorted(labeled_dir.glob("rollout_*.json")):
            rollout_idx = int(rollout_file.stem.split("_")[1])

            with open(rollout_file) as f:
                labeled_data = json.load(f)

            # Load original rollout for prompt_text
            original_path = OOD_ROLLOUTS_DIR / q_id / f"rollout_{rollout_idx}.json"
            if not original_path.exists():
                continue
            with open(original_path) as f:
                rollout = json.load(f)
            prompt_text = rollout["prompt_text"]

            ans_prefixes = extract_answer_prefixes(
                labeled_data, prompt_text, rollout_idx, count_tokens,
            )
            for p in ans_prefixes:
                out_dir = OOD_PREFIXES_DIR / q_id / f"rollout_{rollout_idx}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"prefix_{p['prefix_idx']}.json"
                if out_path.exists():
                    total_skipped += 1
                    continue
                with open(out_path, "w") as f:
                    json.dump(p, f, indent=2)
                total_extracted += 1

    print(f"Extracted {total_extracted} prefixes ({total_skipped} already existed)")


# ── Phase 3: Resample ────────────────────────────────────────────────

def step_resample():
    print("\n" + "=" * 60)
    print(f"Phase 3: Resample OOD ({NUM_RESAMPLES} per prefix, "
          f"{MAX_RESAMPLE_TOKENS} tokens)")
    print("=" * 60)

    prefix_files = sorted(OOD_PREFIXES_DIR.rglob("prefix_*.json"))

    tasks = []
    for pf in prefix_files:
        rel = pf.relative_to(OOD_PREFIXES_DIR)
        out_dir = OOD_RESAMPLES_DIR / rel.parent
        out_path = out_dir / pf.name
        if out_path.exists():
            continue

        with open(pf) as f:
            prefix_data = json.load(f)
        tasks.append((prefix_data, out_dir, pf.name))

    if not tasks:
        print("All OOD resamples exist.")
        return

    print(f"Resampling {len(tasks)} OOD prefix(es)")

    from tinker import ServiceClient, types
    from transformers import AutoTokenizer
    from src2.utils.chat_template import build_thinking_prompt

    print(f"Initializing Tinker client for {SUBJECT_MODEL}...")
    tinker_client = ServiceClient()
    sampling_client = tinker_client.create_sampling_client(
        base_model=SUBJECT_MODEL
    )
    tokenizer = AutoTokenizer.from_pretrained(
        SUBJECT_MODEL, trust_remote_code=True
    )
    print("Tinker client ready.")

    def resample_one(args):
        prefix_data, out_dir, filename = args
        try:
            prompt_str = build_thinking_prompt(
                tokenizer, prefix_data["prompt_text"],
                cot_prefix=prefix_data["prefix_text"],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
            model_input = types.ModelInput.from_ints(tokens)
            params = types.SamplingParams(
                max_tokens=MAX_RESAMPLE_TOKENS, temperature=TEMPERATURE,
            )
            result = sampling_client.sample(
                prompt=model_input, num_samples=NUM_RESAMPLES,
                sampling_params=params,
            ).result()

            result_data = {
                "rollout": prompt_str,
                "answer_text": prefix_data.get("answer_text", ""),
                "is_final": prefix_data.get("is_final", False),
            }
            for i, seq in enumerate(result.sequences):
                result_data[str(i)] = tokenizer.decode(
                    seq.tokens, skip_special_tokens=True
                )

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / filename, "w") as f:
                json.dump(result_data, f, indent=2)
            return True
        except Exception as e:
            print(f"  OOD resample failed: {e}")
            return False

    completed = 0
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(tasks))) as executor:
        futures = {executor.submit(resample_one, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Resampling OOD"):
            completed += 1 if future.result() else 0

    print(f"  {completed}/{len(tasks)} OOD resamples succeeded")


# ── Phase 4: Judge with token-position criteria ──────────────────────

def find_think_close_token_idx(tokenizer, text):
    """Find the token index where </think> starts in the tokenized text."""
    char_pos = text.find("</think>")
    if char_pos == -1:
        return -1
    prefix_text = text[:char_pos]
    with contextlib.redirect_stdout(io.StringIO()):
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    return len(prefix_tokens)


def step_judge():
    print("\n" + "=" * 60)
    print("Phase 4: Judge OOD resamples with token-position criteria")
    print(f"  yes: </think> at token [{YES_TOKEN_MIN}, {YES_TOKEN_MAX}], "
          f">= {YES_THRESHOLD}/50")
    print(f"  no:  </think> at token > {NO_TOKEN_MIN} or absent, "
          f">= {NO_THRESHOLD}/50")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, trust_remote_code=True)

    holdout = select_holdout_questions()
    OOD_VAL_JUDGED_DIR.mkdir(parents=True, exist_ok=True)

    total_judged = 0
    total_yes = 0
    total_no = 0
    total_mixed = 0

    for q_id in sorted(holdout):
        q_resample_dir = OOD_RESAMPLES_DIR / q_id
        if not q_resample_dir.exists():
            print(f"  {q_id}: no resamples found, skipping")
            continue

        for resample_file in sorted(q_resample_dir.rglob("prefix_*.json")):
            rel = resample_file.relative_to(OOD_RESAMPLES_DIR)
            out_path = OOD_VAL_JUDGED_DIR / rel
            if out_path.exists():
                with open(out_path) as f:
                    existing = json.load(f)
                label = existing.get("label", "mixed")
                total_judged += 1
                if label == "yes":
                    total_yes += 1
                elif label == "no":
                    total_no += 1
                else:
                    total_mixed += 1
                continue

            with open(resample_file) as f:
                data = json.load(f)

            # Get prefix metadata
            prefix_file = OOD_PREFIXES_DIR / rel
            prefix_meta = {}
            if prefix_file.exists():
                with open(prefix_file) as f:
                    prefix_meta = json.load(f)

            # Judge each resample
            token_positions = []
            yes_count = 0
            no_count = 0

            for key in sorted(data.keys()):
                if not key.isdigit():
                    continue
                text = data[key]
                tok_idx = find_think_close_token_idx(tokenizer, text)
                token_positions.append(tok_idx)

                if tok_idx != -1 and YES_TOKEN_MIN <= tok_idx <= YES_TOKEN_MAX:
                    yes_count += 1
                if tok_idx == -1 or tok_idx > NO_TOKEN_MIN:
                    no_count += 1

            total = len(token_positions)
            if total == 0:
                continue

            if yes_count >= YES_THRESHOLD:
                label = "yes"
                total_yes += 1
            elif no_count >= NO_THRESHOLD:
                label = "no"
                total_no += 1
            else:
                label = "mixed"
                total_mixed += 1

            result = {
                "prompt_name": q_id,
                "rollout_idx": prefix_meta.get("rollout_idx", -1),
                "prefix_idx": prefix_meta.get("prefix_idx", -1),
                "label": label,
                "yes_count": yes_count,
                "no_count": no_count,
                "total_resamples": total,
                "token_positions": token_positions,
                "token_count": prefix_meta.get("token_count", 0),
                "prefix_text": prefix_meta.get("prefix_text", ""),
                "prompt_text": prefix_meta.get("prompt_text", ""),
            }

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            total_judged += 1

    print(f"\nJudged {total_judged} prefixes: "
          f"{total_yes} yes, {total_no} no, {total_mixed} mixed")


# ── Phase 5: Build balanced OOD val set ──────────────────────────────

def _load_ood_val_judged(holdout):
    """Load all OOD val judged items."""
    items = []
    for q_id in holdout:
        judged_dir = OOD_VAL_JUDGED_DIR / q_id
        if not judged_dir.exists():
            continue
        for jf in sorted(judged_dir.rglob("prefix_*.json")):
            with open(jf) as f:
                item = json.load(f)
            item.setdefault("prompt_name", q_id)

            positions = item.get("token_positions", [])
            if not positions:
                continue

            yes_count = sum(
                1 for p in positions
                if p != -1 and YES_TOKEN_MIN <= p <= YES_TOKEN_MAX
            )
            no_count = sum(
                1 for p in positions
                if p == -1 or p > NO_TOKEN_MIN
            )

            if yes_count >= YES_THRESHOLD:
                label = "yes"
            elif no_count >= NO_THRESHOLD:
                label = "no"
            else:
                continue

            item["label"] = label
            item["yes_count"] = yes_count
            item["no_count"] = no_count

            yes_positions = [
                p for p in positions
                if p != -1 and YES_TOKEN_MIN <= p <= YES_TOKEN_MAX
            ]
            item["mean_yes_position"] = mean(yes_positions) if yes_positions else 0.0

            items.append(item)
    return items


def _dedup_per_rollout(items_by_class):
    """Keep at most 1 prefix per rollout, even across classes."""
    for label in ("yes", "no"):
        items = items_by_class[label]
        by_rollout = defaultdict(list)
        for item in items:
            by_rollout[item["rollout_idx"]].append(item)

        deduped = []
        for rollout_idx, rollout_items in by_rollout.items():
            if label == "yes":
                best = max(rollout_items,
                           key=lambda x: x["mean_yes_position"])
            else:
                best = min(rollout_items,
                           key=lambda x: x["no_count"])
            deduped.append(best)
        items_by_class[label] = deduped

    yes_rollouts = {item["rollout_idx"]: item
                    for item in items_by_class["yes"]}
    no_rollouts = {item["rollout_idx"]: item
                   for item in items_by_class["no"]}
    shared = set(yes_rollouts.keys()) & set(no_rollouts.keys())
    if shared:
        n_yes = len(yes_rollouts)
        n_no = len(no_rollouts)
        for rid in shared:
            if n_yes <= n_no:
                items_by_class["no"] = [
                    i for i in items_by_class["no"]
                    if i["rollout_idx"] != rid
                ]
                n_no -= 1
            else:
                items_by_class["yes"] = [
                    i for i in items_by_class["yes"]
                    if i["rollout_idx"] != rid
                ]
                n_yes -= 1


def _get_bucket(token_count, bucket_size=500):
    return token_count // bucket_size


def step_build():
    print("\n" + "=" * 60)
    print("Phase 5: Build balanced OOD val set (v8)")
    print(f"  Thresholds: YES={YES_THRESHOLD}, NO={NO_THRESHOLD}")
    print(f"  Length: [{LENGTH_MIN}, {LENGTH_MAX})")
    print("=" * 60)

    holdout = select_holdout_questions()
    holdout_set = set(holdout)

    # ── 1. Load candidates ────────────────────────────────────────────
    all_items = _load_ood_val_judged(holdout)
    print(f"\nLoaded {len(all_items)} OOD val judged (yes/no)")

    if not all_items:
        print("No items found. Run the judge phase first.")
        return

    # ── 1b. Filter to length range ────────────────────────────────────
    before = len(all_items)
    all_items = [
        item for item in all_items
        if LENGTH_MIN <= item.get("token_count", 0) < LENGTH_MAX
    ]
    print(f"  After length filter [{LENGTH_MIN}, {LENGTH_MAX}): "
          f"{len(all_items)} / {before}")

    # ── 1b2. Exclude yes prefixes ending with **Final Answer** ────────
    before = len(all_items)
    all_items = [
        item for item in all_items
        if not (item["label"] == "yes"
                and item.get("prefix_text", "").rstrip().endswith("**Final Answer**"))
    ]
    filtered_fa = before - len(all_items)
    print(f"  After excluding yes ending with **Final Answer**: "
          f"{len(all_items)} / {before} (removed {filtered_fa})")

    # ── 1b3. Exclude prefixes with "answer" in last 100 chars ─────────
    before = len(all_items)
    all_items = [
        item for item in all_items
        if "answer" not in item.get("prefix_text", "")[-100:].lower()
    ]
    filtered_ans = before - len(all_items)
    print(f"  After excluding 'answer' in last 100 chars: "
          f"{len(all_items)} / {before} (removed {filtered_ans})")

    # ── 2. Group by prompt, dedup per rollout ─────────────────────────
    by_prompt = defaultdict(lambda: {"yes": [], "no": []})
    for item in all_items:
        by_prompt[item["prompt_name"]][item["label"]].append(item)

    for prompt_name, classes in by_prompt.items():
        _dedup_per_rollout(classes)

    # ── 3. Print per-prompt availability ──────────────────────────────
    all_prompt_names = sorted(by_prompt.keys())
    print(f"\nPer-prompt availability (after rollout dedup) "
          f"[{len(all_prompt_names)} prompts]:")
    prompts_with_both = []
    prompts_single_class = []

    for prompt_name in all_prompt_names:
        n_yes = len(by_prompt[prompt_name]["yes"])
        n_no = len(by_prompt[prompt_name]["no"])
        yes_items = by_prompt[prompt_name]["yes"]
        if yes_items:
            avg_myp = mean(i["mean_yes_position"] for i in yes_items)
            myp_str = f"  mean_yes_pos={avg_myp:.1f}"
        else:
            myp_str = ""
        if n_yes > 0 and n_no > 0:
            prompts_with_both.append(prompt_name)
            print(f"  {prompt_name:30s}  yes={n_yes:<3d} no={n_no:<3d}{myp_str}")
        elif n_yes > 0 or n_no > 0:
            prompts_single_class.append(prompt_name)
            print(f"  {prompt_name:30s}  yes={n_yes:<3d} no={n_no:<3d}{myp_str}"
                  f"  [single class]")

    print(f"\n  Prompts with both classes:  {len(prompts_with_both)}")
    print(f"  Prompts with single class: {len(prompts_single_class)}")

    # ── 4. Select samples: equal yes/no per prompt ────────────────────
    selected = []

    for prompt_name in prompts_with_both:
        yes_items = by_prompt[prompt_name]["yes"]
        no_items = by_prompt[prompt_name]["no"]

        yes_items.sort(key=lambda x: -x["mean_yes_position"])
        no_items.sort(key=lambda x: -x["no_count"])

        n_pairs = min(len(yes_items), len(no_items))
        selected.extend(yes_items[:n_pairs])
        selected.extend(no_items[:n_pairs])

    for prompt_name in prompts_single_class:
        yes_items = by_prompt[prompt_name]["yes"]
        no_items = by_prompt[prompt_name]["no"]
        if yes_items:
            yes_items.sort(key=lambda x: -x["mean_yes_position"])
            selected.append(yes_items[0])
        elif no_items:
            no_items.sort(key=lambda x: -x["no_count"])
            selected.append(no_items[0])

    yes_total = sum(1 for s in selected if s["label"] == "yes")
    no_total = sum(1 for s in selected if s["label"] == "no")
    print(f"\nBefore length balancing: {yes_total} yes / {no_total} no "
          f"= {len(selected)} total")

    # ── 5. Balance by 500-token length buckets ────────────────────────

    def _bucket_stats(items):
        yes_b = defaultdict(int)
        no_b = defaultdict(int)
        for s in items:
            b = _get_bucket(s.get("token_count", 0))
            if s["label"] == "yes":
                yes_b[b] += 1
            else:
                no_b[b] += 1
        return yes_b, no_b

    paired_prompts_set = set(prompts_with_both)

    # Step 5a: trim unpaired singles
    to_remove = set()
    yes_b, no_b = _bucket_stats(selected)
    all_buckets = sorted(set(yes_b.keys()) | set(no_b.keys()))
    print("\n  Length buckets (raw):")
    for b in all_buckets:
        print(f"    [{b*500}-{(b+1)*500}): "
              f"yes={yes_b[b]}, no={no_b[b]}")

    for b in all_buckets:
        ny = yes_b[b]
        nn = no_b[b]
        if ny == nn:
            continue
        majority = "yes" if ny > nn else "no"
        excess = abs(ny - nn)
        unpaired_in_bucket = [
            s for s in selected
            if id(s) not in to_remove
            and s["label"] == majority
            and s["prompt_name"] not in paired_prompts_set
            and _get_bucket(s.get("token_count", 0)) == b
        ]
        if majority == "yes":
            unpaired_in_bucket.sort(key=lambda x: x["mean_yes_position"])
        else:
            unpaired_in_bucket.sort(key=lambda x: x["no_count"])
        for s in unpaired_in_bucket[:excess]:
            to_remove.add(id(s))

    selected = [s for s in selected if id(s) not in to_remove]
    print(f"\n  Removed {len(to_remove)} unpaired singles for bucket balance")

    # Step 5b: pair trimming
    to_remove = set()
    yes_b, no_b = _bucket_stats(selected)
    all_buckets = sorted(set(yes_b.keys()) | set(no_b.keys()))

    for b in all_buckets:
        ny = yes_b.get(b, 0)
        nn = no_b.get(b, 0)
        if ny == nn or min(ny, nn) == 0:
            continue
        majority = "yes" if ny > nn else "no"
        n_pairs_to_remove = (abs(ny - nn)) // 2
        if n_pairs_to_remove == 0:
            continue

        removed = 0
        for prompt_name in list(prompts_with_both):
            if removed >= n_pairs_to_remove:
                break
            prompt_majority = [
                s for s in selected
                if id(s) not in to_remove
                and s["prompt_name"] == prompt_name
                and s["label"] == majority
                and _get_bucket(s.get("token_count", 0)) == b
            ]
            prompt_minority = [
                s for s in selected
                if id(s) not in to_remove
                and s["prompt_name"] == prompt_name
                and s["label"] != majority
            ]
            if prompt_majority and prompt_minority:
                to_remove.add(id(prompt_majority[-1]))
                to_remove.add(id(prompt_minority[-1]))
                removed += 1

    selected = [s for s in selected if id(s) not in to_remove]
    print(f"  Removed {len(to_remove)} samples via pair trimming")

    # Step 5c: add minority singles
    yes_total = sum(1 for s in selected if s["label"] == "yes")
    no_total = sum(1 for s in selected if s["label"] == "no")

    if yes_total != no_total:
        minority = "no" if yes_total > no_total else "yes"
        majority = "yes" if minority == "no" else "no"
        deficit = abs(yes_total - no_total)

        selected_rollout_keys = set(
            f"{s['prompt_name']}/rollout_{s['rollout_idx']}"
            for s in selected
        )

        prompt_bal = defaultdict(lambda: {"yes": 0, "no": 0})
        for s in selected:
            prompt_bal[s["prompt_name"]][s["label"]] += 1

        pool = []
        for prompt_name in all_prompt_names:
            for item in by_prompt[prompt_name][minority]:
                rk = f"{item['prompt_name']}/rollout_{item['rollout_idx']}"
                if rk in selected_rollout_keys:
                    continue
                pb = prompt_bal[prompt_name]
                if pb[minority] < pb[majority] + 1:
                    pool.append(item)

        sort_key = "no_count" if minority == "no" else "mean_yes_position"
        yes_b, no_b = _bucket_stats(selected)

        def _bucket_need(item):
            b = _get_bucket(item.get("token_count", 0))
            ny, nn = yes_b.get(b, 0), no_b.get(b, 0)
            if minority == "no":
                return ny - nn
            else:
                return nn - ny

        pool.sort(key=lambda x: (-_bucket_need(x), -x.get(sort_key, 0)))

        added = 0
        for item in pool:
            if added >= deficit:
                break
            rk = f"{item['prompt_name']}/rollout_{item['rollout_idx']}"
            if rk in selected_rollout_keys:
                continue
            pb = prompt_bal[item["prompt_name"]]
            if pb[minority] >= pb[majority] + 1:
                continue
            selected.append(item)
            selected_rollout_keys.add(rk)
            prompt_bal[item["prompt_name"]][minority] += 1
            b = _get_bucket(item.get("token_count", 0))
            if minority == "no":
                no_b[b] = no_b.get(b, 0) + 1
            else:
                yes_b[b] = yes_b.get(b, 0) + 1
            added += 1

        print(f"  Added {added} {minority} singles to reach balance")

    # Step 5d: stratified bucket trim
    yes_b, no_b = _bucket_stats(selected)
    to_remove = set()
    for b in sorted(yes_b.keys() | no_b.keys()):
        ny, nn = yes_b.get(b, 0), no_b.get(b, 0)
        diff = abs(ny - nn)
        if diff <= 2:
            continue
        majority = "yes" if ny > nn else "no"
        n_to_trim = diff - 2
        candidates = [
            s for s in selected
            if id(s) not in to_remove
            and s["label"] == majority
            and _get_bucket(s.get("token_count", 0)) == b
        ]
        if majority == "yes":
            candidates.sort(key=lambda x: x.get("mean_yes_position", 0))
        else:
            candidates.sort(key=lambda x: -x.get("no_count", 0))
        for s in candidates[:n_to_trim]:
            to_remove.add(id(s))
    if to_remove:
        selected = [s for s in selected if id(s) not in to_remove]
        print(f"  Removed {len(to_remove)} samples via stratified bucket trim")

    # Step 5e: global rebalance
    yes_total = sum(1 for s in selected if s["label"] == "yes")
    no_total = sum(1 for s in selected if s["label"] == "no")
    if yes_total != no_total:
        majority = "yes" if yes_total > no_total else "no"
        excess = abs(yes_total - no_total)
        candidates = [s for s in selected if s["label"] == majority]
        if majority == "yes":
            candidates.sort(key=lambda x: x.get("mean_yes_position", 0))
        else:
            candidates.sort(key=lambda x: -x.get("no_count", 0))
        drop_ids = {id(s) for s in candidates[:excess]}
        selected = [s for s in selected if id(s) not in drop_ids]
        print(f"  Dropped {excess} excess {majority} for global balance")

    yes_total = sum(1 for s in selected if s["label"] == "yes")
    no_total = sum(1 for s in selected if s["label"] == "no")

    print(f"\n  After rebalancing: {yes_total} yes / {no_total} no "
          f"= {len(selected)} total")

    # Final bucket distribution
    yes_b, no_b = _bucket_stats(selected)
    final_buckets = sorted(set(yes_b.keys()) | set(no_b.keys()))
    print("\n  Length buckets (final):")
    for b in final_buckets:
        ny = yes_b[b]
        nn = no_b[b]
        print(f"    [{b*500}-{(b+1)*500}): yes={ny}, no={nn}")

    # Per-prompt detail
    print("\n  Per-prompt selected detail:")
    for prompt_name in sorted(set(s["prompt_name"] for s in selected)):
        prompt_items = [s for s in selected if s["prompt_name"] == prompt_name]
        yes_items = [s for s in prompt_items if s["label"] == "yes"]
        no_items = [s for s in prompt_items if s["label"] == "no"]
        myp_str = ""
        if yes_items:
            myps = [s["mean_yes_position"] for s in yes_items]
            myp_str = f"  mean_yes_pos={mean(myps):.1f}"
        print(f"    {prompt_name:30s}  yes={len(yes_items)} no={len(no_items)}"
              f"{myp_str}")

    # ── 6. Build val set JSON ─────────────────────────────────────────
    prefixes = {}
    for item in selected:
        path_key = (f"{item['prompt_name']}/rollout_{item['rollout_idx']}"
                    f"/prefix_{item['prefix_idx']}")
        prefixes[path_key] = {
            "label": item["label"],
            "prefix_text": item["prefix_text"],
            "prompt_text": item.get("prompt_text", ""),
            "prompt_name": item["prompt_name"],
            "token_length": item.get("token_count", 0),
            "yes_count": item["yes_count"],
            "no_count": item["no_count"],
            "total_resamples": item["total_resamples"],
            "mean_yes_position": item.get("mean_yes_position", 0.0),
            "source": "ood_val",
        }

    val_prompts = set(s["prompt_name"] for s in selected)
    samples_per_prompt = defaultdict(int)
    for s in selected:
        samples_per_prompt[s["prompt_name"]] += 1

    val_set = {
        "description": (
            "v8 OOD validation set: 30 held-out GPQA chemistry questions "
            "(from OOD train v4), token-position judged, least-obvious yes "
            "prefixes, balanced per prompt and by length."
        ),
        "version": "v8",
        "subject_model": SUBJECT_MODEL,
        "labeling_model": LABELING_MODEL,
        "judging_criteria": {
            "yes": f"</think> at token [{YES_TOKEN_MIN}, {YES_TOKEN_MAX}] "
                   f"in >= {YES_THRESHOLD}/{NUM_RESAMPLES} resamples",
            "no": f"</think> at token > {NO_TOKEN_MIN} or absent "
                  f"in >= {NO_THRESHOLD}/{NUM_RESAMPLES} resamples",
        },
        "holdout_questions": sorted(holdout_set),
        "summary": {
            "yes_count": yes_total,
            "no_count": no_total,
            "total": yes_total + no_total,
            "num_prompts": len(val_prompts),
            "samples_per_prompt": dict(sorted(samples_per_prompt.items())),
        },
        "prefixes": prefixes,
    }

    # ── 7. Verification ───────────────────────────────────────────────
    print("\n--- Verification ---")

    # Note: prompts intentionally overlap with OOD train (different label methods)
    print(f"  Using all {len(val_prompts)} chemistry prompts "
          f"(train uses distance-proxy, val uses token-position judging)")

    # At most 1 prefix per rollout
    rollout_keys = [
        f"{s['prompt_name']}/rollout_{s['rollout_idx']}" for s in selected
    ]
    dupes = [k for k in set(rollout_keys) if rollout_keys.count(k) > 1]
    status = "PASS" if not dupes else f"FAIL: {dupes}"
    print(f"  1 prefix per rollout: {status}")

    # Per-prompt balance (diff <= 1)
    imbalanced = []
    for pname in val_prompts:
        p_yes = sum(1 for s in selected
                    if s["prompt_name"] == pname and s["label"] == "yes")
        p_no = sum(1 for s in selected
                   if s["prompt_name"] == pname and s["label"] == "no")
        if abs(p_yes - p_no) > 1:
            imbalanced.append(f"{pname}: {p_yes}y/{p_no}n")
    status = "PASS" if not imbalanced else f"FAIL: {imbalanced}"
    print(f"  Per-prompt balance (diff<=1): {status}")

    # Global balance
    diff = abs(yes_total - no_total)
    status = "PASS" if diff <= 2 else f"WARN (diff={diff})"
    print(f"  Global balance: {yes_total} yes / {no_total} no "
          f"(diff={diff}) {status}")

    # Total >= 50
    total = yes_total + no_total
    status = "PASS" if total >= 50 else f"WARN: only {total} samples"
    print(f"  Total >= 50: {total} {status}")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(VAL_SET_PATH, "w") as f:
        json.dump(val_set, f, indent=2)
    print(f"\nSaved: {VAL_SET_PATH}")
    print(f"  {yes_total} yes + {no_total} no = {len(selected)} samples "
          f"across {len(val_prompts)} prompts")


# ── Phase 6: Rebuild OOD train set ───────────────────────────────────

def step_rebuild_train():
    print("\n" + "=" * 60)
    print("Phase 6: Rebuild OOD train set without held-out questions")
    print("=" * 60)

    holdout = select_holdout_questions()
    holdout_set = set(holdout)

    with open(OOD_TRAIN_V4_PATH) as f:
        ood_train_v4 = json.load(f)

    entries_v4 = ood_train_v4["entries"]
    entries_v5 = [e for e in entries_v4 if e["prompt_name"] not in holdout_set]

    yes_v4 = sum(1 for e in entries_v4 if e["label"] == "yes")
    no_v4 = sum(1 for e in entries_v4 if e["label"] == "no")
    yes_v5 = sum(1 for e in entries_v5 if e["label"] == "yes")
    no_v5 = sum(1 for e in entries_v5 if e["label"] == "no")

    print(f"  v4: {len(entries_v4)} entries ({yes_v4} yes, {no_v4} no)")
    print(f"  v5: {len(entries_v5)} entries ({yes_v5} yes, {no_v5} no)")
    print(f"  Removed: {len(entries_v4) - len(entries_v5)} entries "
          f"from {len(holdout_set)} held-out prompts")

    # Rebalance
    yes_entries = [e for e in entries_v5 if e["label"] == "yes"]
    no_entries = [e for e in entries_v5 if e["label"] == "no"]
    target = min(len(yes_entries), len(no_entries))

    rng = random.Random(SEED)
    if len(yes_entries) > target:
        rng.shuffle(yes_entries)
        yes_entries = yes_entries[:target]
    if len(no_entries) > target:
        rng.shuffle(no_entries)
        no_entries = no_entries[:target]

    entries_v5_balanced = yes_entries + no_entries
    rng.shuffle(entries_v5_balanced)

    remaining_prompts = sorted(set(e["prompt_name"] for e in entries_v5_balanced))

    ood_train_v5 = {
        "summary": {
            "yes_count": len(yes_entries),
            "no_count": len(no_entries),
            "total": len(entries_v5_balanced),
            "label_source": ood_train_v4["summary"].get("label_source", "distance_proxy"),
            "yes_distances": ood_train_v4["summary"].get("yes_distances", []),
            "no_start": ood_train_v4["summary"].get("no_start", 300),
            "no_step": ood_train_v4["summary"].get("no_step", 200),
            "num_prompts": len(remaining_prompts),
            "prompts": remaining_prompts,
            "holdout_questions_removed": sorted(holdout_set),
        },
        "entries": entries_v5_balanced,
    }

    with open(OOD_TRAIN_V5_PATH, "w") as f:
        json.dump(ood_train_v5, f, indent=2)

    print(f"\nSaved: {OOD_TRAIN_V5_PATH}")
    print(f"  {len(yes_entries)} yes + {len(no_entries)} no "
          f"= {len(entries_v5_balanced)} samples "
          f"across {len(remaining_prompts)} prompts")

    # Verify no overlap with holdout
    v5_prompts = set(remaining_prompts)
    overlap = v5_prompts & holdout_set
    status = "PASS" if not overlap else f"FAIL: {overlap}"
    print(f"  No overlap with holdout: {status}")


# ── Main ──────────────────────────────────────────────────────────────

PHASES = {
    "generate": step_generate,
    "label": step_label,
    "extract": step_extract,
    "resample": step_resample,
    "judge": step_judge,
    "build": step_build,
    "rebuild_train": step_rebuild_train,
}


def main():
    parser = argparse.ArgumentParser(
        description="Build ood_answer_val_set_v8: OOD validation set from "
                    "held-out GPQA chemistry questions."
    )
    parser.add_argument(
        "phases", nargs="*", default=["all"],
        help="Phases to run (label, extract, resample, judge, build, "
             "rebuild_train, all)",
    )
    args = parser.parse_args()

    if "all" in args.phases:
        phases = list(PHASES.keys())
    else:
        phases = args.phases

    for phase in phases:
        if phase not in PHASES:
            print(f"Unknown phase: {phase}. "
                  f"Choices: {', '.join(PHASES.keys())}, all")
            continue
        PHASES[phase]()

    print("\nDone.")


if __name__ == "__main__":
    main()
