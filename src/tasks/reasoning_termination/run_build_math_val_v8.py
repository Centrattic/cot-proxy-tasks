#!/usr/bin/env python3
"""
Build answer_val_set_v8: math validation set from 64 unused prompts.

Uses the same v8 methodology as the test set (token-position judging,
same thresholds and filters), but on disjoint prompts.

Phases:
  label    - Label 48 rollouts/prompt with Claude Sonnet 4.5
  extract  - Extract answer-emission prefixes
  resample - Resample 50 continuations per prefix (Qwen3-32B, GPU)
  judge    - Token-position judging → val_judged/
  build    - Build balanced val set
  all      - Run all phases

Usage:
    python -m src2.runs.run_build_math_val_v8 label extract     # local
    python -m src2.runs.run_build_math_val_v8 resample           # GPU pod
    python -m src2.runs.run_build_math_val_v8 judge build        # local
"""

import argparse
import contextlib
import io
import json
import os
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

DATA_DIR = Path("data/reasoning_evals")
VAL_SET_PATH = DATA_DIR / "answer_val_set_v8.json"
VAL_JUDGED_DIR = DATA_DIR / "val_judged"
UNLABELED_DIR = DATA_DIR / "rollouts" / "unlabeled"
LABELED_ANS_DIR = DATA_DIR / "rollouts" / "labeled_answer"
PREFIXES_ANS_DIR = DATA_DIR / "prefixes" / "answer"
RESAMPLES_DIR = DATA_DIR / "resamples"

NUM_ROLLOUTS = 200
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

# 64 unused math prompts (REASONING_PROMPTS - v4 train - v8 test)
VAL_PROMPTS = [
    "age_ratio", "alternating_sum_100", "bagel", "base_conversion",
    "bridge_crossing", "cancer", "catalan_parentheses", "chairland",
    "chromatic_cycle", "coin_flips_3h5", "complete_graph_edges",
    "cone_volume", "cube_space_diagonal", "cubes", "derangement_4",
    "determinant_3x3", "egg_drop_2_100", "expected_dice_sum",
    "floor_division_sum", "four_digit_even", "geometric_sum",
    "grid_paths", "harmonic_speed", "hex", "hexagon_area",
    "inclusion_exclusion", "inscribed_angle_100", "isosceles_perimeter_20",
    "labeled_trees", "last_two_digits", "lattice_rectangle",
    "mixture_salt", "modular_last_digit", "pascals_row_sum",
    "perfect_square_count", "petersen_chromatic", "power_mod_7",
    "power_tower_last_digit", "prisoners_line", "prompt-2",
    "recurrence_sixth", "smallest_prime_100", "sqrt_nested",
    "staircase_8", "stamp_frobenius", "starfish", "stars_bars_15",
    "subset_sum_above", "sum_geometric_3_to_8", "sum_multiples_3_200",
    "sum_squares_formula", "surjection_4_to_3", "system_sum",
    "tangent_distance", "tank_fill", "trailing_zeros", "train_meeting",
    "triangle_integer_sides", "tribonacci_8", "truth_island",
    "turbo", "waffle", "well", "work_together",
]

# Prompts in v4 train set (for verification)
V4_TRAIN_PROMPTS = {
    "prompt-1", "prompt-3", "prompt-4", "prompt-5", "prompt-6",
    "picnic", "sisters", "tricky_sisters", "small_nested",
    "count_solutions", "last_digit", "widow", "chinese-cancer",
    "chinese-math", "minimum_square", "series", "harder_jack",
    "waffle_low", "n_remainder", "leet_speak", "letters",
    "knight_moves", "digital_root", "coin_rows", "snail_wall",
    "power_mod", "rectangle_diagonal", "three_digit_sum",
    "party_handshakes", "polynomial_trick", "clock_angle",
    "socks_guarantee", "locker_problem", "sum_of_cubes",
    "divisor_count", "domino_cover",
}

# Prompts in v8 test set (for verification)
V8_TEST_PROMPTS = {
    "balance_coins", "binary_no_consec", "binary_sum", "birthday_paradox",
    "boat_current_speed", "card_draw_hearts", "chessboard_squares",
    "circular_seating", "colored_balls", "committee_women", "crt_three",
    "dice_even_product", "distinct_necklaces_4r2b", "euler_phi_30",
    "fibonacci_mod_8", "five_digit_palindromes", "frobenius_3_7",
    "gcd_2024", "josephus_7", "lcm_three", "matrix_power_entry",
    "modular_inverse_7_11", "money_change_50", "nim_piles",
    "paint_cube_faces", "partition_ordered", "permutation_mississippi",
    "polygon_diagonals", "probability_no_six", "sum_divisors_72",
    "tetrahedron_volume", "triangle_vertices",
}


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


# ── Phase 1: Label answer emissions ──────────────────────────────────

def step_label():
    print("\n" + "=" * 60)
    print(f"Phase 1: Label answer emissions ({NUM_ROLLOUTS} rollouts/prompt, "
          f"{len(VAL_PROMPTS)} prompts)")
    print("=" * 60)

    from src2.tasks.reasoning_evals.prefix_extraction import parse_annotated_response
    from src2.tasks.reasoning_evals.prompts import (
        ANSWER_EMISSION_LABEL_PROMPT,
        REASONING_PROMPTS,
    )

    # Build task list: first NUM_ROLLOUTS per prompt, skip already labeled
    tasks = []
    skipped = 0
    for name in VAL_PROMPTS:
        src_dir = UNLABELED_DIR / name
        if not src_dir.exists():
            print(f"  WARNING: No rollouts for {name}")
            continue
        for idx in range(NUM_ROLLOUTS):
            src_path = src_dir / f"rollout_{idx}.json"
            dst_path = LABELED_ANS_DIR / name / f"rollout_{idx}.json"
            if not src_path.exists():
                continue
            if dst_path.exists():
                skipped += 1
                continue
            tasks.append((name, idx))

    if skipped:
        print(f"Skipping {skipped} already-labeled rollout(s)")
    if not tasks:
        print("All rollouts already labeled.")
        return

    print(f"Labeling {len(tasks)} rollout(s)")

    def label_one(name, idx):
        try:
            # Load rollout
            with open(UNLABELED_DIR / name / f"rollout_{idx}.json") as f:
                rollout = json.load(f)
            prompt_text = rollout["prompt_text"]
            cot = rollout["chain_of_thought"]

            # Call labeling model
            filled = ANSWER_EMISSION_LABEL_PROMPT.format(
                prompt=prompt_text, thinking_process=cot,
            )
            client = _get_openrouter_client()
            result = client.chat.completions.create(
                model=LABELING_MODEL,
                messages=[{"role": "user", "content": filled}],
            )
            response = result.choices[0].message.content or ""

            # Parse and save
            labeled_data = parse_annotated_response(response)
            out_dir = LABELED_ANS_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"rollout_{idx}.json", "w") as f:
                json.dump(labeled_data, f, indent=2)
            return (name, idx, True, "")
        except Exception as e:
            return (name, idx, False, str(e))

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(tasks))) as executor:
        futures = {
            executor.submit(label_one, n, i): (n, i)
            for n, i in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Labeling"):
            name, idx, success, err = future.result()
            completed += 1
            if not success:
                failed += 1
                print(f"  FAILED {name}/rollout_{idx}: {err}")

    print(f"Labeling complete: {completed - failed}/{len(tasks)} succeeded")


# ── Phase 2: Extract prefixes ────────────────────────────────────────

def step_extract():
    print("\n" + "=" * 60)
    print("Phase 2: Extract answer prefixes")
    print("=" * 60)

    from src2.tasks.reasoning_evals.prefix_extraction import extract_answer_prefixes

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, trust_remote_code=True)

    def count_tokens(text):
        with contextlib.redirect_stdout(io.StringIO()):
            return len(tokenizer.encode(text, add_special_tokens=False))

    total_extracted = 0
    total_skipped = 0

    for name in VAL_PROMPTS:
        labeled_dir = LABELED_ANS_DIR / name
        if not labeled_dir.exists():
            continue

        for rollout_file in sorted(labeled_dir.glob("rollout_*.json")):
            rollout_idx = int(rollout_file.stem.split("_")[1])

            with open(rollout_file) as f:
                labeled_data = json.load(f)

            # Load original rollout for prompt_text
            unlabeled_path = UNLABELED_DIR / name / f"rollout_{rollout_idx}.json"
            if not unlabeled_path.exists():
                continue
            with open(unlabeled_path) as f:
                rollout = json.load(f)
            prompt_text = rollout["prompt_text"]

            ans_prefixes = extract_answer_prefixes(
                labeled_data, prompt_text, rollout_idx, count_tokens,
            )
            for p in ans_prefixes:
                out_dir = PREFIXES_ANS_DIR / name / f"rollout_{rollout_idx}"
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
    print(f"Phase 3: Resample ({NUM_RESAMPLES} per prefix, "
          f"{MAX_RESAMPLE_TOKENS} tokens)")
    print("=" * 60)

    from src2.tasks.reasoning_evals import ReasoningEvalsTask

    task = ReasoningEvalsTask(
        subject_model=SUBJECT_MODEL,
        labeling_model=LABELING_MODEL,
    )
    task._resample_answer(
        num_resamples=NUM_RESAMPLES,
        max_tokens=MAX_RESAMPLE_TOKENS,
        temperature=TEMPERATURE,
        workers=WORKERS,
    )


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
    print("Phase 4: Judge resamples with token-position criteria")
    print(f"  yes: </think> at token [{YES_TOKEN_MIN}, {YES_TOKEN_MAX}], "
          f">= {YES_THRESHOLD}/50")
    print(f"  no:  </think> at token > {NO_TOKEN_MIN} or absent, "
          f">= {NO_THRESHOLD}/50")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, trust_remote_code=True)

    resamples_dir = RESAMPLES_DIR / "answer"
    prefixes_dir = PREFIXES_ANS_DIR
    VAL_JUDGED_DIR.mkdir(parents=True, exist_ok=True)

    total_judged = 0
    total_yes = 0
    total_no = 0
    total_mixed = 0

    for prompt_name in sorted(VAL_PROMPTS):
        prompt_resample_dir = resamples_dir / prompt_name
        if not prompt_resample_dir.exists():
            print(f"  {prompt_name}: no resamples found, skipping")
            continue

        for resample_file in sorted(prompt_resample_dir.rglob("prefix_*.json")):
            # Derive output path
            rel = resample_file.relative_to(resamples_dir)
            out_path = VAL_JUDGED_DIR / rel
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
            prefix_file = prefixes_dir / rel
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

            # Classify
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
                "prompt_name": prompt_name,
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


# ── Phase 5: Build balanced val set ──────────────────────────────────

def _load_val_judged():
    """Load all val judged items, classifying from token_positions."""
    items = []
    for prompt_name in VAL_PROMPTS:
        judged_dir = VAL_JUDGED_DIR / prompt_name
        if not judged_dir.exists():
            continue
        for jf in sorted(judged_dir.rglob("prefix_*.json")):
            with open(jf) as f:
                item = json.load(f)
            item.setdefault("prompt_name", prompt_name)

            # Re-classify from token_positions
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
                continue  # mixed — skip

            item["label"] = label
            item["yes_count"] = yes_count
            item["no_count"] = no_count

            # Mean position of yes-range tokens for least-obvious selection
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

    # Resolve cross-class rollout conflicts
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
    print("Phase 5: Build balanced val set (v8)")
    print(f"  Thresholds: YES={YES_THRESHOLD}, NO={NO_THRESHOLD}")
    print(f"  Length: [{LENGTH_MIN}, {LENGTH_MAX})")
    print("=" * 60)

    # ── 1. Load candidates ────────────────────────────────────────────
    all_items = _load_val_judged()
    print(f"\nLoaded {len(all_items)} val judged (yes/no)")

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

    # ── 4. Select samples: equal yes/no per prompt, allow 1 unpaired ──
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

    # Step 5a: trim unpaired singles from buckets where one class has excess
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

    # Step 5b: for remaining bucket imbalances, remove whole pairs
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

    # Step 5c: enforce exact global balance by adding minority singles
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
            print(f"    + added {minority}: {item['prompt_name']}"
                  f"/rollout_{item['rollout_idx']}"
                  f"/prefix_{item['prefix_idx']}"
                  f" (tokens={item.get('token_count', 0)})")

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

    # Print final bucket distribution
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
            "source": "val",
        }

    val_prompts = set(s["prompt_name"] for s in selected)
    samples_per_prompt = defaultdict(int)
    for s in selected:
        samples_per_prompt[s["prompt_name"]] += 1

    val_set = {
        "description": (
            "v8 math validation set: 64 unused prompts (not in v4 train or "
            "v8 test), token-position judged, least-obvious yes prefixes, "
            "balanced per prompt and by length."
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

    # No overlap with v4 train prompts
    overlap_train = val_prompts & V4_TRAIN_PROMPTS
    status = "PASS" if not overlap_train else f"FAIL: {overlap_train}"
    print(f"  No overlap with v4 train: {status}")

    # No overlap with v8 test prompts
    overlap_test = val_prompts & V8_TEST_PROMPTS
    status = "PASS" if not overlap_test else f"FAIL: {overlap_test}"
    print(f"  No overlap with v8 test: {status}")

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


# ── Main ──────────────────────────────────────────────────────────────

PHASES = {
    "label": step_label,
    "extract": step_extract,
    "resample": step_resample,
    "judge": step_judge,
    "build": step_build,
}


def main():
    parser = argparse.ArgumentParser(
        description="Build answer_val_set_v8: math validation set from "
                    "64 unused prompts."
    )
    parser.add_argument(
        "phases", nargs="*", default=["all"],
        help="Phases to run (label, extract, resample, judge, build, all)",
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
