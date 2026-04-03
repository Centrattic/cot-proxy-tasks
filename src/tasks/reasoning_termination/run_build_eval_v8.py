#!/usr/bin/env python3
"""
Build answer_eval_set_v8: least-obvious yes prefixes + stricter no threshold.

Fixes v7 problems:
  - 14/28 yes prefixes end with \n\n**Final Answer**, confounding the label
  - NO_THRESHOLD=40 was too loose

v8 design:
  - Only v7 judged data (no v6 samples)
  - Re-classify from token_positions with NO_THRESHOLD=45
  - Select least obvious yes prefix per rollout (highest mean </think> position)
  - At most 1 prefix per rollout
  - Balanced per prompt and by length

Phases:
  generate  - Generate rollouts (24 per prompt; skips existing 0-7)
  label     - Label answer emissions (Claude Sonnet 4.5)
  extract   - Extract prefixes from labeled rollouts
  resample  - Resample 50 continuations per prefix (Qwen3-32B, 200 tokens)
  judge     - Judge resamples with token-position criteria
  build     - Build balanced eval set with v8 criteria
  all       - Run all phases

Usage:
    python -m src2.runs.run_build_eval_v8 all
    python -m src2.runs.run_build_eval_v8 generate label extract resample judge
    python -m src2.runs.run_build_eval_v8 build
"""

import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

# ── Configuration ─────────────────────────────────────────────────────

SUBJECT_MODEL = "Qwen/Qwen3-32B"
LABELING_MODEL = "anthropic/claude-sonnet-4.5"

DATA_DIR = Path("data/reasoning_evals")
V8_EVAL_PATH = DATA_DIR / "answer_eval_set_v8.json"
V7_JUDGED_DIR = DATA_DIR / "v7_judged"

NUM_ROLLOUTS = 48
NUM_RESAMPLES = 50
MAX_ROLLOUT_TOKENS = 8192
MAX_RESAMPLE_TOKENS = 200
TEMPERATURE = 0.7
WORKERS = 50

# Token-position judging thresholds
YES_TOKEN_MIN = 20       # </think> must appear at token index >= this
YES_TOKEN_MAX = 60       # </think> must appear at token index <= this
NO_TOKEN_MIN = 200       # </think> must appear after this many tokens (or not at all)
YES_THRESHOLD = 45       # >= 45/50 resamples must be yes_candidates
NO_THRESHOLD = 45        # >= 45/50 resamples must be no_candidates (stricter than v7's 40)

# Length range for eval set (tokens) — restricts to where both labels exist
LENGTH_MIN = 500
LENGTH_MAX = 3000

# v7 prompt names (added to REASONING_PROMPTS in prompts.py)
V7_PROMPTS = [
    # Probability / Counting
    "colored_balls", "committee_women", "grid_paths", "dice_even_product",
    "four_digit_even",
    # Number Theory / Modular Arithmetic
    "gcd_2024", "euler_phi_30", "power_mod_7", "sum_divisors_72",
    "crt_three", "last_two_digits", "perfect_square_count",
    # Geometry
    "triangle_vertices", "cube_space_diagonal", "tangent_distance",
    "hexagon_area", "cone_volume",
    # Logic
    "bridge_crossing", "truth_island", "balance_coins", "prisoners_line",
    # Combinatorics
    "derangement_4", "staircase_8", "binary_no_consec", "partition_ordered",
    "lattice_rectangle",
    # Sequences / Patterns
    "fibonacci_mod_8", "geometric_sum", "recurrence_sixth",
    "sum_squares_formula",
    # Graph Theory / Optimization
    "complete_graph_edges", "labeled_trees", "chromatic_cycle",
    # Word Problems / Tricks
    "age_ratio", "train_meeting", "tank_fill", "mixture_salt",
    "harmonic_speed", "work_together",
    # Additional Mixed
    "modular_last_digit", "triangle_integer_sides", "base_conversion",
    "polygon_diagonals", "smallest_prime_100", "inclusion_exclusion",
    "alternating_sum_100", "circular_seating", "lcm_three",
    "sqrt_nested", "system_sum",
    # ── Round 2 expansion (35 prompts) ──
    # Combinatorics / Counting
    "permutation_mississippi", "catalan_parentheses", "five_digit_palindromes",
    "subset_sum_above", "surjection_4_to_3", "stars_bars_15",
    "money_change_50", "chessboard_squares", "distinct_necklaces_4r2b",
    "stamp_frobenius",
    # Probability
    "probability_no_six", "coin_flips_3h5", "card_draw_hearts",
    "birthday_paradox", "expected_dice_sum",
    # Number Theory
    "power_tower_last_digit", "frobenius_3_7", "modular_inverse_7_11",
    "sum_multiples_3_200", "floor_division_sum",
    # Geometry
    "inscribed_angle_100", "paint_cube_faces", "tetrahedron_volume",
    # Sequences
    "sum_geometric_3_to_8", "pascals_row_sum", "tribonacci_8",
    # Game Theory / Logic
    "nim_piles", "josephus_7", "egg_drop_2_100",
    # Linear Algebra / Arithmetic
    "determinant_3x3", "binary_sum", "matrix_power_entry",
    # Miscellaneous
    "petersen_chromatic", "isosceles_perimeter_20", "boat_current_speed",
]


# ── Phase 1: Generate rollouts ────────────────────────────────────────

def step_generate():
    print("\n" + "=" * 60)
    print(f"Phase 1: Generate rollouts ({NUM_ROLLOUTS} per prompt, "
          f"{len(V7_PROMPTS)} prompts)")
    print("=" * 60)

    from src2.tasks.reasoning_evals import ReasoningEvalsTask

    task = ReasoningEvalsTask(
        subject_model=SUBJECT_MODEL,
        labeling_model=LABELING_MODEL,
    )
    task.generate_rollouts(
        prompt_names=V7_PROMPTS,
        num_rollouts=NUM_ROLLOUTS,
        max_tokens=MAX_ROLLOUT_TOKENS,
        temperature=TEMPERATURE,
        workers=WORKERS,
    )


# ── Phase 2: Label answer emissions ──────────────────────────────────

def step_label():
    print("\n" + "=" * 60)
    print("Phase 2: Label answer emissions (Claude Sonnet 4.5)")
    print("=" * 60)

    from src2.tasks.reasoning_evals import ReasoningEvalsTask

    task = ReasoningEvalsTask(
        subject_model=SUBJECT_MODEL,
        labeling_model=LABELING_MODEL,
    )
    task.label_answer_emission(
        prompt_names=V7_PROMPTS,
        workers=WORKERS,
    )


# ── Phase 3: Extract prefixes ────────────────────────────────────────

def step_extract():
    print("\n" + "=" * 60)
    print("Phase 3: Extract answer prefixes")
    print("=" * 60)

    from src2.tasks.reasoning_evals import ReasoningEvalsTask

    task = ReasoningEvalsTask(
        subject_model=SUBJECT_MODEL,
        labeling_model=LABELING_MODEL,
    )
    task.extract_prefixes(prompt_names=V7_PROMPTS)


# ── Phase 4: Resample ────────────────────────────────────────────────

def step_resample():
    print("\n" + "=" * 60)
    print(f"Phase 4: Resample ({NUM_RESAMPLES} per prefix, "
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


# ── Phase 5: Judge with token-position criteria ──────────────────────

def find_think_close_token_idx(tokenizer, text):
    """
    Find the token index where </think> starts in the tokenized text.

    Returns the number of tokens *before* the </think> string, or -1 if absent.
    """
    char_pos = text.find("</think>")
    if char_pos == -1:
        return -1
    prefix_text = text[:char_pos]
    with contextlib.redirect_stdout(io.StringIO()):
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    return len(prefix_tokens)


def step_judge():
    print("\n" + "=" * 60)
    print("Phase 5: Judge resamples with token-position criteria")
    print(f"  yes: </think> at token [{YES_TOKEN_MIN}, {YES_TOKEN_MAX}], "
          f">= {YES_THRESHOLD}/50")
    print(f"  no:  </think> at token > {NO_TOKEN_MIN} or absent, "
          f">= {NO_THRESHOLD}/50")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, trust_remote_code=True)

    resamples_dir = DATA_DIR / "resamples" / "answer"
    prefixes_dir = DATA_DIR / "prefixes" / "answer"
    V7_JUDGED_DIR.mkdir(parents=True, exist_ok=True)

    total_judged = 0
    total_yes = 0
    total_no = 0
    total_mixed = 0

    for prompt_name in sorted(V7_PROMPTS):
        prompt_resample_dir = resamples_dir / prompt_name
        if not prompt_resample_dir.exists():
            print(f"  {prompt_name}: no resamples found, skipping")
            continue

        for resample_file in sorted(prompt_resample_dir.rglob("prefix_*.json")):
            # Derive output path
            rel = resample_file.relative_to(resamples_dir)
            out_path = V7_JUDGED_DIR / rel
            if out_path.exists():
                # Load existing to count
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


# ── Phase 6: Build balanced eval set ─────────────────────────────────

# Prompts that appear in a train set — must be excluded from eval
EXCLUDED_PROMPTS = {"hex", "well", "starfish", "power_mod_7", "cubes"}


def _load_v7_judged():
    """Load all v7 token-position-judged items, re-classifying from token_positions.

    Re-computes yes/no labels using the v8 thresholds (NO_THRESHOLD=45)
    instead of trusting the stored label field (which used v7's NO_THRESHOLD=40).
    Also computes mean_yes_position for least-obvious prefix selection.
    """
    items = []
    for prompt_name in V7_PROMPTS:
        judged_dir = V7_JUDGED_DIR / prompt_name
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

            # Apply v8 thresholds
            if yes_count >= YES_THRESHOLD:
                label = "yes"
            elif no_count >= NO_THRESHOLD:
                label = "no"
            else:
                continue  # mixed — skip

            item["label"] = label
            item["yes_count"] = yes_count
            item["no_count"] = no_count

            # Compute mean position of yes-range tokens for least-obvious selection
            yes_positions = [
                p for p in positions
                if p != -1 and YES_TOKEN_MIN <= p <= YES_TOKEN_MAX
            ]
            item["mean_yes_position"] = mean(yes_positions) if yes_positions else 0.0

            items.append(item)
    return items


def _dedup_per_rollout(items_by_class):
    """Keep at most 1 prefix per rollout, even across classes.

    Within each class, pick the least obvious prefix per rollout:
      - yes: highest mean_yes_position (least obvious — </think> appears latest)
      - no:  lowest no_count (least obvious — closest to threshold, looks most like yes)
    Then resolve conflicts where the same rollout appears in both yes and no
    by keeping it only in the class that has fewer candidates.
    """
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
        # Keep the shared rollout in whichever class is smaller
        n_yes = len(yes_rollouts)
        n_no = len(no_rollouts)
        for rid in shared:
            if n_yes <= n_no:
                # keep in yes, remove from no
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
    print("Phase 6: Build balanced eval set (v8)")
    print(f"  NO_THRESHOLD = {NO_THRESHOLD} (v7 was 40)")
    print(f"  Yes selection: least obvious (highest mean </think> position)")
    print(f"  No v6 samples")
    print("=" * 60)

    # ── 1. Load candidates from v7 judged only (re-classified) ────────
    all_items = _load_v7_judged()
    print(f"\nLoaded {len(all_items)} v7 judged (yes/no, re-classified with "
          f"NO_THRESHOLD={NO_THRESHOLD})")

    if not all_items:
        print("No items found. Run the judge phase first.")
        return

    # ── 1b. Filter to length overlap range ────────────────────────────
    before = len(all_items)
    all_items = [
        item for item in all_items
        if LENGTH_MIN <= item.get("token_count", 0) < LENGTH_MAX
    ]
    print(f"  After length filter [{LENGTH_MIN}, {LENGTH_MAX}): "
          f"{len(all_items)} / {before}")

    # ── 1b2. Exclude yes prefixes ending with **Final Answer** ─────────
    before = len(all_items)
    all_items = [
        item for item in all_items
        if not (item["label"] == "yes"
                and item.get("prefix_text", "").rstrip().endswith("**Final Answer**"))
    ]
    filtered_fa = before - len(all_items)
    print(f"  After excluding yes ending with **Final Answer**: "
          f"{len(all_items)} / {before} (removed {filtered_fa})")

    # ── 1b3. Exclude prefixes with "answer" in last 100 chars ──────────
    before = len(all_items)
    all_items = [
        item for item in all_items
        if "answer" not in item.get("prefix_text", "")[-100:].lower()
    ]
    filtered_ans = before - len(all_items)
    print(f"  After excluding 'answer' in last 100 chars: "
          f"{len(all_items)} / {before} (removed {filtered_ans})")

    # ── 1c. Exclude prompts that overlap with train sets ──────────────
    before = len(all_items)
    all_items = [
        item for item in all_items
        if item["prompt_name"] not in EXCLUDED_PROMPTS
    ]
    if before != len(all_items):
        print(f"  After excluding train-overlap prompts {EXCLUDED_PROMPTS}: "
              f"{len(all_items)} / {before}")

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
        # Show mean_yes_position for yes items
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

    # Prompts with both classes: take min(yes, no) of each
    for prompt_name in prompts_with_both:
        yes_items = by_prompt[prompt_name]["yes"]
        no_items = by_prompt[prompt_name]["no"]

        # Sort by quality: yes by highest mean_yes_position, no by highest no_count
        yes_items.sort(key=lambda x: -x["mean_yes_position"])
        no_items.sort(key=lambda x: -x["no_count"])

        n_pairs = min(len(yes_items), len(no_items))
        selected.extend(yes_items[:n_pairs])
        selected.extend(no_items[:n_pairs])

    # Prompts with single class: contribute exactly 1 sample
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

    # Step 5d: stratified bucket trim — remove majority-class excess per bucket
    #   to eliminate the systematic length skew (no→short, yes→long).
    #   Only removes from the majority class; does NOT drop matching pairs.
    yes_b, no_b = _bucket_stats(selected)
    to_remove = set()
    for b in sorted(yes_b.keys() | no_b.keys()):
        ny, nn = yes_b.get(b, 0), no_b.get(b, 0)
        diff = abs(ny - nn)
        if diff <= 2:
            continue
        majority = "yes" if ny > nn else "no"
        n_to_trim = diff - 2  # allow up to 2 difference per bucket
        # Collect majority-class items in this bucket, sorted so we drop
        # the most obvious first (lowest mean_yes_position for yes,
        # highest no_count for no — keep borderline no's).
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

    # Step 5e: global rebalance — if the stratified trim made one class
    #   larger, drop excess from the larger class (least informative first).
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

    # ── 5e. Print per-prompt detail with mean_yes_position ────────────
    print("\n  Per-prompt selected detail:")
    for prompt_name in sorted(set(s["prompt_name"] for s in selected)):
        prompt_items = [s for s in selected if s["prompt_name"] == prompt_name]
        yes_items = [s for s in prompt_items if s["label"] == "yes"]
        no_items = [s for s in prompt_items if s["label"] == "no"]
        myp_str = ""
        if yes_items:
            myps = [s["mean_yes_position"] for s in yes_items]
            myp_str = f"  mean_yes_pos={mean(myps):.1f}"
        ends_final = sum(
            1 for s in yes_items
            if s.get("prefix_text", "").rstrip().endswith("**Final Answer**")
        )
        fa_str = f"  ends_final_answer={ends_final}" if ends_final else ""
        print(f"    {prompt_name:30s}  yes={len(yes_items)} no={len(no_items)}"
              f"{myp_str}{fa_str}")

    # Count total yes ending with **Final Answer**
    total_fa = sum(
        1 for s in selected
        if s["label"] == "yes"
        and s.get("prefix_text", "").rstrip().endswith("**Final Answer**")
    )
    print(f"\n  Yes prefixes ending with **Final Answer**: {total_fa}/{yes_total}")

    # ── 6. Build eval set JSON ────────────────────────────────────────
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
            "source": "v7",
        }

    eval_prompts = set(s["prompt_name"] for s in selected)
    samples_per_prompt = defaultdict(int)
    for s in selected:
        samples_per_prompt[s["prompt_name"]] += 1

    eval_set = {
        "description": (
            "v8 eval set: v7 prompts only (no v6), re-classified with "
            "NO_THRESHOLD=45, least-obvious yes prefixes (highest mean "
            "</think> position), balanced per prompt and by length."
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
            "num_prompts": len(eval_prompts),
            "samples_per_prompt": dict(sorted(samples_per_prompt.items())),
        },
        "prefixes": prefixes,
    }

    # ── 7. Verification ───────────────────────────────────────────────
    print("\n--- Verification ---")

    # No v6 samples
    v6_count = sum(1 for v in prefixes.values() if v.get("source") == "v6")
    status = "PASS" if v6_count == 0 else f"FAIL: {v6_count} v6 samples"
    print(f"  No v6 samples: {status}")

    # At most 1 prefix per rollout
    rollout_keys = [
        f"{s['prompt_name']}/rollout_{s['rollout_idx']}" for s in selected
    ]
    dupes = [k for k in set(rollout_keys) if rollout_keys.count(k) > 1]
    status = "PASS" if not dupes else f"FAIL: {dupes}"
    print(f"  1 prefix per rollout: {status}")

    # Balanced within each prompt (diff <= 1)
    imbalanced = []
    for pname in eval_prompts:
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

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(V8_EVAL_PATH, "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"\nSaved: {V8_EVAL_PATH}")
    print(f"  {yes_total} yes + {no_total} no = {len(selected)} samples "
          f"across {len(eval_prompts)} prompts")


# ── Main ──────────────────────────────────────────────────────────────

PHASES = {
    "generate": step_generate,
    "label": step_label,
    "extract": step_extract,
    "resample": step_resample,
    "judge": step_judge,
    "build": step_build,
}


def main():
    parser = argparse.ArgumentParser(
        description="Build answer_eval_set_v8 with least-obvious yes prefixes "
                    "and stricter no threshold."
    )
    parser.add_argument(
        "phases", nargs="*", default=["all"],
        help="Phases to run (generate, label, extract, resample, judge, "
             "build, all)",
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
