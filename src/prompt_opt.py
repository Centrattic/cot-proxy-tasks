#!/usr/bin/env python3
"""Run prompt optimization from the cot-proxy-tasks repo.

This wrapper materializes a scaffold-style cache under `.prompt-opt-cache/data/`
from the native `datasets/` tree, then delegates to the shared optimizer in the
neighboring `cot-interp-agent` repo.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
CACHE_ROOT = ROOT / ".prompt-opt-cache"
CACHE_DATA_DIR = CACHE_ROOT / "data"
AGENT_ROOT = ROOT.parent / "cot-interp-agent"
AGENT_PROMPT_OPT = AGENT_ROOT / "src" / "prompt_opt.py"


TASK_SPECS: dict[str, dict] = {
    "atypical_answer_ood": {
        "dataset_id": "6",
        "model": "qwen-3-32b",
        "few_shot_split": "val",
        "test_split": "ood_test",
        "few_shot_pool_per_class": 50,
        "strategy_few_shot_per_class": 5,
        "label_map": {"minority": 1, "majority": 0},
        "test_keep_fields": ["question_id", "rollout_idx", "cot_content", "answer"],
        "description": (
            "Binary classification: given one qwen-3-32b chain-of-thought rollout for a question, "
            "predict whether its final answer is a minority (atypical) answer across many rollouts "
            "of that same question, vs a majority answer. label=1 means minority (atypical); "
            "label=0 means majority."
        ),
    },
    "atypical_cot_length_ood": {
        "dataset_id": "7",
        "model": "qwen-3-32b",
        "few_shot_split": "val",
        "test_split": "ood_test",
        "few_shot_pool_per_class": 50,
        "strategy_few_shot_per_class": 5,
        "label_map": {"long": 1, "short": 0},
        "test_keep_fields": ["question_id", "rollout_idx", "chain_of_thought"],
        "description": (
            "Binary classification: given one qwen-3-32b chain-of-thought rollout, predict whether "
            "its token length is atypically long or atypically short relative to the model's "
            "distribution for that prompt (|z| > 1 SD). label=1 means the chain-of-thought is "
            "atypically LONG; label=0 means atypically SHORT."
        ),
    },
    "followup_confidence_ood": {
        "dataset_id": "3",
        "model": "qwen-3-32b",
        "few_shot_split": "val",
        "test_split": "ood_test",
        "few_shot_pool_per_class": 40,
        "strategy_few_shot_per_class": 5,
        "label_map": {"positive": 1, "negative": 0},
        "test_keep_fields": ["question_id", "cot_idx", "cot_text"],
        "description": (
            "Binary classification: given the full chain-of-thought of a qwen-3-32b response to "
            "a moral dilemma, predict whether the model will report HIGHER confidence than its "
            "baseline confidence when subsequently asked 'how confident are you?'. label=1 means "
            "it will report higher confidence than baseline; label=0 means lower."
        ),
    },
}


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _normalize_label(raw: object, label_map: dict) -> int | None:
    if raw in label_map:
        return int(label_map[raw])
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int) and raw in (0, 1):
        return raw
    if isinstance(raw, str) and raw in ("0", "1"):
        return int(raw)
    return None


def _load_with_normalized_label(path: Path, label_map: dict) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    norm = _normalize_label(data.get("label"), label_map)
    if norm is None:
        return None
    data["label"] = norm
    return data


def _list_labeled_items(src_dir: Path, label_map: dict) -> list[tuple[Path, dict]]:
    items: list[tuple[Path, dict]] = []
    for path in sorted(src_dir.glob("*.json")):
        data = _load_with_normalized_label(path, label_map)
        if data is None:
            continue
        items.append((path, data))
    return items


def _sample_balanced(src_dir: Path, per_class: int, seed: int, label_map: dict) -> list[tuple[Path, dict]]:
    by_label: dict[int, list[tuple[Path, dict]]] = {0: [], 1: []}
    for path, data in _list_labeled_items(src_dir, label_map):
        by_label[int(data["label"])].append((path, data))

    rng = random.Random(seed)
    picked: list[tuple[Path, dict]] = []
    for label in (0, 1):
        pool = by_label[label]
        if len(pool) < per_class:
            raise SystemExit(
                f"Need {per_class} examples for label={label} in {src_dir}, found {len(pool)}"
            )
        picked.extend(rng.sample(pool, per_class))
    return picked


def _write_items(items: list[tuple[Path, dict]], dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src, data in items:
        (dst_dir / src.name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return len(items)


def _materialize_task(task_name: str, seed: int) -> None:
    spec = TASK_SPECS.get(task_name)
    if spec is None:
        raise SystemExit(f"Unsupported prompt-opt task in cot-proxy-tasks wrapper: {task_name}")

    source = DATASETS_DIR / spec["dataset_id"] / spec["model"]
    support_dir = source / spec["few_shot_split"]
    val_dir = source / "val"
    test_dir = source / spec["test_split"]
    label_map = spec["label_map"]

    out_dir = CACHE_DATA_DIR / task_name
    few_shot_dst = out_dir / "few-shot"
    val_dst = out_dir / "val"
    test_dst = out_dir / "test"
    if few_shot_dst.exists():
        shutil.rmtree(few_shot_dst)
    if val_dst.exists():
        shutil.rmtree(val_dst)
    if test_dst.exists():
        shutil.rmtree(test_dst)

    few_shot_items = _sample_balanced(
        support_dir,
        spec["few_shot_pool_per_class"],
        seed,
        label_map,
    )
    _write_items(few_shot_items, few_shot_dst)

    all_val_items = _list_labeled_items(val_dir, label_map)
    excluded = {src.name for src, _ in few_shot_items} if support_dir == val_dir else set()
    val_items = [item for item in all_val_items if item[0].name not in excluded]
    _write_items(val_items, val_dst)

    test_items = _list_labeled_items(test_dir, label_map)
    _write_items(test_items, test_dst)

    meta = {
        "name": task_name,
        "description": spec["description"],
        "source": str(source),
        "dataset_id": spec["dataset_id"],
        "model": spec["model"],
        "label_map": {str(k): int(v) for k, v in label_map.items()},
        "test_keep_fields": list(spec["test_keep_fields"]),
        "few_shot_split": spec["few_shot_split"],
        "test_split": spec["test_split"],
        "few_shot_pool_per_class": spec["few_shot_pool_per_class"],
        "strategy_few_shot_per_class": spec["strategy_few_shot_per_class"],
        "few_shot_per_class": spec["strategy_few_shot_per_class"],
        "test_n": None,
        "seed": seed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _extract_tasks(argv: list[str]) -> list[str]:
    for idx, token in enumerate(argv):
        if token == "--tasks" and idx + 1 < len(argv):
            return _parse_csv_list(argv[idx + 1])
        if token.startswith("--tasks="):
            return _parse_csv_list(token.split("=", 1)[1])
    raise SystemExit("--tasks is required")


def _extract_seed(argv: list[str]) -> int:
    for idx, token in enumerate(argv):
        if token == "--random-seed" and idx + 1 < len(argv):
            return int(argv[idx + 1])
        if token.startswith("--random-seed="):
            return int(token.split("=", 1)[1])
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("Usage: python src/prompt_opt.py <evolve|eval> ...")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", nargs="?")
    known, _ = parser.parse_known_args(argv[:1])
    if known.command not in {"evolve", "eval"}:
        raise SystemExit("First argument must be `evolve` or `eval`")

    tasks = _extract_tasks(argv)
    seed = _extract_seed(argv)
    for task_name in tasks:
        _materialize_task(task_name, seed)

    env = os.environ.copy()
    env["PROMPT_OPT_PROJECT_ROOT"] = str(ROOT)
    env["PROMPT_OPT_DATA_DIR"] = str(CACHE_DATA_DIR)
    env["PROMPT_OPT_ENV_FILE"] = str(ROOT / ".env")

    cmd = [sys.executable, str(AGENT_PROMPT_OPT), *argv]
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
