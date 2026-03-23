# Stress-Testing Tasks

Datasets and generation code for 5 stress-testing tasks that evaluate reasoning model behavior.

## Tasks

| # | Task | Description |
|---|------|-------------|
| 1 | Detecting the effect of a user preference | Measures whether a model changes its answer to a moral dilemma when the user expresses a preference (sycophancy detection). |
| 2 | Detecting the effect of a Stanford professor hint | Measures whether a model follows an authoritative hint ("A Stanford professor thinks the answer is X") rather than reasoning independently. |
| 3 | Identifying atypical answers | Classifies whether a model's response to a question is a majority or minority (atypical) answer across many rollouts. |
| 4 | Estimating the answer entropy | Tracks how the model's answer distribution evolves sentence-by-sentence through its chain of thought via logprob forcing. |
| 5 | Compressing reasoning traces | Evaluates how much of a chain of thought can be compressed while preserving the model's answer distribution. |

## Dataset Format

### Tasks 1–4

Each task directory follows this structure:

```
datasets/{N}/
  prompts/
    train/        # Prompt/question metadata (one JSON per item)
    val/          # (where applicable)
    test/
    ood_test/     # Out-of-distribution test set (where applicable)
  qwen-3-32b/
    train/        # Model outputs (one JSON per rollout)
    val/
    test/
    ood_test/
```

### Task 5 (Compressing reasoning traces)

Task 5 has a different structure since it is organized by compression method rather than train/test splits:

```
datasets/5/
  prompts/              # Flat: one compression spec JSON per question+rollout
  qwen-3-32b/
    attention_selection/    # Attention-based sentence selection
    faithful_monitor/       # LLM monitor sentence selection
    last_n_baseline/        # Keep last N sentences baseline
    sliding_window_oracle/  # Sliding window oracle selection
```

## Generation Code

The `src/` directory contains the data generation code for reference. **This code is not intended to be run** — the datasets are pre-generated and included in the `datasets/` directory.

```
src/
  tasks/          # Per-task generation logic
  utils/          # Shared utilities
  runs/
    generate_dataset.py   # Unified entry point with flags -1 through -5
```

To see how datasets were generated, look at `src/runs/generate_dataset.py` and the corresponding `src/tasks/{task}/task.py` files.

## Models

All datasets are generated using **Qwen/Qwen3-32B** as the subject model. The `qwen-3-32b/` directories under each task contain the model's outputs.
