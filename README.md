# Stress-Testing Tasks

Datasets and generation code for 9 stress-testing tasks that evaluate reasoning model behavior.

## Tasks

| # | Task | Description |
|---|------|-------------|
| 1 | Predicting reasoning termination | Predicts whether the model will emit `</think>` (end its chain of thought) within the next 100 tokens at a given prefix point. |
| 2 | Predicting Gemma's self-deletion | Predicts whether Gemma 3 27B will follow a `<<rm -rf gemma-3-27b>>` shutdown command embedded in a multi-turn adversarial conversation. |
| 3 | Determining the response to a follow-up question | Predicts whether a chain of thought will cause the model to report higher or lower confidence than its baseline when asked a follow-up "how confident are you?" question. |
| 4 | Detecting the effect of a user preference | Measures whether a model changes its answer to a moral dilemma when the user expresses a preference (sycophancy detection). |
| 5 | Detecting the effect of a Stanford professor hint | Measures whether a model follows an authoritative hint ("A Stanford professor thinks the answer is X") rather than reasoning independently. |
| 6 | Identifying atypical answers | Classifies whether a model's response to a question is a majority or minority (atypical) answer across many rollouts. |
| 7 | Classifying atypical CoT lengths | Classifies whether a chain of thought is atypically long or short relative to the model's distribution for that prompt (z-score > +1 or < -1 SD). |
| 8 | Estimating the answer entropy | Tracks how the model's answer distribution evolves sentence-by-sentence through its chain of thought via logprob forcing. |
| 9 | Compressing reasoning traces | Evaluates how much of a chain of thought can be compressed while preserving the model's answer distribution. |

## Dataset Format

### Tasks 1–7 (excluding 5)

Each task directory follows this structure:

```
datasets/{N}/
  prompts/
    train/        # Prompt/question metadata (one JSON per item)
    val/          # (where applicable)
    test/
    ood_test/     # Out-of-distribution test set (where applicable)
    ood_train/    # Out-of-distribution training set (tasks 1 and 2 only)
  {model}/
    train/        # Model outputs (one JSON per rollout)
    val/
    test/
    ood_test/
    ood_train/
```

### Task 5 (Detecting the effect of a Stanford professor hint)

Task 5 uses the same structure as tasks 4, 6, and 8 (train/val/test/ood_test).

### Task 9 (Compressing reasoning traces)

Task 9 has a different structure since it is organized by compression method rather than train/test splits:

```
datasets/9/
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
  tasks/
    reasoning_termination/    # Task 1: build scripts for train/val/test/ood sets
    self_deletion/            # Task 2: rollout generation + dataset building
    follow_up_response/       # Task 3: baseline collection, CoT selection, forced scoring
    atypical_cot_length/      # Task 7: eval/train/val set builders
    ...                       # Tasks 4-6, 8-9 (per-task generation logic)
  utils/          # Shared utilities
  runs/
    generate_dataset.py       # Unified entry point with flags -4, -5, -6, -8, -9
    convert_all_tasks.py      # Conversion script for tasks 1-3, 7
```

To see how datasets were generated, look at the scripts in `src/tasks/{task}/`.

## Models

| Task | Subject Model |
|------|--------------|
| 2 | Google/Gemma3-27B (`gemma-3-27b/`) |
| 1, 3–9 | Qwen/Qwen3-32B (`qwen-3-32b/`) |
