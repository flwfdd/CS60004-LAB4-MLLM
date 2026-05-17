#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/bn/codeai-lq/mlx/users/fanliwen.2333/playground/code/CS60004-LAB4-MLLM"
PY_SCRIPT="$ROOT/task_scripts/task1_agent.py"
TRAIN_GQA="$ROOT/data/mm_lab/data/task1/train.jsonl"
TRAIN_A="$ROOT/outputs/task2/task2_A_text_teacher.jsonl"
TRAIN_B="$ROOT/outputs/task2/task2_B_vision_teacher.jsonl"
VAL_JSONL="$ROOT/data/mm_lab/data/val.jsonl"
IMAGE_ROOT="$ROOT/data/mm_lab"
OUT_ROOT="$ROOT/outputs/task2_repro_1024"
LOG_ROOT="$OUT_ROOT/logs"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_exp() {
  local name="$1"
  shift
  local outdir="$OUT_ROOT/$name"
  local logfile="$LOG_ROOT/$name.log"

  echo "[$(date '+%F %T')] START $name"
  uv run python "$PY_SCRIPT" \
    --config-name "$name" \
    --output-dir "$outdir" \
    --metrics-name metrics.json \
    --train-jsonl "$@" \
    --val-jsonl "$VAL_JSONL" \
    --image-root "$IMAGE_ROOT" \
    --freeze-config D_full \
    --epochs 1 \
    --lr 5e-6 \
    --batch-size 16 \
    --micro-batch-size 4 \
    --max-train-samples 1024 \
    --eval-interval 0 \
    --lr-schedule constant \
    --gradient-checkpointing \
    --prompt-mode no_bbox \
    --max-bboxes 12 \
    --use-wandb \
    --wandb-project cs60004-lab4-mllm \
    --wandb-run-prefix "$name" \
    2>&1 | tee "$logfile"
  echo "[$(date '+%F %T')] DONE  $name"
}

run_exp "task2_baseline_1024" "$TRAIN_GQA"
run_exp "task2_A_only_1024" "$TRAIN_A"
run_exp "task2_B_only_1024" "$TRAIN_B"
run_exp "task2_AB_1024" "$TRAIN_A" "$TRAIN_B"
run_exp "task2_1A_1024" "$TRAIN_GQA" "$TRAIN_A"
run_exp "task2_1B_1024" "$TRAIN_GQA" "$TRAIN_B"
run_exp "task2_1AB_1024" "$TRAIN_GQA" "$TRAIN_A" "$TRAIN_B"

echo "All Task2 1024-sample repro experiments finished. Metrics are under $OUT_ROOT/*/metrics.json"
