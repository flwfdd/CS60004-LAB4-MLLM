#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/bn/codeai-lq/mlx/users/fanliwen.2333/playground/code/CS60004-LAB4-MLLM"
PY_SCRIPT="$ROOT/task_scripts/build_task2_vqa_data.py"
SOURCE_JSONL="${SOURCE_JSONL:-$ROOT/data/mm_lab/data/task2/pool.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT/data/mm_lab}"
MODEL_PATH="${MODEL_PATH:-$ROOT/data/models/Qwen3-VL-8B-Instruct}"
OUTPUT="${OUTPUT:-$ROOT/outputs/task2/task2_vqa_teacher_both.jsonl}"
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-0}"
BATCH_SIZE="8"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-2}"
MAX_OBJECTS="${MAX_OBJECTS:-12}"
TEACHER_MODE="${TEACHER_MODE:-both}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "$OUTPUT")"

uv run python "$PY_SCRIPT" \
  --source-jsonl "$SOURCE_JSONL" \
  --image-root "$IMAGE_ROOT" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --max-objects "$MAX_OBJECTS" \
  --teacher-mode "$TEACHER_MODE" \
  --shuffle \
  --seed "$SEED"
