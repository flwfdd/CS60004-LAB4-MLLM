MODEL=internvl2
MODEL_PATH=data/models/InternVL2-2B
MODEL_PATH=outputs/task1/D_full
# MODEL=qwen3vl
# MODEL_PATH=data/models/Qwen3-VL-8B-Instruct

MODE=no_bbox
MODE=bbox_prompt
# MODE=cot

DATASET=data/mm_lab/data/val.jsonl
# DATASET=data/mm_lab/data/task3/train_with_bbox.jsonl
DATASET=data/mm_lab/data/test.jsonl

OUTPUT=outputs/task3/train_${MODEL}_${MODE}.jsonl

python task_scripts/evaluate_vqa.py \
    --model $MODEL \
    --model-path $MODEL_PATH \
    --dataset $DATASET \
    --mode $MODE \
    --output $OUTPUT \
    --max-new-tokens 8 \
    # --limit 100 \
    # --num-repeats 3 \
    # --repeat-do-sample \
    # --temperature 0.8