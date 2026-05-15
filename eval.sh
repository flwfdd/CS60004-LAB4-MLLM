MODEL=internvl2
MODEL_PATH=data/models/InternVL2-2B
MODEL_PATH=outputs/task2/B_connector_language
# MODEL=qwen3vl
# MODEL_PATH=data/models/Qwen3-VL-8B-Instruct

MODE=no_bbox
# MODE=bbox_prompt

DATASET=data/mm_lab/data/val.jsonl

OUTPUT=outputs/task1/val_$MODEL_$MODE.jsonl

python task_scripts/evaluate_vqa.py \
    --model $MODEL \
    --model-path $MODEL_PATH \
    --dataset $DATASET \
    --mode $MODE \
    --output $OUTPUT