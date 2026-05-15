from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
POOL_JSONL = ROOT / "data/mm_lab/data/task2/pool.jsonl"
IMAGE_ROOT = ROOT / "data/mm_lab"
MODEL_PATH = ROOT / "data/models/Qwen3-VL-8B-Instruct"
OUT_A = ROOT / "outputs/task2/task2_A_text_teacher.jsonl"
OUT_B = ROOT / "outputs/task2/task2_B_vision_teacher.jsonl"

LIMIT_IMAGES = 256
BATCH_SIZE = 16
MAX_NEW_TOKENS = 2048
SAMPLES_PER_IMAGE = 8

TASK_TYPES = {
    "image_captioning": "Describe the image in one concise sentence, covering the main scene and subjects.",
    "object_recognition": "Ask the model to identify or list visible object categories in the image.",
    "object_counting": "Ask for the number of visible instances of one object category.",
    "dense_visual_summary": "Summarize the annotated visual content using objects, counts, and the overall scene.",
    "spatial_reasoning": "Ask a simple spatial relation question such as left/right/above/below/near.",
}


TASK_TYPE_BLOCK = "\n".join(f"- {name}: {desc}" for name, desc in TASK_TYPES.items())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt_objects(row: dict, *, keep_bbox: bool) -> str:
    lines = []
    for ann in row.get("annotations", []):
        name = ann["category"].replace("_", " ")
        if keep_bbox:
            lines.append(f"- {name}, bbox: {[round(x, 3) for x in ann['bbox']]}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines) or "- none"


def prompt_a(row: dict) -> str:
    return f"""Objects in the image:
{fmt_objects(row, keep_bbox=True)}
Image caption: {row.get("caption", "")}

Please generate {SAMPLES_PER_IMAGE} diverse multimodal instruction-response samples.

Allowed task types:
{TASK_TYPE_BLOCK}

Requirements:
1. Output a JSON array. Each item must contain task_type, question, and answer.
2. task_type must be one of the allowed task types above. Prefer diverse task types but do not invent new types.
3. question and answer must be in English.
4. For object_recognition, object_counting, and spatial_reasoning, write question as a visual question answering question and make answer a short answer, usually one word, a short phrase, yes/no, or a number.
5. The final question must not contain bbox coordinates or bbox lists.
6. Output JSON only. Do not add explanations."""


def prompt_b(row: dict) -> str:
    return f"""Objects in the image:
{fmt_objects(row, keep_bbox=True)}
Image caption: {row.get("caption", "")}

Please look at the image and use the annotations as hints. Generate {SAMPLES_PER_IMAGE} diverse multimodal instruction-response samples.

Allowed task types:
{TASK_TYPE_BLOCK}

Requirements:
1. Output a JSON array. Each item must contain task_type, question, and answer.
2. task_type must be one of the allowed task types above. Prefer diverse task types but do not invent new types.
3. question and answer must be in English.
4. For object_recognition, object_counting, and spatial_reasoning, write question as a visual question answering question and make answer a short answer, usually one word, a short phrase, yes/no, or a number.
5. The final question must not contain bbox coordinates or bbox lists.
6. Output JSON only. Do not add explanations."""


class QwenTeacher:
    def __init__(self, model_path: Path):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.processor.tokenizer.padding_side = "left"

    def generate(self, prompt: str, image: Path | None = None) -> str:
        return self.generate_batch([prompt], [image])[0]

    def generate_batch(
        self, prompts: list[str], images: list[Path | None]
    ) -> list[str]:
        messages = []
        for prompt, image in zip(prompts, images):
            content = []
            if image is not None:
                content.append({"type": "image", "image": str(image)})
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, ids)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def parse_json_list(text: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    if not text.startswith("["):
        match = re.search(r"\[.*\]", text, flags=re.S)
        text = match.group(0) if match else text
    data = json.loads(text)
    return data if isinstance(data, list) else []


def clean_items(items: list[dict]) -> list[dict]:
    rows = []
    seen = set()
    bad_pattern = re.compile(r"bbox|bounding box|\[[0-9.,\s]+\]|x_min|y_min", re.I)
    for item in items:
        inst = str(item.get("question", "")).strip()
        out = str(item.get("answer", "")).strip()
        task_type = str(item.get("task_type", "")).strip()
        if task_type not in TASK_TYPES:
            continue
        if not inst or not out or bad_pattern.search(inst):
            continue
        key = (task_type, inst.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"task_type": task_type, "question": inst, "answer": out})
    return rows[:SAMPLES_PER_IMAGE]


def build_rows(
    pool: list[dict], teacher: QwenTeacher, *, use_image: bool
) -> list[dict]:
    out = []
    tag = "B" if use_image else "A"
    subset = pool[:LIMIT_IMAGES]
    bar = tqdm(total=len(subset), desc=f"Task2-{tag}", dynamic_ncols=True)
    for start in range(0, len(subset), BATCH_SIZE):
        chunk = subset[start : start + BATCH_SIZE]
        prompts = [prompt_b(row) if use_image else prompt_a(row) for row in chunk]
        images = [IMAGE_ROOT / row["image"] if use_image else None for row in chunk]
        try:
            texts = teacher.generate_batch(prompts, images)
        except Exception as e:
            ids = ", ".join(str(row.get("image_id")) for row in chunk)
            bar.write(f"skip image_id={ids}: {e}")
            bar.update(len(chunk))
            continue

        kept = 0
        for row, text in zip(chunk, texts):
            try:
                items = clean_items(parse_json_list(text.strip()))
            except Exception as e:
                bar.write(f"skip image_id={row.get('image_id')}: {e}")
                continue
            kept += len(items)
            for j, item in enumerate(items):
                out.append(
                    {
                        "id": f"task2-{tag}-{row['image_id']}-{j}",
                        "image": row["image"],
                        "source": (
                            "qwen3vl_vision_teacher"
                            if use_image
                            else "qwen3vl_text_teacher"
                        ),
                        **item,
                    }
                )
        bar.update(len(chunk))
        bar.set_postfix(rows=len(out), last=kept)
    bar.close()
    return out


def main() -> None:
    pool = read_jsonl(POOL_JSONL)
    teacher = QwenTeacher(MODEL_PATH)

    rows_a = build_rows(pool, teacher, use_image=False)
    write_jsonl(OUT_A, rows_a)
    print(f"saved {len(rows_a)} rows -> {OUT_A}")

    rows_b = build_rows(pool, teacher, use_image=True)
    write_jsonl(OUT_B, rows_b)
    print(f"saved {len(rows_b)} rows -> {OUT_B}")


if __name__ == "__main__":
    main()
