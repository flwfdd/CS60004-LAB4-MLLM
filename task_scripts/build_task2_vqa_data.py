from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task2 import QwenTeacher, parse_json_list

DEFAULT_SOURCE_JSONL = ROOT / "data/mm_lab/data/task2/pool.jsonl"
DEFAULT_IMAGE_ROOT = ROOT / "data/mm_lab"
DEFAULT_MODEL_PATH = ROOT / "data/models/Qwen3-VL-8B-Instruct"
DEFAULT_OUTPUT = ROOT / "outputs/task2/task2_vqa_teacher_both.jsonl"

TASK_TYPES = {
    "yes_no": "Ask a grounded yes/no question about clearly visible content.",
    "object": "Ask for the identity of a clearly visible object.",
    "attribute": "Ask for a visible attribute such as material, size, state, or type.",
    "color": "Ask for the visible color of a clearly grounded object.",
    "counting": "Ask for a small exact count of visible objects.",
    "spatial": "Ask about a simple spatial relation such as left, right, above, below, behind, in front of, near, or next to.",
    "location": "Ask where an object is or which side of the image it is on.",
}

TASK_TYPE_POOL = (
    ["yes_no"] * 42
    + ["object"] * 19
    + ["attribute"] * 11
    + ["counting"] * 10
    + ["spatial"] * 12
    + ["color"] * 4
    + ["location"] * 2
)

BAD_ANSWER_RE = re.compile(
    r"\b(many|several|various|multiple|a lot|lots|some|few|none visible|unknown|unclear|cannot tell|can't tell|not sure)\b",
    re.I,
)
BAD_QUESTION_RE = re.compile(
    r"bbox|bounding box|normalized_bbox|x_left_top|y_left_top|x_right_down|y_right_down|annotation|annotations|coordinate|coordinates|\[[0-9.,\s]+\]",
    re.I,
)
BAD_PREFIX_RE = re.compile(r"^(describe|summarize|list|name all|write|tell me about)\b", re.I)
YES_NO_PREFIX_RE = re.compile(r"^(is|are|do|does|did|has|have|had|was|were|can|could|will|would)\b", re.I)
SPATIAL_RE = re.compile(r"\bleft\b|\bright\b|\babove\b|\bbelow\b|\bbehind\b|\bin front of\b|\bnear\b|\bnext to\b|\bbetween\b|\bund?er\b", re.I)
LOCATION_RE = re.compile(r"^(where\b|what side\b|which side\b|on which side\b)", re.I)
COLOR_RE = re.compile(r"what colou?r", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def token_count(text: str) -> int:
    return len([tok for tok in re.split(r"\s+", text.strip()) if tok])


def coarse_region(bbox: list[float]) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if cx < 0.33:
        horiz = "left"
    elif cx > 0.67:
        horiz = "right"
    else:
        horiz = "center"
    if cy < 0.33:
        vert = "top"
    elif cy > 0.67:
        vert = "bottom"
    else:
        vert = "middle"
    if horiz == "center":
        return vert
    if vert == "middle":
        return horiz
    return f"{vert}-{horiz}"


def area_bucket(area_ratio: float) -> str:
    if area_ratio >= 0.15:
        return "large"
    if area_ratio >= 0.03:
        return "medium"
    return "small"


def fmt_annotations(row: dict[str, Any], max_objects: int) -> str:
    anns = sorted(
        row.get("annotations", []),
        key=lambda ann: float(ann.get("area_ratio", 0.0)),
        reverse=True,
    )[:max_objects]
    lines = []
    for ann in anns:
        category = str(ann["category"]).replace("_", " ")
        region = coarse_region(ann["bbox"])
        size = area_bucket(float(ann.get("area_ratio", 0.0)))
        synonyms = [str(x) for x in ann.get("synonyms", []) if str(x).strip()]
        alias = synonyms[1] if len(synonyms) > 1 else (synonyms[0] if synonyms else category)
        lines.append(f"- {category}; region: {region}; size: {size}; alias: {alias}")
    return "\n".join(lines) or "- none"


def desired_task_types(row: dict[str, Any], samples_per_image: int) -> list[str]:
    image_id = int(row.get("image_id", 0))
    start = image_id % len(TASK_TYPE_POOL)
    picked: list[str] = []
    used: set[str] = set()
    idx = start
    while len(picked) < samples_per_image:
        task_type = TASK_TYPE_POOL[idx % len(TASK_TYPE_POOL)]
        idx += 11
        if task_type in used and len(used) < len(TASK_TYPES):
            continue
        picked.append(task_type)
        used.add(task_type)
    return picked


def build_prompt(
    row: dict[str, Any],
    *,
    samples_per_image: int,
    max_objects: int,
    teacher_mode: str,
) -> str:
    requested_types = desired_task_types(row, samples_per_image)
    captions = [str(x).strip() for x in row.get("captions", []) if str(x).strip()]
    if not captions and row.get("caption"):
        captions = [str(row["caption"]).strip()]
    caption_block = "\n".join(f"- {c}" for c in captions[:3]) or "- none"
    negs = [str(x).replace("_", " ") for x in row.get("neg_category_names", []) if str(x).strip()]
    neg_block = ", ".join(negs[:4]) or "none"
    task_block = "\n".join(f"- {name}: {TASK_TYPES[name]}" for name in requested_types)
    mode_line = (
        "Look at the image and use the structured hints as support."
        if teacher_mode == "vision"
        else "Use only the structured hints below. Do not invent unseen objects."
    )
    return f"""You are generating short-answer VQA training data that should match a GQA-style validation distribution.

{mode_line}

Image hints:
Captions:
{caption_block}

Annotated objects:
{fmt_annotations(row, max_objects)}

Absent categories that should stay absent:
- {neg_block}

Generate exactly {samples_per_image} JSON items, one for each requested task type:
{task_block}

Requirements:
1. Output JSON only as an array.
2. Each item must contain task_type, question, answer.
3. Questions must be natural VQA questions, not instructions. Do not use Describe, Summarize, List, or Name all.
4. Questions and answers must be in English.
5. Answers must be very short: yes/no, one word, a short phrase, or a small integer.
6. Counting answers must be exact digits, not vague words.
7. Do not mention bounding boxes, coordinates, annotations, regions, or object indices.
8. Avoid vague answers such as many, several, some, various, cannot tell, or unknown.
9. Keep wording concise and close to GQA style.
10. Make every question answerable from the visible image content.
"""


def validate_by_task_type(task_type: str, question: str, answer: str) -> bool:
    q = question.lower()
    a = answer.lower()
    if task_type == "yes_no":
        return YES_NO_PREFIX_RE.match(q) is not None and a in {"yes", "no"}
    if task_type == "counting":
        return "how many" in q and a.isdigit()
    if task_type == "spatial":
        return SPATIAL_RE.search(q) is not None
    if task_type == "location":
        return LOCATION_RE.search(q) is not None or a in {"left", "right", "top", "bottom", "middle", "center"}
    if task_type == "color":
        return COLOR_RE.search(q) is not None
    return True


def clean_items(
    items: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    teacher_mode: str,
    samples_per_image: int,
) -> list[dict[str, Any]]:
    cleaned = []
    seen_questions: set[str] = set()
    for idx, item in enumerate(items):
        task_type = normalize_text(str(item.get("task_type", ""))).lower()
        question = normalize_text(str(item.get("question", "")))
        answer = normalize_text(str(item.get("answer", ""))).strip("\"'`.,;:!?。")
        if task_type not in TASK_TYPES:
            continue
        if not question or not answer:
            continue
        if BAD_PREFIX_RE.search(question):
            continue
        if BAD_QUESTION_RE.search(question):
            continue
        if BAD_ANSWER_RE.search(answer):
            continue
        if token_count(question) > 22 or len(question) > 160:
            continue
        if token_count(answer) > 3 or len(answer) > 24:
            continue
        if not question.endswith("?"):
            question = question.rstrip(".!") + "?"
        if not validate_by_task_type(task_type, question, answer):
            continue
        q_key = question.lower()
        if q_key in seen_questions:
            continue
        seen_questions.add(q_key)
        cleaned.append(
            {
                "id": f"task2-vqa-{teacher_mode}-{row['image_id']}-{idx}",
                "image": row["image"],
                "question": question,
                "answer": answer,
                "source": f"qwen3vl_task2_vqa_{teacher_mode}",
                "task_type": task_type,
                "teacher_mode": teacher_mode,
                "teacher_model": "Qwen3-VL-8B-Instruct",
                "image_id": row["image_id"],
            }
        )
        if len(cleaned) >= samples_per_image:
            break
    return cleaned


def build_rows_for_mode(
    source_rows: list[dict[str, Any]],
    teacher: QwenTeacher,
    *,
    image_root: Path,
    batch_size: int,
    samples_per_image: int,
    max_objects: int,
    teacher_mode: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    bar = tqdm(total=len(source_rows), desc=f"task2-vqa-{teacher_mode}", dynamic_ncols=True)
    for start in range(0, len(source_rows), batch_size):
        chunk = source_rows[start : start + batch_size]
        prompts = [
            build_prompt(
                row,
                samples_per_image=samples_per_image,
                max_objects=max_objects,
                teacher_mode=teacher_mode,
            )
            for row in chunk
        ]
        images = [image_root / row["image"] for row in chunk] if teacher_mode == "vision" else [None] * len(chunk)
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
                items = parse_json_list(text)
            except Exception as e:
                bar.write(f"skip image_id={row.get('image_id')}: {e}")
                continue
            cleaned = clean_items(
                items,
                row,
                teacher_mode=teacher_mode,
                samples_per_image=samples_per_image,
            )
            out.extend(cleaned)
            kept += len(cleaned)
        bar.update(len(chunk))
        bar.set_postfix(rows=len(out), last=kept)
    bar.close()
    return out


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            row["image"],
            row["question"].strip().lower(),
            row["answer"].strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def summarize(rows: list[dict[str, Any]]) -> str:
    counter = Counter(row["task_type"] for row in rows)
    ordered = ", ".join(f"{k}={counter[k]}" for k in sorted(counter))
    return f"rows={len(rows)} | {ordered}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-image", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=12)
    parser.add_argument("--teacher-mode", choices=["text", "vision", "both"], default="both")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.source_jsonl)
    if args.offset > 0:
        rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]

    teacher = QwenTeacher(args.model_path)
    all_rows: list[dict[str, Any]] = []
    modes = [args.teacher_mode] if args.teacher_mode != "both" else ["text", "vision"]
    for mode in modes:
        mode_rows = build_rows_for_mode(
            rows,
            teacher,
            image_root=args.image_root,
            batch_size=args.batch_size,
            samples_per_image=args.samples_per_image,
            max_objects=args.max_objects,
            teacher_mode=mode,
        )
        print(f"mode={mode} {summarize(mode_rows)}")
        all_rows.extend(mode_rows)

    all_rows = dedupe_rows(all_rows)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(all_rows)
    write_jsonl(args.output, all_rows)
    print(f"saved {len(all_rows)} rows -> {args.output}")
    print(summarize(all_rows))


if __name__ == "__main__":
    main()
