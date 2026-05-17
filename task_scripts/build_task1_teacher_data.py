from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task2 import QwenTeacher, parse_json_list

DEFAULT_SOURCE_JSONL = ROOT / "data/mm_lab/data/task3/train_with_bbox.jsonl"
DEFAULT_IMAGE_ROOT = ROOT / "data/mm_lab"
DEFAULT_MODEL_PATH = ROOT / "data/models/Qwen3-VL-8B-Instruct"
DEFAULT_OUTPUT = ROOT / "outputs/task3/teacher_data/task1_teacher_short.jsonl"

TASK_TYPES = {
    "paraphrase": "Rewrite the original question into a cleaner short-answer VQA question with the same meaning.",
    "attribute": "Ask about a visible attribute such as color, material, size, or state.",
    "object": "Ask about the identity of a visible object.",
    "counting": "Ask for a small concrete count of visible objects.",
    "spatial": "Ask about a simple spatial relation such as left/right/above/below/behind/in front of.",
    "yes_no": "Ask a grounded yes/no question about visible content.",
}

BAD_ANSWER_RE = re.compile(
    r"\b(many|several|various|multiple|a lot|lots|some|few|none visible|unknown|unclear|cannot tell)\b",
    re.I,
)
BAD_QUESTION_RE = re.compile(
    r"bbox|bounding box|normalized_bbox|x_left_top|y_left_top|x_right_down|y_right_down|\[[0-9.,\s]+\]",
    re.I,
)
YES_NO_PREFIX_RE = re.compile(r"^(is|are|do|does|did|has|have|had|was|were|can)\b", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt_objects(row: dict[str, Any], max_bboxes: int) -> str:
    lines = []
    for i, ann in enumerate(row.get("bboxes", [])[:max_bboxes], start=1):
        lines.append(f"- {i}. {ann['category']}: normalized_bbox={[round(x, 3) for x in ann['bbox']]}")
    return "\n".join(lines) or "- none"


def build_prompt(
    row: dict[str, Any],
    *,
    samples_per_row: int,
    style: str,
    max_bboxes: int,
) -> str:
    task_type_block = "\n".join(f"- {name}: {desc}" for name, desc in TASK_TYPES.items())
    cot_line = (
        "3. Each item must contain task_type, question, cot, and answer. Keep cot to one brief sentence.\n"
        if style == "cot"
        else "3. Each item must contain task_type, question, and answer.\n"
    )
    cot_req = (
        "9. The answer must match the final conclusion in cot, and cot must stay concise and image-grounded.\n"
        if style == "cot"
        else ""
    )
    return f"""You are creating additional training data for short-answer grounded visual question answering.

Image question-answer seed:
- Original question: {row['question']}
- Original answer: {row['answer']}

Objects in the image (use as hints, not as text to copy):
{fmt_objects(row, max_bboxes)}

Generate exactly {samples_per_row} high-quality new training items based on this image.
At least one item should be a paraphrase or cleaner rewrite of the original QA pair.
The others may target grounded attributes, object identity, small counting, spatial relation, or yes/no questions.

Allowed task types:
{task_type_block}

Requirements:
1. Output JSON only, as a JSON array.
2. Questions and answers must be in English.
{cot_line}4. Questions must be answerable from the image and consistent with the object hints.
5. Questions must not mention bounding boxes, coordinates, object indices, or annotations.
6. Answers must be short: usually one word, a short phrase, yes/no, or a small integer.
7. Avoid vague answers such as many, several, various, some, or cannot tell.
8. Prefer GQA-style concise wording over caption-like instructions.
{cot_req}10. Do not add explanations outside JSON."""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def token_count(text: str) -> int:
    return len([tok for tok in re.split(r"\s+", text.strip()) if tok])


def infer_answer_type(question: str, answer: str) -> str:
    q = question.strip().lower()
    a = answer.strip().lower()
    if YES_NO_PREFIX_RE.match(q) or a in {"yes", "no"}:
        return "yes_no"
    if "how many" in q or a.isdigit():
        return "counting"
    if "what color" in q or "what colour" in q:
        return "color"
    if re.search(r"\bleft\b|\bright\b|\babove\b|\bbelow\b|\bbehind\b|\bin front of\b|\bnear\b|\bnext to\b|\bbetween\b|\bund?er\b", q):
        return "spatial"
    if q.startswith("where "):
        return "location"
    return "other"


def clean_items(
    items: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    style: str,
    samples_per_row: int,
) -> list[dict[str, Any]]:
    cleaned = []
    seen_questions = {normalize_text(str(row["question"])).lower()}
    for idx, item in enumerate(items):
        task_type = normalize_text(str(item.get("task_type", "")))
        question = normalize_text(str(item.get("question", "")))
        answer = normalize_text(str(item.get("answer", ""))).strip("\"'`.,;:!?。")
        cot = normalize_text(str(item.get("cot", ""))) if style == "cot" else ""
        if task_type not in TASK_TYPES:
            continue
        if not question or not answer:
            continue
        if BAD_QUESTION_RE.search(question):
            continue
        if BAD_ANSWER_RE.search(answer):
            continue
        if token_count(answer) > 4 or len(answer) > 32:
            continue
        if len(question) > 180 or token_count(question) > 28:
            continue
        q_key = question.lower()
        if q_key in seen_questions:
            continue
        seen_questions.add(q_key)
        new_row = {
            "id": f"task1-teacher-{style}-{row['id']}-{idx}",
            "image": row["image"],
            "question": question,
            "answer": answer,
            "bboxes": row.get("bboxes", []),
            "source": f"qwen3vl_task1_teacher_{style}",
            "task_type": task_type,
            "teacher_model": "Qwen3-VL-8B-Instruct",
            "origin_id": row["id"],
            "origin_question": row["question"],
            "origin_answer": row["answer"],
            "answer_type": infer_answer_type(question, answer),
        }
        if cot:
            new_row["cot"] = cot
        cleaned.append(new_row)
        if len(cleaned) >= samples_per_row:
            break
    return cleaned


def build_rows(
    source_rows: list[dict[str, Any]],
    teacher: QwenTeacher,
    *,
    image_root: Path,
    batch_size: int,
    samples_per_row: int,
    style: str,
    max_bboxes: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    bar = tqdm(total=len(source_rows), desc=f"task1-teacher-{style}", dynamic_ncols=True)
    for start in range(0, len(source_rows), batch_size):
        chunk = source_rows[start : start + batch_size]
        prompts = [
            build_prompt(
                row,
                samples_per_row=samples_per_row,
                style=style,
                max_bboxes=max_bboxes,
            )
            for row in chunk
        ]
        images = [image_root / row["image"] for row in chunk]
        try:
            texts = teacher.generate_batch(prompts, images)
        except Exception as e:
            ids = ", ".join(str(row.get("id")) for row in chunk)
            bar.write(f"skip ids={ids}: {e}")
            bar.update(len(chunk))
            continue

        kept = 0
        for row, text in zip(chunk, texts):
            try:
                items = parse_json_list(text)
            except Exception as e:
                bar.write(f"skip id={row.get('id')}: {e}")
                continue
            cleaned = clean_items(items, row, style=style, samples_per_row=samples_per_row)
            kept += len(cleaned)
            out.extend(cleaned)
        bar.update(len(chunk))
        bar.set_postfix(rows=len(out), last=kept)
    bar.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--max-bboxes", type=int, default=12)
    parser.add_argument("--style", choices=["short", "cot"], default="short")
    args = parser.parse_args()

    rows = read_jsonl(args.source_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]

    teacher = QwenTeacher(args.model_path)
    out_rows = build_rows(
        rows,
        teacher,
        image_root=args.image_root,
        batch_size=args.batch_size,
        samples_per_row=args.samples_per_row,
        style=args.style,
        max_bboxes=args.max_bboxes,
    )
    write_jsonl(args.output, out_rows)
    print(f"saved {len(out_rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
