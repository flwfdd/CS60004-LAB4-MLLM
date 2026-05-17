"""
从 evaluate_vqa.py 的输出 jsonl 中提取预测错误的样本，
按 id 从源 jsonl 整行复制，得到可直接喂给 task1.py 的 hard-case 训练 jsonl。
支持按 bbox/no_bbox 与问题类型做筛选。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PRED_PATH = ROOT / "outputs/task1/train_internvl2_bbox_prompt.jsonl"
DEFAULT_SOURCE_PATH = ROOT / "data/mm_lab/data/task1/train.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "outputs/task1/hard_train.jsonl"

YES_NO_PREFIX_RE = re.compile(
    r"^(is|are|do|does|did|has|have|had|was|were|can)\b", re.I
)
SPATIAL_RE = re.compile(
    r"\bleft\b|\bright\b|\babove\b|\bbelow\b|\bbehind\b|\bin front of\b|\bnear\b|\bnext to\b|\bbetween\b|\bund?er\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def classify_answer_type(question: str, answer: str) -> str:
    q = question.strip().lower()
    a = str(answer).strip().lower()
    if YES_NO_PREFIX_RE.match(q) or a in {"yes", "no"}:
        return "yes_no"
    if "how many" in q or a.isdigit():
        return "counting"
    if "what color" in q or "what colour" in q:
        return "color"
    if SPATIAL_RE.search(q):
        return "spatial"
    if q.startswith("where "):
        return "location"
    if q.startswith("what is") or q.startswith("what are") or q.startswith("which"):
        return "object_attribute"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-jsonl", type=Path, default=DEFAULT_PRED_PATH)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--bbox-filter", choices=["all", "bbox", "no_bbox"], default="all"
    )
    parser.add_argument("--answer-types", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    preds = read_jsonl(args.pred_jsonl)
    source_by_id = {str(r["id"]): r for r in read_jsonl(args.source_jsonl)}
    answer_type_filter = set(args.answer_types)

    wrong_counter = Counter()
    kept_counter = Counter()
    rows = []
    for pred in preds:
        if pred.get("correct") is not False:
            continue
        pred_id = str(pred.get("id"))
        source_row = source_by_id.get(pred_id)
        if source_row is None:
            continue

        has_bbox = bool(pred.get("has_bbox"))
        bbox_tag = "bbox" if has_bbox else "no_bbox"
        answer_type = classify_answer_type(
            str(source_row.get("question", "")),
            str(source_row.get("answer", "")),
        )
        wrong_counter[(bbox_tag, answer_type)] += 1

        if args.bbox_filter != "all" and bbox_tag != args.bbox_filter:
            continue
        if answer_type_filter and answer_type not in answer_type_filter:
            continue

        kept_counter[(bbox_tag, answer_type)] += 1
        rows.append(
            {
                **source_row,
                "hard_case_prediction": pred.get("prediction"),
                "hard_case_has_bbox": has_bbox,
                "hard_case_answer_type": answer_type,
                "hard_case_pred_path": str(args.pred_jsonl),
            }
        )

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"pred={len(preds)}, kept={len(rows)} -> {args.output_path}")
    print("wrong case distribution:")
    for (bbox_tag, answer_type), count in sorted(wrong_counter.items()):
        print(f"  {bbox_tag:7s} | {answer_type:16s}: {count}")
    print("kept case distribution:")
    for (bbox_tag, answer_type), count in sorted(kept_counter.items()):
        print(f"  {bbox_tag:7s} | {answer_type:16s}: {count}")


if __name__ == "__main__":
    main()
