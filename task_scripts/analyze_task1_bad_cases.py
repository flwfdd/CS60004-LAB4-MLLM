from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED_JSONL = ROOT / "outputs/task3/train_internvl2_bbox_prompt.jsonl"

YES_NO_PREFIX_RE = re.compile(r"^(is|are|do|does|did|has|have|had|was|were|can)\b", re.I)
SPATIAL_RE = re.compile(
    r"\bleft\b|\bright\b|\babove\b|\bbelow\b|\bbehind\b|\bin front of\b|\bnear\b|\bnext to\b|\bbetween\b|\bund?er\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    parser.add_argument("--pred-jsonl", type=Path, default=DEFAULT_PRED_JSONL)
    parser.add_argument("--sample-per-type", type=int, default=3)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.pred_jsonl)
    wrong_rows = [row for row in rows if row.get("correct") is False]

    bbox_counter = Counter("bbox" if row.get("has_bbox") else "no_bbox" for row in wrong_rows)
    type_counter = Counter(
        classify_answer_type(str(row.get("question", "")), str(row.get("answer", "")))
        for row in wrong_rows
    )
    joint_counter = Counter(
        (
            "bbox" if row.get("has_bbox") else "no_bbox",
            classify_answer_type(str(row.get("question", "")), str(row.get("answer", ""))),
        )
        for row in wrong_rows
    )

    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wrong_rows:
        q_type = classify_answer_type(str(row.get("question", "")), str(row.get("answer", "")))
        if len(samples[q_type]) >= args.sample_per_type:
            continue
        samples[q_type].append(
            {
                "id": row.get("id"),
                "has_bbox": bool(row.get("has_bbox")),
                "question": row.get("question"),
                "gold": row.get("answer"),
                "prediction": row.get("prediction"),
            }
        )

    total = len(rows)
    wrong = len(wrong_rows)
    print(f"predictions={total}")
    print(f"wrong={wrong}")
    print("\nWrong by bbox:")
    for key, count in sorted(bbox_counter.items()):
        print(f"  {key}: {count}")

    print("\nWrong by answer type:")
    for key, count in type_counter.most_common():
        print(f"  {key}: {count}")

    print("\nWrong by bbox x answer type:")
    for (bbox_tag, answer_type), count in sorted(joint_counter.items()):
        print(f"  {bbox_tag:7s} | {answer_type:16s}: {count}")

    print("\nSample wrong cases:")
    for answer_type, rows_for_type in samples.items():
        print(f"\n[{answer_type}]")
        for row in rows_for_type:
            print(f"- id={row['id']} bbox={row['has_bbox']} gold={row['gold']} pred={row['prediction']}")
            print(f"  q: {row['question']}")

    if args.summary_json:
        summary = {
            "predictions": total,
            "wrong": wrong,
            "wrong_by_bbox": dict(bbox_counter),
            "wrong_by_answer_type": dict(type_counter),
            "wrong_by_bbox_answer_type": {
                f"{bbox_tag}:{answer_type}": count
                for (bbox_tag, answer_type), count in joint_counter.items()
            },
            "sample_wrong_cases": dict(samples),
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
