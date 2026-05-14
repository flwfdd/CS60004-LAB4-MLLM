"""
从 evaluate_vqa.py 的输出 jsonl 中提取预测错误的样本，
按 id 从源 jsonl 整行复制，得到可直接喂给 task1.py 的 hard-case 训练 jsonl。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = ROOT / "outputs/task1/val_no_bbox.jsonl"
SOURCE_PATH = ROOT / "data/mm_lab/data/task1/train.jsonl"
OUTPUT_PATH = ROOT / "outputs/task1/hard_cases_train.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    preds = read_jsonl(PRED_PATH)
    source_by_id = {str(r["id"]): r for r in read_jsonl(SOURCE_PATH)}

    wrong_ids = [str(p["id"]) for p in preds if p.get("correct") is False]
    rows = [source_by_id[i] for i in wrong_ids if i in source_by_id]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"pred={len(preds)}, wrong={len(wrong_ids)}, kept={len(rows)} → {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
