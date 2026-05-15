from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = ROOT / "data/mm_lab"
DEFAULT_OUT_ROOT = ROOT / "outputs/task3"

SOURCES = {
    "train": ROOT / "data/mm_lab/data/task3/train_with_bbox.jsonl",
    "val": ROOT / "data/mm_lab/data/val.jsonl",
}

EXPERIMENTS = {
    "same_box": {
        "visual_mode": "same_box",
        "prompt_mode": "no_bbox",
    },
    "color_box": {
        "visual_mode": "color_box",
        "prompt_mode": "no_bbox",
    },
    "color_box_label": {
        "visual_mode": "color_box_label",
        "prompt_mode": "no_bbox",
    },
    "same_box_bbox_prompt": {
        "visual_mode": "same_box",
        "prompt_mode": "bbox_prompt",
    },
    "color_box_color_prompt": {
        "visual_mode": "color_box",
        "prompt_mode": "color_bbox_prompt",
    },
}

PALETTE = [
    ("red", (255, 40, 40)),
    ("blue", (30, 144, 255)),
    ("green", (30, 220, 80)),
    ("yellow", (255, 215, 0)),
    ("magenta", (255, 0, 255)),
    ("cyan", (0, 220, 220)),
    ("orange", (255, 140, 0)),
    ("purple", (160, 80, 255)),
    ("lime", (160, 255, 40)),
    ("pink", (255, 105, 180)),
    ("white", (255, 255, 255)),
    ("black", (0, 0, 0)),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def safe_id(value: Any) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def source_image_path(image: str, image_root: Path) -> Path:
    p = Path(image)
    if p.is_absolute():
        return p
    if image.startswith("data/mm_lab/"):
        return ROOT / image
    return image_root / image.lstrip("/")


def color_for(index: int, visual_mode: str) -> tuple[str, tuple[int, int, int]]:
    if visual_mode == "same_box":
        return "red", (255, 40, 40)
    return PALETTE[index % len(PALETTE)]


def text_fill_for(color: tuple[int, int, int]) -> tuple[int, int, int]:
    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


def annotate_bboxes(
    row: dict[str, Any], visual_mode: str, *, max_bboxes: int
) -> list[dict[str, Any]]:
    annotated = []
    for i, bbox in enumerate(row.get("bboxes", [])[:max_bboxes]):
        color_name, _ = color_for(i, visual_mode)
        annotated.append({**bbox, "color": color_name})
    return annotated


def draw_row_image(
    row: dict[str, Any],
    visual_mode: str,
    image_root: Path,
    out_path: Path,
    *,
    max_bboxes: int,
) -> None:
    src = source_image_path(row["image"], image_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not row.get("bboxes"):
        shutil.copy2(src, out_path)
        return

    image = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(3, min(width, height) // 160)
    font = ImageFont.load_default()

    for i, bbox in enumerate(row.get("bboxes", [])[:max_bboxes]):
        _, color = color_for(i, visual_mode)
        x1, y1, x2, y2 = bbox["bbox"]
        box = [
            max(0, min(width - 1, int(round(x1 * width)))),
            max(0, min(height - 1, int(round(y1 * height)))),
            max(0, min(width - 1, int(round(x2 * width)))),
            max(0, min(height - 1, int(round(y2 * height)))),
        ]
        draw.rectangle(box, outline=color, width=line_width)

        if visual_mode == "color_box_label":
            label = f"{i + 1}: {bbox['category']}"
            text_box = draw.textbbox((box[0], box[1]), label, font=font)
            pad = 3
            bg = [
                text_box[0] - pad,
                text_box[1] - pad,
                text_box[2] + pad,
                text_box[3] + pad,
            ]
            draw.rectangle(bg, fill=color)
            draw.text((box[0], box[1]), label, fill=text_fill_for(color), font=font)

    image.save(out_path, quality=95)


def convert_split(
    source_path: Path,
    split: str,
    visual_mode: str,
    image_root: Path,
    out_root: Path,
    *,
    max_bboxes: int,
    limit: int,
    overwrite: bool,
) -> Path:
    rows = read_jsonl(source_path)
    if limit > 0:
        rows = rows[:limit]

    out_rows = []
    image_dir = out_root / "visual_images" / visual_mode / split
    for row in tqdm(rows, desc=f"{visual_mode}:{split}", dynamic_ncols=True):
        ext = Path(row["image"]).suffix or ".jpg"
        out_image = image_dir / f"{safe_id(row['id'])}{ext}"
        if overwrite or not out_image.exists():
            draw_row_image(
                row, visual_mode, image_root, out_image, max_bboxes=max_bboxes
            )

        converted = {**row}
        converted["image"] = repo_rel(out_image)
        converted["bboxes"] = annotate_bboxes(row, visual_mode, max_bboxes=max_bboxes)
        out_rows.append(converted)

    out_jsonl = out_root / "visual_data" / visual_mode / f"{split}.jsonl"
    write_jsonl(out_jsonl, out_rows)
    return out_jsonl


def write_manifest(out_root: Path) -> None:
    manifest = {
        name: {
            **cfg,
            "train_jsonl": f"outputs/task3/visual_data/{cfg['visual_mode']}/train.jsonl",
            "val_jsonl": f"outputs/task3/visual_data/{cfg['visual_mode']}/val.jsonl",
            "image_root": ".",
        }
        for name, cfg in EXPERIMENTS.items()
    }
    path = out_root / "task3_visual_experiments.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visual-modes",
        nargs="+",
        choices=["same_box", "color_box", "color_box_label"],
        default=["same_box", "color_box", "color_box_label"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-bboxes", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_root = args.image_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    for visual_mode in args.visual_modes:
        for split in args.splits:
            out_jsonl = convert_split(
                SOURCES[split],
                split,
                visual_mode,
                image_root,
                out_root,
                max_bboxes=args.max_bboxes,
                limit=args.limit,
                overwrite=args.overwrite,
            )
            print(f"wrote {out_jsonl}")

    write_manifest(out_root)
    print(f"wrote {out_root / 'task3_visual_experiments.json'}")


if __name__ == "__main__":
    main()
