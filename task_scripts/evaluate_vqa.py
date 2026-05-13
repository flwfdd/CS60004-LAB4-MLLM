"""
VQA 评测脚本：在验证集或测试集上运行 InternVL2-2B 或 Qwen3-VL-8B-Instruct 推理，
并计算 Answer Accuracy（精确匹配）。

支持两种模式：
  no_bbox     仅输入图像 + 问题（Task 1 / Task 2 评测）
  bbox_prompt 将 bbox 坐标文字化后拼入 prompt（Task 3 参考）

运行示例：
  # Task 1/2 验证集评测（InternVL2-2B，微调后的权重）
  python scripts/evaluate_vqa.py \\
      --model internvl2 \\
      --model-path /path/to/your/finetuned_model \\
      --dataset data/task1/val.jsonl \\
      --mode no_bbox \\
      --output results/val_no_bbox.jsonl

  # Task 3 验证集评测（带 bbox 文字 prompt）
  python scripts/evaluate_vqa.py \\
      --model internvl2 \\
      --model-path /path/to/your/finetuned_model \\
      --dataset data/task1/val.jsonl \\
      --mode bbox_prompt \\
      --output results/val_bbox_prompt.jsonl

  # 生成 Leaderboard 提交文件（测试集无答案，只输出预测）
  python scripts/evaluate_vqa.py \\
      --model internvl2 \\
      --model-path /path/to/your/finetuned_model \\
      --dataset data/task1/test.jsonl \\
      --mode no_bbox \\
      --output results/submission.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# 图片根目录：数据包内的 images/ 目录，相对于本脚本所在的 scripts/ 上一级
_PKG_ROOT       = Path(__file__).resolve().parents[1]
GQA_IMAGE_ROOT  = _PKG_ROOT
COCO_IMAGE_ROOT = _PKG_ROOT


# ── I/O helpers ──────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image(image: str) -> Path:
    """将数据文件中的相对路径解析为绝对路径（相对于数据包根目录）。"""
    return _PKG_ROOT / image


# ── 答案归一化与评测 ──────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^(answer|the answer is)\s*[:：]?\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


# ── Prompt 构造 ───────────────────────────────────────────────────────────────

def build_prompt(row: dict[str, Any], mode: str, max_bboxes: int) -> str:
    question = row["question"].strip()
    if mode == "bbox_prompt" and row.get("bboxes"):
        lines = ["Objects in the image:"]
        for i, b in enumerate(row["bboxes"][:max_bboxes], start=1):
            lines.append(f"{i}. {b['category']}: normalized_bbox={b['bbox']}")
        bbox_ctx = "\n".join(lines)
        return f"{bbox_ctx}\n\nQuestion: {question}\nAnswer with the shortest correct answer only."
    return f"Question: {question}\nAnswer with the shortest correct answer only."


# ── 模型封装 ──────────────────────────────────────────────────────────────────

class InternVL2Runner:
    def __init__(self, model_path: str, max_tiles: int = 6):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        self.max_tiles  = max_tiles
        self.image_size = 448
        self.transform  = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True, use_flash_attn=False,
        ).eval().cuda()
        self.model.language_model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

    def _tile(self, image: Image.Image) -> torch.Tensor:
        w, h   = image.size
        ratio  = w / h
        sz     = self.image_size
        targets = sorted(
            {(i, j) for n in range(1, self.max_tiles + 1)
             for i in range(1, n + 1) for j in range(1, n + 1)
             if 1 <= i * j <= self.max_tiles},
            key=lambda x: x[0] * x[1],
        )
        best = min(targets, key=lambda r: abs(ratio - r[0] / r[1]))
        tw, th = sz * best[0], sz * best[1]
        img    = image.resize((tw, th))
        tiles  = []
        cols   = tw // sz
        for k in range(best[0] * best[1]):
            box = ((k % cols) * sz, (k // cols) * sz,
                   (k % cols + 1) * sz, (k // cols + 1) * sz)
            tiles.append(img.crop(box))
        if len(tiles) != 1:
            tiles.append(image.resize((sz, sz)))
        pv = [self.transform(t) for t in tiles]
        return torch.stack(pv).to(torch.bfloat16).cuda()

    def generate(self, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        pixel_values = self._tile(Image.open(image_path).convert("RGB"))
        with torch.inference_mode():
            return self.model.chat(
                self.tokenizer, pixel_values, f"<image>\n{prompt}",
                dict(max_new_tokens=max_new_tokens, do_sample=False),
            ).strip()


class Qwen3VLRunner:
    def __init__(self, model_path: str):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate(self, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text",  "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["internvl2", "qwen3vl"], required=True)
    parser.add_argument("--model-path", required=True,  help="模型本地路径")
    parser.add_argument("--dataset",    type=Path, required=True, help="输入 jsonl 文件")
    parser.add_argument("--mode",       choices=["no_bbox", "bbox_prompt"], required=True)
    parser.add_argument("--output",     type=Path, required=True, help="输出 jsonl 文件")
    parser.add_argument("--limit",          type=int, default=0,  help="调试用：只跑前 N 条")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-bboxes",     type=int, default=12)
    parser.add_argument("--max-tiles",      type=int, default=6,
                        help="InternVL2 动态分辨率最大 tile 数")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    if args.limit > 0:
        rows = rows[:args.limit]

    if args.model == "internvl2":
        runner = InternVL2Runner(args.model_path, args.max_tiles)
    else:
        runner = Qwen3VLRunner(args.model_path)

    results, correct, answered = [], 0, 0
    for idx, row in enumerate(rows, 1):
        prompt     = build_prompt(row, args.mode, args.max_bboxes)
        image_path = resolve_image(row["image"])
        prediction = runner.generate(image_path, prompt, args.max_new_tokens)

        gold       = row.get("answer")
        is_correct = exact_match(prediction, str(gold)) if gold is not None else None
        if is_correct is not None:
            correct  += int(is_correct)
            answered += 1

        results.append({
            "id":         row.get("id"),
            "prediction": prediction,
            "answer":     gold,
            "correct":    is_correct,
        })
        if idx % 50 == 0 or idx == len(rows):
            acc = f"{correct/answered:.3f}" if answered else "n/a"
            print(f"[{idx}/{len(rows)}] accuracy={acc}", flush=True)

    write_jsonl(args.output, results)
    print(f"\nSaved {len(results)} predictions → {args.output}")
    if answered:
        print(f"Answer Accuracy: {correct}/{answered} = {correct/answered:.4f}")


if __name__ == "__main__":
    main()
