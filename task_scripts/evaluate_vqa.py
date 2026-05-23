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
from collections import Counter
from pathlib import Path
from typing import Any, Callable, cast

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.auto import tqdm

_PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = _PKG_ROOT / "data/mm_lab"


# ── I/O helpers ──────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image(image: str, root: Path) -> Path:
    p = (root / image.lstrip("/")).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"找不到图像: {p}")
    return p


# ── 答案归一化与评测 ──────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^(answer|the answer is)\s*[:：]?\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def extract_prediction(text: str, mode: str) -> str:
    text = text.strip()
    if mode != "cot":
        return text

    matches = re.findall(r"Answer\s*[:：]\s*(.+)", text, flags=re.I)
    if matches:
        return matches[-1].splitlines()[0].strip().strip("\"'`.,。")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].strip("\"'`.,。") if lines else text


def vote_predictions(raw_predictions: list[str], mode: str) -> tuple[str, list[str]]:
    predictions = [extract_prediction(pred, mode) for pred in raw_predictions]
    counts = Counter(normalize(pred) for pred in predictions)
    best_norm, _ = counts.most_common(1)[0]
    for pred in predictions:
        if normalize(pred) == best_norm:
            return pred, predictions
    return predictions[0], predictions


# ── Prompt 构造 ───────────────────────────────────────────────────────────────


def build_prompt(row: dict[str, Any], mode: str, max_bboxes: int) -> str:
    format_prompt = "The answer must be a simple short answer, usually one word, a short phrase, yes/no, or a number."
    question = row["question"].strip()
    if mode == "cot":
        prefix = ""
        if row.get("bboxes"):
            lines = ["Objects in the image:"]
            for i, b in enumerate(row["bboxes"][:max_bboxes], start=1):
                lines.append(f"{i}. {b['category']}: normalized_bbox={b['bbox']}")
            prefix = "\n".join(lines) + "\n\n"
        return (
            f"{prefix}Question: {question}\n"
            "Answer the question by reasoning briefly from the image and the "
            "listed object bounding boxes (x_left_top, y_left_top, x_right_down, "
            "y_right_down). Use the boxes only as localization clues; do not "
            "invent new boxes. When an object is important to your reasoning path, "
            "annotate it with its given bounding box. Do not mention the prompt or "
            f"say that the object information was provided. {format_prompt}\n"
            "Your response should follow this format:\n"
            "Thought: <brief reasoning process>\n"
            "Answer: <short answer>"
        )
    if mode == "bbox_prompt" and row.get("bboxes"):
        lines = ["Objects in the image:"]
        for i, b in enumerate(row["bboxes"][:max_bboxes], start=1):
            lines.append(f"{i}. {b['category']}: normalized_bbox={b['bbox']}")
        bbox_ctx = "\n".join(lines)
        return f"{bbox_ctx}\n\nQuestion: {question}\nAnswer with the shortest correct answer only. {format_prompt}"
    if mode == "color_prompt" and row.get("bboxes"):
        lines = ["Objects in the image:"]
        for i, b in enumerate(row["bboxes"][:max_bboxes], start=1):
            color = b.get("color", f"box {i}")
            lines.append(f"{i}. {color} box: {b['category']}")
        bbox_ctx = "\n".join(lines)
        return f"{bbox_ctx}\n\nQuestion: {question}\nAnswer with the shortest correct answer only. {format_prompt}"
    if mode == "color_bbox_prompt" and row.get("bboxes"):
        lines = ["Objects in the image:"]
        for i, b in enumerate(row["bboxes"][:max_bboxes], start=1):
            color = b.get("color", f"box {i}")
            lines.append(
                f"{i}. {color} box: {b['category']}, normalized_bbox={b['bbox']}"
            )
        bbox_ctx = "\n".join(lines)
        return f"{bbox_ctx}\n\nQuestion: {question}\nAnswer with the shortest correct answer only. {format_prompt}"
    return f"Question: {question}\nAnswer with the shortest correct answer only. {format_prompt}"


# ── InternVL2 动态分辨率 tile ────────

_TILE_TRANSFORM: dict[int, object] = {}


def _internvl_tile_transform(image_size: int = 448):
    if image_size not in _TILE_TRANSFORM:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        _TILE_TRANSFORM[image_size] = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return _TILE_TRANSFORM[image_size]


def internvl_dynamic_tile_pixel_values(
    image: Image.Image,
    *,
    max_tiles: int = 6,
    image_size: int = 448,
) -> torch.Tensor:
    transform = cast(
        Callable[[Image.Image], torch.Tensor], _internvl_tile_transform(image_size)
    )
    w, h = image.size
    ratio = w / h
    sz = image_size
    targets = sorted(
        {
            (i, j)
            for n in range(1, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_tiles
        },
        key=lambda x: x[0] * x[1],
    )
    best = min(targets, key=lambda r: abs(ratio - r[0] / r[1]))
    tw, th = sz * best[0], sz * best[1]
    img = image.resize((tw, th))
    tiles: list[Image.Image] = []
    cols = tw // sz
    for k in range(best[0] * best[1]):
        box = (
            (k % cols) * sz,
            (k // cols) * sz,
            (k % cols + 1) * sz,
            (k // cols + 1) * sz,
        )
        tiles.append(img.crop(box))
    if len(tiles) != 1:
        tiles.append(image.resize((sz, sz)))
    return torch.stack([transform(t) for t in tiles])


def internvl_tile_image_file(
    image_path: Path,
    *,
    max_tiles: int,
    image_size: int = 448,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return internvl_dynamic_tile_pixel_values(
        image,
        max_tiles=max_tiles,
        image_size=image_size,
    ).to(dtype=torch.bfloat16)


class InternVL2EvalDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        image_root: Path,
        mode: str,
        max_bboxes: int,
        max_tiles: int,
        image_size: int = 448,
    ):
        self.rows = rows
        self.image_root = image_root
        self.mode = mode
        self.max_bboxes = max_bboxes
        self.max_tiles = max_tiles
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = resolve_image(row["image"], self.image_root)
        pixel_values = internvl_tile_image_file(
            image_path,
            max_tiles=self.max_tiles,
            image_size=self.image_size,
        )
        return {
            "row": row,
            "prompt": build_prompt(row, self.mode, self.max_bboxes),
            "pixel_values": pixel_values,
            "num_patches": pixel_values.shape[0],
        }


def collate_internvl2_eval_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": [item["row"] for item in items],
        "prompts": [item["prompt"] for item in items],
        "pixel_values": torch.cat([item["pixel_values"] for item in items], dim=0),
        "num_patches_list": [item["num_patches"] for item in items],
    }


# ── 模型封装 ──────────────────────────────────────────────────────────────────


class InternVL2Runner:
    def __init__(self, model_path: str, max_tiles: int = 6):
        from transformers import AutoModel, AutoTokenizer

        self.max_tiles = max_tiles
        self.image_size = 448
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
                use_flash_attn=True,
            )
            .eval()
            .cuda()
        )
        self.model.language_model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

    def _tile(self, image: Image.Image) -> torch.Tensor:
        return (
            internvl_dynamic_tile_pixel_values(
                image, max_tiles=self.max_tiles, image_size=self.image_size
            )
            .to(torch.bfloat16)
            .cuda()
        )

    def generate_preprocessed_batch(
        self,
        pixel_values: torch.Tensor,
        num_patches_list: list[int],
        prompts: list[str],
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> list[str]:
        pixel_values = pixel_values.cuda(non_blocking=True)
        gen_cfg: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens, do_sample=do_sample
        )
        if do_sample:
            gen_cfg["temperature"] = temperature
        with torch.inference_mode():
            preds = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                prompts,
                gen_cfg,
                num_patches_list=num_patches_list,
            )
        return [p.strip() for p in preds]

    def generate_batch(
        self,
        image_paths: list[Path],
        prompts: list[str],
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> list[str]:
        pv_list, num_patches_list = [], []
        for p in image_paths:
            pv = self._tile(Image.open(p).convert("RGB"))
            pv_list.append(pv)
            num_patches_list.append(pv.shape[0])
        pixel_values = torch.cat(pv_list, dim=0)
        return self.generate_preprocessed_batch(
            pixel_values,
            num_patches_list,
            prompts,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    def generate(
        self,
        image_path: Path,
        prompt: str,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> str:
        return self.generate_batch(
            [image_path],
            [prompt],
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )[0]


class Qwen3VLRunner:
    def __init__(self, model_path: str):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate(
        self,
        image_path: Path,
        prompt: str,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        gen_cfg: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens, do_sample=do_sample
        )
        if do_sample:
            gen_cfg["temperature"] = temperature
        with torch.inference_mode():
            ids = self.model.generate(**inputs, **gen_cfg)
        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, ids)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


# ── 主流程 ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["internvl2", "qwen3vl"], required=True)
    parser.add_argument("--model-path", required=True, help="模型本地路径")
    parser.add_argument("--dataset", type=Path, required=True, help="输入 jsonl 文件")
    parser.add_argument(
        "--mode",
        choices=["no_bbox", "bbox_prompt", "color_prompt", "color_bbox_prompt", "cot"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True, help="输出 jsonl 文件")
    parser.add_argument("--limit", type=int, default=0, help="调试用：只跑前 N 条")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-bboxes", type=int, default=12)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--repeat-do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--max-tiles", type=int, default=6, help="InternVL2 动态分辨率最大 tile 数"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="InternVL2 评测并行样本数（batch_chat），Qwen3-VL 暂时仅支持 1",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="InternVL2 图片预处理 DataLoader worker 数；设为 0 则关闭并行预处理",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="每个 worker 预取 batch 数，仅在 --num-workers > 0 时生效",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="关闭 DataLoader pin_memory；默认开启以加速 CPU 到 GPU 拷贝",
    )
    parser.set_defaults(pin_memory=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="含 images/ 的目录，默认 <repo>/data/mm_lab",
    )
    args = parser.parse_args()

    image_root = args.image_root.expanduser().resolve()

    rows = read_jsonl(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.num_repeats < 1:
        raise ValueError("--num-repeats must be >= 1")

    results = []
    correct_total = correct_bbox = correct_no = 0
    n_total = n_bbox = n_no = 0

    def _record_batch(
        chunk: list[dict[str, Any]],
        raw_by_sample: list[list[str]],
    ) -> None:
        nonlocal correct_total, correct_bbox, correct_no, n_total, n_bbox, n_no
        for row, raw_predictions in zip(chunk, raw_by_sample):
            prediction, prediction_votes = vote_predictions(raw_predictions, args.mode)
            gold = row.get("answer")
            is_correct = (
                exact_match(prediction, str(gold)) if gold is not None else None
            )
            has_bbox = bool(row.get("bboxes"))
            if is_correct is not None:
                n_total += 1
                correct_total += int(is_correct)
                if has_bbox:
                    n_bbox += 1
                    correct_bbox += int(is_correct)
                else:
                    n_no += 1
                    correct_no += int(is_correct)

            results.append(
                {
                    **row,
                    "prediction": prediction,
                    "raw_prediction": raw_predictions[0],
                    "raw_predictions": raw_predictions,
                    "prediction_votes": prediction_votes,
                    "correct": is_correct,
                    "has_bbox": has_bbox,
                }
            )

    def _update_progress(chunk_size: int) -> None:
        pbar.update(chunk_size)
        pbar.set_postfix(
            total=f"{correct_total/n_total:.3f}" if n_total else "n/a",
            bbox=f"{correct_bbox/n_bbox:.3f}" if n_bbox else "n/a",
            no_bbox=f"{correct_no/n_no:.3f}" if n_no else "n/a",
            refresh=False,
        )

    pbar = tqdm(
        total=len(rows),
        desc=f"eval",
        dynamic_ncols=True,
    )
    if args.model == "internvl2":
        runner = InternVL2Runner(args.model_path, args.max_tiles)
        bs = max(int(args.eval_batch_size), 1)
        num_workers = max(int(args.num_workers), 0)
        loader_kwargs: dict[str, Any] = dict(
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_internvl2_eval_batch,
            pin_memory=bool(args.pin_memory),
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = max(int(args.prefetch_factor), 1)
            loader_kwargs["persistent_workers"] = True
        dataset = InternVL2EvalDataset(
            rows,
            image_root=image_root,
            mode=args.mode,
            max_bboxes=args.max_bboxes,
            max_tiles=args.max_tiles,
            image_size=runner.image_size,
        )
        loader = DataLoader(dataset, **loader_kwargs)
        for batch in loader:
            chunk = batch["rows"]
            raw_by_sample = [[] for _ in chunk]
            for _ in range(args.num_repeats):
                preds = runner.generate_preprocessed_batch(
                    batch["pixel_values"],
                    batch["num_patches_list"],
                    batch["prompts"],
                    args.max_new_tokens,
                    do_sample=args.repeat_do_sample,
                    temperature=args.temperature,
                )
                for raw_list, pred in zip(raw_by_sample, preds):
                    raw_list.append(pred)
            _record_batch(chunk, raw_by_sample)
            _update_progress(len(chunk))
    else:
        runner = Qwen3VLRunner(args.model_path)
        for row in rows:
            prompt = build_prompt(row, args.mode, args.max_bboxes)
            path = resolve_image(row["image"], image_root)
            raw_predictions = [
                runner.generate(
                    path,
                    prompt,
                    args.max_new_tokens,
                    do_sample=args.repeat_do_sample,
                    temperature=args.temperature,
                )
                for _ in range(args.num_repeats)
            ]
            _record_batch([row], [raw_predictions])
            _update_progress(1)
    pbar.close()

    write_jsonl(args.output, results)
    print(f"\nSaved {len(results)} predictions → {args.output}")
    if n_total:

        def _line(name: str, c: int, n: int) -> str:
            return f"  {name:8s}: {c}/{n} = {c / n:.4f}" if n else f"  {name:8s}: n/a"

        print("Answer Accuracy:")
        print(_line("total", correct_total, n_total))
        print(_line("bbox", correct_bbox, n_bbox))
        print(_line("no_bbox", correct_no, n_no))


if __name__ == "__main__":
    main()
