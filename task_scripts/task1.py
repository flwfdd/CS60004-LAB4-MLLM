"""
Task 1：InternVL2-2B 在 GQA VQA 上按一种冻结策略做监督微调
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "models" / "InternVL2-2B"))
sys.path.insert(0, str(ROOT / "task_scripts"))

from conversation import get_conv_template
from evaluate_vqa import build_prompt, exact_match, internvl_dynamic_tile_pixel_values
from task0 import count_parameters, set_trainable_modules


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_image(rel: str, image_root: Path) -> Path:
    return (image_root / rel.lstrip("/")).resolve()


def _pad_to(t: torch.Tensor, length: int, value: int) -> torch.Tensor:
    if t.shape[0] >= length:
        return t[:length]
    pad = torch.full((length - t.shape[0],), value, dtype=t.dtype)
    return torch.cat([t, pad], dim=0)


def collate_micro_batch(
    rows: list[dict],
    tokenizer,
    model,
    image_root: Path,
    *,
    max_tiles: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """把若干条样本拼成一个 micro-batch；右侧填充 input_ids / attention_mask / labels。"""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    pv_list, ids_list, mask_list, lbl_list = [], [], [], []
    for row in rows:
        path = resolve_image(row["image"], image_root)
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到图像：{path}\n请检查 cfg.image_root 与 jsonl 中的相对路径。"
            )
        pv = internvl_dynamic_tile_pixel_values(
            Image.open(path).convert("RGB"), max_tiles=max_tiles
        ).to(device=device, dtype=torch.bfloat16)
        ids, mask, lbl = tokenize_sft(
            tokenizer,
            model,
            num_patches=pv.shape[0],
            question=row["question"],
            answer=str(row["answer"]),
        )
        pv_list.append(pv)
        ids_list.append(ids.squeeze(0))
        mask_list.append(mask.squeeze(0))
        lbl_list.append(lbl.squeeze(0))

    max_len = max(t.shape[0] for t in ids_list)
    input_ids = torch.stack([_pad_to(t, max_len, pad_id) for t in ids_list]).to(device)
    attention_mask = torch.stack([_pad_to(t, max_len, 0) for t in mask_list]).to(device)
    labels = torch.stack([_pad_to(t, max_len, -100) for t in lbl_list]).to(device)
    pixel_values = torch.cat(pv_list, dim=0)
    image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=device)
    return pixel_values, input_ids, attention_mask, labels, image_flags


def tokenize_sft(
    tokenizer, model, num_patches: int, question: str, answer: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vqa = (
        f"Question: {question.strip()}\n"
        "Answer with the shortest correct answer only."
    )
    user_msg = f"<image>\n{vqa}"
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], user_msg)
    template.append_message(template.roles[1], None)
    prefix = template.get_prompt()
    img_tokens = (
        "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
    )
    prefix = prefix.replace("<image>", img_tokens, 1)
    suffix = answer.strip() + template.sep
    full = prefix + suffix

    pref = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
    full_t = tokenizer(full, return_tensors="pt", add_special_tokens=True)
    input_ids = full_t.input_ids
    attention_mask = full_t.attention_mask
    plen = pref.input_ids.shape[1]
    if not torch.equal(input_ids[:, :plen], pref.input_ids):
        raise ValueError("SFT 前缀与整句 token 边界不一致，请检查模板与分词器。")
    labels = input_ids.clone()
    labels[:, :plen] = -100
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


@torch.inference_mode()
def val_accuracy(
    model,
    tokenizer,
    rows: list[dict],
    image_root: Path,
    device: torch.device,
    *,
    max_tiles: int,
    limit: int,
    max_new_tokens: int,
    batch_size: int = 1,
    desc: str = "eval",
) -> dict:
    """返回 dict: total / bbox / no_bbox 三档准确率以及样本数。

    batch_size>1 时走 InternVL2 自带的 batch_chat，一次性把多个样本的 tile
    在 ViT 上拼接做特征提取，并在 LLM 上做左 pad 的并行生成。
    """
    was_training = model.training
    model.eval()
    correct_total = correct_bbox = correct_no = 0
    n_total = n_bbox = n_no = 0
    subset = rows[:limit]
    gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
    bs = max(int(batch_size), 1)

    bar = tqdm(total=len(subset), desc=desc, leave=False, dynamic_ncols=True)
    for start in range(0, len(subset), bs):
        chunk = subset[start : start + bs]

        pv_list, num_patches_list, questions = [], [], []
        for row in chunk:
            path = resolve_image(row["image"], image_root)
            pv = internvl_dynamic_tile_pixel_values(
                Image.open(path).convert("RGB"), max_tiles=max_tiles
            ).to(device=device, dtype=torch.bfloat16)
            pv_list.append(pv)
            num_patches_list.append(pv.shape[0])
            questions.append(build_prompt(row, "no_bbox", max_bboxes=12))

        pixel_values = torch.cat(pv_list, dim=0)
        preds = [
            p.strip()
            for p in model.batch_chat(
                tokenizer,
                pixel_values,
                questions,
                gen_cfg,
                num_patches_list=num_patches_list,
            )
        ]

        for row, pred in zip(chunk, preds):
            ok = exact_match(pred, str(row["answer"]))
            has_bbox = bool(row.get("bboxes"))
            n_total += 1
            correct_total += int(ok)
            if has_bbox:
                n_bbox += 1
                correct_bbox += int(ok)
            else:
                n_no += 1
                correct_no += int(ok)

        bar.update(len(chunk))
        bar.set_postfix(acc=f"{correct_total / n_total:.3f}")
    bar.close()
    if was_training:
        model.train()

    def _ratio(c: int, n: int) -> float:
        return c / n if n else 0.0

    return {
        "accuracy": _ratio(correct_total, n_total),
        "accuracy_bbox": _ratio(correct_bbox, n_bbox),
        "accuracy_no_bbox": _ratio(correct_no, n_no),
        "n_total": n_total,
        "n_bbox": n_bbox,
        "n_no_bbox": n_no,
    }


def train_one_config(
    cfg_name: str,
    train_cfg: dict,
    cfg: SimpleNamespace,
    device: torch.device,
    image_root: Path,
    *,
    log_wandb: bool = False,
) -> dict:
    from transformers import AutoModel, AutoTokenizer

    torch.cuda.reset_peak_memory_stats(device)

    model = AutoModel.from_pretrained(
        cfg.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
        low_cpu_mem_usage=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True, use_fast=False
    )
    model.language_model.config.use_cache = False
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model = set_trainable_modules(model, **train_cfg).to(device)
    # 默认先全部关掉，再按 train_cfg 选择性开启，避免给被冻结的子模块上无意义的 GC
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if cfg.gradient_checkpointing:
        # 用 HF 官方的递归 enable 才能传到 InternVisionEncoder / InternLM2Model 这种内部子模块
        def _enable_gc(submodule):
            try:
                submodule.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                submodule.gradient_checkpointing_enable()

        if train_cfg.get("train_vision", False):
            _enable_gc(model.vision_model)
        if train_cfg.get("train_language", False):
            _enable_gc(model.language_model)
            # GC 要求至少一路输入张量 requires_grad，否则 backward 不会重算激活
            if hasattr(model.language_model, "enable_input_require_grads"):
                model.language_model.enable_input_require_grads()
    model.train()

    if cfg.batch_size <= 0 or cfg.micro_batch_size <= 0:
        raise ValueError("batch_size 与 micro_batch_size 必须为正整数。")
    if cfg.batch_size % cfg.micro_batch_size != 0:
        raise ValueError(
            f"batch_size({cfg.batch_size}) 必须是 micro_batch_size({cfg.micro_batch_size}) 的整数倍。"
        )
    accumulation_steps = cfg.batch_size // cfg.micro_batch_size

    total, trainable, ratio = count_parameters(model)
    train_rows = load_jsonl(Path(cfg.train_jsonl))
    if cfg.max_train_samples > 0:
        train_rows = train_rows[: cfg.max_train_samples]

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=0.01
    )
    opt.zero_grad(set_to_none=True)

    val_rows = load_jsonl(Path(cfg.val_jsonl))
    final_val_n = (
        len(val_rows) if cfg.val_limit <= 0 else min(cfg.val_limit, len(val_rows))
    )
    eval_val_n = (
        len(val_rows)
        if cfg.eval_val_limit <= 0
        else min(cfg.eval_val_limit, len(val_rows))
    )

    eval_time_total = 0.0

    def _run_eval(at_optim_step: int, n_samples: int, tag: str) -> dict:
        nonlocal eval_time_total
        t_eval = time.perf_counter()
        m = val_accuracy(
            model,
            tokenizer,
            val_rows,
            image_root,
            device,
            max_tiles=cfg.max_tiles,
            limit=n_samples,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=cfg.eval_batch_size,
            desc=f"eval[{tag}@{at_optim_step}]",
        )
        eval_time_total += time.perf_counter() - t_eval
        msg = (
            f"[{tag} @ optim_step={at_optim_step}] "
            f"total={m['accuracy']:.4f} (n={m['n_total']}), "
            f"bbox={m['accuracy_bbox']:.4f} (n={m['n_bbox']}), "
            f"no_bbox={m['accuracy_no_bbox']:.4f} (n={m['n_no_bbox']})"
        )
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)
        if log_wandb:
            import wandb

            wandb.log(
                {
                    "val/accuracy": m["accuracy"],
                    "val/accuracy_bbox": m["accuracy_bbox"],
                    "val/accuracy_no_bbox": m["accuracy_no_bbox"],
                    "val/samples": m["n_total"],
                    "val/samples_bbox": m["n_bbox"],
                    "val/samples_no_bbox": m["n_no_bbox"],
                },
                step=max(at_optim_step, 1),
            )
        return m

    total_micro = cfg.epochs * (len(train_rows) // cfg.micro_batch_size)
    pbar = tqdm(
        total=total_micro,
        desc=f"train[{cfg_name}]",
        dynamic_ncols=True,
        leave=True,
    )

    t0 = time.perf_counter()
    micro_step, optim_step = 0, 0
    last_loss = float("nan")
    last_val: dict | None = None

    if cfg.eval_at_start:
        last_val = _run_eval(0, eval_val_n, "pre-train")

    for _ in range(cfg.epochs):
        for i in range(0, len(train_rows), cfg.micro_batch_size):
            chunk = train_rows[i : i + cfg.micro_batch_size]
            if len(chunk) < cfg.micro_batch_size:
                break  # drop_last，保持梯度累积窗口完整

            pv, input_ids, attention_mask, labels, flags = collate_micro_batch(
                chunk,
                tokenizer,
                model,
                image_root,
                max_tiles=cfg.max_tiles,
                device=device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    pixel_values=pv,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=flags,
                    labels=labels,
                    use_cache=False,
                )
            loss = out.loss / accumulation_steps
            loss.backward()
            last_loss = loss.detach().float().item() * accumulation_steps
            micro_step += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{last_loss:.4f}",
                optim=optim_step,
                val=f"{last_val['accuracy']:.3f}" if last_val is not None else "n/a",
            )

            if micro_step % accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                optim_step += 1
                if log_wandb:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": last_loss,
                            "train/micro_step": micro_step,
                        },
                        step=optim_step,
                    )
                if cfg.eval_interval > 0 and optim_step % cfg.eval_interval == 0:
                    last_val = _run_eval(optim_step, eval_val_n, "periodic")

    pbar.close()

    train_time = time.perf_counter() - t0 - eval_time_total
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    val_n = final_val_n
    final_metrics = val_accuracy(
        model,
        tokenizer,
        val_rows,
        image_root,
        device,
        max_tiles=cfg.max_tiles,
        limit=val_n,
        max_new_tokens=cfg.max_new_tokens,
        batch_size=cfg.eval_batch_size,
        desc="eval[final]",
    )

    out_dir = Path(cfg.output_dir) / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    del model, tokenizer, opt
    torch.cuda.empty_cache()

    summary = {
        "config": cfg_name,
        "answer_accuracy": final_metrics["accuracy"],
        "answer_accuracy_bbox": final_metrics["accuracy_bbox"],
        "answer_accuracy_no_bbox": final_metrics["accuracy_no_bbox"],
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio_pct": ratio,
        "peak_gpu_memory_gb": peak_mem_gb,
        "train_time_sec": train_time,
        "checkpoint": str(out_dir),
        "val_samples": final_metrics["n_total"],
        "val_samples_bbox": final_metrics["n_bbox"],
        "val_samples_no_bbox": final_metrics["n_no_bbox"],
        "batch_size": cfg.batch_size,
        "micro_batch_size": cfg.micro_batch_size,
        "optim_steps": optim_step,
        "micro_steps": micro_step,
    }
    if log_wandb:
        import wandb

        wandb.log(
            {
                "val/accuracy": final_metrics["accuracy"],
                "val/accuracy_bbox": final_metrics["accuracy_bbox"],
                "val/accuracy_no_bbox": final_metrics["accuracy_no_bbox"],
                "val/samples": final_metrics["n_total"],
                "val/samples_bbox": final_metrics["n_bbox"],
                "val/samples_no_bbox": final_metrics["n_no_bbox"],
                "model/trainable_params": trainable,
                "model/trainable_ratio_pct": ratio,
                "sys/peak_gpu_memory_gb": peak_mem_gb,
                "train/wall_time_sec": train_time,
                "train/optim_steps": optim_step,
                "train/micro_steps": micro_step,
            },
            step=max(optim_step, 1),
        )
    return summary


FREEZE_CONFIGS = {
    "A_connector_only": dict(
        train_vision=False, train_connector=True, train_language=False
    ),
    "B_connector_language": dict(
        train_vision=False, train_connector=True, train_language=True
    ),
    "C_vision_connector": dict(
        train_vision=True, train_connector=True, train_language=False
    ),
    "D_full": dict(train_vision=True, train_connector=True, train_language=True),
}


def main() -> None:
    cfg = SimpleNamespace(
        model_path=str(ROOT / "data/models/InternVL2-2B"),
        # train_jsonl=str(ROOT / "data/mm_lab/data/task1/train.jsonl"),
        train_jsonl=str(ROOT / "outputs/task1/hard_train.jsonl"),
        val_jsonl=str(ROOT / "data/mm_lab/data/val.jsonl"),
        output_dir=str(ROOT / "outputs/task1"),
        image_root=str(ROOT / "data/mm_lab"),
        freeze_config="D_full",
        epochs=1,
        lr=1e-5,
        batch_size=64,
        micro_batch_size=1,
        max_train_samples=512,
        val_limit=0,  # 训练结束后最终评测条数，0全量
        eval_interval=0,  # 每多少次optim_step阶段性评测，0关闭
        eval_val_limit=100,  # 阶段性评测样本数
        eval_batch_size=8,  # eval 并行样本数（InternVL2 batch_chat）
        eval_at_start=False,
        gradient_checkpointing=True,
        max_new_tokens=16,
        max_tiles=6,
        use_wandb=True,
        wandb_project="cs60004-lab4-mllm",
        wandb_run_prefix="task1-hard",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("no cuda")

    name = cfg.freeze_config
    if name not in FREEZE_CONFIGS:
        raise ValueError(
            f"未知 freeze_config={name!r}，请填 {list(FREEZE_CONFIGS)!r} 之一。"
        )
    train_cfg = FREEZE_CONFIGS[name]
    image_root = Path(cfg.image_root).resolve()

    print(f"\n========== {name} ==========")
    if cfg.use_wandb:
        import wandb

        prefix = (cfg.wandb_run_prefix or "task1").strip() or "task1"
        wandb.init(
            project=cfg.wandb_project,
            name=f"{prefix}_{name}",
            config={
                "freeze_config": name,
                **train_cfg,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "micro_batch_size": cfg.micro_batch_size,
                "max_train_samples": cfg.max_train_samples,
                "val_limit": cfg.val_limit,
                "eval_interval": cfg.eval_interval,
                "eval_val_limit": cfg.eval_val_limit,
                "eval_batch_size": cfg.eval_batch_size,
                "eval_at_start": cfg.eval_at_start,
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "max_new_tokens": cfg.max_new_tokens,
                "max_tiles": cfg.max_tiles,
                "model_path": cfg.model_path,
                "train_jsonl": cfg.train_jsonl,
                "val_jsonl": cfg.val_jsonl,
                "image_root": cfg.image_root,
            },
        )
    try:
        summ = train_one_config(
            name, train_cfg, cfg, device, image_root, log_wandb=cfg.use_wandb
        )
    finally:
        print(
            f"Answer Accuracy (val):\n"
            f"  total   = {summ['answer_accuracy']:.4f} (n={summ['val_samples']})\n"
            f"  bbox    = {summ['answer_accuracy_bbox']:.4f} (n={summ['val_samples_bbox']})\n"
            f"  no_bbox = {summ['answer_accuracy_no_bbox']:.4f} (n={summ['val_samples_no_bbox']})\n"
            f"Trainable: {summ['trainable_params']:,} / {summ['total_params']:,} "
            f"({summ['trainable_ratio_pct']:.4f}%)\n"
            f"Peak GPU memory: {summ['peak_gpu_memory_gb']:.2f} GiB\n"
            f"Train wall time: {summ['train_time_sec']:.1f} s\n"
            f"Saved: {summ['checkpoint']}"
        )
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = Path(cfg.output_dir) / "task1_metrics.json"
        metrics_path.write_text(json.dumps([summ], indent=2), encoding="utf-8")
        print(f"\nWrote {metrics_path}")
        if cfg.use_wandb:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
