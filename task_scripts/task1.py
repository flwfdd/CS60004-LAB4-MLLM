import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "models" / "InternVL2-2B"))
sys.path.insert(0, str(ROOT / "task_scripts"))

from conversation import get_conv_template
from evaluate_vqa import build_prompt, exact_match, internvl_dynamic_tile_pixel_values
from task0 import count_parameters, set_trainable_modules


def read_jsonl(fn):
    ans = []
    with Path(fn).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ans.append(json.loads(line))
    return ans


def interleave_files(files, max_n):
    pools = [read_jsonl(x) for x in files]
    if not pools:
        raise ValueError("train_jsonl 不能为空")

    k = len(pools)
    if max_n and max_n > 0:
        each, left = divmod(max_n, k)
        need = [each + (i < left) for i in range(k)]
    else:
        m = min(len(x) for x in pools)
        need = [m] * k

    for fn, rows, n in zip(files, pools, need):
        if len(rows) < n:
            raise ValueError(f"{fn}: 只有 {len(rows)} 条，不够取 {n} 条")

    mixed = []
    for i in range(max(need)):
        for rows, n in zip(pools, need):
            if i < n:
                mixed.append(rows[i])
    return mixed


def image_path(name, root):
    return (Path(root) / str(name).lstrip("/")).resolve()


def pad_1d(x, n, val):
    if x.numel() >= n:
        return x[:n]
    return torch.cat([x, torch.full((n - x.numel(),), val, dtype=x.dtype)], 0)


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def sft_tokens(tok, template_name, sys_msg, img_tok_n, patch_n, question, answer):
    conv = get_conv_template(template_name)
    conv.system_message = sys_msg
    conv.append_message(conv.roles[0], "<image>\n" + question.strip())
    conv.append_message(conv.roles[1], None)

    prefix = conv.get_prompt()
    img = "<img>" + "<IMG_CONTEXT>" * (img_tok_n * patch_n) + "</img>"
    prefix = prefix.replace("<image>", img, 1)
    full = prefix + answer.strip() + conv.sep

    a = tok(prefix, return_tensors="pt", add_special_tokens=True)
    b = tok(full, return_tensors="pt", add_special_tokens=True)
    plen = a.input_ids.shape[1]
    if not torch.equal(b.input_ids[:, :plen], a.input_ids):
        raise RuntimeError("tokenizer 把 prefix/full 切坏了")

    labels = b.input_ids.clone()
    labels[:, :plen] = -100
    labels[b.attention_mask == 0] = -100
    return b.input_ids, b.attention_mask, labels


class TrainCollate:
    def __init__(self, tok, model, cfg, image_root):
        self.tok = tok
        self.template = model.template
        self.sys_msg = model.system_message
        self.img_tok_n = model.num_image_token
        self.root = Path(image_root)
        self.cfg = cfg

    def __call__(self, rows):
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        pixels, ids, masks, labels = [], [], [], []

        for r in rows:
            q, a = r.get("question"), r.get("answer")
            if q is None or a is None:
                raise KeyError("训练数据要有 question/answer")

            fn = image_path(r["image"], self.root)
            if not fn.is_file():
                raise FileNotFoundError(f"图像不存在: {fn}")
            with Image.open(fn) as im:
                pv = internvl_dynamic_tile_pixel_values(
                    im.convert("RGB"), max_tiles=self.cfg.max_tiles
                ).to(dtype=torch.bfloat16)

            prompt = build_prompt(r, self.cfg.prompt_mode, max_bboxes=self.cfg.max_bboxes)
            x, m, y = sft_tokens(
                self.tok,
                self.template,
                self.sys_msg,
                self.img_tok_n,
                pv.shape[0],
                prompt,
                str(a),
            )
            pixels.append(pv)
            ids.append(x.squeeze(0))
            masks.append(m.squeeze(0))
            labels.append(y.squeeze(0))

        n = max(x.shape[0] for x in ids)
        pix = torch.cat(pixels, 0)
        return {
            "pixel_values": pix,
            "input_ids": torch.stack([pad_1d(x, n, pad_id) for x in ids]),
            "attention_mask": torch.stack([pad_1d(x, n, 0) for x in masks]),
            "labels": torch.stack([pad_1d(x, n, -100) for x in labels]),
            "image_flags": torch.ones(pix.shape[0], dtype=torch.long),
            "sample_count": len(rows),
        }


class ValCollate:
    def __init__(self, cfg, image_root):
        self.cfg = cfg
        self.root = Path(image_root)

    def __call__(self, rows):
        pixels, patches, questions = [], [], []
        for r in rows:
            fn = image_path(r["image"], self.root)
            if not fn.is_file():
                raise FileNotFoundError(f"图像不存在: {fn}")
            with Image.open(fn) as im:
                pv = internvl_dynamic_tile_pixel_values(
                    im.convert("RGB"), max_tiles=self.cfg.max_tiles
                ).to(dtype=torch.bfloat16)
            pixels.append(pv)
            patches.append(pv.shape[0])
            questions.append(
                build_prompt(r, self.cfg.prompt_mode, max_bboxes=self.cfg.max_bboxes)
            )
        return {
            "rows": rows,
            "pixel_values": torch.cat(pixels, 0),
            "num_patches_list": patches,
            "questions": questions,
            "sample_count": len(rows),
        }


def loader_args(collate, batch, workers, cfg, shuffle=False, drop_last=False):
    kw = dict(
        batch_size=batch,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=max(int(workers), 0),
        collate_fn=collate,
        pin_memory=bool(cfg.pin_memory),
    )
    if kw["num_workers"] > 0:
        kw["prefetch_factor"] = max(int(cfg.prefetch_factor), 1)
        kw["persistent_workers"] = True
    return kw


@torch.inference_mode()
def eval_acc(model, tok, rows, cfg, image_root, device, limit, desc="eval"):
    old_train = model.training
    model.eval()

    rows = rows[:limit]
    good = good_box = good_plain = 0
    n = n_box = n_plain = 0
    gen_cfg = dict(max_new_tokens=cfg.max_new_tokens, do_sample=False)

    bs = max(int(cfg.eval_batch_size), 1)
    dl = DataLoader(
        Rows(rows),
        **loader_args(ValCollate(cfg, image_root), bs, cfg.num_workers, cfg),
    )

    bar = tqdm(total=len(rows), desc=desc, leave=False, dynamic_ncols=True)
    for batch in dl:
        pv = batch["pixel_values"].to(device=device, non_blocking=True)
        out = model.batch_chat(
            tok,
            pv,
            batch["questions"],
            gen_cfg,
            num_patches_list=batch["num_patches_list"],
        )
        out = [x.strip() for x in out]

        for r, pred in zip(batch["rows"], out):
            ok = exact_match(pred, str(r["answer"]))
            has_box = bool(r.get("bboxes"))
            n += 1
            good += int(ok)
            if has_box:
                n_box += 1
                good_box += int(ok)
            else:
                n_plain += 1
                good_plain += int(ok)

        bar.update(int(batch["sample_count"]))
        bar.set_postfix(acc=f"{good / max(n, 1):.3f}")
    bar.close()

    if old_train:
        model.train()

    div = lambda a, b: a / b if b else 0.0
    return {
        "accuracy": div(good, n),
        "accuracy_bbox": div(good_box, n_box),
        "accuracy_no_bbox": div(good_plain, n_plain),
        "n_total": n,
        "n_bbox": n_box,
        "n_no_bbox": n_plain,
    }


def run_one(name, train_bits, cfg, device, image_root, log_wandb=False):
    from transformers import AutoModel, AutoTokenizer
    from transformers import get_constant_schedule, get_cosine_schedule_with_warmup

    torch.cuda.reset_peak_memory_stats(device)

    model = AutoModel.from_pretrained(
        cfg.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
        low_cpu_mem_usage=False,
    )
    tok = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True, use_fast=False)
    model.language_model.config.use_cache = False
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    model = set_trainable_modules(model, **train_bits).to(device)

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if cfg.gradient_checkpointing:
        def gc_on(m):
            try:
                m.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                m.gradient_checkpointing_enable()

        if train_bits.get("train_vision"):
            gc_on(model.vision_model)
        if train_bits.get("train_language"):
            gc_on(model.language_model)
            if hasattr(model.language_model, "enable_input_require_grads"):
                model.language_model.enable_input_require_grads()
    model.train()

    if cfg.batch_size <= 0 or cfg.micro_batch_size <= 0:
        raise ValueError("batch_size/micro_batch_size 必须是正数")
    if cfg.batch_size % cfg.micro_batch_size:
        raise ValueError("batch_size 要能被 micro_batch_size 整除")
    grad_acc = cfg.batch_size // cfg.micro_batch_size

    total_p, train_p, train_ratio = count_parameters(model)
    train_rows = interleave_files(cfg.train_jsonl, cfg.max_train_samples)
    val_rows = read_jsonl(cfg.val_jsonl)

    train_dl = DataLoader(
        Rows(train_rows),
        **loader_args(
            TrainCollate(tok, model, cfg, image_root),
            cfg.micro_batch_size,
            cfg.num_workers,
            cfg,
            shuffle=False,
            drop_last=True,
        ),
    )

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=0.01
    )
    opt.zero_grad(set_to_none=True)

    train_samples_per_epoch = (len(train_rows) // cfg.micro_batch_size) * cfg.micro_batch_size
    total_micro = cfg.epochs * (train_samples_per_epoch // cfg.micro_batch_size)
    total_optim = max(total_micro // grad_acc, 1)
    warmup = int(total_optim * float(cfg.lr_warmup_ratio))

    if cfg.lr_schedule == "constant":
        sched = get_constant_schedule(opt)
    elif cfg.lr_schedule == "warmup_cosine":
        sched = get_cosine_schedule_with_warmup(opt, warmup, total_optim)
    else:
        raise ValueError(f"lr_schedule 不存在: {cfg.lr_schedule}")

    eval_n = len(val_rows) if cfg.eval_val_limit <= 0 else min(cfg.eval_val_limit, len(val_rows))
    final_n = len(val_rows) if cfg.val_limit <= 0 else min(cfg.val_limit, len(val_rows))
    eval_time = 0.0
    best = None
    pbar = tqdm(
        total=cfg.epochs * train_samples_per_epoch,
        desc=f"train[{name}]",
        dynamic_ncols=True,
        leave=True,
    )

    def save_to(where):
        where = Path(where)
        where.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(where)
        tok.save_pretrained(where)
        return where

    def keep_best(m, step, tag):
        nonlocal best
        if best is not None and m["accuracy"] <= best["accuracy"]:
            return
        where = save_to(Path(cfg.output_dir) / f"{name}_best")
        best = {
            "accuracy": float(m["accuracy"]),
            "accuracy_bbox": float(m["accuracy_bbox"]),
            "accuracy_no_bbox": float(m["accuracy_no_bbox"]),
            "n_total": int(m["n_total"]),
            "n_bbox": int(m["n_bbox"]),
            "n_no_bbox": int(m["n_no_bbox"]),
            "optim_step": int(step),
            "tag": tag,
            "checkpoint": str(where),
        }
        pbar.write(f"[best] {tag}/{step}: {m['accuracy']:.4f} -> {where}")

    def run_val(step, sample_n, tag):
        nonlocal eval_time
        t = time.perf_counter()
        m = eval_acc(model, tok, val_rows, cfg, image_root, device, sample_n, f"eval[{tag}@{step}]")
        eval_time += time.perf_counter() - t
        pbar.write(
            f"[{tag} @ {step}] total={m['accuracy']:.4f}({m['n_total']}), "
            f"bbox={m['accuracy_bbox']:.4f}({m['n_bbox']}), "
            f"no_bbox={m['accuracy_no_bbox']:.4f}({m['n_no_bbox']})"
        )
        if log_wandb:
            import wandb
            wandb.log({
                "val/accuracy": m["accuracy"],
                "val/accuracy_bbox": m["accuracy_bbox"],
                "val/accuracy_no_bbox": m["accuracy_no_bbox"],
                "val/samples": m["n_total"],
                "val/samples_bbox": m["n_bbox"],
                "val/samples_no_bbox": m["n_no_bbox"],
            }, step=max(step, 1))
        keep_best(m, step, tag)
        return m

    micro = optim = 0
    last_loss = float("nan")
    if cfg.eval_at_start:
        run_val(0, eval_n, "pre")

    t0 = time.perf_counter()
    for _epoch in range(cfg.epochs):
        for batch in train_dl:
            pv = batch["pixel_values"].to(device=device, non_blocking=True)
            input_ids = batch["input_ids"].to(device=device, non_blocking=True)
            mask = batch["attention_mask"].to(device=device, non_blocking=True)
            labels = batch["labels"].to(device=device, non_blocking=True)
            flags = batch["image_flags"].to(device=device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    pixel_values=pv,
                    input_ids=input_ids,
                    attention_mask=mask,
                    image_flags=flags,
                    labels=labels,
                    use_cache=False,
                )
            loss = out.loss / grad_acc
            loss.backward()
            last_loss = float(loss.detach().float().item() * grad_acc)
            micro += 1

            pbar.update(int(batch["sample_count"]))
            pbar.set_postfix(loss=f"{last_loss:.4f}", optim=optim, lr=f"{opt.param_groups[0]['lr']:.2e}")

            if micro % grad_acc == 0:
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                optim += 1

                if log_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": last_loss,
                        "train/micro_step": micro,
                        "train/lr": opt.param_groups[0]["lr"],
                    }, step=optim)
                if cfg.eval_interval > 0 and optim % cfg.eval_interval == 0:
                    run_val(optim, eval_n, "periodic")
    pbar.close()

    train_time = time.perf_counter() - t0 - eval_time
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    final = eval_acc(model, tok, val_rows, cfg, image_root, device, final_n, "eval[final]")

    out_dir = save_to(Path(cfg.output_dir) / name)
    if best is None:
        keep_best(final, optim, "final")

    final_lr = opt.param_groups[0]["lr"]
    ans = {
        "config": name,
        "answer_accuracy": final["accuracy"],
        "answer_accuracy_bbox": final["accuracy_bbox"],
        "answer_accuracy_no_bbox": final["accuracy_no_bbox"],
        "total_params": total_p,
        "trainable_params": train_p,
        "trainable_ratio_pct": train_ratio,
        "peak_gpu_memory_gb": peak_mem,
        "train_time_sec": train_time,
        "checkpoint": str(out_dir),
        "val_samples": final["n_total"],
        "val_samples_bbox": final["n_bbox"],
        "val_samples_no_bbox": final["n_no_bbox"],
        "batch_size": cfg.batch_size,
        "micro_batch_size": cfg.micro_batch_size,
        "prompt_mode": cfg.prompt_mode,
        "max_bboxes": cfg.max_bboxes,
        "optim_steps": optim,
        "micro_steps": micro,
        "lr_schedule": cfg.lr_schedule,
        "lr_warmup_ratio": cfg.lr_warmup_ratio,
        "final_lr": final_lr,
        "best_checkpoint": None if best is None else best["checkpoint"],
        "best_val_accuracy": None if best is None else best["accuracy"],
        "best_val_accuracy_bbox": None if best is None else best["accuracy_bbox"],
        "best_val_accuracy_no_bbox": None if best is None else best["accuracy_no_bbox"],
        "best_val_samples": None if best is None else best["n_total"],
        "best_at_optim_step": None if best is None else best["optim_step"],
        "best_eval_tag": None if best is None else best["tag"],
    }

    if log_wandb:
        import wandb
        wandb.log({
            "val/accuracy": final["accuracy"],
            "val/accuracy_bbox": final["accuracy_bbox"],
            "val/accuracy_no_bbox": final["accuracy_no_bbox"],
            "val/samples": final["n_total"],
            "val/samples_bbox": final["n_bbox"],
            "val/samples_no_bbox": final["n_no_bbox"],
            "model/trainable_params": train_p,
            "model/trainable_ratio_pct": train_ratio,
            "train/lr": final_lr,
            "sys/peak_gpu_memory_gb": peak_mem,
            "train/wall_time_sec": train_time,
            "train/optim_steps": optim,
            "train/micro_steps": micro,
        }, step=max(optim, 1))

    del model, tok, opt
    torch.cuda.empty_cache()
    return ans


FREEZE = {
    "A_connector_only": dict(train_vision=False, train_connector=True, train_language=False),
    "B_connector_language": dict(train_vision=False, train_connector=True, train_language=True),
    "C_vision_connector": dict(train_vision=True, train_connector=True, train_language=False),
    "D_full": dict(train_vision=True, train_connector=True, train_language=True),
}


def cfg_here():
    return NS(
        # model_path=str(ROOT / "data/models/InternVL2-2B"),
        model_path=str(
            ROOT
            / "outputs/task3/D_full"
            # / "outputs/agent_runs/baseline_bbox_20260517_023836/baseline_bbox_20260517_023836"
        ),
        train_jsonl=[
            # str(ROOT / "data/mm_lab/data/task1/train.jsonl"),
            # str(ROOT / "outputs/task3/hard_train.jsonl"),
            # str(ROOT / "outputs/task1/hard_train.jsonl"),
            # str(ROOT / "outputs/task2/task2_A_text_teacher.jsonl"),
            # str(ROOT / "outputs/task2/task2_B_vision_teacher.jsonl"),
            # str(ROOT / "data/mm_lab/data/val.jsonl"),
            # str(ROOT / "data/mm_lab/data/task3/train_with_bbox.jsonl"),
            # str(ROOT / "outputs/task3/hard_train.jsonl"),
            # str(ROOT / "outputs/task2/task2_vqa_teacher_both_1000.jsonl")
            str(ROOT / "data/mm_lab/data/task3/train_with_bbox_shuffle.jsonl"),
        ],
        val_jsonl=str(ROOT / "data/mm_lab/data/val.jsonl"),
        # val_jsonl=str(ROOT / "outputs/task1/hard_val.jsonl"),
        output_dir=str(ROOT / "outputs/task3"),
        image_root=str(ROOT / "data/mm_lab"),
        freeze_config="D_full",
        epochs=1,
        lr=5e-6,
        batch_size=32,
        micro_batch_size=4,
        max_train_samples=0,
        val_limit=0,  # 训练结束后最终评测条数，0全量
        eval_interval=10,  # 每多少次optim_step阶段性评测，0关闭
        eval_val_limit=0,  # 阶段性评测样本数
        eval_batch_size=8,  # eval 并行样本数
        eval_at_start=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        lr_schedule="warmup_cosine",  # constant, warmup_cosine
        lr_warmup_ratio=0.05,
        gradient_checkpointing=True,
        max_new_tokens=8,
        max_tiles=6,
        prompt_mode="bbox_prompt",  # no_bbox, bbox_prompt, color_bbox_prompt
        max_bboxes=12,
        use_wandb=True,
        wandb_project="cs60004-lab4-mllm",
        wandb_run_prefix="task3-baseonhard13-epoch2",
    )


def main():
    cfg = cfg_here()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("no cuda")

    name = cfg.freeze_config
    if name not in FREEZE:
        raise ValueError(f"freeze_config 写错了: {name}; 可选 {list(FREEZE)}")
    bits = FREEZE[name]
    image_root = Path(cfg.image_root).resolve()

    if cfg.use_wandb:
        import wandb
        run_prefix = (cfg.wandb_run_prefix or "task1").strip() or "task1"
        wandb.init(
            project=cfg.wandb_project,
            name=f"{run_prefix}_{name}",
            config={
                "freeze_config": name,
                **bits,
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
                "num_workers": cfg.num_workers,
                "prefetch_factor": cfg.prefetch_factor,
                "pin_memory": cfg.pin_memory,
                "lr_schedule": cfg.lr_schedule,
                "lr_warmup_ratio": cfg.lr_warmup_ratio,
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "max_new_tokens": cfg.max_new_tokens,
                "max_tiles": cfg.max_tiles,
                "prompt_mode": cfg.prompt_mode,
                "max_bboxes": cfg.max_bboxes,
                "model_path": cfg.model_path,
                "train_jsonl": list(cfg.train_jsonl),
                "val_jsonl": cfg.val_jsonl,
                "image_root": cfg.image_root,
            },
        )

    summ = None
    try:
        summ = run_one(name, bits, cfg, device, image_root, log_wandb=cfg.use_wandb)
    finally:
        if summ is not None:
            print(
                "Answer Accuracy (val):\n"
                f"  total   = {summ['answer_accuracy']:.4f} (n={summ['val_samples']})\n"
                f"  bbox    = {summ['answer_accuracy_bbox']:.4f} (n={summ['val_samples_bbox']})\n"
                f"  no_bbox = {summ['answer_accuracy_no_bbox']:.4f} (n={summ['val_samples_no_bbox']})\n"
                f"Trainable: {summ['trainable_params']:,} / {summ['total_params']:,} "
                f"({summ['trainable_ratio_pct']:.4f}%)\n"
                f"Peak GPU memory: {summ['peak_gpu_memory_gb']:.2f} GiB\n"
                f"Train wall time: {summ['train_time_sec']:.1f} s\n"
                f"Saved: {summ['checkpoint']}"
            )
            out = Path(cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            fn = out / "task1_metrics.json"
            fn.write_text(json.dumps([summ], indent=2), encoding="utf-8")
            print(f"\nWrote {fn}")
        if cfg.use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
