"""
Train a Transformer LM on pre-tokenized uint16 data (see tokenize_data.py).

Defaults are the TinyStories config from the handout (~17M non-embedding... see printed count):
vocab 10k, context 256, d_model 512, 4 layers, 16 heads, d_ff 1344, RoPE theta 10000.

    uv run python -m cs336_basics.train \
        --train-data data/tinystories_train.bin --valid-data data/tinystories_valid.bin \
        --out-dir runs/ts_base --device mps --max-iters 5000

Resume with --resume (loads <out-dir>/ckpt_latest.pt).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint


@dataclass
class Config:
    # data / io
    train_data: str = "data/tinystories_train.bin"
    valid_data: str = "data/tinystories_valid.bin"
    out_dir: str = "runs/default"
    resume: bool = False
    # model
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    # ablation switches
    use_rope: bool = True
    use_norm: bool = True
    norm_position: str = "pre"
    ffn_type: str = "swiglu"
    # optimization
    batch_size: int = 64
    max_iters: int = 5000
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    warmup_iters: int = 200
    cosine_iters: int = -1  # -1 => max_iters
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    # logging / eval / checkpoints
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 50
    ckpt_interval: int = 1000
    wandb_project: str = ""  # set to enable W&B
    wandb_run_name: str = ""
    # system
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype: str = "float32"  # autocast dtype: float32 | bfloat16
    compile: bool = False
    seed: int = 0


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    for name, default in asdict(Config()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    return Config(**vars(parser.parse_args()))


def load_tokens(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


@torch.no_grad()
def evaluate(model, data, cfg: Config, autocast) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.batch_size, cfg.context_length, cfg.device)
        with autocast():
            losses.append(cross_entropy(model(x), y).item())
    model.train()
    return float(np.mean(losses))


def main():
    cfg = parse_args()
    if cfg.cosine_iters < 0:
        cfg.cosine_iters = cfg.max_iters
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_data = load_tokens(cfg.train_data)
    valid_data = load_tokens(cfg.valid_data)
    print(f"train tokens: {len(train_data):,}  valid tokens: {len(valid_data):,}")

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        use_rope=cfg.use_rope,
        use_norm=cfg.use_norm,
        norm_position=cfg.norm_position,
        ffn_type=cfg.ffn_type,
        device=cfg.device,
    )
    print(f"params: {model.num_parameters():,} total, {model.num_parameters(non_embedding=True):,} non-embedding")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    start_iter = 0
    latest = out_dir / "ckpt_latest.pt"
    if cfg.resume and latest.exists():
        start_iter = load_checkpoint(latest, model, optimizer)
        print(f"resumed from {latest} at iter {start_iter}")

    train_model = torch.compile(model) if cfg.compile else model

    if cfg.dtype == "bfloat16" and cfg.device != "cpu":
        dev_type = "cuda" if cfg.device.startswith("cuda") else cfg.device

        def autocast():
            return torch.autocast(device_type=dev_type, dtype=torch.bfloat16)
    else:
        import contextlib

        autocast = contextlib.nullcontext

    wandb_run = None
    if cfg.wandb_project:
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name or None, config=asdict(cfg))

    log_f = open(out_dir / "log.jsonl", "a")

    def log(record: dict):
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()
        if wandb_run is not None:
            wandb_run.log(record, step=record["iter"])

    tokens_per_iter = cfg.batch_size * cfg.context_length
    t_start = time.time()
    t_last = t_start
    model.train()
    for it in range(start_iter, cfg.max_iters):
        lr = get_lr_cosine_schedule(it, cfg.lr_max, cfg.lr_min, cfg.warmup_iters, cfg.cosine_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        with autocast():
            loss = cross_entropy(train_model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = gradient_clipping(model.parameters(), cfg.grad_clip)
        optimizer.step()

        step = it + 1
        if step % cfg.log_interval == 0 or step == 1:
            loss_val = loss.item()
            now = time.time()
            n = cfg.log_interval if step > 1 else 1
            tok_s = n * tokens_per_iter / max(now - t_last, 1e-9)
            t_last = now
            rec = dict(
                iter=step,
                train_loss=loss_val,
                lr=lr,
                grad_norm=float(grad_norm),
                tokens=step * tokens_per_iter,
                tok_per_s=tok_s,
                wallclock=now - t_start,
            )
            log(rec)
            print(
                f"it {step:6d} | loss {loss_val:.4f} | lr {lr:.2e} | gnorm {float(grad_norm):.2f} | "
                f"{tok_s / 1e3:.1f}k tok/s | {now - t_start:.0f}s",
                flush=True,
            )
            if not math.isfinite(loss_val):
                print("loss diverged; stopping")
                break

        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            val = evaluate(train_model, valid_data, cfg, autocast)
            log(dict(iter=step, val_loss=val, wallclock=time.time() - t_start))
            print(f"it {step:6d} | val loss {val:.4f}", flush=True)

        if step % cfg.ckpt_interval == 0 or step == cfg.max_iters:
            save_checkpoint(model, optimizer, step, latest)

    log_f.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
