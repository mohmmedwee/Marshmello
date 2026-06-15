"""
GPT pretraining loop: next-token cross-entropy, checkpoints, MPS support.

Run from project root:
  python 13_gpt_pretraining/training/trainer.py
  python 13_gpt_pretraining/training/trainer.py --quick
  python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 300
  python 13_gpt_pretraining/training/trainer.py --resume checkpoints/large_50m/latest.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))
sys.path.insert(0, str(PROJECT_ROOT / "11_scale_model"))

from bpe_demo import BPETokenizer  # noqa: E402
from estimate_params import estimate_for_gpt_model, gpt_parameter_breakdown  # noqa: E402

from config import (  # noqa: E402
    CORPUS_PATH,
    DEFAULT_CONFIG,
    GPTConfig,
    checkpoint_dir_for,
    latest_checkpoint_for,
    resolve_config,
)
from data.prepare_corpus import prepare_corpus  # noqa: E402
from model.gpt import GPT, count_parameters  # noqa: E402
from tokenizer.bpe_io import load_tokenizer  # noqa: E402
from tokenizer.encode import corpus_to_ids_safe  # noqa: E402
from tokenizer.train_bpe import train_bpe  # noqa: E402
from training.dataset import GPTDataset, split_token_ids  # noqa: E402


def pick_device(prefer_mps: bool = True, force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    needles = (
        "out of memory",
        "insufficient memory",
        "memory allocation",
        "mps backend out of memory",
        "kIOGPUCommandBufferCallbackErrorOutOfMemory",
    )
    return any(n in message for n in needles)


def estimate_memory_mb(model: GPT, batch_size: int, block_size: int) -> float:
    """Rough training memory: params + gradients + Adam states + activations."""
    params = count_parameters(model)
    param_bytes = params * 4
    adam_bytes = param_bytes * 2
    activation_bytes = batch_size * block_size * model.blocks[0].norm1.normalized_shape[0] * 4
    activation_bytes *= len(model.blocks) * 8
    total = param_bytes * 2 + adam_bytes + activation_bytes
    return total / (1024**2)


def training_probe_step(
    model: GPT,
    train_set: GPTDataset,
    vocab_size: int,
    cfg: GPTConfig,
    device: torch.device,
    batch_size: int,
) -> None:
    """Run one micro-batch forward+backward to detect OOM before training."""
    model.train()
    model.zero_grad(set_to_none=True)
    x, y = train_set.get_batch(batch_size, cfg.block_size)
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    model.zero_grad(set_to_none=True)
    clear_device_cache(device)


def find_working_batch_size(
    model: GPT,
    train_set: GPTDataset,
    vocab_size: int,
    cfg: GPTConfig,
    device: torch.device,
    *,
    start_batch_size: int,
    min_batch_size: int = 1,
) -> int:
    batch_size = start_batch_size
    while batch_size >= min_batch_size:
        try:
            training_probe_step(model, train_set, vocab_size, cfg, device, batch_size)
            return batch_size
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            print(
                f"  OOM at batch_size={batch_size} on {device} — "
                f"retrying with batch_size={max(min_batch_size, batch_size // 2)}"
            )
            model.zero_grad(set_to_none=True)
            clear_device_cache(device)
            batch_size = max(min_batch_size, batch_size // 2)
    raise RuntimeError(f"Could not find a working batch size >= {min_batch_size} on {device}")


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: GPTDataset,
    val_data: GPTDataset,
    cfg: GPTConfig,
    device: torch.device,
    batches: int = 20,
) -> tuple[float, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, dataset in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(batches):
            x, y = dataset.get_batch(cfg.batch_size, cfg.block_size)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, dataset.vocab_size),
                y.view(-1),
            )
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out["train"], out["val"]


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: GPTConfig,
    train_loss: float,
    val_loss: float,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg.__dict__,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(payload, path)
    return path.stat().st_size


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("step", 0))


def ensure_tokenizer(cfg: GPTConfig) -> BPETokenizer:
    from config import TOKENIZER_PATH

    if TOKENIZER_PATH.exists():
        print(f"Loading tokenizer from {TOKENIZER_PATH}")
        return load_tokenizer(TOKENIZER_PATH)
    print("Tokenizer not found — training BPE...")
    return train_bpe(target_vocab_size=cfg.target_vocab_size, verbose=True)


def build_model(vocab_size: int, cfg: GPTConfig, device: torch.device) -> GPT:
    model = GPT(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(device)
    return model


def apply_config_overrides(
    cfg: GPTConfig,
    *,
    quick: bool = False,
    max_steps: int | None = None,
    learning_rate: float | None = None,
) -> GPTConfig:
    updates: dict[str, object] = {}
    if max_steps is not None:
        updates["max_steps"] = max_steps
    if learning_rate is not None:
        updates["learning_rate"] = learning_rate
    if quick:
        updates.update(
            {
                "max_steps": max_steps or 300,
                "eval_every": 50,
                "log_every": 10,
                "checkpoint_every": 150,
                "batch_size": max(1, cfg.batch_size // 2),
            }
        )
    if not updates:
        return cfg
    return GPTConfig(**{**cfg.__dict__, **updates})


def train(
    cfg: GPTConfig,
    resume_path: Path | None = None,
    quick: bool = False,
    force_cpu: bool = False,
    *,
    auto_batch_size: bool = True,
    max_steps: int | None = None,
    learning_rate: float | None = None,
    corpus_path: Path | None = None,
) -> None:
    torch.manual_seed(cfg.seed)
    explicit_step_budget = max_steps is not None
    cfg = apply_config_overrides(
        cfg,
        quick=quick,
        max_steps=max_steps,
        learning_rate=learning_rate,
    )

    device = pick_device(force_cpu=force_cpu)
    ckpt_dir = checkpoint_dir_for(cfg)
    title = "Phase 15: GPT 50M Pretraining" if cfg.config_name == "large_50m" else "Phase 13: GPT Pretraining"
    print(title)
    print("=" * 60)
    print(f"Config:              {cfg.config_name}")
    print(f"Device:              {device}")
    print()

    if corpus_path is None:
        corpus_path = prepare_corpus()
    elif not corpus_path.exists():
        raise FileNotFoundError(f"Corpus override not found: {corpus_path}")

    bpe = ensure_tokenizer(cfg)
    text = corpus_path.read_text(encoding="utf-8")
    all_ids = corpus_to_ids_safe(bpe, text)
    train_ids, val_ids = split_token_ids(all_ids, val_ratio=cfg.val_ratio)

    train_set = GPTDataset(train_ids, bpe.vocab_size)
    val_set = GPTDataset(val_ids, bpe.vocab_size)

    model = build_model(bpe.vocab_size, cfg, device)
    actual_params = count_parameters(model)
    actual_bd = gpt_parameter_breakdown(model)
    est = estimate_for_gpt_model(model, cfg.__dict__)

    batch_size = cfg.batch_size
    if auto_batch_size and device.type in {"mps", "cuda"}:
        print(f"Probing batch size (start={batch_size})...")
        batch_size = find_working_batch_size(
            model, train_set, bpe.vocab_size, cfg, device, start_batch_size=batch_size
        )
        if batch_size != cfg.batch_size:
            cfg = GPTConfig(**{**cfg.__dict__, "batch_size": batch_size})
            print(f"Using batch_size={batch_size} after OOM fallback")

    print(f"BPE vocab size:     {bpe.vocab_size:,}")
    print(f"Corpus path:        {corpus_path}")
    print(f"Train BPE tokens:   {len(train_ids):,}")
    print(f"Val BPE tokens:     {len(val_ids):,}")
    print(f"Block size:         {cfg.block_size}")
    print(f"Batch size:         {cfg.batch_size}")
    print(f"Learning rate:      {cfg.learning_rate:.2e}")
    print(f"Grad accumulation:  {cfg.gradient_accumulation_steps}")
    print(f"Effective batch:    {cfg.effective_batch_size} sequences")
    print(f"Tokens / opt step:  {cfg.tokens_per_optimizer_step:,}")
    print(f"Model parameters:   {actual_params:,}  (~{actual_params / 1e6:.2f}M)")
    print(f"Analytic estimate:  {est.total:,}  (~{est.total / 1e6:.2f}M)")
    if actual_bd.total != est.total:
        err = abs(est.total - actual_bd.total) / max(actual_bd.total, 1) * 100
        print(f"Estimate error:     {err:.3f}%")
    print(f"Est. train memory:  ~{estimate_memory_mb(model, cfg.batch_size, cfg.block_size):.0f} MB")
    print(f"Checkpoint dir:     {ckpt_dir}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.max_steps, 1)
    )

    start_step = 0
    if resume_path and resume_path.exists():
        start_step = load_checkpoint(resume_path, model, optimizer, device)
        if learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate
        print(f"Resumed from {resume_path} at step {start_step}\n")

    model.train()
    best_val = float("inf")
    tokens_per_step = cfg.tokens_per_optimizer_step
    accum = max(1, cfg.gradient_accumulation_steps)
    step = start_step
    target_step = start_step + cfg.max_steps if resume_path and explicit_step_budget else cfg.max_steps
    steps_to_run = max(0, target_step - start_step)
    print(f"Training target step: {target_step:,} ({steps_to_run:,} optimizer steps to run)")
    t0 = time.perf_counter()

    while step < target_step:
        step += 1
        step_t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(accum):
            try:
                x, y = train_set.get_batch(cfg.batch_size, cfg.block_size)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, bpe.vocab_size), y.view(-1))
                (loss / accum).backward()
                loss_sum += loss.item()
            except RuntimeError as exc:
                if not auto_batch_size or not is_oom_error(exc) or cfg.batch_size <= 1:
                    raise
                new_batch = max(1, cfg.batch_size // 2)
                print(f"  OOM during training — reducing batch_size {cfg.batch_size} → {new_batch}")
                model.zero_grad(set_to_none=True)
                clear_device_cache(device)
                cfg = GPTConfig(**{**cfg.__dict__, "batch_size": new_batch})
                tokens_per_step = cfg.tokens_per_optimizer_step
                x, y = train_set.get_batch(cfg.batch_size, cfg.block_size)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, bpe.vocab_size), y.view(-1))
                (loss / accum).backward()
                loss_sum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        avg_loss = loss_sum / accum
        step_dt = time.perf_counter() - step_t0
        tokens_sec = tokens_per_step / max(step_dt, 1e-6)

        if step % cfg.log_every == 0 or step == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:5d} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | {tokens_sec:,.0f} tok/s | "
                f"bs={cfg.batch_size} accum={accum}"
            )

        if step % cfg.eval_every == 0 or step == target_step:
            train_loss, val_loss = estimate_loss(model, train_set, val_set, cfg, device)
            elapsed = time.perf_counter() - t0
            avg_tps = (step - start_step) * tokens_per_step / max(elapsed, 1e-6)
            print(
                f"  eval step {step} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | avg {avg_tps:,.0f} tok/s | "
                f"mem ~{estimate_memory_mb(model, cfg.batch_size, cfg.block_size):.0f} MB"
            )
            if val_loss < best_val:
                best_val = val_loss

        if step % cfg.checkpoint_every == 0 or step == target_step:
            train_loss, val_loss = estimate_loss(model, train_set, val_set, cfg, device, batches=10)
            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            size = save_checkpoint(ckpt_path, model, optimizer, step, cfg, train_loss, val_loss)
            latest = save_checkpoint(
                latest_checkpoint_for(cfg), model, optimizer, step, cfg, train_loss, val_loss
            )
            print(
                f"  checkpoint → {ckpt_path.name} "
                f"({format_bytes(size)}) | latest {format_bytes(latest)}"
            )

    print()
    print(f"Training complete. Best val loss: {best_val:.4f}")
    print(f"Latest checkpoint: {latest_checkpoint_for(cfg)}")
    print()
    print("Generate text:")
    if cfg.config_name == "large_50m":
        print(
            '  python 13_gpt_pretraining/generate.py --config large_50m '
            '--prompt "Artificial intelligence" --temperature 0.8 --top-k 40'
        )
    else:
        print(
            '  python 13_gpt_pretraining/generate.py '
            '--prompt "Artificial intelligence" --temperature 0.8 --top-k 40'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT pretraining (Phase 13 / 15)")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Model config: default | large_50m",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--quick", action="store_true", help="Short run for smoke tests")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override training steps; with --resume, run this many additional steps",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Alias for --steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Override training corpus path instead of 13_gpt_pretraining/data/corpus.txt",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU (debug)")
    parser.add_argument(
        "--no-auto-batch",
        action="store_true",
        help="Disable automatic batch-size fallback on OOM",
    )
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    max_steps = args.steps if args.steps is not None else args.max_steps
    resume = Path(args.resume) if args.resume else None
    train(
        cfg,
        resume_path=resume,
        quick=args.quick,
        force_cpu=args.cpu,
        auto_batch_size=not args.no_auto_batch,
        max_steps=max_steps,
        learning_rate=args.lr,
        corpus_path=args.corpus,
    )


if __name__ == "__main__":
    main()
