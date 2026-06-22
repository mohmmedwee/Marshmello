"""
Benchmark Phase 13 GPT throughput and memory before scaling to Phase 15 (50M).

Run from project root:
  python 13_gpt_pretraining/benchmark.py
  python 13_gpt_pretraining/benchmark.py --steps 30 --warmup 10
  python 13_gpt_pretraining/benchmark.py --cpu
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))
sys.path.insert(0, str(PROJECT_ROOT / "11_scale_model"))

from bpe_demo import BPETokenizer  # noqa: E402
from estimate_params import (  # noqa: E402
    estimate_for_gpt_model,
    gpt_parameter_breakdown,
    print_parameter_comparison,
)

from config import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    GPTConfig,
    resolve_config,
)
from model.gpt import GPT, count_parameters  # noqa: E402
from tokenizer.bpe_io import load_tokenizer  # noqa: E402
from training.trainer import (  # noqa: E402
    build_model,
    clear_device_cache,
    format_bytes,
    is_oom_error,
    load_checkpoint,
    pick_device,
    save_checkpoint,
)


@dataclass
class BenchmarkResult:
    model_label: str
    parameters: int
    device: str
    batch_size: int
    block_size: int
    forward_tokens_per_sec: float
    training_tokens_per_sec: float
    generation_tokens_per_sec: float
    peak_memory_mb: float
    checkpoint_size_bytes: int
    checkpoint_model_only_bytes: int


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def read_peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024**2)
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024**2)
    try:
        import resource

        # macOS reports bytes; Linux reports kilobytes.
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return usage / (1024**2)
        return usage / 1024
    except ImportError:
        return 0.0


def format_model_label(params: int) -> str:
    if params >= 1_000_000:
        return f"~{params / 1_000_000:.0f}M"
    if params >= 1_000:
        return f"~{params / 1_000:.0f}K"
    return str(params)


def load_vocab_size(cfg: GPTConfig) -> tuple[int, BPETokenizer | None]:
    from config import TOKENIZER_PATH

    if TOKENIZER_PATH.exists():
        bpe = load_tokenizer(TOKENIZER_PATH)
        return bpe.vocab_size, bpe
    return cfg.vocab_size, None


def make_batch(
    batch_size: int,
    block_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    return x, y


def benchmark_forward(
    model: GPT,
    x: torch.Tensor,
    *,
    steps: int,
    warmup: int,
    tokens_per_step: int,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync_device(x.device)

        start = time.perf_counter()
        for _ in range(steps):
            model(x)
        sync_device(x.device)
        elapsed = time.perf_counter() - start

    return (steps * tokens_per_step) / max(elapsed, 1e-9)


def benchmark_training(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    *,
    steps: int,
    warmup: int,
    tokens_per_step: int,
) -> float:
    model.train()
    for _ in range(warmup):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    sync_device(x.device)

    start = time.perf_counter()
    for _ in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    sync_device(x.device)
    elapsed = time.perf_counter() - start

    return (steps * tokens_per_step) / max(elapsed, 1e-9)


def benchmark_generation(
    model: GPT,
    *,
    prompt_len: int,
    new_tokens: int,
    vocab_size: int,
    device: torch.device,
    warmup: int,
) -> float:
    model.eval()
    ids = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    def run_tokens(count: int) -> float:
        local_ids = ids.clone()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(count):
                context = local_ids[:, -model.block_size :]
                logits = model(context)[:, -1, :]
                next_id = int(torch.argmax(logits, dim=-1).item())
                next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
                local_ids = torch.cat([local_ids, next_tensor], dim=1)
        sync_device(device)
        return time.perf_counter() - start

    for _ in range(max(1, warmup // 2)):
        run_tokens(min(8, new_tokens))
    sync_device(device)

    elapsed = run_tokens(new_tokens)
    return new_tokens / max(elapsed, 1e-9)


def measure_checkpoint_sizes(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    cfg: GPTConfig,
) -> tuple[int, int]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        full_path = tmp_dir / "full.pt"
        model_only_path = tmp_dir / "model.pt"

        save_checkpoint(full_path, model, optimizer, 0, cfg, 0.0, 0.0)
        torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, model_only_path)

        return full_path.stat().st_size, model_only_path.stat().st_size


def find_benchmark_batch(
    model: GPT,
    cfg: GPTConfig,
    vocab_size: int,
    device: torch.device,
    *,
    start_batch_size: int,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    batch_size = start_batch_size
    while batch_size >= 1:
        try:
            x, y = make_batch(batch_size, cfg.block_size, vocab_size, device)
            model.train()
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
            clear_device_cache(device)
            return batch_size, x, y
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            print(
                f"  OOM at batch_size={batch_size} — "
                f"retrying with {max(1, batch_size // 2)}"
            )
            model.zero_grad(set_to_none=True)
            clear_device_cache(device)
            batch_size = max(1, batch_size // 2)
    raise RuntimeError("Could not find a working benchmark batch size")


def run_benchmark(
    cfg: GPTConfig,
    *,
    steps: int = 20,
    warmup: int = 5,
    batch_size: int | None = None,
    gen_tokens: int = 64,
    force_cpu: bool = False,
    checkpoint: Path | None = None,
    auto_batch_size: bool = True,
) -> tuple[BenchmarkResult, GPT]:
    device = pick_device(force_cpu=force_cpu)
    start_batch = batch_size or cfg.batch_size

    vocab_size, _bpe = load_vocab_size(cfg)
    if checkpoint and checkpoint.exists():
        model = build_model(vocab_size, cfg, device)
        load_checkpoint(checkpoint, model, optimizer=None, device=device)
    else:
        model = build_model(vocab_size, cfg, device)

    if auto_batch_size and device.type in {"mps", "cuda"} and batch_size is None:
        start_batch, x, y = find_benchmark_batch(
            model, cfg, vocab_size, device, start_batch_size=start_batch
        )
    else:
        start_batch = batch_size or cfg.batch_size
        x, y = make_batch(start_batch, cfg.block_size, vocab_size, device)

    params = count_parameters(model)
    tokens_per_step = start_batch * cfg.block_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    reset_peak_memory(device)
    forward_tps = benchmark_forward(
        model, x, steps=steps, warmup=warmup, tokens_per_step=tokens_per_step
    )
    training_tps = benchmark_training(
        model,
        x,
        y,
        optimizer,
        vocab_size,
        steps=steps,
        warmup=warmup,
        tokens_per_step=tokens_per_step,
    )
    generation_tps = benchmark_generation(
        model,
        prompt_len=min(32, cfg.block_size // 2),
        new_tokens=gen_tokens,
        vocab_size=vocab_size,
        device=device,
        warmup=warmup,
    )
    peak_mb = read_peak_memory_mb(device)
    full_ckpt, model_ckpt = measure_checkpoint_sizes(model, optimizer, cfg)

    return BenchmarkResult(
        model_label=format_model_label(params),
        parameters=params,
        device=str(device),
        batch_size=start_batch,
        block_size=cfg.block_size,
        forward_tokens_per_sec=forward_tps,
        training_tokens_per_sec=training_tps,
        generation_tokens_per_sec=generation_tps,
        peak_memory_mb=peak_mb,
        checkpoint_size_bytes=full_ckpt,
        checkpoint_model_only_bytes=model_ckpt,
    ), model


def print_report(result: BenchmarkResult, cfg: GPTConfig, model: GPT) -> None:
    actual = gpt_parameter_breakdown(model)
    estimated = estimate_for_gpt_model(model, cfg.__dict__)
    tied = model.lm_head.weight is model.token_emb.weight
    lm_head_bias = model.lm_head.bias is not None
    if cfg.config_name == "large_300m":
        phase_label = "Phase 19A GPT 300M Benchmark"
    elif cfg.config_name == "large_50m":
        phase_label = "Phase 15 GPT 50M Benchmark"
    else:
        phase_label = "Phase 13 GPT Benchmark"

    print(phase_label)
    print("=" * 44)
    print(f"Model:                 {result.model_label}")
    print(f"Parameters:            {result.parameters:,}")
    print(f"Analytic estimate:     {estimated.total:,}")
    delta_pct = abs(estimated.total - actual.total) / max(actual.total, 1) * 100
    print(f"Estimate error:        {delta_pct:.3f}%")
    print(f"Device:                {result.device}")
    print(f"Batch size:            {result.batch_size}")
    print(f"Block size:            {result.block_size}")
    print()
    print(f"Forward tokens/sec:    {result.forward_tokens_per_sec:,.0f}")
    print(f"Training tokens/sec:   {result.training_tokens_per_sec:,.0f}")
    print(f"Generation tokens/sec: {result.generation_tokens_per_sec:,.0f}")
    print()
    print(f"Peak memory:           {result.peak_memory_mb:.1f} MB")
    print(f"Checkpoint size:       {format_bytes(result.checkpoint_size_bytes)} (full train state)")
    print(f"Model weights only:    {format_bytes(result.checkpoint_model_only_bytes)}")
    print()
    print("Config:")
    print(
        f"  d_model={cfg.d_model} layers={cfg.num_layers} "
        f"heads={cfg.num_heads} d_ff={cfg.d_ff} vocab={model.vocab_size}"
    )
    print()
    print_parameter_comparison(
        actual,
        estimated,
        weight_tied=tied,
        lm_head_bias=lm_head_bias,
    )
    print()
    if cfg.config_name == "large_300m":
        print("Compare with 50M: python 13_gpt_pretraining/benchmark.py --config large_50m")
        print("Phase 19A docs: 19A_scale_to_300m/README.md")
    elif cfg.config_name == "large_50m":
        print("Compare with Phase 13 baseline: python 13_gpt_pretraining/benchmark.py --config default")
        print("Scale up: python 13_gpt_pretraining/benchmark.py --config large_300m")
        print("Phase 15 docs: 15_scale_to_50m/README.md")
    else:
        print("Scale up: python 13_gpt_pretraining/benchmark.py --config large_50m")
        print("Phase 15 docs: 15_scale_to_50m/README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GPT throughput and memory.")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Model config: default | large_50m | large_300m",
    )
    parser.add_argument("--steps", type=int, default=20, help="Timed benchmark steps")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps before timing")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gen-tokens", type=int, default=64, help="Tokens for generation benchmark")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Optional checkpoint (default: build fresh model from config)",
    )
    parser.add_argument("--quick", action="store_true", help="Fewer steps for smoke test")
    parser.add_argument(
        "--no-auto-batch",
        action="store_true",
        help="Disable automatic batch-size fallback on OOM",
    )
    args = parser.parse_args()

    steps = 5 if args.quick else args.steps
    warmup = 2 if args.quick else args.warmup
    gen_tokens = 16 if args.quick else args.gen_tokens
    cfg = resolve_config(args.config)

    checkpoint = args.checkpoint
    if checkpoint is None and DEFAULT_CHECKPOINT.exists() and cfg.config_name == "default":
        checkpoint = DEFAULT_CHECKPOINT
    elif checkpoint is None and cfg.config_name == "large_50m":
        from config import latest_checkpoint_for

        candidate = latest_checkpoint_for(cfg)
        checkpoint = candidate if candidate.exists() else None

    result, model = run_benchmark(
        cfg,
        steps=steps,
        warmup=warmup,
        batch_size=args.batch_size,
        gen_tokens=gen_tokens,
        force_cpu=args.cpu,
        checkpoint=checkpoint if checkpoint and checkpoint.exists() else None,
        auto_batch_size=not args.no_auto_batch,
    )
    print_report(result, cfg, model)


if __name__ == "__main__":
    main()
