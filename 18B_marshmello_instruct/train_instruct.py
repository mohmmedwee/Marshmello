#!/usr/bin/env python3
"""
Phase 18B: SFT fine-tuning for Marshmello-45M-Instruct.

Loads Marshmello-45M-Base-v2 (large_50m checkpoint), trains only on assistant
response targets in Phase 17 chat JSONL, and writes instruct checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE13_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from config import GPTConfig, latest_checkpoint_for, resolve_config  # noqa: E402
from model.gpt import GPT, count_parameters  # noqa: E402
from tokenizer.bpe_io import load_tokenizer  # noqa: E402
from training.trainer import load_checkpoint, pick_device  # noqa: E402

TOKENIZER_PATH = PHASE13_ROOT / "tokenizer" / "tokenizer.json"
CHAT_DATA_PATH = PROJECT_ROOT / "17_instruction_dataset" / "processed" / "chat.jsonl"
CHECKPOINT_DIR = PHASE_ROOT / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"

USER_TAG = "<USER>"
ASSISTANT_TAG = "<ASSISTANT>"
END_TAG = "<END>"


def find_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    for i in range(0, len(values) - len(needle) + 1):
        if values[i : i + len(needle)] == needle:
            return i
    return None


def encode_text(bpe, text: str) -> list[int]:
    try:
        return bpe.corpus_to_ids(text)
    except KeyError as exc:
        raise RuntimeError(
            "Tokenizer cannot encode SFT text. Rebuild the tokenizer after running "
            "18A_large_pretraining_corpus/build_corpus.py so chat tags and technical "
            f"characters are present. Missing token: {exc}"
        ) from exc


class SFTDataset:
    """Fixed-length SFT examples with assistant-only loss weights."""

    def __init__(
        self,
        texts: list[str],
        bpe,
        block_size: int,
        *,
        first_token_weight: float = 8.0,
        end_weight: float = 2.0,
        pad_id: int = 0,
    ) -> None:
        self.block_size = block_size
        self.pad_id = pad_id
        self.vocab_size = bpe.vocab_size
        self.assistant_ids = encode_text(bpe, ASSISTANT_TAG)
        self.end_ids = encode_text(bpe, END_TAG)
        self.examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for text in texts:
            ids = encode_text(bpe, text)
            if len(ids) < 2:
                continue
            ids = ids[: block_size + 1]
            weights = self._weights_for_ids(
                ids,
                first_token_weight=first_token_weight,
                end_weight=end_weight,
            )
            if not any(weights):
                continue
            if len(ids) < block_size + 1:
                pad = [pad_id] * (block_size + 1 - len(ids))
                ids = ids + pad
                weights = weights + [0.0] * (block_size - len(weights))

            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            w = torch.tensor(weights[:block_size], dtype=torch.float)
            self.examples.append((x, y, w))

        if not self.examples:
            raise ValueError("No SFT examples produced non-empty assistant loss weights")

    def _weights_for_ids(
        self,
        ids: list[int],
        *,
        first_token_weight: float,
        end_weight: float,
    ) -> list[float]:
        weights = [0.0] * max(0, len(ids) - 1)
        assistant_start = find_subsequence(ids, self.assistant_ids)
        if assistant_start is None:
            return weights
        first_answer_target_pos = assistant_start + len(self.assistant_ids) - 1
        end_start = find_subsequence(ids, self.end_ids)
        end_last_target_pos = len(weights) - 1
        if end_start is not None:
            end_last_target_pos = min(len(weights) - 1, end_start + len(self.end_ids) - 2)

        for pos in range(first_answer_target_pos, end_last_target_pos + 1):
            weights[pos] = 1.0
        if 0 <= first_answer_target_pos < len(weights):
            weights[first_answer_target_pos] = first_token_weight
        if end_start is not None:
            for pos in range(max(first_answer_target_pos, end_start - 1), end_last_target_pos + 1):
                weights[pos] = end_weight
        return weights

    def __len__(self) -> int:
        return len(self.examples)

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.examples), (batch_size,))
        xs, ys, ws = zip(*(self.examples[int(i)] for i in idx))
        return torch.stack(xs).to(device), torch.stack(ys).to(device), torch.stack(ws).to(device)


def load_chat_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            if text:
                texts.append(text)
    if not texts:
        raise ValueError(f"No chat texts found in {path}")
    return texts


def split_texts(texts: list[str], val_ratio: float = 0.05) -> tuple[list[str], list[str]]:
    split_at = max(1, int(len(texts) * (1.0 - val_ratio)))
    return texts[:split_at], texts[split_at:] or texts[-1:]


def weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
    loss = loss.view(weights.shape)
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


@torch.no_grad()
def estimate_loss(model: GPT, train_set: SFTDataset, val_set: SFTDataset, batch_size: int, device: torch.device, batches: int = 10) -> tuple[float, float]:
    model.eval()
    values: dict[str, float] = {}
    for name, dataset in [("train", train_set), ("val", val_set)]:
        losses = []
        for _ in range(batches):
            x, y, w = dataset.get_batch(batch_size, device)
            logits = model(x)
            losses.append(weighted_cross_entropy(logits, y, w).item())
        values[name] = sum(losses) / len(losses)
    model.train()
    return values["train"], values["val"]


def save_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer, step: int, cfg: GPTConfig, train_loss: float, val_loss: float) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg.__dict__,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "phase": "18B_marshmello_instruct",
    }
    torch.save(payload, path)
    return path.stat().st_size


def build_model(vocab_size: int, cfg: GPTConfig, device: torch.device) -> GPT:
    return GPT(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(device)


def train(args: argparse.Namespace) -> None:
    cfg = resolve_config(args.config)
    cfg = replace(cfg, max_steps=args.steps, learning_rate=args.lr or cfg.learning_rate)
    device = pick_device(force_cpu=args.cpu)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")
    bpe = load_tokenizer(TOKENIZER_PATH)

    base_checkpoint = latest_checkpoint_for(cfg)
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_checkpoint}\n"
            "Run Phase 18A, retrain tokenizer, then pretrain base first:\n"
            "  python 18A_large_pretraining_corpus/build_corpus.py --target-words 1000000\n"
            "  python 13_gpt_pretraining/tokenizer/train_bpe.py\n"
            "  python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 3000"
        )

    model = build_model(bpe.vocab_size, cfg, device)
    load_checkpoint(base_checkpoint, model, optimizer=None, device=device)

    texts = load_chat_texts(args.data)
    train_texts, val_texts = split_texts(texts, val_ratio=0.05)
    train_set = SFTDataset(train_texts, bpe, cfg.block_size)
    val_set = SFTDataset(val_texts, bpe, cfg.block_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    print("Phase 18B: Marshmello-45M-Instruct SFT")
    print("=" * 60)
    print(f"Config:          {cfg.config_name}")
    print(f"Device:          {device}")
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Tokenizer:       {TOKENIZER_PATH}")
    print(f"Chat data:       {args.data}")
    print(f"Train examples:  {len(train_set):,}")
    print(f"Val examples:    {len(val_set):,}")
    print(f"Parameters:      {count_parameters(model):,}")
    print(f"Checkpoint path: {LATEST_CHECKPOINT}")
    print()

    model.train()
    t0 = time.perf_counter()
    last_train_loss = 0.0
    last_val_loss = 0.0
    for step in range(1, cfg.max_steps + 1):
        step_t0 = time.perf_counter()
        x, y, weights = train_set.get_batch(cfg.batch_size, device)
        logits = model(x)
        loss = weighted_cross_entropy(logits, y, weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        tokens_sec = cfg.batch_size * cfg.block_size / max(time.perf_counter() - step_t0, 1e-6)
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d} | train loss {loss.item():.4f} | {tokens_sec:,.0f} tok/s")

        if step % args.eval_every == 0 or step == cfg.max_steps:
            last_train_loss, last_val_loss = estimate_loss(
                model, train_set, val_set, cfg.batch_size, device, batches=args.eval_batches
            )
            elapsed = time.perf_counter() - t0
            avg_tps = step * cfg.batch_size * cfg.block_size / max(elapsed, 1e-6)
            print(
                f"  eval step {step} | train loss {last_train_loss:.4f} | "
                f"val loss {last_val_loss:.4f} | avg {avg_tps:,.0f} tok/s"
            )

        if step % args.checkpoint_every == 0 or step == cfg.max_steps:
            size = save_checkpoint(LATEST_CHECKPOINT, model, optimizer, step, cfg, last_train_loss, last_val_loss)
            print(f"  checkpoint path: {LATEST_CHECKPOINT} ({size / 1024**2:.1f} MB)")

    print("\nTraining complete.")
    print(f"Latest checkpoint: {LATEST_CHECKPOINT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Marshmello-45M-Instruct.")
    parser.add_argument("--config", default="large_50m")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--data", type=Path, default=CHAT_DATA_PATH)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
