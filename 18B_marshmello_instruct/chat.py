#!/usr/bin/env python3
"""Chat with Marshmello-45M-Instruct."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PHASE13_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from config import resolve_config  # noqa: E402
from model.gpt import GPT  # noqa: E402
from tokenizer.bpe_io import load_tokenizer  # noqa: E402
from tokenizer.decode import decode_ids_pretty  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from train_instruct import (  # noqa: E402
    CHECKPOINT_DIR,
    END_TAG,
    TOKENIZER_PATH,
    encode_text,
    format_chat_prompt,
)

USER_TAG = "<USER>"
ASSISTANT_TAG = "<ASSISTANT>"


def load_model(checkpoint_path: Path, config_name: str, device: torch.device) -> tuple[GPT, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    bpe = load_tokenizer(TOKENIZER_PATH)
    cfg = resolve_config(config_name)
    model = GPT(
        vocab_size=bpe.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        block_size=cfg.block_size,
        dropout=0.0,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, bpe


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, indices, values)
    return filtered


@torch.no_grad()
def generate_reply(
    model: GPT,
    bpe,
    prompt: str,
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 40,
    greedy: bool = False,
) -> str:
    prefix = format_chat_prompt(prompt)
    ids = encode_text(bpe, prefix)
    end_ids = encode_text(bpe, END_TAG)
    prompt_len = len(ids)

    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.block_size :]], dtype=torch.long, device=next(model.parameters()).device)
        logits = model.generate_step_logits(context)[0]
        if greedy:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = apply_top_k(logits / max(temperature, 1e-6), top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
        if len(ids) >= len(end_ids) and ids[-len(end_ids) :] == end_ids:
            break

    answer_ids = ids[prompt_len:]
    text = decode_ids_pretty(bpe, answer_ids)
    return text.split(END_TAG, 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Marshmello-45M-Instruct.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", default="large_50m")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "latest.pt")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = pick_device(force_cpu=args.cpu)
    model, bpe = load_model(args.checkpoint, args.config, device)
    reply = generate_reply(
        model,
        bpe,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )
    print(reply)


if __name__ == "__main__":
    main()
