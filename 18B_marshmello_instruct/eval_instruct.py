#!/usr/bin/env python3
"""Compare Marshmello-45M-Base and Marshmello-45M-Instruct on fixed prompts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PHASE13_ROOT))

from config import latest_checkpoint_for, resolve_config  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from chat import generate_reply, load_model  # noqa: E402

PROMPTS = [
    "What is AI?",
    "Explain database indexes.",
    "Write a Python function to reverse a string.",
    "Explain Docker simply.",
    "What is BPE?",
]


def cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs instruct checkpoints.")
    parser.add_argument("--config", default="large_50m")
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--instruct-checkpoint", type=Path, default=PHASE_ROOT / "checkpoints" / "latest.pt")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    base_path = args.base_checkpoint or latest_checkpoint_for(cfg)
    device = pick_device(force_cpu=args.cpu)

    base_model, bpe = load_model(base_path, args.config, device)
    instruct_model, _ = load_model(args.instruct_checkpoint, args.config, device)

    rows = []
    for prompt in PROMPTS:
        base_reply = generate_reply(
            base_model,
            bpe,
            prompt,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
        )
        instruct_reply = generate_reply(
            instruct_model,
            bpe,
            prompt,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
        )
        rows.append([prompt, base_reply[:220], instruct_reply[:220]])

    print("Phase 18B: Base vs Instruct evaluation")
    print("=" * 60)
    print(f"Base checkpoint:     {base_path}")
    print(f"Instruct checkpoint: {args.instruct_checkpoint}")
    print()
    print("| Prompt | Marshmello-45M-Base | Marshmello-45M-Instruct |")
    print("|---|---|---|")
    for prompt, base_reply, instruct_reply in rows:
        print(f"| {cell(prompt)} | {cell(base_reply)} | {cell(instruct_reply)} |")


if __name__ == "__main__":
    main()
