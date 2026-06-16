#!/usr/bin/env python3
"""Compare Marshmello old tokenizer vs tokenizer v2 on chat-oriented prompts."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE13_ROOT))
sys.path.insert(0, str(PHASE_ROOT))

from tokenizer.bpe_io import load_tokenizer  # noqa: E402
from train_tokenizer_v2 import (  # noqa: E402
    DEFAULT_OLD_TOKENIZER,
    DEFAULT_OUTPUT,
    INCOMPATIBILITY_WARNING,
    merge_rank_map,
    safe_tokenize_to_tokens,
)

SAMPLES = (
    "What is AI?",
    "Explain database indexes.",
    "Write a Python function to reverse a string.",
    "<USER> What is AI? <ASSISTANT>",
)


def compact_tokens(tokens: list[str], limit: int) -> str:
    shown = tokens[:limit]
    suffix = " ..." if len(tokens) > limit else ""
    return f"{shown}{suffix}"


def skipped_summary(skipped: Counter[str]) -> str:
    if not skipped:
        return "none"
    return ", ".join(f"{char!r} x{count}" for char, count in skipped.most_common(8))


def print_tokenizer_row(
    *,
    label: str,
    tokens: list[str],
    skipped: Counter[str],
    max_tokens: int,
) -> None:
    print(f"  {label}:")
    print(f"    token count: {len(tokens)}")
    print(f"    skipped:     {skipped_summary(skipped)}")
    print(f"    pieces:      {compact_tokens(tokens, max_tokens)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old tokenizer vs tokenizer v2.")
    parser.add_argument("--old-tokenizer", type=Path, default=DEFAULT_OLD_TOKENIZER)
    parser.add_argument("--v2-tokenizer", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=40)
    args = parser.parse_args()

    if not args.old_tokenizer.exists():
        raise FileNotFoundError(f"Old tokenizer not found: {args.old_tokenizer}")
    if not args.v2_tokenizer.exists():
        raise FileNotFoundError(
            f"Tokenizer v2 not found: {args.v2_tokenizer}. "
            "Run python 18D_tokenizer_v2/train_tokenizer_v2.py first."
        )

    old = load_tokenizer(args.old_tokenizer)
    v2 = load_tokenizer(args.v2_tokenizer)
    old_ranks = merge_rank_map(old)
    v2_ranks = merge_rank_map(v2)

    print("Phase 18D: old tokenizer vs tokenizer v2")
    print("=" * 60)
    print(f"Old tokenizer: {args.old_tokenizer} ({old.vocab_size:,} vocab)")
    print(f"V2 tokenizer:  {args.v2_tokenizer} ({v2.vocab_size:,} vocab)")
    print()

    for sample in SAMPLES:
        old_tokens, old_skipped = safe_tokenize_to_tokens(old, sample, old_ranks)
        v2_tokens, v2_skipped = safe_tokenize_to_tokens(v2, sample, v2_ranks)
        ratio = len(old_tokens) / len(v2_tokens) if v2_tokens else 0.0

        print(f"Text: {sample!r}")
        print_tokenizer_row(
            label="old",
            tokens=old_tokens,
            skipped=old_skipped,
            max_tokens=args.max_tokens,
        )
        print_tokenizer_row(
            label="v2",
            tokens=v2_tokens,
            skipped=v2_skipped,
            max_tokens=args.max_tokens,
        )
        print(f"  old/v2 token ratio: {ratio:.2f}x")
        print()

    print(f"Important: {INCOMPATIBILITY_WARNING}")


if __name__ == "__main__":
    main()
