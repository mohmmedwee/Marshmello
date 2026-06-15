#!/usr/bin/env python3
"""
Build a raw-text + chat-format corpus for continued base pretraining.

This is not SFT: the output is plain text for next-token prediction on every
token, with no assistant masking.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
DEFAULT_RAW_CORPUS = PROJECT_ROOT / "13_gpt_pretraining" / "data" / "corpus.txt"
DEFAULT_CHAT_DATA = PROJECT_ROOT / "17_instruction_dataset" / "processed" / "chat.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "13_gpt_pretraining" / "data" / "corpus_chat_mixed.txt"


def normalize_block(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def word_count(text: str) -> int:
    return len(text.split())


def load_raw_paragraphs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    paragraphs = [normalize_block(part) for part in re.split(r"\n\s*\n", text)]
    return [paragraph for paragraph in paragraphs if paragraph]


def load_chat_examples(path: Path, max_examples: int | None) -> list[str]:
    examples: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = normalize_block(str(record.get("text", "")))
            if text:
                examples.append(text)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def select_chat_examples_for_ratio(
    raw_paragraphs: list[str],
    chat_examples: list[str],
    chat_ratio: float,
) -> list[str]:
    raw_words = sum(word_count(paragraph) for paragraph in raw_paragraphs)
    target_chat_words = int(raw_words * chat_ratio / (1.0 - chat_ratio))

    selected: list[str] = []
    selected_words = 0
    for example in chat_examples:
        selected.append(example)
        selected_words += word_count(example)
        if selected_words >= target_chat_words:
            break
    return selected


def write_mixed_corpus(
    *,
    raw_paragraphs: list[str],
    chat_examples: list[str],
    output_path: Path,
    seed: int,
) -> str:
    blocks = [("raw", paragraph) for paragraph in raw_paragraphs]
    blocks.extend(("chat", example) for example in chat_examples)

    rng = random.Random(seed)
    rng.shuffle(blocks)

    text = "\n\n".join(block for _, block in blocks).strip() + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


def print_report(
    *,
    raw_paragraphs: list[str],
    selected_chat: list[str],
    output_path: Path,
    written_text: str,
    requested_ratio: float,
) -> None:
    raw_words = sum(word_count(paragraph) for paragraph in raw_paragraphs)
    chat_words = sum(word_count(example) for example in selected_chat)
    total_words = raw_words + chat_words
    actual_ratio = chat_words / total_words if total_words else 0.0

    print("Phase 18C: base chat adaptation corpus")
    print("=" * 60)
    print("Objective: next-token prediction on all tokens (not SFT)")
    print(f"Output:             {output_path}")
    print(f"Raw paragraphs:     {len(raw_paragraphs):,}")
    print(f"Chat examples:      {len(selected_chat):,}")
    print(f"Raw words:          {raw_words:,}")
    print(f"Chat words:         {chat_words:,}")
    print(f"Requested chat mix: {requested_ratio:.1%}")
    print(f"Actual chat mix:    {actual_ratio:.1%}")
    print(f"Written words:      {len(written_text.split()):,}")
    print(f"Written chars:      {len(written_text):,}")
    print()
    print("Train with:")
    print(
        "  python 13_gpt_pretraining/training/trainer.py "
        "--config large_50m "
        "--resume 13_gpt_pretraining/checkpoints/large_50m/latest.pt "
        "--corpus 13_gpt_pretraining/data/corpus_chat_mixed.txt "
        "--steps 1000 --lr 5e-5"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 18C chat-mixed corpus.")
    parser.add_argument("--raw-corpus", type=Path, default=DEFAULT_RAW_CORPUS)
    parser.add_argument("--chat-data", type=Path, default=DEFAULT_CHAT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chat-ratio", type=float, default=0.2)
    parser.add_argument("--max-chat-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.chat_ratio < 1.0:
        raise ValueError("--chat-ratio must be between 0 and 1")
    if args.max_chat_examples is not None and args.max_chat_examples <= 0:
        raise ValueError("--max-chat-examples must be positive when provided")
    if not args.raw_corpus.exists():
        raise FileNotFoundError(f"Raw corpus not found: {args.raw_corpus}")
    if not args.chat_data.exists():
        raise FileNotFoundError(f"Chat data not found: {args.chat_data}")

    raw_paragraphs = load_raw_paragraphs(args.raw_corpus)
    chat_examples = load_chat_examples(args.chat_data, args.max_chat_examples)
    if not raw_paragraphs:
        raise ValueError(f"No raw paragraphs found in {args.raw_corpus}")
    if not chat_examples:
        raise ValueError(f"No chat examples found in {args.chat_data}")

    selected_chat = select_chat_examples_for_ratio(
        raw_paragraphs=raw_paragraphs,
        chat_examples=chat_examples,
        chat_ratio=args.chat_ratio,
    )
    written_text = write_mixed_corpus(
        raw_paragraphs=raw_paragraphs,
        chat_examples=selected_chat,
        output_path=args.output,
        seed=args.seed,
    )
    print_report(
        raw_paragraphs=raw_paragraphs,
        selected_chat=selected_chat,
        output_path=args.output,
        written_text=written_text,
        requested_ratio=args.chat_ratio,
    )


if __name__ == "__main__":
    main()
