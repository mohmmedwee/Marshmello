#!/usr/bin/env python3
"""Evaluate the Phase 18E teacher checkpoint on direct assistant prompts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE18B_ROOT = PROJECT_ROOT / "18B_marshmello_instruct"
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE18B_ROOT))
sys.path.insert(0, str(PHASE13_ROOT))

from chat import load_model  # noqa: E402
from train_instruct import generate_assistant_reply, repeated_ngram  # noqa: E402
from training.trainer import pick_device  # noqa: E402

DEFAULT_CHECKPOINT = PHASE_ROOT / "checkpoints" / "teacher_latest.pt"

PROMPT_KEYWORDS = {
    "What is AI?": ("ai", "computer", "thinking", "tasks"),
    "What is machine learning?": ("machine", "learning", "data", "patterns"),
    "What is attention?": ("attention", "tokens", "matter", "model"),
    "Explain database indexes.": ("database", "index", "rows", "faster"),
    "What is a transformer?": ("transformer", "attention", "tokens", "sequence"),
    "Who are you?": ("marshmello", "assistant", "answer", "directly"),
}

BAD_LOOP = "computer is a computer"


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def on_topic(text: str, keywords: tuple[str, ...]) -> bool:
    lower = text.casefold()
    return sum(1 for keyword in keywords if keyword in lower) >= 2


def check_answer(prompt: str, answer: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    words = word_count(answer)
    if not 4 <= words <= 70:
        failures.append(f"not short ({words} words)")
    if not on_topic(answer, PROMPT_KEYWORDS[prompt]):
        failures.append("off topic")
    if repeated_ngram(answer, n=3) is not None:
        failures.append("repeated 3-gram")
    if BAD_LOOP in answer.casefold():
        failures.append("computer loop")
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 18E teacher checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default="large_50m")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    device = pick_device(force_cpu=args.cpu)
    model, bpe = load_model(args.checkpoint, args.config, device)

    all_ok = True
    print("Phase 18E: teacher checkpoint eval")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    for prompt in PROMPT_KEYWORDS:
        answer = generate_assistant_reply(
            model,
            bpe,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.2,
            top_k=10,
            repetition_penalty=1.3,
        )
        ok, failures = check_answer(prompt, answer)
        all_ok = all_ok and ok
        result = "PASS" if ok else "FAIL"
        reason = "ok" if ok else ", ".join(failures)
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Result: {result} - {reason}")
        print()

    print(f"Final: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
