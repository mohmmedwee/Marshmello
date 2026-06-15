#!/usr/bin/env python3
"""Evaluate whether a base checkpoint understands the chat boundary format."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE13_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from config import latest_checkpoint_for, resolve_config  # noqa: E402
from generate import (  # noqa: E402
    encode_prompt,
    generate,
    generated_suffix,
    load_model_and_tokenizer,
)
from tokenizer.decode import decode_ids_pretty  # noqa: E402
from training.trainer import pick_device  # noqa: E402

PROMPT = "<USER> What is AI? <ASSISTANT>"
ANSWER_PATTERNS = (
    re.compile(r"\bai\b", flags=re.IGNORECASE),
    re.compile(r"\bartificial intelligence\b", flags=re.IGNORECASE),
    re.compile(r"\bintelligence\b", flags=re.IGNORECASE),
    re.compile(r"\bmachine learning\b", flags=re.IGNORECASE),
)
BAD_START_RE = re.compile(
    r"^(<user>|<assistant>|<end>|===|topic:|database|sql|python|software "
    r"engineering|operating systems|containers|this text is about)\b",
    flags=re.IGNORECASE,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def answer_suffix(bpe, prompt: str, full_text: str) -> str:
    prompt_ids = encode_prompt(bpe, prompt)
    prompt_decoded = decode_ids_pretty(bpe, prompt_ids)
    return normalize(generated_suffix(full_text, prompt_decoded))


def looks_like_answer(text: str) -> tuple[bool, str]:
    text = normalize(text)
    if not text:
        return False, "empty generation after chat prompt"
    if BAD_START_RE.search(text):
        return False, "starts like corpus continuation or another chat tag"

    first_words = " ".join(text.split()[:30])
    if any(pattern.search(first_words) for pattern in ANSWER_PATTERNS):
        return True, "starts with AI-related answer content"
    return False, "first words do not look related to the user question"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 18C chat-format adaptation.")
    parser.add_argument("--config", default="large_50m")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Print the check result but do not exit nonzero on failure",
    )
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    checkpoint = args.checkpoint or latest_checkpoint_for(cfg)
    device = pick_device(force_cpu=args.cpu)
    model, bpe, model_cfg = load_model_and_tokenizer(checkpoint, device, cfg)

    result = generate(
        model,
        bpe,
        PROMPT,
        max_new_tokens=args.max_new_tokens,
        greedy=True,
        stop_sequence="<END>",
        stop_on_eos_token=True,
        stop_on_sentence_end=False,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        device=device,
    )
    suffix = answer_suffix(bpe, PROMPT, result.text)
    if not suffix:
        suffix = answer_suffix(bpe, PROMPT, result.raw_text)

    ok, reason = looks_like_answer(suffix)

    print("Phase 18C: chat-format boundary eval")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Config:     {model_cfg.config_name}")
    print(f"Device:     {device}")
    print(f"Prompt:     {PROMPT}")
    print(f"Generated:  {suffix}")
    print(f"Result:     {'PASS' if ok else 'FAIL'} - {reason}")

    if not ok and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
