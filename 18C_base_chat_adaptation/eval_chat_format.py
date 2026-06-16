#!/usr/bin/env python3
"""Evaluate whether a base checkpoint understands the chat boundary format."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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
EXACT_STARTS = (
    "ai is",
    "artificial intelligence is",
)
TOPIC_KEYWORDS = (
    "artificial",
    "intelligence",
    "machine",
    "learning",
    "data",
    "algorithm",
    "prediction",
    "pattern",
)
RAW_CORPUS_START_PHRASES = (
    "teams should write",
    "example note",
    "database systems need",
)


@dataclass(frozen=True)
class SemanticEval:
    boundary_ok: bool
    exact_start_ok: bool
    topic_keywords_found: list[str]
    semantic_ok: bool

    @property
    def topic_keyword_count(self) -> int:
        return len(self.topic_keywords_found)

    @property
    def final_ok(self) -> bool:
        return self.boundary_ok and (self.exact_start_ok or self.semantic_ok)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def answer_suffix(bpe, prompt: str, full_text: str) -> str:
    prompt_ids = encode_prompt(bpe, prompt)
    prompt_decoded = decode_ids_pretty(bpe, prompt_ids)
    return normalize(generated_suffix(full_text, prompt_decoded))


def exact_start_ok(text: str) -> bool:
    lower = normalize(text).casefold()
    return lower.startswith(EXACT_STARTS)


def find_topic_keywords(text: str) -> list[str]:
    lower = normalize(text).casefold()
    found: list[str] = []
    for keyword in TOPIC_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}s?\b", lower):
            found.append(keyword)
    return found


def boundary_ok(text: str) -> bool:
    lower = normalize(text).casefold()
    if not lower:
        return False
    if "<user>" in lower:
        return False
    return not any(lower.startswith(phrase) for phrase in RAW_CORPUS_START_PHRASES)


def evaluate_generated_text(text: str, *, min_keywords: int) -> SemanticEval:
    topic_keywords_found = find_topic_keywords(text)
    return SemanticEval(
        boundary_ok=boundary_ok(text),
        exact_start_ok=exact_start_ok(text),
        topic_keywords_found=topic_keywords_found,
        semantic_ok=len(topic_keywords_found) >= min_keywords,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 18C chat-format adaptation.")
    parser.add_argument(
        "--config",
        default="large_50m",
        help="Model config used when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path to evaluate (default: latest checkpoint for --config).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--min-keywords", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Print the check result but do not exit nonzero on failure",
    )
    args = parser.parse_args()
    if args.min_keywords < 0:
        raise ValueError("--min-keywords must be >= 0")

    cfg = resolve_config(args.config)
    checkpoint = args.checkpoint or latest_checkpoint_for(cfg)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
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

    semantic_eval = evaluate_generated_text(suffix, min_keywords=args.min_keywords)
    final_result = "PASS" if semantic_eval.final_ok else "FAIL"

    print("Phase 18C: chat-format boundary eval")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Config:     {model_cfg.config_name}")
    print(f"Device:     {device}")
    print(f"Prompt:     {PROMPT}")
    print(f"Generated:  {suffix}")
    print(f"boundary_ok:          {semantic_eval.boundary_ok}")
    print(f"exact_start_ok:       {semantic_eval.exact_start_ok}")
    print(f"topic_keywords_found: {semantic_eval.topic_keywords_found}")
    print(f"topic_keyword_count:  {semantic_eval.topic_keyword_count}")
    print(f"semantic_ok:          {semantic_eval.semantic_ok}")
    print(f"final_result:         {final_result}")

    if not semantic_eval.final_ok and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
