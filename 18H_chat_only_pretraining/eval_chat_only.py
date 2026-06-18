#!/usr/bin/env python3
"""
Evaluate a checkpoint after Phase 18H chat-only continued pretraining.

For each prompt the model is asked to answer and stop at ``<END>``. Success
means the model behaves like a chat assistant rather than a raw corpus
continuation:

    - output before <END> is on topic
    - output stops (or can be truncated) at <END>
    - no raw corpus continuation after <END>
    - no repeated 3-grams
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
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

STOP_SEQUENCE = "<END>"

# Each prompt pairs the chat turn with the topic keywords that signal an
# on-topic answer. A match on any keyword counts the answer as on topic.
PROMPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "<USER> What is AI? <ASSISTANT>",
        ("ai", "artificial", "intelligence", "machine", "computer", "software", "data", "learning", "tasks"),
    ),
    (
        "<USER> Who are you? <ASSISTANT>",
        ("i", "assistant", "marshmello", "help", "model", "trained", "answer", "questions"),
    ),
    (
        "<USER> Explain database indexes. <ASSISTANT>",
        ("index", "indexes", "database", "query", "queries", "lookup", "search", "table", "column", "faster", "speed", "rows"),
    ),
)

# Phrases that indicate the model fell back into raw corpus text.
RAW_CORPUS_PHRASES = (
    "teams should write",
    "example note",
    "database systems need",
    "documentation for",
)


@dataclass
class PromptEval:
    prompt: str
    answer: str
    after_end: str
    on_topic: bool
    topic_keywords_found: list[str]
    stops_at_end: bool
    clean_after_end: bool
    no_repeated_3gram: bool
    repeated_3gram: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.on_topic
            and self.stops_at_end
            and self.clean_after_end
            and self.no_repeated_3gram
        )


@dataclass
class EvalSummary:
    results: list[PromptEval] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def answer_suffix(bpe, prompt: str, full_text: str) -> str:
    prompt_ids = encode_prompt(bpe, prompt)
    prompt_decoded = decode_ids_pretty(bpe, prompt_ids)
    return normalize(generated_suffix(full_text, prompt_decoded))


def find_topic_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    lower = normalize(text).casefold()
    found: list[str] = []
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", lower):
            found.append(keyword)
    return found


def first_repeated_3gram(text: str) -> str | None:
    words = re.findall(r"[a-z0-9']+", normalize(text).casefold())
    seen: set[tuple[str, str, str]] = set()
    for i in range(len(words) - 2):
        gram = (words[i], words[i + 1], words[i + 2])
        if gram in seen:
            return " ".join(gram)
        seen.add(gram)
    return None


def is_clean_after_end(after_end: str) -> bool:
    lower = normalize(after_end).casefold()
    if not lower:
        return True
    if "<user>" in lower or "<assistant>" in lower:
        return False
    return not any(phrase in lower for phrase in RAW_CORPUS_PHRASES)


def evaluate_prompt(bpe, prompt: str, keywords: tuple[str, ...], result) -> PromptEval:
    answer = answer_suffix(bpe, prompt, result.text)
    if not answer:
        answer = answer_suffix(bpe, prompt, result.raw_text)

    # Everything the model emitted after the first <END> in the raw stream.
    raw_suffix = answer_suffix(bpe, prompt, result.raw_text)
    end_idx = raw_suffix.find(STOP_SEQUENCE)
    if end_idx == -1:
        after_end = ""
    else:
        after_end = raw_suffix[end_idx + len(STOP_SEQUENCE):]

    stops_at_end = STOP_SEQUENCE in result.raw_text or result.matched_stop_sequence == STOP_SEQUENCE
    topic_keywords_found = find_topic_keywords(answer, keywords)
    repeated_3gram = first_repeated_3gram(answer)

    return PromptEval(
        prompt=prompt,
        answer=answer,
        after_end=normalize(after_end),
        on_topic=len(topic_keywords_found) > 0,
        topic_keywords_found=topic_keywords_found,
        stops_at_end=stops_at_end,
        clean_after_end=is_clean_after_end(after_end),
        no_repeated_3gram=repeated_3gram is None,
        repeated_3gram=repeated_3gram,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 18H chat-only pretraining.")
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
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    device = pick_device(force_cpu=args.cpu)
    model, bpe, model_cfg = load_model_and_tokenizer(checkpoint, device, cfg)

    summary = EvalSummary()
    for prompt, keywords in PROMPTS:
        result = generate(
            model,
            bpe,
            prompt,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
            stop_sequence=STOP_SEQUENCE,
            stop_on_eos_token=True,
            stop_on_sentence_end=False,
            repetition_penalty=1.1,
            presence_penalty=0.0,
            device=device,
        )
        summary.results.append(evaluate_prompt(bpe, prompt, keywords, result))

    print("Phase 18H: chat-only pretraining eval")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Config:     {model_cfg.config_name}")
    print(f"Device:     {device}")
    for r in summary.results:
        print("-" * 60)
        print(f"Prompt:               {r.prompt}")
        print(f"Answer:               {r.answer}")
        print(f"on_topic:             {r.on_topic} {r.topic_keywords_found}")
        print(f"stops_at_end:         {r.stops_at_end}")
        print(f"clean_after_end:      {r.clean_after_end}")
        if not r.clean_after_end:
            print(f"  after <END>:        {r.after_end[:120]!r}")
        print(f"no_repeated_3gram:    {r.no_repeated_3gram}")
        if r.repeated_3gram:
            print(f"  repeated 3-gram:    {r.repeated_3gram!r}")
        print(f"result:               {'PASS' if r.passed else 'FAIL'}")
    print("=" * 60)
    final = "PASS" if summary.all_passed else "FAIL"
    passed_count = sum(1 for r in summary.results if r.passed)
    print(f"final_result:         {final} ({passed_count}/{len(summary.results)})")

    if not summary.all_passed and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
