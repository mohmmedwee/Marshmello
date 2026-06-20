#!/usr/bin/env python3
"""
Evaluate question-to-answer routing after the Phase 18I routing fix.

The model can already produce short answers and stop at ``<END>``; the failure
this phase targets is *routing*: "What is AI?" coming back with a tokenizer or
database answer. So each prompt is checked two ways:

    - on topic: the answer mentions at least one keyword for the right concept
    - correctly routed: the answer contains none of the competing concept's
      signature terms (an AI answer must not mention tokenizer, BPE, SQL, or
      database)

A prompt fails if it is off topic, mis-routed, too long, or loops.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE18B_ROOT = PROJECT_ROOT / "18B_marshmello_instruct"
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE18B_ROOT))
sys.path.insert(0, str(PHASE13_ROOT))

from chat import load_model  # noqa: E402
from train_instruct import (  # noqa: E402
    ROUTING_LATEST_CHECKPOINT,
    generate_assistant_reply,
    repeated_ngram,
)
from training.trainer import pick_device  # noqa: E402

DEFAULT_CHECKPOINT = ROUTING_LATEST_CHECKPOINT


@dataclass(frozen=True)
class PromptSpec:
    prompt: str
    expected: tuple[str, ...]
    forbidden: tuple[str, ...]


# Forbidden = signature terms of competing concepts. The AI prompts encode the
# exact requirement: fail if an AI answer contains tokenizer, BPE, SQL, database.
AI_FORBIDDEN = ("tokenizer", "token", "bpe", "subword", "sql", "database", "index", "attention")

PROMPTS: tuple[PromptSpec, ...] = (
    PromptSpec(
        "What is AI?",
        expected=("ai", "artificial", "intelligence", "computer", "machine", "reason", "software"),
        forbidden=AI_FORBIDDEN,
    ),
    PromptSpec(
        "Define artificial intelligence.",
        expected=("ai", "artificial", "intelligence", "computer", "machine", "reason", "software"),
        forbidden=AI_FORBIDDEN,
    ),
    PromptSpec(
        "Who are you?",
        expected=("marshmello", "assistant", "answer", "helper", "small"),
        forbidden=("tokenizer", "bpe", "sql", "database", "index", "attention"),
    ),
    PromptSpec(
        "What is attention?",
        expected=("attention", "tokens", "token", "focus", "weigh", "model", "relevant"),
        forbidden=("tokenizer", "bpe", "sql", "database", "index"),
    ),
    PromptSpec(
        "Explain database indexes.",
        expected=("index", "indexes", "database", "rows", "query", "queries", "table", "faster"),
        forbidden=("attention", "tokenizer", "bpe", "subword", "artificial intelligence"),
    ),
    PromptSpec(
        "What is a tokenizer?",
        expected=("tokenizer", "token", "tokens", "text", "subword", "pieces", "bpe"),
        forbidden=("attention", "database", "sql", "artificial intelligence"),
    ),
)


# Concept nouns that, when an answer *starts* with them followed immediately by
# a bare article, signal a dropped verb (e.g. "Attention a model ...").
MALFORMED_START_NOUNS = (
    "attention", "self-attention", "tokenizer", "tokenization", "ai", "index", "database"
)
REPEATED_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
MALFORMED_START_RE = re.compile(
    rf"^\s*(?:{'|'.join(MALFORMED_START_NOUNS)})\s+(?:a|an|the)\b",
    re.IGNORECASE,
)


@dataclass
class PromptResult:
    prompt: str
    answer: str
    on_topic: bool
    matched: list[str]
    routed_ok: bool
    leaked: list[str]
    short_ok: bool
    no_loop: bool
    grammar_ok: bool
    grammar_issue: str | None

    @property
    def passed(self) -> bool:
        return (
            self.on_topic
            and self.routed_ok
            and self.short_ok
            and self.no_loop
            and self.grammar_ok
        )


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def contains_term(text: str, term: str) -> bool:
    lower = text.casefold()
    if " " in term:
        return term in lower
    return re.search(rf"\b{re.escape(term)}\b", lower) is not None


def grammar_issue(answer: str) -> str | None:
    """Return a short reason if the answer is grammatically malformed, else None."""
    repeated = REPEATED_WORD_RE.search(answer)
    if repeated:
        return f"repeated word {repeated.group(1)!r}"
    if MALFORMED_START_RE.search(answer):
        return "malformed phrase (noun followed by bare article, missing verb)"
    return None


def evaluate(spec: PromptSpec, answer: str, *, min_keywords: int) -> PromptResult:
    matched = [k for k in spec.expected if contains_term(answer, k)]
    leaked = [k for k in spec.forbidden if contains_term(answer, k)]
    words = word_count(answer)
    issue = grammar_issue(answer)
    return PromptResult(
        prompt=spec.prompt,
        answer=answer,
        on_topic=len(matched) >= min_keywords,
        matched=matched,
        routed_ok=not leaked,
        leaked=leaked,
        short_ok=4 <= words <= 70,
        no_loop=repeated_ngram(answer, n=3) is None,
        grammar_ok=issue is None,
        grammar_issue=issue,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 18I routing fix.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default="large_50m")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--min-keywords", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    if args.min_keywords < 1:
        raise ValueError("--min-keywords must be >= 1")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train the routing checkpoint first:\n"
            "  python 18B_marshmello_instruct/train_instruct.py --mode routing"
        )

    device = pick_device(force_cpu=args.cpu)
    model, bpe = load_model(args.checkpoint, args.config, device)

    all_ok = True
    print("Phase 18I: routing-fix eval")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    for spec in PROMPTS:
        answer = generate_assistant_reply(
            model,
            bpe,
            spec.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.3,
        )
        result = evaluate(spec, answer, min_keywords=args.min_keywords)
        all_ok = all_ok and result.passed
        failures = []
        if not result.on_topic:
            failures.append(f"off topic (matched {result.matched})")
        if not result.routed_ok:
            failures.append(f"mis-routed, leaked {result.leaked}")
        if not result.short_ok:
            failures.append("not short")
        if not result.no_loop:
            failures.append("repeated 3-gram")
        if not result.grammar_ok:
            failures.append(result.grammar_issue or "bad grammar")
        print("-" * 60)
        print(f"Prompt:   {spec.prompt}")
        print(f"Answer:   {result.answer}")
        print(f"on_topic: {result.on_topic} {result.matched}")
        print(f"routed:   {result.routed_ok}" + (f" leaked={result.leaked}" if result.leaked else ""))
        print(f"grammar:  {result.grammar_ok}" + (f" ({result.grammar_issue})" if result.grammar_issue else ""))
        print(f"Result:   {'PASS' if result.passed else 'FAIL'}"
              + ("" if result.passed else f" - {', '.join(failures)}"))

    print("=" * 60)
    print(f"Final: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
