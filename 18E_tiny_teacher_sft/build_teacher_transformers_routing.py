#!/usr/bin/env python3
"""Build transformers-only routing micro-dataset for light fine-tuning."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
OUTPUT = PHASE_ROOT / "data" / "teacher_transformers_routing.jsonl"
REPORT = PHASE_ROOT / "reports" / "teacher_transformers_routing_report.json"

DOMAIN = "transformers_llms"
MIN_WORDS = 12
MAX_WORDS = 25

INDEX_CONTRAST = (
    "Attention mixes token representations by learned relevance; a database index organizes column values to speed row lookup.",
    "Tokenization splits text into model token units; a database index speeds lookups in stored table rows.",
    "A transformer processes token sequences with attention layers; a database index speeds row lookup in tables.",
    "Self-attention weights tokens within one sequence; a database index maps column values to table rows for faster queries.",
    "Multi-head attention combines parallel attention patterns; a database index is a separate structure for database row lookup.",
    "Positional encoding marks token order in a sequence; a database index orders keys to accelerate table access.",
    "A tokenizer maps text to token IDs; a database index maps indexed column values to row locations in storage.",
    "A model vocabulary lists token strings and IDs; a database index lists column keys pointing to table rows.",
    "A context window limits how many tokens a model sees; a database index limits how many table rows a query must scan.",
    "A causal mask blocks future tokens in model attention; a database index does not process language tokens at all.",
    "An attention head computes one attention pattern; a database index is an auxiliary structure for relational queries.",
    "BPE builds subword tokens from text statistics; a database index builds lookup paths over stored relational columns.",
)

CONCEPT_ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "transformer",
        "A transformer is a neural architecture that processes sequences with attention and feed-forward layers.",
        (
            "What is a transformer?",
            "Explain what a transformer is in one sentence.",
            "State a precise one-sentence definition of a transformer.",
            "How is a transformer different from a database index?",
        ),
    ),
    (
        "attention",
        "Attention lets each token combine information from other tokens in the same sequence using learned relevance weights.",
        (
            "What is attention?",
            "Explain attention in transformers in one sentence.",
            "How is attention different from a database index?",
            "Why should attention not be confused with a database index?",
        ),
    ),
    (
        "self_attention",
        "Self-attention lets each token combine information from other tokens in the same sequence.",
        (
            "What is self-attention?",
            "Define self-attention in one sentence.",
            "How is self-attention different from a database index?",
        ),
    ),
    (
        "multi_head_attention",
        "Multi-head attention runs multiple attention heads in parallel and combines their outputs into one representation.",
        (
            "What is multi-head attention?",
            "Explain multi-head attention in one sentence.",
            "How is multi-head attention different from a database index?",
        ),
    ),
    (
        "positional_encoding",
        "Positional encoding adds information about token order to representations processed by a transformer.",
        (
            "What is positional encoding?",
            "Define positional encoding in one sentence.",
            "How is positional encoding different from a database index?",
        ),
    ),
    (
        "tokenizer",
        "A tokenizer is the component that converts text to token IDs and token IDs back to text.",
        (
            "What is a tokenizer?",
            "Explain what a tokenizer does in one sentence.",
            "How is a tokenizer different from a database index?",
        ),
    ),
    (
        "tokenization",
        "Tokenization is the process of converting text into a sequence of token units that a model can encode.",
        (
            "What is tokenization?",
            "Explain tokenization in one sentence.",
            "Why is tokenization not SQL indexing?",
            "How is tokenization different from a database index?",
        ),
    ),
    (
        "bpe",
        "BPE is a subword tokenization method that merges frequent character pairs to build a compact vocabulary.",
        (
            "What is BPE?",
            "Explain BPE tokenization in one sentence.",
            "Why is BPE not SQL indexing?",
            "How is BPE different from a database index?",
        ),
    ),
    (
        "vocabulary",
        "A model vocabulary is the fixed mapping between supported token strings and integer token IDs.",
        (
            "What is a model vocabulary?",
            "Define model vocabulary in one sentence.",
            "How is vocabulary different from a database index?",
        ),
    ),
    (
        "context_window",
        "A context window is the maximum number of tokens a model can process together for one prediction.",
        (
            "What is a context window?",
            "Explain context window in one sentence.",
            "How is a context window different from a database index?",
        ),
    ),
    (
        "causal_mask",
        "A causal mask prevents a language model position from attending to future tokens during sequence processing.",
        (
            "What is a causal mask?",
            "Define causal mask in one sentence.",
            "How is a causal mask different from a database index?",
        ),
    ),
    (
        "attention_head",
        "An attention head is one set of query, key, and value projections that computes an attention pattern.",
        (
            "What is an attention head?",
            "Explain an attention head in one sentence.",
            "How is an attention head different from a database index?",
        ),
    ),
)

COMPARISON_RE = re.compile(
    r"\b("
    r"different|distinguish|versus|\bvs\b|not be confused|instead of|"
    r"how is .+ different|why is .+ not|why should .+ not"
    r")\b",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def is_comparison(instruction: str) -> bool:
    return bool(COMPARISON_RE.search(instruction))


def response_for(instruction: str, definition: str, contrast_idx: int) -> str:
    if is_comparison(instruction):
        return INDEX_CONTRAST[contrast_idx % len(INDEX_CONTRAST)]
    return definition


def build_rows() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    contrast_idx = 0

    for concept, definition, instructions in CONCEPT_ROWS:
        words = word_count(definition)
        if not MIN_WORDS <= words <= MAX_WORDS:
            raise ValueError(f"{concept} definition out of range: {words} words")
        if len(split_sentences(definition)) != 1:
            raise ValueError(f"{concept} definition must be one sentence")

        for instruction in instructions:
            response = response_for(instruction, definition, contrast_idx)
            if is_comparison(instruction):
                contrast_idx += 1
            w = word_count(response)
            if not MIN_WORDS <= w <= MAX_WORDS:
                raise ValueError(f"{concept} response out of range ({w}): {response}")
            if len(split_sentences(response)) != 1:
                raise ValueError(f"{concept} response must be one sentence: {response}")

            key = re.sub(r"\s+", " ", instruction.casefold())
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "domain": DOMAIN,
                    "concept": concept,
                }
            )

    return rows


def validate(rows: list[dict]) -> None:
    concepts = {row["concept"] for row in rows}
    required = {
        "transformer",
        "attention",
        "self_attention",
        "multi_head_attention",
        "positional_encoding",
        "tokenizer",
        "tokenization",
        "bpe",
        "vocabulary",
        "context_window",
        "causal_mask",
        "attention_head",
    }
    missing = required - concepts
    if missing:
        raise ValueError(f"Missing concepts: {sorted(missing)}")
    if not any("database index" in row["instruction"].casefold() for row in rows):
        raise ValueError("Missing database index contrast instructions")
    if not any("not sql indexing" in row["instruction"].casefold() for row in rows):
        raise ValueError("Missing SQL indexing contrast instructions")


def main() -> None:
    rows = build_rows()
    validate(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    word_counts = [word_count(row["response"]) for row in rows]
    meta = {
        "output": str(OUTPUT),
        "total": len(rows),
        "domain": DOMAIN,
        "word_count_min": min(word_counts),
        "word_count_max": max(word_counts),
        "word_count_mean": round(sum(word_counts) / len(word_counts), 2),
        "by_concept": dict(Counter(row["concept"] for row in rows)),
        "comparison_count": sum(1 for row in rows if is_comparison(row["instruction"])),
    }
    REPORT.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Phase 18E: teacher_transformers_routing")
    print("=" * 60)
    print(f"Output: {OUTPUT}")
    print(f"Rows:   {len(rows)}")
    print(f"Words:  {meta['word_count_min']}-{meta['word_count_max']} (mean {meta['word_count_mean']})")
    print(f"Contrast vs database index: {meta['comparison_count']}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
