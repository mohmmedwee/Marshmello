#!/usr/bin/env python3
"""Build single-sentence teacher dataset from teacher_extended.jsonl."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
INPUT = PHASE_ROOT / "data" / "teacher_extended.jsonl"
OUTPUT = PHASE_ROOT / "data" / "teacher_extended_short.jsonl"
REPORT = PHASE_ROOT / "reports" / "teacher_extended_short_report.json"

FILLER_PHRASES = (
    "It helps the system answer or act more usefully.",
    "It is useful for building models that work beyond one example.",
    "It is part of how language models read prompts and produce answers.",
    "It helps keep data useful, reliable, or fast to access.",
    "It helps teams build software that is easier to change and operate.",
    "It helps Python code stay clear and practical.",
)

MIN_WORDS = 12
MAX_WORDS = 35

COMPARISON_RE = re.compile(
    r"\b("
    r"compare|contrast|different|distinguish|versus|\bvs\b|"
    r"without mixing|not be confused|instead of|operating difference|"
    r"would not be an example|do not confuse|should not be confused|"
    r"how is .+ different|what distinguishes|mixing their roles"
    r")\b",
    re.IGNORECASE,
)

CORE_DOMAINS = frozenset({"ai_basics", "databases", "transformers_llms"})

DOMAIN_SIGNATURES: dict[str, tuple[str, ...]] = {
    "databases": (
        "database index",
        "database",
        "sql",
        "primary key",
        "foreign key",
        "b-tree",
        "b tree",
        "table scan",
        "query plan",
        "transaction",
        "normalization",
        "denormalization",
        "acid",
        "schema",
        "relational",
        "row",
        "column",
        "join",
        "select ",
        "where clause",
        "group by",
        "index lookup",
        "full table scan",
        "hash index",
        "composite index",
        "deadlock",
        "nosql",
        "replication",
        "migration",
    ),
    "transformers_llms": (
        "transformer",
        "attention",
        "self-attention",
        "self attention",
        "tokenizer",
        "tokenization",
        "tokenize",
        " bpe",
        "byte pair",
        "embedding",
        "positional encoding",
        "positional embedding",
        "causal mask",
        "decoder-only",
        "context window",
        "softmax",
        "logits",
        "greedy decoding",
        "top-k",
        "temperature",
        "next token",
        "language model",
        "vocabulary",
        "subword",
    ),
    "ai_basics": (
        "machine learning",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "neural network",
        "gradient descent",
        "overfitting",
        "underfitting",
        "classification",
        "regression",
        "training data",
        "validation set",
        "test set",
        "artificial intelligence",
        " inference",
        "loss function",
        "precision",
        "recall",
        "f1 score",
        "bias-variance",
        "generalization",
        "model evaluation",
    ),
}


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def clean_filler(text: str) -> str:
    out = text.strip()
    for phrase in FILLER_PHRASES:
        out = out.replace(phrase, " ")
    return re.sub(r"\s+", " ", out).strip()


def first_sentence(text: str) -> str:
    sentences = split_sentences(clean_filler(text))
    return sentences[0] if sentences else clean_filler(text)


def is_comparison(instruction: str) -> bool:
    return bool(COMPARISON_RE.search(instruction))


def normalize_domain(domain: str) -> str | None:
    if domain in CORE_DOMAINS:
        return domain
    if domain in {"machine_learning"}:
        return "ai_basics"
    return None


def contains_signature(text: str, terms: tuple[str, ...]) -> bool:
    lowered = f" {text.casefold()} "
    for term in terms:
        if term.strip() and term.casefold() in lowered:
            return True
    return False


def cross_domain_violation(domain: str, instruction: str, response: str) -> str | None:
    core = normalize_domain(domain)
    if core is None or is_comparison(instruction):
        return None
    for other_domain, terms in DOMAIN_SIGNATURES.items():
        if other_domain == core:
            continue
        if contains_signature(response, terms):
            return other_domain
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_short_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    stats: Counter[str] = Counter()
    seen: set[str] = set()

    for row in rows:
        stats["input"] += 1
        instruction = row["instruction"].strip()
        domain = row.get("domain", "general")
        response = first_sentence(row["response"])
        words = word_count(response)

        if words < MIN_WORDS:
            stats["reject_short"] += 1
            continue
        if words > MAX_WORDS:
            stats["reject_long"] += 1
            continue

        violation = cross_domain_violation(domain, instruction, response)
        if violation:
            stats[f"reject_cross_domain_{violation}"] += 1
            continue

        key = re.sub(r"\s+", " ", instruction.casefold())
        if key in seen:
            stats["reject_duplicate"] += 1
            continue
        seen.add(key)

        kept.append({"instruction": instruction, "response": response, "domain": domain})
        stats["kept"] += 1

    return kept, dict(stats)


def validate(rows: list[dict]) -> None:
    if len(rows) < 400:
        raise ValueError(f"Too few examples after filtering: {len(rows)}")
    counts = Counter(r["domain"] for r in rows)
    for domain in ("databases", "transformers_llms", "ai_basics"):
        if counts[domain] < 80:
            raise ValueError(f"Domain {domain} too small: {counts[domain]}")
    for idx, row in enumerate(rows, start=1):
        w = word_count(row["response"])
        if not MIN_WORDS <= w <= MAX_WORDS:
            raise ValueError(f"Row {idx} word count {w} outside {MIN_WORDS}-{MAX_WORDS}")
        if len(split_sentences(row["response"])) != 1:
            raise ValueError(f"Row {idx} is not single sentence")


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}. Run build_teacher_extended.py first.")

    rows = load_jsonl(INPUT)
    for row in rows:
        if row.get("split") == "held_out":
            raise ValueError("Input must not contain held_out eval rows")

    short_rows, filter_stats = build_short_rows(rows)
    validate(short_rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in short_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    domain_counts = Counter(r["domain"] for r in short_rows)
    word_counts = [word_count(r["response"]) for r in short_rows]
    meta = {
        "input": str(INPUT),
        "output": str(OUTPUT),
        "total": len(short_rows),
        "filter_stats": filter_stats,
        "domain_counts": dict(domain_counts),
        "word_count_min": min(word_counts),
        "word_count_max": max(word_counts),
        "word_count_mean": round(sum(word_counts) / len(word_counts), 2),
        "eval_leakage": "none — derived from teacher_extended (train-only sources)",
    }
    REPORT.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Phase 18E: teacher_extended_short")
    print("=" * 60)
    print(f"Output: {OUTPUT}")
    print(f"Kept:   {len(short_rows)} / {len(rows)}")
    print(f"Words:  {meta['word_count_min']}-{meta['word_count_max']} (mean {meta['word_count_mean']})")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
