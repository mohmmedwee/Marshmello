#!/usr/bin/env python3
"""Boost teacher short data for weak routing domains (databases, transformers_llms)."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
CORE_BUILD = PROJECT_ROOT / "18J_marshmello_core_sft" / "build_marshmello_core_data.py"
BASE_SHORT = PHASE_ROOT / "data" / "teacher_extended_short.jsonl"
OUTPUT = PHASE_ROOT / "data" / "teacher_extended_short.jsonl"
REPORT = PHASE_ROOT / "reports" / "teacher_weak_boost_report.json"

BOOST_DOMAINS = frozenset({"databases", "transformers_llms"})
MIN_WORDS = 12
MAX_WORDS = 35

FILLER_PHRASES = (
    "It helps the system answer or act more usefully.",
    "It is useful for building models that work beyond one example.",
    "It is part of how language models read prompts and produce answers.",
    "It helps keep data useful, reliable, or fast to access.",
)

COMPARISON_RE = re.compile(
    r"\b("
    r"compare|contrast|different|distinguish|versus|\bvs\b|"
    r"without mixing|not be confused|instead of|operating difference|"
    r"would not be an example|do not confuse|should not be confused|"
    r"how is .+ different|what distinguishes|mixing their roles"
    r")\b",
    re.IGNORECASE,
)

DOMAIN_SIGNATURES: dict[str, tuple[str, ...]] = {
    "databases": (
        "database index",
        "database",
        "sql",
        "primary key",
        "foreign key",
        "b-tree",
        "table scan",
        "transaction",
        "normalization",
        "acid",
        "schema",
        "relational",
        "join",
        "query plan",
    ),
    "transformers_llms": (
        "transformer",
        "attention",
        "self-attention",
        "tokenizer",
        "tokenization",
        "embedding",
        "positional encoding",
        "positional embedding",
        "causal mask",
        "context window",
        "language model",
        "vocabulary",
        "softmax",
        "fine-tuning",
        "pretraining",
    ),
    "ai_basics": (
        "machine learning",
        "supervised learning",
        "artificial intelligence",
        "neural network",
        "gradient descent",
        "overfitting",
        "underfitting",
        "training data",
        " inference",
    ),
}

EXTRA_PARAPHRASES = (
    "State a precise one-sentence definition of {label}.",
    "In one sentence, define {label}.",
    "Give the shortest accurate explanation of {label}.",
    "For a new learner, what does {alias} mean in one sentence?",
)

SHORT_CANONICAL: tuple[tuple[str, str, str], ...] = (
    ("What is SQL?", "SQL is a language for defining, querying, and modifying data in relational database systems.", "databases"),
    ("Explain database indexes.", "A database index is an auxiliary data structure that speeds row lookup by organizing selected column values.", "databases"),
    ("What is a transformer?", "A transformer is a neural architecture that processes sequences with attention and feed-forward layers.", "transformers_llms"),
    ("What is attention?", "Self-attention lets each token combine information from other tokens in the same sequence.", "transformers_llms"),
    ("What is BPE?", "A BPE tokenizer builds common subword pieces from text and maps them to token IDs.", "transformers_llms"),
    ("What is tokenization?", "Tokenization is the process of converting text into a sequence of token units that a model can encode.", "transformers_llms"),
    ("What is a database?", "A database is an organized system for storing, retrieving, and managing data.", "databases"),
    ("What is a primary key?", "A primary key is a column or column set that uniquely identifies each row in a table.", "databases"),
    ("What is ACID?", "ACID stands for atomicity, consistency, isolation, and durability, four properties associated with reliable database transactions.", "databases"),
)


def load_core_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("marshmello_core_build", CORE_BUILD)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {CORE_BUILD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["marshmello_core_build"] = module
    spec.loader.exec_module(module)
    return module


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


def contains_signature(text: str, terms: tuple[str, ...]) -> bool:
    lowered = f" {text.casefold()} "
    return any(term.strip() and term.casefold() in lowered for term in terms)


def cross_domain_violation(domain: str, instruction: str, response: str) -> str | None:
    if is_comparison(instruction):
        return None
    for other_domain, terms in DOMAIN_SIGNATURES.items():
        if other_domain == domain:
            continue
        if contains_signature(response, terms):
            return other_domain
    return None


def pad_definition(concept) -> str:
    text = first_sentence(concept.definition)
    if word_count(text) >= MIN_WORDS:
        return text
    extra = first_sentence(concept.importance)
    if extra and word_count(extra) <= 12:
        combined = f"{text} {extra}"
        if MIN_WORDS <= word_count(combined) <= MAX_WORDS:
            return combined
    return text


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_boost_rows(core) -> list[dict]:
    concepts = core.concept_map()
    rows: list[dict] = []

    for concept in core.CONCEPTS:
        if concept.domain not in BOOST_DOMAINS:
            continue
        response = pad_definition(concept)
        words = word_count(response)
        if words < MIN_WORDS or words > MAX_WORDS:
            continue
        if len(split_sentences(response)) != 1:
            continue

        templates = [
            ("definition", "normal"),
            ("definition", "paraphrase"),
            ("why", "normal"),
            ("why", "paraphrase"),
            ("how", "normal"),
            ("how", "paraphrase"),
        ]
        for question_type, routing_variant in templates:
            instruction = core.format_question(
                concept,
                question_type,
                routing_variant,
                core.TRAIN_QUESTION_TEMPLATES,
                concepts,
            )
            rows.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "domain": concept.domain,
                    "source": "weak_boost_train_template",
                }
            )

        for template in EXTRA_PARAPHRASES:
            instruction = template.format(label=concept.label, alias=concept.alias)
            rows.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "domain": concept.domain,
                    "source": "weak_boost_paraphrase",
                }
            )

    for instruction, response, domain in SHORT_CANONICAL:
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "domain": domain,
                "source": "weak_boost_canonical",
            }
        )

    return rows


def merge_rows(base: list[dict], boost: list[dict]) -> tuple[list[dict], dict]:
    merged: dict[str, dict] = {}
    stats: Counter[str] = Counter()

    for row in base:
        key = re.sub(r"\s+", " ", row["instruction"].casefold())
        merged[key] = {
            "instruction": row["instruction"].strip(),
            "response": row["response"].strip(),
            "domain": row["domain"],
        }
        stats["base_kept"] += 1

    for row in boost:
        stats["boost_input"] += 1
        instruction = row["instruction"].strip()
        response = row["response"].strip()
        domain = row["domain"]
        words = word_count(response)

        if words < MIN_WORDS or words > MAX_WORDS:
            stats["boost_reject_length"] += 1
            continue
        if len(split_sentences(response)) != 1:
            stats["boost_reject_multisentence"] += 1
            continue
        violation = cross_domain_violation(domain, instruction, response)
        if violation:
            stats[f"boost_reject_cross_{violation}"] += 1
            continue

        key = re.sub(r"\s+", " ", instruction.casefold())
        if key in merged:
            stats["boost_skip_duplicate"] += 1
            continue
        merged[key] = {"instruction": instruction, "response": response, "domain": domain}
        stats["boost_added"] += 1

    return list(merged.values()), dict(stats)


def validate(rows: list[dict]) -> None:
    if len(rows) < 400:
        raise ValueError(f"Too few examples: {len(rows)}")
    counts = Counter(r["domain"] for r in rows)
    for domain in ("databases", "transformers_llms", "ai_basics"):
        if counts[domain] < 80:
            raise ValueError(f"Domain {domain} too small: {counts[domain]}")


def main() -> None:
    if not BASE_SHORT.exists():
        raise FileNotFoundError(f"Missing base dataset: {BASE_SHORT}")

    core = load_core_module()
    base_rows = load_jsonl(BASE_SHORT)
    boost_rows = build_boost_rows(core)
    merged, stats = merge_rows(base_rows, boost_rows)
    validate(merged)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in sorted(merged, key=lambda item: (item["domain"], item["instruction"])):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    domain_counts = Counter(r["domain"] for r in merged)
    meta = {
        "base": str(BASE_SHORT),
        "output": str(OUTPUT),
        "total": len(merged),
        "base_count": len(base_rows),
        "boost_generated": len(boost_rows),
        "merge_stats": stats,
        "domain_counts": dict(domain_counts),
        "eval_leakage": "none — train templates and new paraphrases only",
    }
    REPORT.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Phase 18E: teacher weak-domain boost")
    print("=" * 60)
    print(f"Output: {OUTPUT}")
    print(f"Total:  {len(merged)} (was {len(base_rows)}, +{stats.get('boost_added', 0)} new)")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
