#!/usr/bin/env python3
"""
Phase 18A: Build a larger local pretraining corpus for Marshmello-45M-Base-v2.

No internet is required. The script mixes existing Phase 13 text with generated
technical documentation paragraphs and writes 13_gpt_pretraining/data/corpus.txt.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
PHASE14_ROOT = PROJECT_ROOT / "14_dataset_pipeline"
CORPUS_PATH = PHASE13_ROOT / "data" / "corpus.txt"
REPORT_PATH = PHASE_ROOT / "reports" / "corpus_report.json"

sys.path.insert(0, str(PHASE14_ROOT))

try:
    from scripts.common import estimate_bpe_tokens  # type: ignore
except Exception:
    estimate_bpe_tokens = None


DOMAIN_TOPICS: dict[str, list[str]] = {
    "software_engineering": [
        "code review",
        "dependency injection",
        "unit testing",
        "observability",
        "API design",
        "refactoring",
        "continuous integration",
        "service boundaries",
    ],
    "databases": [
        "indexes",
        "transactions",
        "query planning",
        "schema migrations",
        "replication",
        "normalization",
        "partitioning",
        "backup recovery",
    ],
    "ai_ml": [
        "transformers",
        "embeddings",
        "gradient descent",
        "validation splits",
        "tokenization",
        "attention",
        "fine-tuning",
        "evaluation metrics",
    ],
    "cybersecurity": [
        "least privilege",
        "phishing defense",
        "password hashing",
        "encryption",
        "threat modeling",
        "audit logging",
        "vulnerability patching",
        "incident response",
    ],
    "python_api": [
        "Python packages",
        "REST endpoints",
        "request validation",
        "async workers",
        "error handling",
        "configuration",
        "serialization",
        "client SDKs",
    ],
}

TEMPLATES: list[str] = [
    (
        "{topic_title} is important in {domain_text} because production systems "
        "need predictable behavior, clear failure modes, and maintainable code. "
        "A practical implementation starts with a small interface, adds tests for "
        "common and edge cases, and documents the assumptions that future engineers "
        "must preserve."
    ),
    (
        "When teams work on {topic}, they should measure the result instead of "
        "trusting intuition. Useful measurements include latency, correctness, "
        "resource usage, error rates, and the cost of operational maintenance. "
        "These signals turn architectural debates into concrete tradeoffs."
    ),
    (
        "A common mistake with {topic} is optimizing the visible syntax while "
        "ignoring the data flow. Good technical design traces inputs, outputs, "
        "state changes, and ownership. That view makes it easier to test the "
        "system and to debug failures under load."
    ),
    (
        "Documentation for {topic} should explain the problem, the chosen design, "
        "alternatives that were rejected, and the risks that remain. Short examples "
        "help readers connect the concept to real code, real queries, real models, "
        "or real operations."
    ),
    (
        "Reliable {topic} work depends on feedback loops. Engineers make a small "
        "change, run focused checks, inspect metrics, and then decide whether to "
        "continue, roll back, or redesign. This habit is more useful than large "
        "untested rewrites."
    ),
]

SPECIAL_TOKEN_NOTE = """
Instruction tuning data later uses literal tags such as <USER>, <ASSISTANT>,
and <END>. A base tokenizer should see these characters during pretraining so
fine-tuning can encode chat-formatted examples without unknown symbols.
"""


@dataclass(frozen=True)
class Paragraph:
    domain: str
    text: str


def normalize_paragraph(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def estimated_tokens(text: str) -> int:
    if estimate_bpe_tokens is not None:
        return int(estimate_bpe_tokens(word_count(text)))
    return int(word_count(text) * 1.35)


def split_existing_corpus(path: Path) -> list[Paragraph]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    paragraphs = []
    for raw in re.split(r"\n\s*\n", text):
        paragraph = normalize_paragraph(raw)
        if paragraph:
            paragraphs.append(Paragraph("existing_phase13", paragraph))
    return paragraphs


def generated_paragraphs(cycle: int) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    for domain, topics in DOMAIN_TOPICS.items():
        domain_text = domain.replace("_", " ")
        for topic in topics:
            topic_title = topic[:1].upper() + topic[1:]
            for idx, template in enumerate(TEMPLATES):
                text = template.format(
                    topic=topic,
                    topic_title=topic_title,
                    domain_text=domain_text,
                )
                text += (
                    f" Example note {cycle}-{idx}: teams should write this guidance "
                    "in plain language, connect it to tests or metrics, and revisit it "
                    "when the system or dataset changes."
                )
                paragraphs.append(Paragraph(domain, normalize_paragraph(text)))
    paragraphs.append(Paragraph("instruction_format", normalize_paragraph(SPECIAL_TOKEN_NOTE)))
    return paragraphs


def build_corpus(target_words: int, seed: int) -> tuple[list[Paragraph], dict[str, object]]:
    rng = random.Random(seed)
    candidates = split_existing_corpus(CORPUS_PATH)
    source_candidates = len(candidates)
    seen: set[str] = {p.text.casefold() for p in candidates}

    total_words = sum(word_count(p.text) for p in candidates)
    cycle = 0
    duplicate_candidates = 0

    while total_words < target_words:
        for paragraph in generated_paragraphs(cycle):
            key = paragraph.text.casefold()
            if key in seen:
                duplicate_candidates += 1
                continue
            seen.add(key)
            candidates.append(paragraph)
            total_words += word_count(paragraph.text)
            if total_words >= target_words:
                break
        cycle += 1

    rng.shuffle(candidates)
    input_candidates = source_candidates + cycle * (
        len(DOMAIN_TOPICS) * len(next(iter(DOMAIN_TOPICS.values()))) * len(TEMPLATES) + 1
    )
    duplicate_ratio = duplicate_candidates / max(input_candidates, 1)
    report = corpus_report(candidates, duplicate_ratio)
    report["target_words"] = target_words
    report["generated_cycles"] = cycle
    return candidates, report


def corpus_report(paragraphs: list[Paragraph], duplicate_ratio: float) -> dict[str, object]:
    text = "\n\n".join(p.text for p in paragraphs)
    domain_counts = Counter(p.domain for p in paragraphs)
    total_words = word_count(text)
    return {
        "total_words": total_words,
        "estimated_bpe_tokens": estimated_tokens(text),
        "domain_distribution": dict(sorted(domain_counts.items())),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "unique_paragraph_count": len({p.text.casefold() for p in paragraphs}),
    }


def write_outputs(paragraphs: list[Paragraph], report: dict[str, object]) -> None:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text("\n\n".join(p.text for p in paragraphs) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_summary(report: dict[str, object]) -> None:
    print("Phase 18A: Large-scale base pretraining corpus")
    print("=" * 60)
    print(f"Corpus path:              {CORPUS_PATH}")
    print(f"Report path:              {REPORT_PATH}")
    print(f"Total words:              {report['total_words']:,}")
    print(f"Estimated BPE tokens:     {report['estimated_bpe_tokens']:,}")
    print(f"Unique paragraphs:        {report['unique_paragraph_count']:,}")
    print(f"Duplicate ratio:          {report['duplicate_ratio']}")
    print("\nDomain distribution:")
    for domain, count in dict(report["domain_distribution"]).items():
        print(f"  {domain}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Marshmello-45M-Base-v2 corpus.")
    parser.add_argument("--target-words", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paragraphs, report = build_corpus(target_words=args.target_words, seed=args.seed)
    write_outputs(paragraphs, report)
    print_summary(report)


if __name__ == "__main__":
    main()
