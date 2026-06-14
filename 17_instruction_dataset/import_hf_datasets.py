#!/usr/bin/env python3
"""
Phase 17B: Online Hugging Face instruction dataset import.

Downloads public instruction datasets, normalizes their schemas, filters noisy
pairs, dedupes globally, and writes raw/hf_imported.jsonl for Phase 17.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PHASE_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PHASE_ROOT / "raw" / "hf_imported.jsonl"

DATASETS = (
    "tatsu-lab/alpaca",
    "databricks/databricks-dolly-15k",
    "sahil2801/CodeAlpaca-20k",
)

DOMAINS = {
    "software_engineering",
    "databases",
    "ai",
    "cybersecurity",
    "general",
}

SOFTWARE_KEYWORDS = {
    "api",
    "class",
    "code",
    "debug",
    "function",
    "git",
    "javascript",
    "python",
    "refactor",
    "software",
    "test",
    "typescript",
}
DATABASE_KEYWORDS = {
    "database",
    "index",
    "join",
    "query",
    "sql",
    "table",
    "transaction",
}
AI_KEYWORDS = {
    "ai",
    "attention",
    "embedding",
    "language model",
    "machine learning",
    "model",
    "neural",
    "token",
    "transformer",
}
CYBERSECURITY_KEYWORDS = {
    "authentication",
    "cryptography",
    "cybersecurity",
    "encryption",
    "exploit",
    "malware",
    "password",
    "phishing",
    "security",
    "vulnerability",
}


@dataclass(frozen=True)
class ImportedPair:
    instruction: str
    response: str
    domain: str
    source: str


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_for_dedupe(text: str) -> str:
    return normalize_text(text).casefold()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def infer_domain(text: str, source: str) -> str:
    """Heuristic domain tagger for imported instruction data."""
    lowered = text.casefold()
    if "codealpaca" in source.casefold():
        return "software_engineering"

    def count_keywords(keywords: set[str]) -> int:
        count = 0
        for keyword in keywords:
            pattern = r"(?<!\w)" + re.escape(keyword.casefold()) + r"(?!\w)"
            count += int(re.search(pattern, lowered) is not None)
        return count

    scores = {
        "software_engineering": count_keywords(SOFTWARE_KEYWORDS),
        "databases": count_keywords(DATABASE_KEYWORDS),
        "ai": count_keywords(AI_KEYWORDS),
        "cybersecurity": count_keywords(CYBERSECURITY_KEYWORDS),
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score > 0 else "general"


def join_instruction(instruction: object, input_text: object = "") -> str:
    instruction_text = normalize_text(instruction)
    input_part = normalize_text(input_text)
    if input_part:
        return f"{instruction_text}\n\nInput:\n{input_part}"
    return instruction_text


def normalize_record(record: dict[str, object], source: str) -> ImportedPair | None:
    """Normalize supported HF dataset schemas to instruction/response/domain/source."""
    if source == "databricks/databricks-dolly-15k":
        instruction = join_instruction(record.get("instruction"), record.get("context"))
        response = normalize_text(record.get("response"))
    else:
        instruction = join_instruction(record.get("instruction"), record.get("input"))
        response = normalize_text(record.get("output"))

    if not instruction or not response:
        return None

    domain = infer_domain(f"{instruction} {response}", source)
    if domain not in DOMAINS:
        domain = "general"
    return ImportedPair(
        instruction=instruction,
        response=response,
        domain=domain,
        source=source,
    )


def filter_and_dedupe(
    pairs: Iterable[ImportedPair],
    *,
    min_response_words: int,
    max_response_words: int,
    max_examples: int,
) -> tuple[list[ImportedPair], dict[str, object]]:
    kept: list[ImportedPair] = []
    seen_instructions: set[str] = set()
    seen_responses: set[str] = set()
    stats = Counter(
        input_pairs=0,
        empty_removed=0,
        short_responses_removed=0,
        long_responses_removed=0,
        duplicate_instructions_removed=0,
        duplicate_responses_removed=0,
    )
    source_counts: Counter[str] = Counter()

    for pair in pairs:
        stats["input_pairs"] += 1
        if not pair.instruction or not pair.response:
            stats["empty_removed"] += 1
            continue

        response_words = word_count(pair.response)
        if response_words < min_response_words:
            stats["short_responses_removed"] += 1
            continue
        if response_words > max_response_words:
            stats["long_responses_removed"] += 1
            continue

        instruction_key = normalize_for_dedupe(pair.instruction)
        if instruction_key in seen_instructions:
            stats["duplicate_instructions_removed"] += 1
            continue

        response_key = normalize_for_dedupe(pair.response)
        if response_key in seen_responses:
            stats["duplicate_responses_removed"] += 1
            continue

        seen_instructions.add(instruction_key)
        seen_responses.add(response_key)
        kept.append(pair)
        source_counts[pair.source] += 1
        if len(kept) >= max_examples:
            break

    stats["output_pairs"] = len(kept)
    return kept, {**dict(stats), "source_counts": dict(sorted(source_counts.items()))}


def iter_hf_records(dataset_names: Iterable[str]) -> Iterable[ImportedPair]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets`."
        ) from exc

    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...")
        dataset = load_dataset(dataset_name, split="train")
        for record in dataset:
            pair = normalize_record(dict(record), dataset_name)
            if pair is not None:
                yield pair


def write_jsonl(path: Path, pairs: Iterable[ImportedPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def import_datasets(
    *,
    output_path: Path,
    max_examples: int,
    min_response_words: int,
    max_response_words: int,
    dataset_names: Iterable[str] = DATASETS,
) -> dict[str, object]:
    pairs, stats = filter_and_dedupe(
        iter_hf_records(dataset_names),
        min_response_words=min_response_words,
        max_response_words=max_response_words,
        max_examples=max_examples,
    )
    write_jsonl(output_path, pairs)
    return stats


def print_summary(stats: dict[str, object], output_path: Path) -> None:
    print("\nPhase 17B: Hugging Face import")
    print("=" * 60)
    print(f"Output pairs: {stats['output_pairs']}")
    print(f"Output: {output_path}")
    print("\nSource counts:")
    for source, count in dict(stats["source_counts"]).items():
        print(f"  {source}: {count}")
    print("\nFilters:")
    for key, value in stats.items():
        if key != "source_counts":
            print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import HF instruction datasets for Phase 17.")
    parser.add_argument("--max-examples", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--min-response-words", type=int, default=10)
    parser.add_argument("--max-response-words", type=int, default=300)
    args = parser.parse_args()

    stats = import_datasets(
        output_path=args.output,
        max_examples=args.max_examples,
        min_response_words=args.min_response_words,
        max_response_words=args.max_response_words,
    )
    print_summary(stats, args.output)


if __name__ == "__main__":
    main()
