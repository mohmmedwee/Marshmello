#!/usr/bin/env python3
"""
Phase 17: Instruction dataset pipeline.

Converts raw instruction/response/domain JSONL into cleaned instruction pairs
for Marshmello-45M-Instruct, plus a chat-formatted export used by SFT.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PHASE_ROOT = Path(__file__).resolve().parent
SEED_RAW_PATH = PHASE_ROOT / "raw" / "seed_instructions.jsonl"
HF_IMPORTED_PATH = PHASE_ROOT / "raw" / "hf_imported.jsonl"
RAW_PATH = HF_IMPORTED_PATH
PROCESSED_DIR = PHASE_ROOT / "processed"
REPORTS_DIR = PHASE_ROOT / "reports"
INSTRUCTIONS_PATH = PROCESSED_DIR / "instructions.jsonl"
CHAT_PATH = PROCESSED_DIR / "chat.jsonl"
STATS_PATH = REPORTS_DIR / "instruction_stats.json"

USER_TAG = "<USER>"
ASSISTANT_TAG = "<ASSISTANT>"
END_TAG = "<END>"

DOMAINS = {
    "software_engineering",
    "databases",
    "ai",
    "cybersecurity",
    "general",
}


@dataclass(frozen=True)
class InstructionPair:
    instruction: str
    response: str
    domain: str
    source: str = "unknown"


def normalize_text(text: str) -> str:
    """Collapse whitespace and trim outer space."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_dedupe(text: str) -> str:
    """Case-insensitive normalized key for exact duplicate detection."""
    return normalize_text(text).casefold()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must be a JSON object")
            yield record


def parse_pair(record: dict[str, object]) -> InstructionPair | None:
    instruction = normalize_text(str(record.get("instruction", "")))
    response = normalize_text(str(record.get("response", "")))
    domain = normalize_text(str(record.get("domain", "")))
    source = normalize_text(str(record.get("source", "unknown"))) or "unknown"

    if not instruction or not response:
        return None
    if domain not in DOMAINS:
        return None
    return InstructionPair(
        instruction=instruction,
        response=response,
        domain=domain,
        source=source,
    )


def clean_pairs(
    records: Iterable[dict[str, object]],
    *,
    min_response_words: int,
) -> tuple[list[InstructionPair], dict[str, int]]:
    """Clean, domain-check, dedupe instructions, dedupe responses, and filter shorts."""
    kept: list[InstructionPair] = []
    seen_instructions: set[str] = set()
    seen_responses: set[str] = set()
    stats = Counter(
        input_records=0,
        invalid_records=0,
        short_responses_removed=0,
        duplicate_instructions_removed=0,
        duplicate_responses_removed=0,
    )

    for record in records:
        stats["input_records"] += 1
        pair = parse_pair(record)
        if pair is None:
            stats["invalid_records"] += 1
            continue

        if word_count(pair.response) < min_response_words:
            stats["short_responses_removed"] += 1
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

    stats["output_pairs"] = len(kept)
    return kept, dict(stats)


def format_chat(pair: InstructionPair) -> str:
    return f"{USER_TAG}\n{pair.instruction}\n{ASSISTANT_TAG}\n{pair.response}\n{END_TAG}"


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def dataset_statistics(pairs: list[InstructionPair]) -> dict[str, object]:
    total_response_words = sum(word_count(pair.response) for pair in pairs)
    domain_counts = Counter(pair.domain for pair in pairs)
    source_counts = Counter(pair.source for pair in pairs)
    total = len(pairs)
    return {
        "total_pairs": total,
        "average_response_length_words": round(total_response_words / total, 2) if total else 0.0,
        "domain_distribution": dict(sorted(domain_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }


def pair_output_record(pair: InstructionPair) -> dict[str, str]:
    """Required training JSONL schema; source stays in reports only."""
    return {
        "instruction": pair.instruction,
        "response": pair.response,
        "domain": pair.domain,
    }


def run_pipeline(
    *,
    input_path: Path,
    output_path: Path,
    chat_path: Path,
    stats_path: Path,
    min_response_words: int,
) -> dict[str, object]:
    pairs, cleaning_stats = clean_pairs(
        read_jsonl(input_path),
        min_response_words=min_response_words,
    )
    write_jsonl(output_path, (pair_output_record(pair) for pair in pairs))
    write_jsonl(
        chat_path,
        ({"text": format_chat(pair), "domain": pair.domain, "source": pair.source} for pair in pairs),
    )

    stats = dataset_statistics(pairs)
    report = {
        **stats,
        "cleaning": cleaning_stats,
        "outputs": {
            "instructions_jsonl": str(output_path),
            "chat_jsonl": str(chat_path),
        },
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def print_summary(report: dict[str, object]) -> None:
    print("Phase 17: Instruction dataset pipeline")
    print("=" * 60)
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Average response length: {report['average_response_length_words']} words")
    print("\nDomain distribution:")
    for domain, count in dict(report["domain_distribution"]).items():
        print(f"  {domain}: {count}")
    print("\nSource counts:")
    for source, count in dict(report["source_counts"]).items():
        print(f"  {source}: {count}")
    print("\nCleaning:")
    for key, value in dict(report["cleaning"]).items():
        print(f"  {key}: {value}")
    print("\nOutputs:")
    for key, value in dict(report["outputs"]).items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 17 instruction dataset files.")
    default_input = HF_IMPORTED_PATH if HF_IMPORTED_PATH.exists() else SEED_RAW_PATH
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=INSTRUCTIONS_PATH)
    parser.add_argument("--chat-output", type=Path, default=CHAT_PATH)
    parser.add_argument("--stats-output", type=Path, default=STATS_PATH)
    parser.add_argument("--min-response-words", type=int, default=6)
    args = parser.parse_args()

    report = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        chat_path=args.chat_output,
        stats_path=args.stats_output,
        min_response_words=args.min_response_words,
    )
    print_summary(report)


if __name__ == "__main__":
    main()
