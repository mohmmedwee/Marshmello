#!/usr/bin/env python3
"""
Build a chat-only corpus for a short continued-pretraining stage.

Base v2 reaches a good validation loss after the expanded mixed pretraining,
but it still answers chat prompts weakly and then keeps emitting raw corpus
after ``<END>``. This corpus contains nothing but chat turns so a short
continued-pretraining run can strongly teach the boundary format:

    <USER> instruction <ASSISTANT> response <END>

This is not SFT: the output is plain text for next-token prediction on every
token, with no assistant masking. The teacher examples are repeated heavily so
the small, direct-answer behavior dominates the signal.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
DEFAULT_TEACHER_DATA = PROJECT_ROOT / "18E_tiny_teacher_sft" / "data" / "teacher.jsonl"
DEFAULT_INSTRUCTION_DATA = PROJECT_ROOT / "17_instruction_dataset" / "processed" / "chat.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "13_gpt_pretraining" / "data" / "corpus_chat_only.txt"
DEFAULT_REPORT = PHASE_ROOT / "reports" / "chat_only_corpus_report.json"

MIN_RESPONSE_WORDS = 8
MAX_RESPONSE_WORDS = 80
DEFAULT_MAX_INSTRUCTION_WORDS = 80
SCAFFOLDING_TOKENS = ("Input:", "Output:", "Instruction:", "Response:")


@dataclass
class FilterStats:
    total_input_examples: int = 0
    dropped_long_instructions: int = 0
    dropped_scaffolding_instruction: int = 0
    dropped_scaffolding_response: int = 0
    max_instruction_words_seen_input: int = 0
    max_instruction_words_seen_kept: int = 0
    kept_instruction_domains: set[str] | None = None

    def __post_init__(self) -> None:
        if self.kept_instruction_domains is None:
            self.kept_instruction_domains = set()

    def observe_input(self, instruction: str) -> int:
        instruction_words = word_count(instruction)
        self.max_instruction_words_seen_input = max(
            self.max_instruction_words_seen_input, instruction_words
        )
        return instruction_words

    def observe_kept(self, instruction_words: int) -> None:
        self.max_instruction_words_seen_kept = max(
            self.max_instruction_words_seen_kept, instruction_words
        )

    def should_drop(self, instruction: str, response: str, limit: int) -> bool:
        instruction_words = word_count(instruction)
        too_long = instruction_words > limit
        scaffolded_instruction = contains_scaffolding(instruction)
        scaffolded_response = contains_scaffolding(response)
        if too_long:
            self.dropped_long_instructions += 1
        if scaffolded_instruction:
            self.dropped_scaffolding_instruction += 1
        if scaffolded_response:
            self.dropped_scaffolding_response += 1
        return too_long or scaffolded_instruction or scaffolded_response


def normalize(text: str) -> str:
    """Collapse all whitespace so each example is a single clean line."""
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def contains_scaffolding(text: str) -> bool:
    folded = text.casefold()
    return any(token.casefold() in folded for token in SCAFFOLDING_TOKENS)


def format_example(instruction: str, response: str) -> str:
    return f"<USER> {instruction} <ASSISTANT> {response} <END>"


def response_in_range(response: str) -> bool:
    return MIN_RESPONSE_WORDS <= word_count(response) <= MAX_RESPONSE_WORDS


def load_teacher_examples(
    path: Path,
    max_instruction_words: int = DEFAULT_MAX_INSTRUCTION_WORDS,
    stats: FilterStats | None = None,
) -> list[tuple[str, str]]:
    """Load (instruction, response) pairs from teacher.jsonl."""
    stats = stats or FilterStats()
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats.total_input_examples += 1
            record = json.loads(line)
            instruction = normalize(str(record.get("instruction", "")))
            response = normalize(str(record.get("response", "")))
            if not instruction or not response:
                continue
            instruction_words = stats.observe_input(instruction)
            if stats.should_drop(instruction, response, max_instruction_words):
                continue
            if not response_in_range(response):
                continue
            pairs.append((instruction, response))
            stats.observe_kept(instruction_words)
    return pairs


CHAT_PATTERN = re.compile(
    r"<USER>\s*(?P<instruction>.*?)\s*<ASSISTANT>\s*(?P<response>.*?)\s*(?:<END>)?\s*$",
    re.DOTALL,
)


def parse_chat_text(text: str) -> tuple[str, str] | None:
    """Extract (instruction, response) from a chat.jsonl ``text`` field."""
    match = CHAT_PATTERN.search(text)
    if not match:
        return None
    instruction = normalize(match.group("instruction"))
    response = normalize(match.group("response"))
    if not instruction or not response:
        return None
    return instruction, response


def load_instruction_examples(
    path: Path,
    limit: int | None,
    max_instruction_words: int = DEFAULT_MAX_INSTRUCTION_WORDS,
    stats: FilterStats | None = None,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Load, shuffle, and cap clean instruction pairs from chat.jsonl."""
    stats = stats or FilterStats()
    candidates: list[tuple[str, str, str | None]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats.total_input_examples += 1
            record = json.loads(line)
            parsed = parse_chat_text(str(record.get("text", "")))
            if parsed is None:
                continue
            instruction, response = parsed
            stats.observe_input(instruction)
            if stats.should_drop(instruction, response, max_instruction_words):
                continue
            if not response_in_range(response):
                continue
            domain = normalize(str(record.get("domain", ""))) or None
            candidates.append((instruction, response, domain))

    random.Random(seed).shuffle(candidates)
    if limit is not None:
        candidates = candidates[:limit]

    pairs: list[tuple[str, str]] = []
    for instruction, response, domain in candidates:
        pairs.append((instruction, response))
        instruction_words = word_count(instruction)
        if domain is not None:
            assert stats.kept_instruction_domains is not None
            stats.kept_instruction_domains.add(domain)
        stats.observe_kept(instruction_words)
    return pairs


def shuffle_without_consecutive_duplicates(blocks: list[str], seed: int) -> list[str]:
    """Shuffle repeated blocks and arrange them without adjacent duplicates."""
    shuffled = list(blocks)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled

    counts = Counter(shuffled)
    if max(counts.values()) > (len(shuffled) + 1) // 2:
        raise ValueError("Cannot prevent consecutive duplicates with these repeat counts")

    first_position: dict[str, int] = {}
    for index, block in enumerate(shuffled):
        first_position.setdefault(block, index)

    heap = [
        (-count, first_position[block], block)
        for block, count in counts.items()
    ]
    heapq.heapify(heap)
    arranged: list[str] = []
    previous: tuple[int, int, str] | None = None

    while heap:
        count, order, block = heapq.heappop(heap)
        arranged.append(block)
        count += 1
        if previous is not None:
            heapq.heappush(heap, previous)
            previous = None
        if count < 0:
            previous = (count, order, block)

    if previous is not None:
        raise ValueError("Could not arrange blocks without consecutive duplicates")
    return arranged


def build_blocks(
    *,
    teacher_pairs: list[tuple[str, str]],
    instruction_pairs: list[tuple[str, str]],
    teacher_repeat: int,
    instruction_repeat: int,
    seed: int = 42,
) -> list[str]:
    blocks: list[str] = []
    for instruction, response in teacher_pairs:
        block = format_example(instruction, response)
        blocks.extend(block for _ in range(teacher_repeat))
    for instruction, response in instruction_pairs:
        block = format_example(instruction, response)
        blocks.extend(block for _ in range(instruction_repeat))
    return shuffle_without_consecutive_duplicates(blocks, seed)


def render_corpus(blocks: list[str]) -> str:
    return "\n\n".join(blocks).strip() + "\n"


def write_corpus(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def max_consecutive_duplicate_blocks(blocks: list[str]) -> int:
    if not blocks:
        return 0
    longest_run = 1
    current_run = 1
    previous = blocks[0].casefold()
    for block in blocks[1:]:
        normalized = block.casefold()
        if normalized == previous:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
        previous = normalized
    return longest_run


def build_report(
    *,
    output_path: Path,
    written_text: str,
    teacher_pairs: list[tuple[str, str]],
    instruction_pairs: list[tuple[str, str]],
    teacher_repeat: int,
    instruction_repeat: int,
    blocks: list[str],
    filter_stats: FilterStats,
    max_instruction_words: int,
    seed: int,
) -> dict[str, object]:
    teacher_examples = len(teacher_pairs) * teacher_repeat
    instruction_examples = len(instruction_pairs) * instruction_repeat
    total_examples = len(blocks)
    total_words = sum(word_count(block) for block in blocks)
    normalized = [block.casefold() for block in blocks]
    duplicate_blocks = len(normalized) - len(set(normalized))
    max_duplicate_run = max_consecutive_duplicate_blocks(blocks)
    return {
        "output": str(output_path),
        "total_input_examples": filter_stats.total_input_examples,
        "kept_teacher_examples": len(teacher_pairs),
        "kept_instruction_examples": len(instruction_pairs),
        "final_examples": total_examples,
        "total_words": total_words,
        "teacher_examples_after_repeat": teacher_examples,
        "instruction_examples_after_repeat": instruction_examples,
        "teacher_repeat": teacher_repeat,
        "instruction_repeat": instruction_repeat,
        "teacher_ratio_after_repeat": round(
            teacher_examples / total_examples if total_examples else 0.0, 6
        ),
        "min_response_words": MIN_RESPONSE_WORDS,
        "max_response_words": MAX_RESPONSE_WORDS,
        "max_instruction_words": max_instruction_words,
        "dropped_long_instructions": filter_stats.dropped_long_instructions,
        "dropped_scaffolding_instruction": (
            filter_stats.dropped_scaffolding_instruction
        ),
        "dropped_scaffolding_response": filter_stats.dropped_scaffolding_response,
        "max_instruction_words_seen_input": (
            filter_stats.max_instruction_words_seen_input
        ),
        "max_instruction_words_seen_kept": (
            filter_stats.max_instruction_words_seen_kept
        ),
        "duplicate_blocks": duplicate_blocks,
        "max_consecutive_duplicate_blocks": max_duplicate_run,
        "unique_instruction_domains": sorted(
            filter_stats.kept_instruction_domains or set()
        ),
        "shuffle_seed": seed,
        "written_chars": len(written_text),
    }


def validate_report(report: dict[str, object]) -> None:
    if int(report["max_consecutive_duplicate_blocks"]) > 1:
        raise ValueError(
            "Build failed: consecutive duplicate blocks remain after reordering"
        )
    if float(report["teacher_ratio_after_repeat"]) < 0.30:
        raise ValueError(
            "Build failed: teacher_ratio_after_repeat must be at least 0.30"
        )


def write_report(report: dict[str, object], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def print_report(report: dict[str, object]) -> None:
    print("Phase 18H: chat-only continued-pretraining corpus")
    print("=" * 60)
    print("Objective: next-token prediction on chat turns only (not SFT)")
    print(f"Output:                {report['output']}")
    print(f"Input examples:        {report['total_input_examples']:,}")
    print(f"Final examples:        {report['final_examples']:,}")
    print(f"Total words:           {report['total_words']:,}")
    print(f"Teacher examples:      {report['teacher_examples_after_repeat']:,} "
          f"({report['kept_teacher_examples']:,} x {report['teacher_repeat']})")
    print(f"Instruction examples:  {report['instruction_examples_after_repeat']:,} "
          f"({report['kept_instruction_examples']:,} x {report['instruction_repeat']})")
    print(f"Teacher ratio:         {report['teacher_ratio_after_repeat']:.2%}")
    print(f"Response word window:  {report['min_response_words']}-{report['max_response_words']}")
    print(f"Instruction max:       {report['max_instruction_words']} words")
    print(f"Dropped long:          {report['dropped_long_instructions']:,}")
    print(f"Scaffold instructions: {report['dropped_scaffolding_instruction']:,}")
    print(f"Scaffold responses:    {report['dropped_scaffolding_response']:,}")
    print(f"Longest input:         {report['max_instruction_words_seen_input']:,} words")
    print(f"Longest kept:          {report['max_instruction_words_seen_kept']:,} words")
    print(f"Instruction domains:   {', '.join(report['unique_instruction_domains']) or 'n/a'}")
    print(f"Duplicate blocks:      {report['duplicate_blocks']:,}")
    print(f"Max duplicate run:     {report['max_consecutive_duplicate_blocks']:,}")
    print()
    print("Train with:")
    print(
        "  python 13_gpt_pretraining/training/trainer.py \\\n"
        "    --config large_50m \\\n"
        "    --resume 13_gpt_pretraining/checkpoints/large_50m/step_006500.pt \\\n"
        "    --corpus 13_gpt_pretraining/data/corpus_chat_only.txt \\\n"
        "    --steps 1000 \\\n"
        "    --lr 2e-5"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 18H chat-only corpus.")
    parser.add_argument("--teacher-data", type=Path, default=DEFAULT_TEACHER_DATA)
    parser.add_argument("--instruction-data", type=Path, default=DEFAULT_INSTRUCTION_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--teacher-repeat", type=int, default=20)
    parser.add_argument("--instruction-repeat", type=int, default=1)
    parser.add_argument(
        "--max-instruction-words",
        type=int,
        default=DEFAULT_MAX_INSTRUCTION_WORDS,
        help="Drop USER instructions longer than this many words (default: 80).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to shuffle examples after repetition (default: 42).",
    )
    parser.add_argument(
        "--max-instruction-examples",
        type=int,
        default=5000,
        help="Cap the number of curated instruction pairs (default: 5000).",
    )
    args = parser.parse_args()

    if args.teacher_repeat <= 0 or args.instruction_repeat <= 0:
        raise ValueError("repeat counts must be positive")
    if args.max_instruction_words <= 0:
        raise ValueError("--max-instruction-words must be positive")
    if args.max_instruction_examples is not None and args.max_instruction_examples <= 0:
        raise ValueError("--max-instruction-examples must be positive")
    if not args.teacher_data.exists():
        raise FileNotFoundError(f"Teacher data not found: {args.teacher_data}")
    if not args.instruction_data.exists():
        raise FileNotFoundError(f"Instruction data not found: {args.instruction_data}")

    filter_stats = FilterStats()
    teacher_pairs = load_teacher_examples(
        args.teacher_data,
        args.max_instruction_words,
        filter_stats,
    )
    instruction_pairs = load_instruction_examples(
        args.instruction_data,
        args.max_instruction_examples,
        args.max_instruction_words,
        filter_stats,
        args.seed,
    )
    if not teacher_pairs:
        raise ValueError(f"No usable teacher examples in {args.teacher_data}")
    if not instruction_pairs:
        raise ValueError(f"No usable instruction examples in {args.instruction_data}")

    blocks = build_blocks(
        teacher_pairs=teacher_pairs,
        instruction_pairs=instruction_pairs,
        teacher_repeat=args.teacher_repeat,
        instruction_repeat=args.instruction_repeat,
        seed=args.seed,
    )
    written_text = render_corpus(blocks)
    report = build_report(
        output_path=args.output,
        written_text=written_text,
        teacher_pairs=teacher_pairs,
        instruction_pairs=instruction_pairs,
        teacher_repeat=args.teacher_repeat,
        instruction_repeat=args.instruction_repeat,
        blocks=blocks,
        filter_stats=filter_stats,
        max_instruction_words=args.max_instruction_words,
        seed=args.seed,
    )
    validate_report(report)
    write_corpus(written_text, args.output)
    write_report(report, args.report)
    print_report(report)


if __name__ == "__main__":
    main()
