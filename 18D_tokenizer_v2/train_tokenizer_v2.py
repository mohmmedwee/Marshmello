#!/usr/bin/env python3
"""Train Marshmello tokenizer v2 on base corpus plus chat-format instructions."""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE13_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from bpe_demo import BPETokenizer  # noqa: E402
from tokenizer.bpe_io import load_tokenizer, save_tokenizer  # noqa: E402

DEFAULT_BASE_CORPUS = PROJECT_ROOT / "13_gpt_pretraining" / "data" / "corpus.txt"
DEFAULT_CHAT_JSONL = PROJECT_ROOT / "17_instruction_dataset" / "processed" / "chat.jsonl"
DEFAULT_OLD_TOKENIZER = PROJECT_ROOT / "13_gpt_pretraining" / "tokenizer" / "tokenizer.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "13_gpt_pretraining" / "tokenizer" / "tokenizer_v2.json"
DEFAULT_REPORT_OUTPUT = PHASE_ROOT / "reports" / "tokenizer_v2_report.json"

SPECIAL_MARKERS = ("<USER>", "<ASSISTANT>", "<END>")
COMMON_PUNCTUATION = ("?", ":", ";", "-", "'", '"', "(", ")", "[", "]", "{", "}", "/", "\\", "`")
INCOMPATIBILITY_WARNING = (
    "tokenizer_v2 changes token IDs and vocab size. Old checkpoints are not "
    "compatible; re-pretrain Marshmello base with the new vocab before SFT."
)

Pair = tuple[str, str]
HeapEntry = tuple[int, Pair]
WordCache = dict[str, tuple[int, Counter[str]]]


@dataclass(frozen=True)
class ChatExample:
    text: str
    instruction: str
    response: str


@dataclass(frozen=True)
class BPETrainingStats:
    target_vocab_size: int
    initial_vocab_size: int
    final_vocab_size: int
    learned_merges: int
    duplicate_merge_tokens: int
    stop_reason: str

    @property
    def requested_unique_merges(self) -> int:
        return max(0, self.target_vocab_size - self.initial_vocab_size)


def parse_chat_parts(text: str) -> tuple[str, str]:
    """Extract instruction and response from Phase 17 chat text."""
    user_idx = text.find("<USER>")
    assistant_idx = text.find("<ASSISTANT>")
    if assistant_idx < 0:
        return "", ""

    instruction_start = user_idx + len("<USER>") if user_idx >= 0 else 0
    instruction = text[instruction_start:assistant_idx].strip()

    response_start = assistant_idx + len("<ASSISTANT>")
    end_idx = text.find("<END>", response_start)
    response_end = end_idx if end_idx >= 0 else len(text)
    response = text[response_start:response_end].strip()
    return instruction, response


def read_chat_examples(path: Path, max_chats: int) -> list[ChatExample]:
    examples: list[ChatExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}: {exc}") from exc
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            instruction, response = parse_chat_parts(text)
            examples.append(ChatExample(text=text, instruction=instruction, response=response))
            if len(examples) >= max_chats:
                break
    if not examples:
        raise ValueError(f"No chat examples found in {path}")
    return examples


def required_symbol_text(repetitions: int) -> str:
    """Force special markers and punctuation into the character vocabulary."""
    units = [*SPECIAL_MARKERS, *COMMON_PUNCTUATION]
    return "\n".join(" ".join(units) for _ in range(repetitions))


def build_training_text(
    *,
    base_corpus_path: Path,
    chat_jsonl_path: Path,
    max_chats: int,
    required_repetitions: int,
) -> tuple[str, str, list[ChatExample]]:
    base_text = base_corpus_path.read_text(encoding="utf-8")
    chat_examples = read_chat_examples(chat_jsonl_path, max_chats=max_chats)
    chat_text = "\n\n".join(example.text for example in chat_examples)
    parts = [base_text.strip(), required_symbol_text(required_repetitions), chat_text.strip()]
    training_text = "\n\n".join(part for part in parts if part).strip() + "\n"
    return training_text, base_text, chat_examples


def word_to_symbols(word: str) -> list[str]:
    if word == "\n":
        return ["\n"]
    return [*word, BPETokenizer.END]


def count_symbol_pairs(symbols: list[str]) -> Counter[Pair]:
    return Counter(zip(symbols, symbols[1:]))


def merge_symbols(symbols: list[str], pair: Pair, replacement: str) -> list[str]:
    first, second = pair
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
            merged.append(replacement)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


def push_pair(heap: list[HeapEntry], pair_counts: Counter[Pair], pair: Pair) -> None:
    count = pair_counts.get(pair, 0)
    if count > 0:
        heapq.heappush(heap, (-count, pair))


def pop_best_pair(heap: list[HeapEntry], pair_counts: Counter[Pair]) -> tuple[Pair | None, int]:
    while heap:
        negative_count, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if count > 0 and count == -negative_count:
            return pair, count
    return None, 0


def train_bpe_fast(
    text: str,
    *,
    target_vocab_size: int,
    verbose: bool,
) -> tuple[BPETokenizer, BPETrainingStats]:
    """Train a BPETokenizer-compatible merge table with incremental pair updates."""
    word_freq = Counter(text.split())
    if not word_freq:
        raise ValueError("Cannot train tokenizer on empty text")

    word_items = list(word_freq.items())
    word_symbols = [word_to_symbols(word) for word, _ in word_items]
    word_freqs = [freq for _, freq in word_items]

    vocab: set[str] = set()
    pair_counts: Counter[Pair] = Counter()
    pair_to_word_ids: DefaultDict[Pair, set[int]] = defaultdict(set)

    for word_id, (symbols, freq) in enumerate(zip(word_symbols, word_freqs, strict=True)):
        vocab.update(symbols)
        for pair, pair_count in count_symbol_pairs(symbols).items():
            pair_counts[pair] += pair_count * freq
            pair_to_word_ids[pair].add(word_id)

    initial_vocab_size = len(vocab)
    heap: list[HeapEntry] = [(-count, pair) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)

    merges: list[Pair] = []
    duplicate_merge_tokens = 0
    stop_reason = "target_vocab_reached"

    if verbose:
        print("Phase 18D: tokenizer v2 BPE training")
        print("=" * 60)
        print(f"Unique training words: {len(word_freq):,}")
        print(f"Initial vocab:         {initial_vocab_size:,}")
        print(f"Target vocab:          {target_vocab_size:,}")

    if initial_vocab_size >= target_vocab_size:
        stop_reason = "initial_vocab_already_reaches_target"

    while len(vocab) < target_vocab_size:
        best_pair, best_count = pop_best_pair(heap, pair_counts)
        if best_pair is None:
            stop_reason = (
                "no_adjacent_symbol_pairs_left: every unique training word has "
                "collapsed to one symbol"
            )
            break

        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        if new_token in vocab:
            duplicate_merge_tokens += 1
        vocab.add(new_token)

        affected_word_ids = list(pair_to_word_ids.get(best_pair, set()))
        changed_pairs: set[Pair] = set()

        for word_id in affected_word_ids:
            symbols = word_symbols[word_id]
            old_pairs = count_symbol_pairs(symbols)
            if best_pair not in old_pairs:
                continue

            freq = word_freqs[word_id]
            for pair, pair_count in old_pairs.items():
                pair_counts[pair] -= pair_count * freq
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                word_ids = pair_to_word_ids.get(pair)
                if word_ids is not None:
                    word_ids.discard(word_id)
                    if not word_ids:
                        pair_to_word_ids.pop(pair, None)
                changed_pairs.add(pair)

            new_symbols = merge_symbols(symbols, best_pair, new_token)
            word_symbols[word_id] = new_symbols

            for pair, pair_count in count_symbol_pairs(new_symbols).items():
                pair_counts[pair] += pair_count * freq
                pair_to_word_ids[pair].add(word_id)
                changed_pairs.add(pair)

        for pair in changed_pairs:
            push_pair(heap, pair_counts, pair)

        if verbose and (
            len(merges) <= 8 or len(merges) % 500 == 0 or len(vocab) >= target_vocab_size
        ):
            print(
                f"  Merge {len(merges):5d}: {best_pair!r} -> {new_token!r} "
                f"(count={best_count:,}, vocab={len(vocab):,})"
            )

    bpe = BPETokenizer()
    bpe.merges = merges
    bpe.vocab = vocab
    bpe._initial_vocab_size = initial_vocab_size
    bpe.build_index()

    stats = BPETrainingStats(
        target_vocab_size=target_vocab_size,
        initial_vocab_size=initial_vocab_size,
        final_vocab_size=bpe.vocab_size,
        learned_merges=len(merges),
        duplicate_merge_tokens=duplicate_merge_tokens,
        stop_reason=stop_reason,
    )
    return bpe, stats


def merge_rank_map(bpe: BPETokenizer) -> dict[Pair, int]:
    ranks: dict[Pair, int] = {}
    for rank, pair in enumerate(bpe.merges):
        ranks.setdefault(pair, rank)
    return ranks


def encode_word_with_ranks(
    word: str,
    ranks: dict[Pair, int],
    end_marker: str = BPETokenizer.END,
) -> list[str]:
    """Encode one word by repeatedly applying the earliest available merge."""
    if word == "\n":
        return ["\n"]
    symbols = [*word, end_marker]
    while len(symbols) > 1:
        best_pair: Pair | None = None
        best_rank: int | None = None
        for pair in zip(symbols, symbols[1:]):
            rank = ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_pair = pair
                best_rank = rank
        if best_pair is None:
            break
        symbols = merge_symbols(symbols, best_pair, best_pair[0] + best_pair[1])
    return symbols


def known_single_chars(bpe: BPETokenizer) -> set[str]:
    return {token for token in bpe.vocab if len(token) == 1}


def safe_tokenize_to_tokens(
    bpe: BPETokenizer,
    text: str,
    ranks: dict[Pair, int] | None = None,
) -> tuple[list[str], Counter[str]]:
    if not hasattr(bpe, "stoi"):
        bpe.build_index()
    ranks = ranks if ranks is not None else merge_rank_map(bpe)
    known = known_single_chars(bpe)
    tokens: list[str] = []
    skipped: Counter[str] = Counter()
    for word in text.split():
        missing = [ch for ch in word if ch not in known]
        if missing:
            skipped.update(missing)
        safe_word = "".join(ch for ch in word if ch in known)
        if not safe_word:
            continue
        tokens.extend(token for token in encode_word_with_ranks(safe_word, ranks, bpe.END) if token in bpe.stoi)
    return tokens, skipped


def count_text_tokens_safe(
    bpe: BPETokenizer,
    text: str,
    *,
    ranks: dict[Pair, int],
    cache: WordCache,
) -> tuple[int, Counter[str]]:
    if not hasattr(bpe, "stoi"):
        bpe.build_index()
    known = known_single_chars(bpe)
    total = 0
    skipped: Counter[str] = Counter()

    for word in text.split():
        cached = cache.get(word)
        if cached is None:
            missing = Counter(ch for ch in word if ch not in known)
            safe_word = "".join(ch for ch in word if ch in known)
            if safe_word:
                token_count = sum(
                    1
                    for token in encode_word_with_ranks(safe_word, ranks, bpe.END)
                    if token in bpe.stoi
                )
            else:
                token_count = 0
            cached = (token_count, missing)
            cache[word] = cached

        token_count, missing = cached
        total += token_count
        if missing:
            skipped.update(missing)

    return total, skipped


def top_char_report(chars: Counter[str], limit: int = 20) -> list[dict[str, object]]:
    return [
        {
            "char": char,
            "repr": repr(char),
            "codepoint": f"U+{ord(char):04X}",
            "count": count,
        }
        for char, count in chars.most_common(limit)
    ]


def build_coverage_report(
    *,
    v2: BPETokenizer,
    old: BPETokenizer,
    stats: BPETrainingStats,
    minimum_vocab_size: int,
    base_text: str,
    training_text: str,
    chat_examples: list[ChatExample],
    base_corpus_path: Path,
    chat_jsonl_path: Path,
    old_tokenizer_path: Path,
    output_path: Path,
) -> dict[str, object]:
    v2_ranks = merge_rank_map(v2)
    old_ranks = merge_rank_map(old)
    v2_cache: WordCache = {}
    old_cache: WordCache = {}

    totals = Counter()
    v2_skipped: Counter[str] = Counter()
    old_skipped: Counter[str] = Counter()
    parse_failures = 0

    for example in chat_examples:
        if not example.instruction or not example.response:
            parse_failures += 1

        v2_full, skipped = count_text_tokens_safe(v2, example.text, ranks=v2_ranks, cache=v2_cache)
        totals["v2_full"] += v2_full
        v2_skipped.update(skipped)

        old_full, skipped = count_text_tokens_safe(old, example.text, ranks=old_ranks, cache=old_cache)
        totals["old_full"] += old_full
        old_skipped.update(skipped)

        v2_instruction, skipped = count_text_tokens_safe(
            v2,
            example.instruction,
            ranks=v2_ranks,
            cache=v2_cache,
        )
        totals["v2_instruction"] += v2_instruction

        v2_response, skipped = count_text_tokens_safe(
            v2,
            example.response,
            ranks=v2_ranks,
            cache=v2_cache,
        )
        totals["v2_response"] += v2_response

        old_instruction, skipped = count_text_tokens_safe(
            old,
            example.instruction,
            ranks=old_ranks,
            cache=old_cache,
        )
        totals["old_instruction"] += old_instruction

        old_response, skipped = count_text_tokens_safe(
            old,
            example.response,
            ranks=old_ranks,
            cache=old_cache,
        )
        totals["old_response"] += old_response

    chat_count = len(chat_examples)
    compression_ratio = (
        totals["old_full"] / totals["v2_full"] if totals["v2_full"] else 0.0
    )
    token_reduction = (
        1.0 - totals["v2_full"] / totals["old_full"] if totals["old_full"] else 0.0
    )
    v2_chars = known_single_chars(v2)

    return {
        "tokenizer": {
            "output": str(output_path),
            "target_vocab_size": stats.target_vocab_size,
            "minimum_vocab_size": minimum_vocab_size,
            "initial_vocab_size": stats.initial_vocab_size,
            "final_vocab_size": stats.final_vocab_size,
            "requested_unique_merges": stats.requested_unique_merges,
            "learned_merges": stats.learned_merges,
            "duplicate_merge_tokens": stats.duplicate_merge_tokens,
            "stop_reason": stats.stop_reason,
        },
        "training_data": {
            "base_corpus": str(base_corpus_path),
            "chat_jsonl": str(chat_jsonl_path),
            "chat_examples_used": chat_count,
            "base_words": len(base_text.split()),
            "chat_words": sum(len(example.text.split()) for example in chat_examples),
            "training_words": len(training_text.split()),
            "training_chars": len(training_text),
        },
        "coverage": {
            "examples_evaluated": chat_count,
            "parse_failures": parse_failures,
            "avg_tokens_per_instruction": round(totals["v2_instruction"] / chat_count, 2),
            "avg_tokens_per_response": round(totals["v2_response"] / chat_count, 2),
            "old_avg_tokens_per_instruction": round(totals["old_instruction"] / chat_count, 2),
            "old_avg_tokens_per_response": round(totals["old_response"] / chat_count, 2),
            "v2_total_chat_tokens": totals["v2_full"],
            "old_total_chat_tokens": totals["old_full"],
            "compression_ratio_old_to_v2": round(compression_ratio, 4),
            "token_reduction_vs_old": round(token_reduction, 4),
            "unknown_or_skipped_characters": {
                "v2_total": sum(v2_skipped.values()),
                "old_total": sum(old_skipped.values()),
            },
            "top_missing_chars": {
                "v2": top_char_report(v2_skipped),
                "old": top_char_report(old_skipped),
            },
        },
        "required_symbols": {
            "special_markers": list(SPECIAL_MARKERS),
            "common_punctuation": list(COMMON_PUNCTUATION),
            "common_punctuation_present_as_chars": {
                char: char in v2_chars for char in COMMON_PUNCTUATION
            },
        },
        "old_tokenizer": str(old_tokenizer_path),
        "warning": INCOMPATIBILITY_WARNING,
    }


def print_report_summary(report: dict[str, object], report_output: Path) -> None:
    tokenizer = dict(report["tokenizer"])
    training = dict(report["training_data"])
    coverage = dict(report["coverage"])
    skipped = dict(coverage["unknown_or_skipped_characters"])

    print()
    print("Tokenizer v2 summary")
    print("=" * 60)
    print(f"Output:                 {tokenizer['output']}")
    print(f"Report:                 {report_output}")
    print(f"Final vocab:            {tokenizer['final_vocab_size']:,}")
    print(f"Learned merges:         {tokenizer['learned_merges']:,}")
    print(f"Stop reason:            {tokenizer['stop_reason']}")
    print(f"Base words:             {training['base_words']:,}")
    print(f"Chat examples used:     {training['chat_examples_used']:,}")
    print(f"Training words:         {training['training_words']:,}")
    print(f"Avg instruction tokens: {coverage['avg_tokens_per_instruction']}")
    print(f"Avg response tokens:    {coverage['avg_tokens_per_response']}")
    print(f"V2 skipped chars:       {skipped['v2_total']:,}")
    print(f"Old skipped chars:      {skipped['old_total']:,}")
    print(f"Compression old/v2:     {coverage['compression_ratio_old_to_v2']}x")
    print(f"Token reduction vs old: {coverage['token_reduction_vs_old']:.2%}")
    print()
    print(f"Important: {INCOMPATIBILITY_WARNING}")


def validate_args(args: argparse.Namespace) -> None:
    if args.target_vocab_size < args.min_vocab_size:
        raise ValueError("--target-vocab-size must be >= --min-vocab-size")
    if args.max_chats <= 0:
        raise ValueError("--max-chats must be positive")
    if args.required_repetitions <= 0:
        raise ValueError("--required-repetitions must be positive")
    for label, path in (
        ("base corpus", args.base_corpus),
        ("chat data", args.chat_data),
        ("old tokenizer", args.old_tokenizer),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Marshmello tokenizer v2.")
    parser.add_argument("--base-corpus", type=Path, default=DEFAULT_BASE_CORPUS)
    parser.add_argument("--chat-data", type=Path, default=DEFAULT_CHAT_JSONL)
    parser.add_argument("--old-tokenizer", type=Path, default=DEFAULT_OLD_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--target-vocab-size", type=int, default=8000)
    parser.add_argument("--min-vocab-size", type=int, default=5000)
    parser.add_argument("--max-chats", type=int, default=50_000)
    parser.add_argument("--required-repetitions", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    validate_args(args)

    training_text, base_text, chat_examples = build_training_text(
        base_corpus_path=args.base_corpus,
        chat_jsonl_path=args.chat_data,
        max_chats=args.max_chats,
        required_repetitions=args.required_repetitions,
    )
    bpe, stats = train_bpe_fast(
        training_text,
        target_vocab_size=args.target_vocab_size,
        verbose=not args.quiet,
    )
    save_tokenizer(bpe, args.output)

    old = load_tokenizer(args.old_tokenizer)
    report = build_coverage_report(
        v2=bpe,
        old=old,
        stats=stats,
        minimum_vocab_size=args.min_vocab_size,
        base_text=base_text,
        training_text=training_text,
        chat_examples=chat_examples,
        base_corpus_path=args.base_corpus,
        chat_jsonl_path=args.chat_data,
        old_tokenizer_path=args.old_tokenizer,
        output_path=args.output,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_report_summary(report, args.report_output)

    if bpe.vocab_size < args.min_vocab_size:
        raise SystemExit(
            f"Tokenizer v2 vocab size {bpe.vocab_size:,} is below minimum "
            f"{args.min_vocab_size:,}. Stop reason: {stats.stop_reason}"
        )


if __name__ == "__main__":
    main()
