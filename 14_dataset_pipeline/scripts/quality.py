"""Step 4 — quality scoring and filtering with domain tagging."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from scripts.common import (  # noqa: E402
    DEDUPED_DIR,
    Document,
    detect_language,
    ensure_dirs,
    infer_domain,
    iter_jsonl,
    word_count,
    write_jsonl,
)

SPAM_CHAR_RUN = re.compile(r"(.)\1{7,}")
LOREM = re.compile(r"\blorem ipsum\b", re.IGNORECASE)


def line_repetition_ratio(text: str) -> float:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(lines)


def unique_word_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def char_repetition_ratio(text: str) -> float:
    if not text:
        return 0.0
    max_run = 1
    current = 1
    for prev, ch in zip(text, text[1:]):
        if prev == ch:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run / len(text)


def score_document(text: str) -> dict[str, float | int | str]:
    words = word_count(text)
    language, language_confidence = detect_language(text)
    return {
        "word_count": words,
        "unique_word_ratio": round(unique_word_ratio(text), 4),
        "line_repetition_ratio": round(line_repetition_ratio(text), 4),
        "char_repetition_ratio": round(char_repetition_ratio(text), 4),
        "language": language,
        "language_confidence": round(language_confidence, 4),
    }


def passes_quality(
    text: str,
    scores: dict[str, float | int | str],
    *,
    min_words: int,
    min_unique_ratio: float,
    max_line_repetition: float,
    min_language_confidence: float,
) -> tuple[bool, str | None]:
    if LOREM.search(text):
        return False, "lorem_ipsum"
    if SPAM_CHAR_RUN.search(text):
        return False, "char_spam"
    if int(scores["word_count"]) < min_words:
        return False, "too_short"
    if float(scores["unique_word_ratio"]) < min_unique_ratio:
        return False, "low_unique_words"
    if float(scores["line_repetition_ratio"]) > max_line_repetition:
        return False, "repeated_lines"
    if float(scores["language_confidence"]) < min_language_confidence:
        return False, "low_language_confidence"
    return True, None


def run_quality(
    input_path: Path,
    output_path: Path,
    *,
    min_words: int = 20,
    min_unique_ratio: float = 0.25,
    max_line_repetition: float = 0.35,
    min_language_confidence: float = 0.30,
) -> dict[str, int]:
    ensure_dirs(output_path.parent)
    stats = {
        "input_documents": 0,
        "output_documents": 0,
        "filtered": 0,
        "filter_reasons": Counter(),
    }

    def records():
        for payload in iter_jsonl(input_path):
            stats["input_documents"] += 1
            doc = Document.from_dict(payload)
            scores = score_document(doc.text)
            ok, reason = passes_quality(
                doc.text,
                scores,
                min_words=min_words,
                min_unique_ratio=min_unique_ratio,
                max_line_repetition=max_line_repetition,
                min_language_confidence=min_language_confidence,
            )
            if not ok:
                stats["filtered"] += 1
                if reason:
                    stats["filter_reasons"][reason] += 1
                continue

            doc.domain = infer_domain(doc.text)
            doc.language = str(scores["language"])
            doc.meta["quality"] = scores
            stats["output_documents"] += 1
            yield doc.to_dict()

    write_jsonl(output_path, records())
    stats["filter_reasons"] = dict(stats["filter_reasons"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality scoring and domain tagging.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEDUPED_DIR / "quality.jsonl")
    parser.add_argument("--min-words", type=int, default=20)
    args = parser.parse_args()

    stats = run_quality(args.input, args.output, min_words=args.min_words)
    print(
        f"Quality filter kept {stats['output_documents']} of {stats['input_documents']} "
        f"(filtered {stats['filtered']}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
