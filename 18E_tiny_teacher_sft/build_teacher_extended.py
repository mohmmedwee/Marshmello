#!/usr/bin/env python3
"""Merge tiny teacher data with 18J core SFT train examples (no eval leakage)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
ORIGINAL_TEACHER = PHASE_ROOT / "data" / "teacher.jsonl"
CORE_SFT = PROJECT_ROOT / "18J_marshmello_core_sft" / "data" / "marshmello_core_sft.jsonl"
OUTPUT = PHASE_ROOT / "data" / "teacher_extended.jsonl"
REPORT = PHASE_ROOT / "reports" / "teacher_extended_report.json"

FILLER_PHRASES = (
    "It helps the system answer or act more usefully.",
    "It is useful for building models that work beyond one example.",
    "It is part of how language models read prompts and produce answers.",
    "It helps keep data useful, reliable, or fast to access.",
    "It helps teams build software that is easier to change and operate.",
    "It helps Python code stay clear and practical.",
)

MAX_RESPONSE_WORDS = 55
MIN_RESPONSE_WORDS = 8
MAX_SENTENCES = 3


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def clean_filler(text: str) -> str:
    out = text.strip()
    for phrase in FILLER_PHRASES:
        out = out.replace(phrase, " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def shorten_response(text: str) -> str:
    text = clean_filler(text)
    sentences = split_sentences(text)
    if not sentences:
        return text
    kept: list[str] = []
    words = 0
    for sentence in sentences:
        kept.append(sentence)
        words += word_count(sentence)
        if len(kept) >= MAX_SENTENCES or words >= MAX_RESPONSE_WORDS:
            break
    result = " ".join(kept).strip()
    if word_count(result) < MIN_RESPONSE_WORDS and len(sentences) > len(kept):
        result = " ".join(sentences[: min(len(sentences), MAX_SENTENCES)]).strip()
    return result


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def merge_rows(*sources: list[dict]) -> list[dict]:
    by_instruction: dict[str, dict] = {}
    for rows in sources:
        for row in rows:
            instruction = row["instruction"].strip()
            response = shorten_response(row["response"])
            domain = row.get("domain", "general")
            if not instruction or word_count(response) < MIN_RESPONSE_WORDS:
                continue
            key = normalize_instruction(instruction)
            candidate = {"instruction": instruction, "response": response, "domain": domain}
            existing = by_instruction.get(key)
            if existing is None or word_count(response) < word_count(existing["response"]):
                by_instruction[key] = candidate
    return list(by_instruction.values())


def validate(rows: list[dict]) -> None:
    if len(rows) < 500:
        raise ValueError(f"Expected at least 500 examples, got {len(rows)}")
    domains = Counter(r["domain"] for r in rows)
    for domain in ("databases", "transformers_llms", "ai_basics"):
        if domains[domain] < 100:
            raise ValueError(f"Domain {domain} too small: {domains[domain]}")
    for idx, row in enumerate(rows, start=1):
        w = word_count(row["response"])
        if w > 70:
            raise ValueError(f"Row {idx} response too long: {w} words")


def main() -> None:
    from build_teacher_data import build_examples as build_original

    original = build_original()
    for row in original:
        row["response"] = shorten_response(row["response"])

    core_rows = []
    for row in load_jsonl(CORE_SFT):
        if row.get("split") == "held_out":
            raise ValueError("Core SFT file must not contain held_out rows")
        core_rows.append(
            {
                "instruction": row["instruction"],
                "response": row["response"],
                "domain": row["domain"],
            }
        )

    merged = merge_rows(original, core_rows)
    validate(merged)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(r["domain"] for r in merged)
    meta = {
        "output": str(OUTPUT),
        "total": len(merged),
        "from_original": len(original),
        "from_core_sft_train": len(core_rows),
        "domain_counts": dict(counts),
    }
    REPORT.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Phase 18E: extended teacher dataset")
    print("=" * 60)
    print(f"Output:  {OUTPUT}")
    print(f"Total:   {len(merged)}")
    print(f"  original tiny teacher: {len(original)}")
    print(f"  core SFT train rows:   {len(core_rows)}")
    for domain, count in sorted(counts.items()):
        print(f"  {domain}: {count}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
