#!/usr/bin/env python3
"""Build short teacher math examples and merge into teacher_extended_short.jsonl."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
MATH_OUTPUT = PHASE_ROOT / "data" / "teacher_math.jsonl"
SHORT_INPUT = PHASE_ROOT / "data" / "teacher_extended_short.jsonl"
SHORT_BACKUP = PHASE_ROOT / "data" / "teacher_extended_short_before_math.jsonl"
SHORT_OUTPUT = SHORT_INPUT
REPORT = PHASE_ROOT / "reports" / "teacher_math_report.json"

DOMAIN = "math_basics"
SEED = 42
MIN_WORDS = 12
MAX_WORDS = 25

WORD_RE = re.compile(r"\b[\w'-]+\b")

PROMPT_TEMPLATES = (
    "What is {a} plus {b}?",
    "What is the sum of {a} and {b}?",
    "Calculate {a} + {b}.",
    "Can you add {a} and {b}?",
    "can you do sum {a}+{b}=",
    "What is {a} minus {b}?",
    "What is {a} times {b}?",
    "Compute {a} * {b}.",
    "What is {a} multiplied by {b}?",
)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def number_words(n: int) -> str:
    ones = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    if n < 20:
        return ones[n]
    if n < 100:
        t, o = divmod(n, 10)
        return tens[t] if o == 0 else f"{tens[t]} {ones[o]}"
    return str(n)


def pad_addition(a: int, b: int, result: int) -> str:
    return (
        f"The sum of {a} and {b} is {result}, which follows from basic integer addition."
    )


def pad_subtraction(a: int, b: int, result: int) -> str:
    return (
        f"Subtracting {b} from {a} gives {result}, which is a basic integer subtraction result."
    )


def pad_multiplication(a: int, b: int, result: int) -> str:
    return (
        f"Multiplying {a} by {b} gives {result}, which is a basic integer multiplication result."
    )


def validate_response(text: str) -> None:
    words = word_count(text)
    if not MIN_WORDS <= words <= MAX_WORDS:
        raise ValueError(f"Response word count {words} outside {MIN_WORDS}-{MAX_WORDS}: {text}")


def build_math_rows(rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    seen_instructions: set[str] = set()

    add_pairs = [(a, b) for a in range(1, 50) for b in range(1, 50)]
    rng.shuffle(add_pairs)
    for a, b in add_pairs[:40]:
        result = a + b
        templates = [PROMPT_TEMPLATES[0], PROMPT_TEMPLATES[1], PROMPT_TEMPLATES[2], PROMPT_TEMPLATES[4]]
        for template in templates:
            instruction = template.format(a=a, b=b)
            key = re.sub(r"\s+", " ", instruction.casefold())
            if key in seen_instructions:
                continue
            response = pad_addition(a, b, result)
            validate_response(response)
            seen_instructions.add(key)
            rows.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "domain": DOMAIN,
                    "concept": "integer_addition",
                }
            )

    sub_pairs = [(a, b) for a in range(10, 100) for b in range(1, a)]
    rng.shuffle(sub_pairs)
    for a, b in sub_pairs[:25]:
        result = a - b
        instruction = PROMPT_TEMPLATES[5].format(a=a, b=b)
        key = re.sub(r"\s+", " ", instruction.casefold())
        if key in seen_instructions:
            continue
        response = pad_subtraction(a, b, result)
        validate_response(response)
        seen_instructions.add(key)
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "domain": DOMAIN,
                "concept": "integer_subtraction",
            }
        )

    mul_pairs = [(a, b) for a in range(2, 13) for b in range(2, 13)]
    rng.shuffle(mul_pairs)
    for a, b in mul_pairs[:25]:
        result = a * b
        for template in (PROMPT_TEMPLATES[6], PROMPT_TEMPLATES[7], PROMPT_TEMPLATES[8]):
            instruction = template.format(a=a, b=b)
            key = re.sub(r"\s+", " ", instruction.casefold())
            if key in seen_instructions:
                continue
            response = pad_multiplication(a, b, result)
            validate_response(response)
            seen_instructions.add(key)
            rows.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "domain": DOMAIN,
                    "concept": "integer_multiplication",
                }
            )
            break

    # Exact user-style prompts for common sums
    extras = (
        ("can you do sum 12+14=", 12, 14, "add"),
        ("What is 12 plus 14?", 12, 14, "add"),
        ("Calculate 12 + 14.", 12, 14, "add"),
        ("Can you add 7 and 8?", 7, 8, "add"),
        ("What is 25 minus 9?", 25, 9, "sub"),
        ("What is 6 times 7?", 6, 7, "mul"),
    )
    for instruction, a, b, kind in extras:
        key = re.sub(r"\s+", " ", instruction.casefold())
        if key in seen_instructions:
            continue
        if kind == "add":
            response = pad_addition(a, b, a + b)
        elif kind == "sub":
            response = pad_subtraction(a, b, a - b)
        else:
            response = pad_multiplication(a, b, a * b)
        validate_response(response)
        seen_instructions.add(key)
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "domain": DOMAIN,
                "concept": f"integer_{kind}",
            }
        )

    return rows


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_rows(base: list[dict], extra: list[dict]) -> tuple[list[dict], int]:
    merged: dict[str, dict] = {}
    for row in base:
        key = re.sub(r"\s+", " ", row["instruction"].strip().casefold())
        merged[key] = row
    added = 0
    for row in extra:
        key = re.sub(r"\s+", " ", row["instruction"].strip().casefold())
        if key in merged:
            continue
        merged[key] = row
        added += 1
    return list(merged.values()), added


def main() -> None:
    rng = random.Random(SEED)
    math_rows = build_math_rows(rng)

    MATH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with MATH_OUTPUT.open("w", encoding="utf-8") as handle:
        for row in math_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not SHORT_INPUT.exists():
        raise FileNotFoundError(f"Missing base dataset: {SHORT_INPUT}")

    base_rows = load_jsonl(SHORT_INPUT)
    SHORT_BACKUP.write_text(SHORT_INPUT.read_text(), encoding="utf-8")
    merged_rows, added = merge_rows(base_rows, math_rows)
    merged_rows.sort(key=lambda row: (row.get("domain", ""), row["instruction"]))

    with SHORT_OUTPUT.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "math_output": str(MATH_OUTPUT),
        "math_rows": len(math_rows),
        "base_rows": len(base_rows),
        "merged_total": len(merged_rows),
        "added_to_short": added,
        "backup": str(SHORT_BACKUP),
        "domain_counts": dict(Counter(r["domain"] for r in merged_rows)),
        "concept_counts": dict(Counter(r.get("concept", "?") for r in math_rows)),
    }
    REPORT.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Phase 18E: teacher math supplement")
    print("=" * 60)
    print(f"Math file:   {MATH_OUTPUT} ({len(math_rows)} rows)")
    print(f"Merged into: {SHORT_OUTPUT}")
    print(f"  base:  {len(base_rows)}")
    print(f"  added: {added}")
    print(f"  total: {len(merged_rows)}")
    print(f"Backup: {SHORT_BACKUP}")
    print("Next: retrain teacher from chat base, then eval chat + 18J/18K")


if __name__ == "__main__":
    main()
