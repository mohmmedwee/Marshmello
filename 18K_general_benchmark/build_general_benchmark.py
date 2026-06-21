#!/usr/bin/env python3
"""Build Phase 18K held-out general assistant benchmark from marshmello_all_sft.jsonl."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
SOURCE = PROJECT_ROOT / "data" / "marshmello_all_sft.jsonl"
DATA_DIR = PHASE_ROOT / "data"
REPORT_DIR = PHASE_ROOT / "reports"
DEFAULT_EVAL = DATA_DIR / "general_eval.jsonl"
DEFAULT_TRAIN = DATA_DIR / "general_train.jsonl"
DEFAULT_REPORT = REPORT_DIR / "general_dataset_report.md"
DEFAULT_META = REPORT_DIR / "general_dataset_report.json"

PER_BUCKET = 100
SEED = 42

BUCKETS: dict[str, frozenset[str]] = {
    "ai": frozenset(
        {
            "ai_basics",
            "machine_learning",
            "deep_learning",
            "nlp",
            "statistics",
            "math_basics",
            "math_for_ml",
            "logic_reasoning",
            "transformers_llms",
        }
    ),
    "databases": frozenset({"databases", "sql", "data_structures"}),
    "programming": frozenset(
        {
            "python",
            "javascript",
            "html_css",
            "web_development",
            "software_engineering",
            "algorithms",
        }
    ),
    "system_design": frozenset(
        {
            "system_design",
            "devops",
            "linux",
            "networking",
            "operating_systems",
            "cybersecurity",
            "cloud_computing",
        }
    ),
    "general_knowledge": frozenset(
        {
            "daily_life",
            "general_science",
            "geography_basics",
            "history_basics",
            "study_skills",
            "writing_communication",
            "business_basics",
            "computer_vision",
        }
    ),
}

STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with",
        "is", "are", "was", "were", "be", "by", "as", "at", "it", "that", "this",
        "from", "can", "may", "such", "using", "used", "when", "which", "their",
        "they", "them", "into", "through", "also", "than", "other", "more",
        "most", "one", "two", "how", "what", "why",
    }
)

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_instruction(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def bucket_for(domain: str) -> str | None:
    for bucket, domains in BUCKETS.items():
        if domain in domains:
            return bucket
    return None


def expected_keywords_for(record: dict) -> list[str]:
    concept = str(record.get("concept", "")).strip()
    concept_parts = [part.strip() for part in re.split(r"[/,;]+", concept) if part.strip()]
    keywords: list[str] = []
    for part in concept_parts:
        normalized = normalize_instruction(part)
        if normalized and normalized not in keywords:
            keywords.append(normalized)
    for token in WORD_RE.findall(str(record.get("response", "")).casefold()):
        if len(token) >= 4 and token not in STOPWORDS and token not in keywords:
            keywords.append(token)
        if len(keywords) >= 10:
            break
    if not keywords and concept:
        keywords = [normalize_instruction(concept)]
    return keywords[:10]


def sample_bucket(rows: list[dict], bucket: str, count: int, rng: random.Random) -> list[dict]:
    pool = rows[:]
    rng.shuffle(pool)
    picked: list[dict] = []
    seen: set[str] = set()
    for row in pool:
        key = normalize_instruction(str(row["instruction"]))
        if not key or key in seen:
            continue
        seen.add(key)
        picked.append(row)
        if len(picked) == count:
            break
    if len(picked) != count:
        raise ValueError(f"Bucket {bucket}: need {count} unique rows, found {len(picked)}")
    return picked


def build_eval_rows(source_rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    by_bucket: dict[str, list[dict]] = {name: [] for name in BUCKETS}
    for row in source_rows:
        bucket = bucket_for(str(row.get("domain", "")))
        if bucket is not None:
            by_bucket[bucket].append(row)

    eval_rows: list[dict] = []
    for bucket in BUCKETS:
        selected = sample_bucket(by_bucket[bucket], bucket, PER_BUCKET, rng)
        for index, row in enumerate(selected, start=1):
            eval_rows.append(
                {
                    "id": f"18k-eval-{bucket}-{index:03d}",
                    "instruction": row["instruction"],
                    "reference_response": row["response"],
                    "domain": row["domain"],
                    "concept": row.get("concept", ""),
                    "question_type": row.get("question_type", ""),
                    "difficulty": row.get("difficulty", ""),
                    "benchmark_bucket": bucket,
                    "expected_keywords": expected_keywords_for(row),
                    "source": "marshmello_all_sft_v1",
                    "split": "held_out",
                }
            )

    eval_instructions = {normalize_instruction(row["instruction"]) for row in eval_rows}
    train_rows = [
        {**row, "split": "train"}
        for row in source_rows
        if normalize_instruction(str(row["instruction"])) not in eval_instructions
    ]
    return eval_rows, train_rows


def validate(eval_rows: list[dict], train_rows: list[dict]) -> None:
    required = {
        "id",
        "instruction",
        "reference_response",
        "domain",
        "concept",
        "expected_keywords",
        "benchmark_bucket",
        "split",
    }
    if len(eval_rows) != PER_BUCKET * len(BUCKETS):
        raise ValueError(f"Expected {PER_BUCKET * len(BUCKETS)} eval rows, got {len(eval_rows)}")
    bucket_counts = Counter(row["benchmark_bucket"] for row in eval_rows)
    for bucket in BUCKETS:
        if bucket_counts[bucket] != PER_BUCKET:
            raise ValueError(f"Bucket {bucket}: expected {PER_BUCKET}, got {bucket_counts[bucket]}")
    eval_keys = {normalize_instruction(row["instruction"]) for row in eval_rows}
    if len(eval_keys) != len(eval_rows):
        raise ValueError("Eval instructions must be unique")
    overlap = eval_keys & {normalize_instruction(str(row["instruction"])) for row in train_rows}
    if overlap:
        raise ValueError(f"Train/eval instruction overlap: {len(overlap)}")
    for index, row in enumerate(eval_rows, start=1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Eval row {index} missing fields: {sorted(missing)}")
        if not row["expected_keywords"]:
            raise ValueError(f"Eval row {index} has empty expected_keywords")


def write_report(
    path: Path,
    meta_path: Path,
    eval_rows: list[dict],
    train_rows: list[dict],
    source_count: int,
    seed: int,
) -> None:
    bucket_domains = Counter((row["benchmark_bucket"], row["domain"]) for row in eval_rows)
    meta = {
        "phase": "18K_general_benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "eval_rows": len(eval_rows),
        "train_rows": len(train_rows),
        "source_rows": source_count,
        "per_bucket": PER_BUCKET,
        "seed": seed,
        "buckets": {name: sorted(domains) for name, domains in BUCKETS.items()},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 18K — General Benchmark Dataset Report",
        "",
        f"- Generated: {meta['generated_at']}",
        f"- Source: `{SOURCE.name}` ({source_count} rows)",
        f"- Held-out eval: **{len(eval_rows)}** questions",
        f"- Train pool (non-overlapping): **{len(train_rows)}** rows",
        f"- Seed: {seed}",
        "",
        "## Buckets (100 each)",
        "",
        "| Bucket | Label | Source domains |",
        "|---|---|---|",
        "| ai | AI / ML | "
        + ", ".join(sorted(BUCKETS["ai"]))
        + " |",
        "| databases | Databases / SQL | "
        + ", ".join(sorted(BUCKETS["databases"]))
        + " |",
        "| programming | Programming | "
        + ", ".join(sorted(BUCKETS["programming"]))
        + " |",
        "| system_design | System Design / DevOps | "
        + ", ".join(sorted(BUCKETS["system_design"]))
        + " |",
        "| general_knowledge | General / writing / daily life | "
        + ", ".join(sorted(BUCKETS["general_knowledge"]))
        + " |",
        "",
        "## Domain mix in eval",
        "",
        "| Bucket | Domain | Count |",
        "|---|---|---:|",
    ]
    for (bucket, domain), count in sorted(bucket_domains.items()):
        lines.append(f"| {bucket} | {domain} | {count} |")
    lines += ["", f"JSON meta: `{meta_path}`"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Missing source dataset: {args.source}")

    source_rows = read_jsonl(args.source)
    rng = random.Random(args.seed)
    eval_rows, train_rows = build_eval_rows(source_rows, rng)
    validate(eval_rows, train_rows)

    write_jsonl(args.eval_output, eval_rows)
    write_jsonl(args.train_output, train_rows)
    write_report(args.report, args.meta, eval_rows, train_rows, len(source_rows), args.seed)

    print("Phase 18K: Marshmello General Benchmark")
    print("=" * 60)
    print(f"Eval:  {args.eval_output} ({len(eval_rows)} rows)")
    print(f"Train: {args.train_output} ({len(train_rows)} rows)")
    print(f"Report: {args.report}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
