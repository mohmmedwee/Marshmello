#!/usr/bin/env python3
"""Run the full Phase 14 dataset pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_ROOT))

from scripts.clean import run_clean  # noqa: E402
from scripts.common import (  # noqa: E402
    CLEANED_DIR,
    DEDUPED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    SHARDS_DIR,
    ensure_dirs,
)
from scripts.dedupe import run_dedupe  # noqa: E402
from scripts.ingest import run_ingest  # noqa: E402
from scripts.quality import run_quality  # noqa: E402
from scripts.shard import run_shard  # noqa: E402
from scripts.stats import print_summary, run_stats, write_reports  # noqa: E402

INGESTED_DIR = PHASE_ROOT / "ingested"


def run_pipeline(
    *,
    raw_dir: Path,
    shard_mb: float,
    min_words: int,
    skip_ingest: bool = False,
) -> dict[str, object]:
    ensure_dirs(CLEANED_DIR, DEDUPED_DIR, SHARDS_DIR, REPORTS_DIR, INGESTED_DIR)

    ingested_path = INGESTED_DIR / "documents.jsonl"
    cleaned_path = CLEANED_DIR / "documents.jsonl"
    deduped_path = DEDUPED_DIR / "documents.jsonl"
    filtered_path = DEDUPED_DIR / "filtered.jsonl"

    if skip_ingest:
        if not cleaned_path.exists():
            raise FileNotFoundError(
                f"--skip-ingest requires existing cleaned data at {cleaned_path}"
            )
        print("Step 1-2/6 — Ingest/Clean skipped (using existing cleaned data)")
        ingest_stats = {"files": 0, "documents": 0, "skipped": True}
        clean_stats = {"skipped": True}
    else:
        print("Step 1/6 — Ingest")
        ingest_stats = run_ingest(raw_dir, ingested_path)
        print(f"  {ingest_stats['documents']} documents from {ingest_stats['files']} files")

        print("Step 2/6 — Clean")
        clean_stats = run_clean(ingested_path, cleaned_path)
        print(f"  kept {clean_stats['output_documents']} (removed {clean_stats['removed']})")

    print("Step 3/6 — Dedupe")
    dedupe_stats = run_dedupe(cleaned_path, deduped_path)
    print(
        f"  kept {dedupe_stats['output_documents']} "
        f"(removed {dedupe_stats['duplicates_removed']} duplicates)"
    )

    print("Step 4/6 — Quality + domain tagging")
    quality_stats = run_quality(deduped_path, filtered_path, min_words=min_words)
    print(f"  kept {quality_stats['output_documents']} (filtered {quality_stats['filtered']})")

    print("Step 5/6 — Shard")
    max_bytes = int(shard_mb * 1024 * 1024)
    shard_stats = run_shard(filtered_path, SHARDS_DIR, max_bytes=max_bytes)
    shard_paths = [Path(p) for p in shard_stats["shard_paths"]]
    print(f"  {shard_stats['shards']} shard(s), max {shard_mb} MB each")

    print("Step 6/6 — Statistics")
    report, domain_distribution = run_stats(
        shard_paths=shard_paths,
        dedupe_stats=dedupe_stats,
        quality_stats=quality_stats,
        ingest_stats=ingest_stats,
    )
    report["clean"] = clean_stats
    report["shard"] = shard_stats
    report_paths = write_reports(report, domain_distribution, REPORTS_DIR)
    print_summary(report)
    print(f"\nReports written to {report_paths[0]} and {report_paths[1]}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 14 dataset pipeline end-to-end.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--shard-mb",
        type=float,
        default=1.0,
        help="Max shard size in MB (default 1 for demo; use 100 in production)",
    )
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse existing cleaned/documents.jsonl (skip ingest+clean)",
    )
    args = parser.parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        shard_mb=args.shard_mb,
        min_words=args.min_words,
        skip_ingest=args.skip_ingest,
    )


if __name__ == "__main__":
    main()
