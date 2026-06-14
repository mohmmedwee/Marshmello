"""Step 7 — dataset statistics and report generation."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from scripts.common import (  # noqa: E402
    REPORTS_DIR,
    ensure_dirs,
    iter_jsonl,
    word_count,
    write_json,
)


def estimate_bpe_tokens(text: str) -> int:
    """Rough token estimate (~1.35 tokens per word for English BPE)."""
    return int(word_count(text) * 1.35)


def run_stats(
    *,
    shard_paths: list[Path],
    dedupe_stats: dict[str, int | float],
    quality_stats: dict[str, int],
    ingest_stats: dict[str, int],
) -> dict[str, object]:
    documents = 0
    words = 0
    tokens = 0
    domains: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    sources: Counter[str] = Counter()

    for shard_path in shard_paths:
        if not shard_path.exists():
            continue
        for payload in iter_jsonl(shard_path):
            text = str(payload.get("text", ""))
            documents += 1
            words += word_count(text)
            tokens += estimate_bpe_tokens(text)
            domains[str(payload.get("domain", "general"))] += 1
            languages[str(payload.get("language", "unknown"))] += 1
            sources[str(payload.get("source", "unknown"))] += 1

    domain_distribution = {
        domain: round(count / documents, 4) if documents else 0.0
        for domain, count in domains.items()
    }

    report = {
        "documents": documents,
        "words": words,
        "estimated_bpe_tokens": tokens,
        "sources": dict(sources),
        "languages": dict(languages),
        "domains": dict(domains),
        "domain_percentages": {
            domain: f"{pct * 100:.1f}%"
            for domain, pct in sorted(domain_distribution.items(), key=lambda x: -x[1])
        },
        "ingest": ingest_stats,
        "quality": quality_stats,
        "dedupe": dedupe_stats,
        "duplicates_removed": dedupe_stats.get("duplicates_removed", 0),
        "duplicate_ratio": dedupe_stats.get("duplicate_ratio", 0.0),
        "duplicate_ratio_percent": f"{float(dedupe_stats.get('duplicate_ratio', 0.0)) * 100:.1f}%",
        "shard_count": len(shard_paths),
    }
    return report, domain_distribution


def write_reports(
    report: dict[str, object],
    domain_distribution: dict[str, float],
    reports_dir: Path,
) -> tuple[Path, Path]:
    ensure_dirs(reports_dir)
    report_path = reports_dir / "dataset_report.json"
    domain_path = reports_dir / "domain_distribution.json"
    write_json(report_path, report)
    write_json(domain_path, {"domains": domain_distribution, "percentages": report["domain_percentages"]})
    return report_path, domain_path


def print_summary(report: dict[str, object]) -> None:
    print("\n=== Dataset Report ===")
    print(f"Documents: {report['documents']}")
    print(f"Words: {report['words']:,}")
    print(f"BPE Tokens (est.): {report['estimated_bpe_tokens']:,}")
    print("\nDomains:")
    for domain, pct in report["domain_percentages"].items():
        print(f"  {domain}: {pct}")
    print(f"\nDuplicates removed: {report['duplicate_ratio_percent']}")
    print(f"Languages: {report['languages']}")
    print(f"Shards: {report['shard_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset statistics reports.")
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    shard_paths = sorted(args.shards_dir.glob("shard_*.jsonl"))
    report, domain_distribution = run_stats(
        shard_paths=shard_paths,
        dedupe_stats={},
        quality_stats={},
        ingest_stats={},
    )
    paths = write_reports(report, domain_distribution, args.reports_dir)
    print_summary(report)
    print(f"\nReports -> {paths[0]}, {paths[1]}")


if __name__ == "__main__":
    main()
