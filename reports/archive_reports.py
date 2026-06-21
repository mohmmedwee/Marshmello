#!/usr/bin/env python3
"""Archive Marshmello eval reports into reports/archive/<run_id>/."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "reports" / "archive"
LATEST_SUMMARY = PROJECT_ROOT / "reports" / "latest_eval_summary.md"

REPORT_FILES = [
    "18J_marshmello_core_sft/data/reports/marshmello_core_eval_comparison.md",
    "18J_marshmello_core_sft/data/reports/marshmello_core_eval_fix_report.md",
    "18J_marshmello_core_sft/data/reports/marshmello_core_eval_teacher_best_score_report.md",
    "18J_marshmello_core_sft/data/reports/marshmello_core_eval_report.md",
    "reports/marshmello_pipeline_report.md",
]


def archive(run_id: str | None, label: str, metrics: dict | None = None) -> Path:
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    dest = ARCHIVE_ROOT / run_id
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in REPORT_FILES:
        src = PROJECT_ROOT / rel
        if src.exists():
            target = dest / Path(rel).name
            shutil.copy2(src, target)
            copied.append(target.name)
    meta = {
        "run_id": run_id,
        "label": label,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "files": copied,
        "metrics": metrics or {},
    }
    (dest / "archive_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dest


def write_latest_summary(rows: list[dict], best: dict) -> None:
    lines = [
        "# Marshmello — Latest Eval Summary",
        "",
        f"- Updated: {datetime.now(timezone.utc).isoformat()}",
        f"- Best routing (18J): **{100 * best.get('routing_accuracy', 0):.1f}%** "
        f"(`{best.get('checkpoint', '?')}`)",
        "",
        "## All runs this session",
        "",
        "| Label | Checkpoint | Routing | Concept | Hallucination |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('label', '?')} | `{row.get('checkpoint', '?')}` | "
            f"{100 * row.get('routing_accuracy', 0):.1f}% | "
            f"{100 * row.get('concept_accuracy', 0):.1f}% | "
            f"{100 * row.get('hallucination_rate', 0):.1f}% |"
        )
    lines += [
        "",
        "## Archives",
        "",
        f"See `reports/archive/` — latest folder: `{best.get('archive', '?')}`",
    ]
    LATEST_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    args = parser.parse_args()
    metrics = None
    if args.metrics_json and args.metrics_json.exists():
        metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    dest = archive(args.run_id, args.label, metrics)
    print(dest)


if __name__ == "__main__":
    main()
