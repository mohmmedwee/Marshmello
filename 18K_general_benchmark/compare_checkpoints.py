#!/usr/bin/env python3
"""Compare three Marshmello checkpoints on the Phase 18K general benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluate_general import evaluate_checkpoint, pct

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
REPORT_DIR = PHASE_ROOT / "reports"
DATA_DIR = PHASE_ROOT / "data"
DEFAULT_EVAL = DATA_DIR / "general_eval.jsonl"

CHECKPOINTS: tuple[tuple[str, Path], ...] = (
    ("base", PROJECT_ROOT / "13_gpt_pretraining" / "checkpoints" / "large_50m" / "latest.pt"),
    (
        "best_18j_routing",
        PROJECT_ROOT / "18B_marshmello_instruct" / "checkpoints" / "best_18j_routing.pt",
    ),
    (
        "teacher_latest",
        PROJECT_ROOT / "18E_tiny_teacher_sft" / "checkpoints" / "teacher_latest.pt",
    ),
)


def write_comparison_markdown(path: Path, payloads: list[dict]) -> None:
    lines = [
        "# Phase 18K — General Benchmark Comparison",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Eval: `{DEFAULT_EVAL.name}` (500 held-out, greedy decoding)",
        "",
        "## Overall",
        "",
        "| Checkpoint | Domain score | Keyword recall | Token overlap | Hallucination | Empty | Repetition |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        summary = payload["summary"]
        lines.append(
            f"| {payload['label']} | {pct(float(summary['domain_score']))} | "
            f"{pct(float(summary['keyword_recall']))} | "
            f"{pct(float(summary['reference_token_overlap']))} | "
            f"{pct(float(summary['hallucination_rate']))} | "
            f"{pct(float(summary['empty_response_rate']))} | "
            f"{pct(float(summary['repetition_rate']))} |"
        )

    lines += [
        "",
        "## By benchmark bucket",
        "",
    ]
    for payload in payloads:
        lines += [f"### {payload['label']}", "", "| Bucket | Domain score | Hallucination |", "|---|---:|---:|"]
        for bucket, metrics in payload["summary"]["by_bucket"].items():
            lines.append(
                f"| {bucket} | {pct(float(metrics['domain_score']))} | "
                f"{pct(float(metrics['hallucination_rate']))} |"
            )
        lines.append("")

    lines += ["## Sample outputs (10 per checkpoint)", ""]
    for payload in payloads:
        lines += [f"### {payload['label']}", ""]
        for sample in payload["sample_outputs"]:
            lines += [
                f"#### {sample['id']}",
                "",
                f"**Q:** {sample['instruction']}",
                "",
                f"**Generated:** {sample['generated_response']}",
                "",
                f"**Reference:** {sample['reference_response']}",
                "",
            ]

    lines += [
        "## How to read this with 18J",
        "",
        "- **18J** measures core concept routing (currently ~18% on `best_18j_routing`).",
        "- **18K** measures general assistant quality on held-out broad SFT-style questions.",
        "- A useful assistant can have low 18J routing but higher 18K scores after broad SFT.",
        "- A 300M decision should consider **both** benchmarks, not 18J alone.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--report-md", type=Path, default=REPORT_DIR / "comparison.md")
    parser.add_argument("--report-json", type=Path, default=REPORT_DIR / "comparison.json")
    parser.add_argument("--config", default="large_50m")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payloads: list[dict] = []

    for label, checkpoint in CHECKPOINTS:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {label}: {checkpoint}")
        print(f"\n=== 18K eval: {label} ===", flush=True)
        payload = evaluate_checkpoint(
            eval_data=args.eval_data,
            checkpoint=checkpoint,
            label=label,
            config=args.config,
            max_new_tokens=args.max_new_tokens,
            cpu=args.cpu,
            results_json=REPORT_DIR / f"{label}_results.json",
            report_md=REPORT_DIR / f"{label}_report.md",
            predictions_output=REPORT_DIR / f"{label}_predictions.jsonl",
        )
        payloads.append(payload)
        summary = payload["summary"]
        print(
            f"{label}: domain_score={pct(float(summary['domain_score']))} "
            f"hallucination={pct(float(summary['hallucination_rate']))}",
            flush=True,
        )

    comparison = {
        "phase": "18K_general_benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_data": str(args.eval_data),
        "checkpoints": [
            {"label": label, "path": str(path)} for label, path in CHECKPOINTS
        ],
        "results": [
            {
                "label": payload["label"],
                "checkpoint": payload["checkpoint"],
                "summary": payload["summary"],
                "sample_outputs": payload["sample_outputs"],
            }
            for payload in payloads
        ],
    }
    args.report_json.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    write_comparison_markdown(args.report_md, payloads)

    print(f"\nComparison markdown: {args.report_md}")
    print(f"Comparison JSON: {args.report_json}")


if __name__ == "__main__":
    main()
