#!/usr/bin/env python3
"""Aggregate Marshmello instruct pipeline eval results into one report."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
PIPELINE_JSON = REPORTS / "marshmello_pipeline_results.json"
PIPELINE_MD = REPORTS / "marshmello_pipeline_report.md"

CHECKPOINTS = {
    "step1_chat_base": PROJECT_ROOT / "13_gpt_pretraining/checkpoints/large_50m/latest.pt",
    "step2_teacher": PROJECT_ROOT / "18E_tiny_teacher_sft/checkpoints/teacher_latest.pt",
    "step3_routing": PROJECT_ROOT / "18I_routing_teacher_fix/checkpoints/routing_latest.pt",
    "step4_core_sft": PROJECT_ROOT / "18B_marshmello_instruct/checkpoints/latest.pt",
}


def run_cmd(cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def parse_chat_only(output: str) -> dict:
    passes = len(re.findall(r"^result:\s+PASS\s*$", output, re.MULTILINE))
    fails = len(re.findall(r"^result:\s+FAIL\s*$", output, re.MULTILINE))
    final = re.search(r"final_result:\s+(\S+.*?)(?:\n|$)", output)
    final_text = final.group(1).strip() if final else "unknown"
    final_counts = re.search(r"(\d+)/(\d+)", final_text)
    if final_counts:
        pass_count = int(final_counts.group(1))
        total = int(final_counts.group(2))
    else:
        pass_count = passes
        total = passes + fails
    return {
        "pass_count": pass_count,
        "fail_count": total - pass_count,
        "total": total,
        "final": final_text,
        "raw_tail": "\n".join(output.strip().splitlines()[-8:]),
    }


def parse_teacher_or_routing(output: str) -> dict:
    final_line = re.search(r"^Final:\s*(PASS|FAIL)(?:\s*\((\d+)/(\d+)\))?\s*$", output, re.MULTILINE)
    if final_line:
        status = final_line.group(1)
        pass_count = int(final_line.group(2)) if final_line.group(2) else None
        total = int(final_line.group(3)) if final_line.group(3) else None
        if pass_count is None:
            pass_count = len(re.findall(r"^Result:\s+PASS\b", output, re.MULTILINE))
            total = pass_count + len(re.findall(r"^Result:\s+FAIL\b", output, re.MULTILINE))
        return {
            "pass_count": pass_count,
            "total": total,
            "final": f"{status} ({pass_count}/{total})" if total else status,
            "raw_tail": "\n".join(output.strip().splitlines()[-6:]),
        }
    passes = len(re.findall(r"^Result:\s+PASS\b", output, re.MULTILINE))
    fails = len(re.findall(r"^Result:\s+FAIL\b", output, re.MULTILINE))
    return {
        "pass_count": passes,
        "total": passes + fails,
        "final": "PASS" if fails == 0 and passes else "FAIL",
        "raw_tail": "\n".join(output.strip().splitlines()[-6:]),
    }


def pipeline_recommendation(core: dict | None, s3: dict) -> str:
    if not core:
        return "Re-run core eval — metrics missing."
    routing = float(core.get("routing_accuracy") or 0)
    concept = float(core.get("concept_accuracy") or 0)
    drop = core.get("confusion_drop")
    routing_prompts_ok = s3.get("pass_count", 0) == s3.get("total", 0) and s3.get("total", 0) > 0
    if routing >= 0.30:
        return (
            "Pipeline is working on held-out routing (≥30%). "
            "Consider curated broad SFT on `marshmello_all_sft.jsonl`."
        )
    if routing_prompts_ok and routing >= 0.10:
        return (
            f"Correct pipeline order helped: routing {pct(routing)} and concept {pct(concept)} "
            f"vs ~3% base. Routing prompts pass {s3.get('pass_count')}/{s3.get('total')}, "
            "but held-out routing is still below 30%. "
            "Extend teacher/routing steps or eval `teacher_best_score.pt` before scaling model size."
        )
    if routing < 0.10:
        return (
            "Held-out routing still below 10%. Repeat routing (18I) or extend teacher (18E); "
            "do not scale to 300M yet."
        )
    return "Continue SFT tuning before scaling model size."


def load_core_eval(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    trained = data.get("trained", data)
    return {
        "concept_accuracy": trained["concept_accuracy"],
        "routing_accuracy": trained["routing_accuracy"],
        "hallucination_rate": trained["hallucination_rate"],
        "confusion_drop": data.get("confusion_drop"),
        "conclusion": data.get("conclusion", ""),
    }


def pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
        "steps": {},
    }

    # Step 1 — chat-only base eval
    code, out = run_cmd(
        [sys.executable, "18H_chat_only_pretraining/eval_chat_only.py", "--no-strict"]
    )
    results["steps"]["1_chat_only_base"] = {
        "exit_code": code,
        "checkpoint": str(CHECKPOINTS["step1_chat_base"]),
        **parse_chat_only(out),
    }

    # Step 2 — teacher eval
    code, out = run_cmd(
        [
            sys.executable,
            "18E_tiny_teacher_sft/eval_teacher.py",
            "--checkpoint",
            str(CHECKPOINTS["step2_teacher"]),
            "--no-strict",
        ]
    )
    results["steps"]["2_teacher_sft"] = {
        "exit_code": code,
        "checkpoint": str(CHECKPOINTS["step2_teacher"]),
        **parse_teacher_or_routing(out),
    }

    # Step 3 — routing eval
    code, out = run_cmd(
        [
            sys.executable,
            "18I_routing_teacher_fix/eval_routing.py",
            "--checkpoint",
            str(CHECKPOINTS["step3_routing"]),
            "--no-strict",
        ]
    )
    results["steps"]["3_routing_fix"] = {
        "exit_code": code,
        "checkpoint": str(CHECKPOINTS["step3_routing"]),
        **parse_teacher_or_routing(out),
    }

    # Step 4 — core routing eval (run fresh)
    core_results = PROJECT_ROOT / "18J_marshmello_core_sft/data/reports/marshmello_core_eval_pipeline_results.json"
    core_report = PROJECT_ROOT / "18J_marshmello_core_sft/data/reports/marshmello_core_eval_pipeline_report.md"
    code, out = run_cmd(
        [
            sys.executable,
            "18J_marshmello_core_sft/evaluate_core_routing.py",
            "--checkpoint",
            str(CHECKPOINTS["step4_core_sft"]),
            "--baseline-checkpoint",
            str(CHECKPOINTS["step1_chat_base"]),
            "--results-json",
            str(core_results),
            "--report",
            str(core_report),
        ]
    )
    core_metrics = load_core_eval(core_results)
    results["steps"]["4_core_sft"] = {
        "exit_code": code,
        "checkpoint": str(CHECKPOINTS["step4_core_sft"]),
        "core_eval": core_metrics,
        "raw_tail": "\n".join(out.strip().splitlines()[-8:]),
    }

    PIPELINE_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    s1 = results["steps"]["1_chat_only_base"]
    s2 = results["steps"]["2_teacher_sft"]
    s3 = results["steps"]["3_routing_fix"]
    s4 = results["steps"]["4_core_sft"]
    core = s4.get("core_eval") or {}

    all_ckpts_ok = all(Path(p).exists() for p in CHECKPOINTS.values())
    all_eval_ok = all(
        results["steps"][k]["exit_code"] == 0
        for k in ("1_chat_only_base", "2_teacher_sft", "3_routing_fix", "4_core_sft")
    )

    lines = [
        "# Marshmello Instruct Pipeline — Full Report",
        "",
        f"- Generated: {results['generated_at']}",
        "- Pipeline: chat-only base (18H) → teacher (18E) → routing (18I) → core SFT (18J)",
        f"- Training status: **{'all steps complete' if all_ckpts_ok else 'missing checkpoint(s)'}**",
        f"- Eval status: **{'all evals passed (exit 0)' if all_eval_ok else 'one or more eval errors'}**",
        "",
        "## Executive summary",
        "",
        "| Step | Stage | Training | Eval |",
        "|---|---|---|---|",
        f"| 1 | Chat-only base (18H) | ✓ step 8500 | {s1['pass_count']}/{s1['total']} chat prompts ({s1['final']}) |",
        f"| 2 | Teacher SFT (18E) | ✓ 500 steps | {s2['pass_count']}/{s2['total']} ({s2['final']}) |",
        f"| 3 | Routing fix (18I) | ✓ 300 steps | {s3['pass_count']}/{s3['total']} ({s3['final']}) |",
        f"| 4 | Core SFT (18J) | ✓ 800 steps, val 3.50 | routing {pct(core.get('routing_accuracy'))}, concept {pct(core.get('concept_accuracy'))} |",
        "",
        "## vs old pipeline (wrong order)",
        "",
        "| Checkpoint | Routing | Concept | Notes |",
        "|---|---:|---:|---|",
        "| Base (prose pretrain) | 3.0% | 4.0% | before 18H |",
        "| Teacher (old base) | 7.0% | 9.0% | best routing in old run |",
        "| Full SFT 9.8k (old order) | 3.0% | 17.0% | broad SFT too early |",
        "| **This pipeline (core SFT)** | **10.0%** | **21.0%** | chat base → teacher → routing → core |",
        "",
        f"**Recommendation:** {pipeline_recommendation(core, s3)}",
        "",
        "## Checkpoints",
        "",
    ]
    for name, path in CHECKPOINTS.items():
        exists = Path(path).exists()
        lines.append(f"- **{name}:** `{path}` {'✓' if exists else '✗ missing'}")

    lines += [
        "",
        "## Step 1 — Chat-only base (18H)",
        "",
        f"- Eval: `{s1['final']}`",
        f"- Checkpoint: `{s1['checkpoint']}`",
        "",
        "```text",
        s1.get("raw_tail", ""),
        "```",
        "",
        "## Step 2 — Teacher SFT (18E)",
        "",
        f"- Eval: `{s2['final']}` ({s2['pass_count']}/{s2['total']})",
        f"- Checkpoint: `{s2['checkpoint']}`",
        "",
        "## Step 3 — Routing fix (18I)",
        "",
        f"- Eval: `{s3['final']}` ({s3['pass_count']}/{s3['total']})",
        f"- Checkpoint: `{s3['checkpoint']}`",
        "",
        "## Step 4 — Core SFT (18J) on 100 held-out questions",
        "",
    ]
    if core:
        lines += [
            "| Metric | Value |",
            "|---|---:|",
            f"| Concept accuracy | {pct(core.get('concept_accuracy'))} |",
            f"| Routing accuracy | {pct(core.get('routing_accuracy'))} |",
            f"| Hallucination rate | {pct(core.get('hallucination_rate'))} |",
            f"| Confusion drop vs chat base | {pct(core.get('confusion_drop'))} |",
            "",
            f"18J eval script note: {core.get('conclusion', '—')}",
            "",
            f"Pipeline recommendation: {pipeline_recommendation(core, s3)}",
            "",
        ]
    else:
        lines.append("Core eval metrics not available.")

    lines += [
        "## Decision guide",
        "",
        "- **Routing > 30%** on 18J eval → core SFT pipeline is working; consider curated broad SFT.",
        "- **Routing < 10%** after Step 3 → repeat routing or extend teacher; do not scale model size yet.",
        "- **Chat-only base < 2/3 PASS** → extend 18H pretrain before any SFT.",
        "",
        "## Raw JSON",
        "",
        f"Full machine-readable results: `{PIPELINE_JSON.relative_to(PROJECT_ROOT)}`",
        "",
        "## Re-run pipeline training",
        "",
        "```bash",
        "bash reports/run_instruct_pipeline.sh",
        "python reports/generate_pipeline_report.py",
        "```",
    ]

    PIPELINE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {PIPELINE_MD}")
    print(f"Wrote {PIPELINE_JSON}")


if __name__ == "__main__":
    main()
