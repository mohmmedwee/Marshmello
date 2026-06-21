#!/usr/bin/env python3
"""Evaluate one checkpoint on the Phase 18K general assistant benchmark."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
REPORT_DIR = PHASE_ROOT / "reports"
DATA_DIR = PHASE_ROOT / "data"
DEFAULT_EVAL = DATA_DIR / "general_eval.jsonl"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "18B_marshmello_instruct" / "checkpoints" / "best_18j_routing.pt"

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
EXPECTED_BUCKETS = ("ai", "databases", "programming", "system_design", "general_knowledge")
SAMPLE_COUNT = 10


@dataclass
class ExampleScore:
    id: str
    instruction: str
    benchmark_bucket: str
    domain: str
    concept: str
    generated_response: str
    reference_response: str
    keyword_recall: float
    reference_token_overlap: float
    response_length_sane: bool
    repetition: bool
    empty_response: bool
    hallucination: bool
    domain_score: float
    matched_keywords: list[str]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def word_count(text: str) -> int:
    return len(word_tokens(text))


def phrase_present(text: str, phrase: str) -> bool:
    normalized_text = f" {normalize(text)} "
    normalized_phrase = normalize(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def repeated_ngram(text: str, n: int = 3) -> bool:
    tokens = word_tokens(text)
    seen: set[tuple[str, ...]] = set()
    for index in range(len(tokens) - n + 1):
        ngram = tuple(tokens[index : index + n])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


def token_f1(generated: str, reference: str) -> float:
    generated_counts = Counter(word_tokens(generated))
    reference_counts = Counter(word_tokens(reference))
    if not generated_counts or not reference_counts:
        return 0.0
    common = sum((generated_counts & reference_counts).values())
    precision = common / sum(generated_counts.values())
    recall = common / sum(reference_counts.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reference_field(record: dict[str, Any]) -> str:
    return str(record.get("reference_response", record.get("response", "")))


def keywords_field(record: dict[str, Any]) -> list[str]:
    raw = record.get("expected_keywords", record.get("keywords", []))
    return [str(item) for item in raw]


def length_sane(generated: str, reference: str) -> bool:
    gen_words = word_count(generated)
    ref_words = max(1, word_count(reference))
    if gen_words < 4:
        return False
    if gen_words > max(160, ref_words * 3):
        return False
    return gen_words >= max(4, int(ref_words * 0.15))


def hallucination_proxy(generated: str, reference: str, repetition: bool, empty: bool) -> bool:
    overlap = token_f1(generated, reference)
    if empty or repetition:
        return True
    return overlap < 0.12 and word_count(generated) >= 8


def domain_score(
    keyword_recall: float,
    reference_token_overlap: float,
    response_length_sane: bool,
    repetition: bool,
    empty_response: bool,
    hallucination: bool,
) -> float:
    sanity = 1.0 if response_length_sane and not empty_response and not repetition else 0.0
    grounded = 0.0 if hallucination else 1.0
    return 0.35 * keyword_recall + 0.35 * reference_token_overlap + 0.15 * sanity + 0.15 * grounded


def validate_eval(records: list[dict[str, Any]]) -> None:
    if len(records) != 500:
        raise ValueError(f"Expected 500 held-out questions, found {len(records)}")
    bucket_counts = Counter(str(record["benchmark_bucket"]) for record in records)
    for bucket in EXPECTED_BUCKETS:
        if bucket_counts[bucket] != 100:
            raise ValueError(f"Bucket {bucket}: expected 100, found {bucket_counts[bucket]}")
    instructions = [normalize(str(record["instruction"])) for record in records]
    if len(set(instructions)) != len(records):
        raise ValueError("Eval instructions must be unique")
    for index, record in enumerate(records, start=1):
        if not reference_field(record).strip():
            raise ValueError(f"Row {index} missing reference_response")
        if not keywords_field(record):
            raise ValueError(f"Row {index} missing expected_keywords")


def score_example(record: dict[str, Any], generated: str) -> ExampleScore:
    reference = reference_field(record)
    keywords = keywords_field(record)
    matched = [keyword for keyword in keywords if phrase_present(generated, keyword)]
    keyword_recall = len(matched) / max(1, len(keywords))
    reference_token_overlap = token_f1(generated, reference)
    empty_response = word_count(generated) < 4
    repetition = repeated_ngram(generated)
    response_length_sane = length_sane(generated, reference)
    hallucination = hallucination_proxy(generated, reference, repetition, empty_response)
    score = domain_score(
        keyword_recall,
        reference_token_overlap,
        response_length_sane,
        repetition,
        empty_response,
        hallucination,
    )
    return ExampleScore(
        id=str(record["id"]),
        instruction=str(record["instruction"]),
        benchmark_bucket=str(record["benchmark_bucket"]),
        domain=str(record.get("domain", "")),
        concept=str(record.get("concept", "")),
        generated_response=generated,
        reference_response=reference,
        keyword_recall=keyword_recall,
        reference_token_overlap=reference_token_overlap,
        response_length_sane=response_length_sane,
        repetition=repetition,
        empty_response=empty_response,
        hallucination=hallucination,
        domain_score=score,
        matched_keywords=matched,
    )


def grouped_summary(scores: list[ExampleScore], field: str) -> dict[str, dict[str, float | int]]:
    by_key: dict[str, list[ExampleScore]] = {}
    for score in scores:
        by_key.setdefault(getattr(score, field), []).append(score)
    result: dict[str, dict[str, float | int]] = {}
    for key, items in sorted(by_key.items()):
        count = len(items)
        result[key] = {
            "count": count,
            "keyword_recall": sum(item.keyword_recall for item in items) / count,
            "reference_token_overlap": sum(item.reference_token_overlap for item in items) / count,
            "response_length_sanity_rate": sum(item.response_length_sane for item in items) / count,
            "repetition_rate": sum(item.repetition for item in items) / count,
            "empty_response_rate": sum(item.empty_response for item in items) / count,
            "hallucination_rate": sum(item.hallucination for item in items) / count,
            "domain_score": sum(item.domain_score for item in items) / count,
        }
    return result


def summarize(scores: list[ExampleScore]) -> dict[str, Any]:
    count = len(scores)
    return {
        "count": count,
        "keyword_recall": sum(s.keyword_recall for s in scores) / count,
        "reference_token_overlap": sum(s.reference_token_overlap for s in scores) / count,
        "response_length_sanity_rate": sum(s.response_length_sane for s in scores) / count,
        "repetition_rate": sum(s.repetition for s in scores) / count,
        "empty_response_rate": sum(s.empty_response for s in scores) / count,
        "hallucination_rate": sum(s.hallucination for s in scores) / count,
        "domain_score": sum(s.domain_score for s in scores) / count,
        "by_bucket": grouped_summary(scores, "benchmark_bucket"),
        "by_domain": grouped_summary(scores, "domain"),
    }


def pick_samples(scores: list[ExampleScore], count: int = SAMPLE_COUNT) -> list[dict[str, Any]]:
    if len(scores) <= count:
        return [asdict(score) for score in scores]
    step = max(1, len(scores) // count)
    picked = [scores[index] for index in range(0, len(scores), step)][:count]
    return [asdict(score) for score in picked]


def load_prediction_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in read_jsonl(path):
        identifier = str(record.get("id", ""))
        response = str(record.get("generated_response", record.get("response", "")))
        if not identifier:
            raise ValueError(f"Prediction record missing id: {record}")
        result[identifier] = response
    return result


def generate_predictions(
    records: list[dict[str, Any]],
    checkpoint: Path,
    config: str,
    max_new_tokens: int,
    cpu: bool,
) -> dict[str, str]:
    phase_root = PROJECT_ROOT / "18B_marshmello_instruct"
    phase13_root = PROJECT_ROOT / "13_gpt_pretraining"
    sys.path.insert(0, str(phase_root))
    sys.path.insert(0, str(phase13_root))
    from chat import generate_reply, load_model  # type: ignore
    from training.trainer import pick_device  # type: ignore

    device = pick_device(force_cpu=cpu)
    model, bpe = load_model(checkpoint, config, device)
    predictions: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        answer = generate_reply(
            model,
            bpe,
            str(record["instruction"]),
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        predictions[str(record["id"])] = answer
        print(f"[{index:03d}/{len(records):03d}] {record['id']}: {answer[:100]}", flush=True)
    return predictions


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown_report(
    path: Path,
    label: str,
    checkpoint: Path,
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    lines = [
        f"# Phase 18K — General Benchmark ({label})",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Checkpoint: `{checkpoint}`",
        f"- Questions: {summary['count']}",
        f"- Decoding: greedy",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Keyword recall | {pct(float(summary['keyword_recall']))} |",
        f"| Reference token overlap | {pct(float(summary['reference_token_overlap']))} |",
        f"| Response length sanity | {pct(float(summary['response_length_sanity_rate']))} |",
        f"| Repetition rate | {pct(float(summary['repetition_rate']))} |",
        f"| Empty response rate | {pct(float(summary['empty_response_rate']))} |",
        f"| Hallucination proxy | {pct(float(summary['hallucination_rate']))} |",
        f"| Domain score (composite) | {pct(float(summary['domain_score']))} |",
        "",
        "## By benchmark bucket",
        "",
        "| Bucket | Domain score | Keyword | Overlap | Hallucination |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket, metrics in summary["by_bucket"].items():
        lines.append(
            f"| {bucket} | {pct(float(metrics['domain_score']))} | "
            f"{pct(float(metrics['keyword_recall']))} | "
            f"{pct(float(metrics['reference_token_overlap']))} | "
            f"{pct(float(metrics['hallucination_rate']))} |"
        )
    lines += ["", f"## Sample outputs ({len(samples)})", ""]
    for sample in samples:
        lines += [
            f"### {sample['id']}",
            "",
            f"**Q:** {sample['instruction']}",
            "",
            f"**Reference:** {sample['reference_response']}",
            "",
            f"**Generated:** {sample['generated_response']}",
            "",
            f"- keyword recall: {pct(float(sample['keyword_recall']))}, "
            f"overlap: {pct(float(sample['reference_token_overlap']))}, "
            f"domain score: {pct(float(sample['domain_score']))}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_checkpoint(
    *,
    eval_data: Path,
    checkpoint: Path,
    label: str,
    config: str = "large_50m",
    max_new_tokens: int = 120,
    cpu: bool = False,
    predictions_path: Path | None = None,
    results_json: Path | None = None,
    report_md: Path | None = None,
    predictions_output: Path | None = None,
) -> dict[str, Any]:
    eval_records = read_jsonl(eval_data)
    validate_eval(eval_records)

    if predictions_path:
        predictions = load_prediction_map(predictions_path)
    else:
        predictions = generate_predictions(eval_records, checkpoint, config, max_new_tokens, cpu)
        if predictions_output:
            write_jsonl(
                predictions_output,
                [
                    {
                        "id": record["id"],
                        "instruction": record["instruction"],
                        "benchmark_bucket": record["benchmark_bucket"],
                        "generated_response": predictions[str(record["id"])],
                    }
                    for record in eval_records
                ],
            )

    scores = [score_example(record, predictions[str(record["id"])]) for record in eval_records]
    summary = summarize(scores)
    samples = pick_samples(scores)

    payload = {
        "phase": "18K_general_benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "checkpoint": str(checkpoint),
        "eval_data": str(eval_data),
        "summary": summary,
        "sample_outputs": samples,
        "examples": [asdict(score) for score in scores],
    }

    if results_json:
        results_json.parent.mkdir(parents=True, exist_ok=True)
        results_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if report_md:
        write_markdown_report(report_md, label, checkpoint, summary, samples)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="checkpoint")
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default="large_50m")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--results-json", type=Path, default=REPORT_DIR / "general_eval_results.json")
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "general_eval_report.md")
    parser.add_argument("--predictions-output", type=Path, default=REPORT_DIR / "general_predictions.jsonl")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate_eval(read_jsonl(args.eval_data))
        print(f"Validated {len(read_jsonl(args.eval_data))} held-out general questions")
        return

    payload = evaluate_checkpoint(
        eval_data=args.eval_data,
        checkpoint=args.checkpoint,
        label=args.label,
        config=args.config,
        max_new_tokens=args.max_new_tokens,
        cpu=args.cpu,
        predictions_path=args.predictions,
        results_json=args.results_json,
        report_md=args.report,
        predictions_output=args.predictions_output,
    )
    summary = payload["summary"]
    print(f"Keyword recall:           {pct(float(summary['keyword_recall']))}")
    print(f"Reference token overlap:  {pct(float(summary['reference_token_overlap']))}")
    print(f"Domain score:             {pct(float(summary['domain_score']))}")
    print(f"Hallucination proxy:      {pct(float(summary['hallucination_rate']))}")
    print(f"JSON results: {args.results_json}")
    print(f"Markdown report: {args.report}")


if __name__ == "__main__":
    main()
