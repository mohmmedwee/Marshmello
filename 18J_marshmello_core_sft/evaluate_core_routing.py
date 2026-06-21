#!/usr/bin/env python3
"""Evaluate Marshmello concept routing on the held-out core SFT set."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
DATA_DIR = PHASE_ROOT / "data"
DEFAULT_EVAL = DATA_DIR / "marshmello_core_eval.jsonl"
DEFAULT_SFT = DATA_DIR / "marshmello_core_sft.jsonl"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "18B_marshmello_instruct" / "checkpoints" / "latest.pt"
DEFAULT_BASELINE = PROJECT_ROOT / "13_gpt_pretraining" / "checkpoints" / "large_50m" / "latest.pt"
DEFAULT_RESULTS = DATA_DIR / "reports" / "marshmello_core_eval_results.json"
DEFAULT_REPORT = DATA_DIR / "reports" / "marshmello_core_eval_report.md"
DEFAULT_PREDICTIONS = DATA_DIR / "reports" / "marshmello_core_predictions.jsonl"
DEFAULT_BASELINE_PREDICTIONS = DATA_DIR / "reports" / "marshmello_core_baseline_predictions.jsonl"

EXPECTED_EVAL_DISTRIBUTION = {
    "ai_basics": 40,
    "transformers_llms": 30,
    "databases": 30,
}
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass
class ExampleScore:
    id: str
    instruction: str
    expected_domain: str
    expected_concept: str
    predicted_domain: str
    predicted_concept: str
    question_type: str
    routing_variant: str
    generated_response: str
    reference_response: str
    concept_correct: bool
    routing_correct: bool
    exact_answer_overlap: bool
    keyword_overlap: float
    reference_token_overlap: float
    hallucination: bool
    expected_score: float
    predicted_score: float
    matched_expected_keywords: list[str]
    matched_hard_negatives: list[str]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def phrase_present(text: str, phrase: str) -> bool:
    normalized_text = f" {normalize(text)} "
    normalized_phrase = normalize(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def phrase_position(text: str, phrase: str) -> int:
    normalized_text = normalize(text)
    normalized_phrase = normalize(phrase)
    if not normalized_phrase:
        return 10**9
    position = normalized_text.find(normalized_phrase)
    return position if position >= 0 else 10**9


def phrase_word_position(text: str, phrase: str) -> int:
    text_words = word_tokens(text)
    phrase_words = word_tokens(phrase)
    if not phrase_words:
        return 10**9
    for index in range(len(text_words) - len(phrase_words) + 1):
        if text_words[index : index + len(phrase_words)] == phrase_words:
            return index
    return 10**9


def repeated_ngram(text: str, n: int = 3) -> bool:
    tokens = word_tokens(text)
    seen: set[tuple[str, ...]] = set()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


def exact_overlap(generated: str, reference: str) -> bool:
    return normalize(generated) == normalize(reference) and bool(normalize(reference))


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


def validate_data(
    eval_records: list[dict[str, Any]],
    sft_records: list[dict[str, Any]],
) -> None:
    if len(eval_records) != 100:
        raise ValueError(f"Expected 100 held-out questions, found {len(eval_records)}")
    for domain, expected in EXPECTED_EVAL_DISTRIBUTION.items():
        actual = sum(record.get("domain") == domain for record in eval_records)
        if actual != expected:
            raise ValueError(f"{domain}: expected {expected} eval questions, found {actual}")
    eval_questions = {normalize(str(record["instruction"])) for record in eval_records}
    train_questions = {normalize(str(record["instruction"])) for record in sft_records}
    if len(eval_questions) != len(eval_records):
        raise ValueError("Evaluation questions are not unique")
    overlap = eval_questions & train_questions
    if overlap:
        raise ValueError(f"Evaluation questions overlap training: {sorted(overlap)[:3]}")
    required = {
        "id",
        "instruction",
        "response",
        "domain",
        "concept",
        "question_type",
        "routing_variant",
        "hard_negative_concepts",
        "keywords",
    }
    for index, record in enumerate(eval_records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Eval record {index} is missing {sorted(missing)}")


def build_profiles(
    sft_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, float], dict[str, str]]:
    profiles: dict[str, set[str]] = defaultdict(set)
    domains: dict[str, str] = {}
    primary_keywords: dict[str, str] = {}
    for record in [*sft_records, *eval_records]:
        concept = str(record["concept"])
        domains[concept] = str(record["domain"])
        keywords = [str(keyword) for keyword in record.get("keywords", [])]
        profiles[concept].update(keywords)
        if keywords:
            primary_keywords.setdefault(concept, keywords[0])
        profiles[concept].add(concept.replace("_", " "))

    concept_count = len(profiles)
    document_frequency: Counter[str] = Counter()
    for phrases in profiles.values():
        for phrase in {normalize(item) for item in phrases if normalize(item)}:
            document_frequency[phrase] += 1
    weights = {
        phrase: math.log((1 + concept_count) / (1 + frequency)) + 1.0
        for phrase, frequency in document_frequency.items()
    }
    return dict(profiles), domains, weights, primary_keywords


def concept_scores(
    answer: str,
    profiles: dict[str, set[str]],
    weights: dict[str, float],
    primary_keywords: dict[str, str],
    allowed_contrast: str | None,
) -> tuple[dict[str, float], dict[str, list[str]], dict[str, float]]:
    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}
    positions: dict[str, float] = {}
    for concept, phrases in profiles.items():
        matched = sorted(
            {phrase for phrase in phrases if phrase_present(answer, phrase)},
            key=lambda item: (phrase_position(answer, item), -len(item)),
        )
        primary = normalize(primary_keywords.get(concept, ""))
        score = 0.0
        for phrase in matched:
            position = phrase_position(answer, phrase)
            position_boost = 1.0 + (2.0 / (1.0 + position / 24.0))
            primary_word_count = max(1, len(primary.split()))
            primary_boost = (
                3.0 * (1.0 + 0.75 * (primary_word_count - 1))
                if normalize(phrase) == primary
                else 1.0
            )
            score += weights.get(normalize(phrase), 1.0) * position_boost * primary_boost
        # Contrast questions legitimately mention a competing concept. Discount it
        # so the concept being explained remains the routing target.
        if allowed_contrast and concept == allowed_contrast:
            score *= 0.20
        scores[concept] = score
        matches[concept] = matched
        anchor_phrases = {
            primary_keywords.get(concept, ""),
            concept.replace("_", " "),
        }
        anchor_ranks = [
            phrase_word_position(answer, phrase) / max(1, len(word_tokens(phrase)))
            for phrase in anchor_phrases
            if phrase_present(answer, phrase)
        ]
        positions[concept] = min(anchor_ranks, default=float(10**9))
    return scores, matches, positions


def predict_concept(
    scores: dict[str, float],
    matches: dict[str, list[str]],
    positions: dict[str, float],
) -> str:
    if not scores or max(scores.values(), default=0.0) <= 0.0:
        return "__unknown__"
    return max(
        scores,
        key=lambda concept: (
            positions[concept] < 10**9,
            -positions[concept],
            scores[concept],
            len(matches[concept]),
            concept,
        ),
    )


def score_example(
    record: dict[str, Any],
    answer: str,
    profiles: dict[str, set[str]],
    domains: dict[str, str],
    weights: dict[str, float],
    primary_keywords: dict[str, str],
) -> ExampleScore:
    expected = str(record["concept"])
    allowed_contrast = record.get("contrast_concept")
    scores, matches, positions = concept_scores(
        answer,
        profiles,
        weights,
        primary_keywords,
        str(allowed_contrast) if allowed_contrast else None,
    )
    predicted = predict_concept(scores, matches, positions)
    expected_keywords = [str(item) for item in record.get("keywords", [])]
    matched_expected = [keyword for keyword in expected_keywords if phrase_present(answer, keyword)]
    negatives = [str(item) for item in record.get("hard_negative_concepts", [])]
    matched_negatives = [
        concept
        for concept in negatives
        if concept in scores and scores[concept] > 0 and concept != allowed_contrast
    ]
    expected_score = scores.get(expected, 0.0)
    predicted_score = scores.get(predicted, 0.0)
    concept_correct = expected_score > 0.0
    routing_correct = predicted == expected
    malformed = not normalize(answer) or repeated_ngram(answer)
    exact = exact_overlap(answer, str(record["response"]))
    hallucination = (not exact) and (malformed or not concept_correct or (
        predicted in set(negatives) and predicted != allowed_contrast
    ))
    return ExampleScore(
        id=str(record["id"]),
        instruction=str(record["instruction"]),
        expected_domain=str(record["domain"]),
        expected_concept=expected,
        predicted_domain=domains.get(predicted, "__unknown__"),
        predicted_concept=predicted,
        question_type=str(record["question_type"]),
        routing_variant=str(record["routing_variant"]),
        generated_response=answer,
        reference_response=str(record["response"]),
        concept_correct=concept_correct,
        routing_correct=routing_correct,
        exact_answer_overlap=exact,
        keyword_overlap=len(matched_expected) / max(1, len(expected_keywords)),
        reference_token_overlap=token_f1(answer, str(record["response"])),
        hallucination=hallucination,
        expected_score=expected_score,
        predicted_score=predicted_score,
        matched_expected_keywords=matched_expected,
        matched_hard_negatives=matched_negatives,
    )


def confusion_matrix(scores: list[ExampleScore]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for score in scores:
        matrix[score.expected_concept][score.predicted_concept] += 1
    return {
        expected: dict(sorted(predictions.items()))
        for expected, predictions in sorted(matrix.items())
    }


def domain_confusion_matrix(scores: list[ExampleScore]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for score in scores:
        matrix[score.expected_domain][score.predicted_domain] += 1
    return {
        expected: dict(sorted(predictions.items()))
        for expected, predictions in sorted(matrix.items())
    }


def grouped_accuracy(scores: list[ExampleScore], field: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[ExampleScore]] = defaultdict(list)
    for score in scores:
        groups[str(getattr(score, field))].append(score)
    return {
        key: {
            "count": len(group),
            "concept_accuracy": sum(item.concept_correct for item in group) / len(group),
            "routing_accuracy": sum(item.routing_correct for item in group) / len(group),
            "hallucination_rate": sum(item.hallucination for item in group) / len(group),
        }
        for key, group in sorted(groups.items())
    }


def summarize(scores: list[ExampleScore]) -> dict[str, Any]:
    total = len(scores)
    if total == 0:
        raise ValueError("No evaluation scores")
    return {
        "count": total,
        "concept_accuracy": sum(score.concept_correct for score in scores) / total,
        "routing_accuracy": sum(score.routing_correct for score in scores) / total,
        "concept_confusion_rate": 1.0 - (sum(score.routing_correct for score in scores) / total),
        "exact_answer_overlap": sum(score.exact_answer_overlap for score in scores) / total,
        "keyword_overlap": sum(score.keyword_overlap for score in scores) / total,
        "reference_token_overlap": sum(score.reference_token_overlap for score in scores) / total,
        "hallucination_rate": sum(score.hallucination for score in scores) / total,
        "by_domain": grouped_accuracy(scores, "expected_domain"),
        "by_question_type": grouped_accuracy(scores, "question_type"),
        "by_routing_variant": grouped_accuracy(scores, "routing_variant"),
        "confusion_matrix": confusion_matrix(scores),
        "domain_confusion_matrix": domain_confusion_matrix(scores),
    }


def load_prediction_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in read_jsonl(path):
        identifier = str(record.get("id", ""))
        response = str(record.get("generated_response", record.get("response", "")))
        if not identifier or not response:
            raise ValueError(f"Prediction records require id and generated_response: {record}")
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


def evaluate_prediction_map(
    records: list[dict[str, Any]],
    predictions: dict[str, str],
    profiles: dict[str, set[str]],
    domains: dict[str, str],
    weights: dict[str, float],
    primary_keywords: dict[str, str],
) -> tuple[list[ExampleScore], dict[str, Any]]:
    missing = [str(record["id"]) for record in records if str(record["id"]) not in predictions]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} eval records: {missing[:5]}")
    scores = [
        score_example(
            record,
            predictions[str(record["id"])],
            profiles,
            domains,
            weights,
            primary_keywords,
        )
        for record in records
    ]
    return scores, summarize(scores)


def conclusion_for(
    trained: dict[str, Any],
    baseline: dict[str, Any] | None,
    routing_threshold: float,
    significant_drop: float,
) -> tuple[str, float | None]:
    drop = None
    if baseline is not None:
        drop = float(baseline["concept_confusion_rate"]) - float(trained["concept_confusion_rate"])
    if float(trained["routing_accuracy"]) > routing_threshold and drop is not None and drop >= significant_drop:
        return "Data/training was the primary bottleneck.", drop
    if float(trained["routing_accuracy"]) <= routing_threshold:
        return "Model capacity is likely the bottleneck and scaling to 300M–500M is justified.", drop
    return (
        "Inconclusive: routing cleared 80%, but the measured confusion drop was not large enough "
        "to isolate data/training as the primary bottleneck.",
        drop,
    )


def top_confusions(matrix: dict[str, dict[str, int]], limit: int = 15) -> list[tuple[str, str, int]]:
    rows = [
        (expected, predicted, count)
        for expected, predictions in matrix.items()
        for predicted, count in predictions.items()
        if expected != predicted
    ]
    return sorted(rows, key=lambda item: (-item[2], item[0], item[1]))[:limit]


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def metric_table(summary: dict[str, Any]) -> list[str]:
    return [
        f"| Concept accuracy | {pct(float(summary['concept_accuracy']))} |",
        f"| Routing accuracy | {pct(float(summary['routing_accuracy']))} |",
        f"| Concept confusion rate | {pct(float(summary['concept_confusion_rate']))} |",
        f"| Exact answer overlap | {pct(float(summary['exact_answer_overlap']))} |",
        f"| Keyword overlap | {pct(float(summary['keyword_overlap']))} |",
        f"| Reference token overlap | {pct(float(summary['reference_token_overlap']))} |",
        f"| Hallucination rate | {pct(float(summary['hallucination_rate']))} |",
    ]


def write_markdown_report(
    path: Path,
    trained: dict[str, Any],
    baseline: dict[str, Any] | None,
    conclusion: str,
    confusion_drop: float | None,
    checkpoint: Path | None,
    baseline_checkpoint: Path | None,
    routing_threshold: float,
    significant_drop: float,
) -> None:
    lines = [
        "# Marshmello Core Routing Evaluation",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Fine-tuned checkpoint: `{checkpoint}`" if checkpoint else "- Fine-tuned predictions: supplied JSONL",
        (
            f"- Baseline checkpoint: `{baseline_checkpoint}`"
            if baseline_checkpoint
            else "- Baseline: not evaluated"
        ),
        "- Decoding: greedy",
        "",
        "## Fine-tuned metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *metric_table(trained),
    ]
    if baseline is not None:
        lines.extend(
            [
                "",
                "## Baseline metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
                *metric_table(baseline),
                "",
                f"Absolute concept-confusion drop: {pct(confusion_drop or 0.0)}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            conclusion,
            "",
            f"The configured success rule requires routing accuracy above {pct(routing_threshold)} "
            f"and an absolute confusion-rate drop of at least {pct(significant_drop)}.",
            "",
            "## Domain breakdown",
            "",
            "| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for domain, metrics in trained["by_domain"].items():
        lines.append(
            f"| {domain} | {metrics['count']} | {pct(float(metrics['concept_accuracy']))} | "
            f"{pct(float(metrics['routing_accuracy']))} | {pct(float(metrics['hallucination_rate']))} |"
        )
    lines.extend(
        [
            "",
            "## Largest concept confusions",
            "",
            "| Expected | Predicted | Count |",
            "|---|---|---:|",
        ]
    )
    confusions = top_confusions(trained["confusion_matrix"])
    if confusions:
        for expected, predicted, count in confusions:
            lines.append(f"| {expected} | {predicted} | {count} |")
    else:
        lines.append("| — | — | 0 |")
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- Concept accuracy: the answer contains at least one expected concept signature.",
            "- Routing accuracy: the expected concept has the highest weighted signature score.",
            "- Exact answer overlap: normalized generated text exactly matches the reference answer.",
            "- Keyword overlap: mean recall of expected concept keywords.",
            "- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.",
            "- The complete concept and domain confusion matrices are stored in the JSON results file.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--sft-data", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--baseline-predictions", type=Path, default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--config", default="large_50m")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--routing-threshold", type=float, default=0.80)
    parser.add_argument("--significant-confusion-drop", type=float, default=0.10)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--baseline-predictions-output",
        type=Path,
        default=DEFAULT_BASELINE_PREDICTIONS,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    eval_records = read_jsonl(args.eval_data)
    sft_records = read_jsonl(args.sft_data)
    validate_data(eval_records, sft_records)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        eval_records = eval_records[: args.limit]
    print(
        f"Validated {len(sft_records)} SFT examples and {len(eval_records)} held-out questions",
        flush=True,
    )
    if args.validate_only:
        return

    profiles, domains, weights, primary_keywords = build_profiles(sft_records, eval_records)
    if args.predictions:
        trained_predictions = load_prediction_map(args.predictions)
        trained_checkpoint: Path | None = None
    else:
        trained_predictions = generate_predictions(
            eval_records,
            args.checkpoint,
            args.config,
            args.max_new_tokens,
            args.cpu,
        )
        trained_checkpoint = args.checkpoint
        write_jsonl(
            args.predictions_output,
            [
                {
                    "id": record["id"],
                    "instruction": record["instruction"],
                    "generated_response": trained_predictions[str(record["id"])],
                }
                for record in eval_records
            ],
        )
    trained_scores, trained_summary = evaluate_prediction_map(
        eval_records, trained_predictions, profiles, domains, weights, primary_keywords
    )

    baseline_scores: list[ExampleScore] | None = None
    baseline_summary: dict[str, Any] | None = None
    baseline_checkpoint: Path | None = None
    if not args.no_baseline:
        if args.baseline_predictions:
            baseline_predictions = load_prediction_map(args.baseline_predictions)
        else:
            baseline_predictions = generate_predictions(
                eval_records,
                args.baseline_checkpoint,
                args.config,
                args.max_new_tokens,
                args.cpu,
            )
            baseline_checkpoint = args.baseline_checkpoint
            write_jsonl(
                args.baseline_predictions_output,
                [
                    {
                        "id": record["id"],
                        "instruction": record["instruction"],
                        "generated_response": baseline_predictions[str(record["id"])],
                    }
                    for record in eval_records
                ],
            )
        baseline_scores, baseline_summary = evaluate_prediction_map(
            eval_records, baseline_predictions, profiles, domains, weights, primary_keywords
        )

    conclusion, confusion_drop = conclusion_for(
        trained_summary,
        baseline_summary,
        args.routing_threshold,
        args.significant_confusion_drop,
    )
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "checkpoint": str(trained_checkpoint) if trained_checkpoint else None,
        "baseline_checkpoint": str(baseline_checkpoint) if baseline_checkpoint else None,
        "success_rule": {
            "routing_accuracy_must_exceed": args.routing_threshold,
            "minimum_absolute_confusion_drop": args.significant_confusion_drop,
        },
        "confusion_drop": confusion_drop,
        "conclusion": conclusion,
        "trained": trained_summary,
        "baseline": baseline_summary,
        "trained_examples": [asdict(score) for score in trained_scores],
        "baseline_examples": (
            [asdict(score) for score in baseline_scores] if baseline_scores is not None else None
        ),
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(
        args.report,
        trained_summary,
        baseline_summary,
        conclusion,
        confusion_drop,
        trained_checkpoint,
        baseline_checkpoint,
        args.routing_threshold,
        args.significant_confusion_drop,
    )

    print(f"Concept accuracy:     {pct(float(trained_summary['concept_accuracy']))}")
    print(f"Routing accuracy:     {pct(float(trained_summary['routing_accuracy']))}")
    print(f"Exact answer overlap: {pct(float(trained_summary['exact_answer_overlap']))}")
    print(f"Keyword overlap:      {pct(float(trained_summary['keyword_overlap']))}")
    print(f"Hallucination rate:   {pct(float(trained_summary['hallucination_rate']))}")
    if confusion_drop is not None:
        print(f"Confusion drop:       {pct(confusion_drop)}")
    print(f"Conclusion: {conclusion}")
    print(f"JSON results: {args.results_json}")
    print(f"Markdown report: {args.report}")

    success = (
        float(trained_summary["routing_accuracy"]) > args.routing_threshold
        and confusion_drop is not None
        and confusion_drop >= args.significant_confusion_drop
    )
    if args.strict and not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
