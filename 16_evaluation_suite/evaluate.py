#!/usr/bin/env python3
"""
Phase 16 — Compare Marshmello-8M vs Marshmello-45M and detect memorization.

Run from project root:
  python 16_evaluation_suite/evaluate.py
  python 16_evaluation_suite/evaluate.py --max-new-tokens 80 --greedy
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
GPT_ROOT = PROJECT_ROOT / "13_gpt_pretraining"

sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(GPT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from config import CORPUS_PATH, resolve_config  # noqa: E402
from generate import (  # noqa: E402
    generate,
    generated_suffix,
    load_model_and_tokenizer,
    decode_ids_pretty,
    encode_prompt,
    format_prompt,
)
from metrics import (  # noqa: E402
    AggregateMetrics,
    SampleMetrics,
    aggregate,
    build_conclusion,
    build_corpus_index,
    score_output,
)
from prompts import EVAL_CONFIGS, MODEL_ALIASES, PROMPT_SUITE  # noqa: E402
from training.trainer import pick_device  # noqa: E402

REPORTS_DIR = PHASE_ROOT / "reports"


@dataclass
class GenerationSettings:
    max_new_tokens: int = 100
    min_new_tokens: int = 20
    greedy: bool = False
    temperature: float = 0.7
    top_k: int = 30
    stop_on_sentence_end: bool = True


def extract_generated_text(bpe, prompt: str, result_text: str) -> str:
    ids = encode_prompt(bpe, format_prompt(prompt))
    prompt_decoded = decode_ids_pretty(bpe, ids)
    suffix = generated_suffix(result_text, prompt_decoded).strip()
    return suffix or result_text.strip()


def load_corpus_text() -> str:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Training corpus not found at {CORPUS_PATH}. "
            "Run prepare_corpus or the Phase 14 export first."
        )
    return CORPUS_PATH.read_text(encoding="utf-8")


def run_model_eval(
    config_name: str,
    corpus_index,
    settings: GenerationSettings,
    device: torch.device,
) -> tuple[list[SampleMetrics], AggregateMetrics]:
    cfg = resolve_config(config_name)
    alias = MODEL_ALIASES.get(config_name, config_name)
    from config import latest_checkpoint_for

    checkpoint = latest_checkpoint_for(cfg)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint for {alias} at {checkpoint}. "
            f"Train with: python 13_gpt_pretraining/training/trainer.py --config {config_name}"
        )

    model, bpe, _model_cfg = load_model_and_tokenizer(checkpoint, device, cfg)
    model.eval()

    samples: list[SampleMetrics] = []
    for item in PROMPT_SUITE:
        result = generate(
            model,
            bpe,
            item.prompt,
            max_new_tokens=settings.max_new_tokens,
            min_new_tokens=settings.min_new_tokens,
            greedy=settings.greedy,
            temperature=settings.temperature,
            top_k=settings.top_k,
            stop_on_sentence_end=settings.stop_on_sentence_end,
            stop_sequence=None,
            device=device,
        )
        output = extract_generated_text(bpe, item.prompt, result.text)
        samples.append(
            score_output(
                prompt=item.prompt,
                domain=item.domain,
                config_name=config_name,
                model_alias=alias,
                output=output,
                corpus=corpus_index,
            )
        )

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return samples, aggregate(samples)


def print_metric_row(label: str, left: str, right: str) -> None:
    print(f"{label:<28} {left:>18} {right:>18}")


def print_side_by_side(
    prompt_label: str,
    left: SampleMetrics,
    right: SampleMetrics,
) -> None:
    print("=" * 88)
    print(f"Prompt: {prompt_label}")
    print("-" * 88)
    print(f"{left.model_alias}:")
    print(f"  {left.output}")
    print()
    print(f"{right.model_alias}:")
    print(f"  {right.output}")
    print("-" * 88)
    print_metric_row(
        "Metric",
        left.model_alias,
        right.model_alias,
    )
    print_metric_row("Words", f"{left.word_length}", f"{right.word_length}")
    print_metric_row(
        "Repeated 4-gram ratio",
        f"{left.repeated_ngram_ratio:.3f}",
        f"{right.repeated_ngram_ratio:.3f}",
    )
    print_metric_row(
        "Exact paragraph match",
        str(left.exact_paragraph_match),
        str(right.exact_paragraph_match),
    )
    print_metric_row(
        "Nearest corpus sim",
        f"{left.nearest_similarity:.3f}",
        f"{right.nearest_similarity:.3f}",
    )
    print_metric_row(
        "Domain consistency",
        f"{left.domain_consistency:.3f}",
        f"{right.domain_consistency:.3f}",
    )
    print()


def print_summary(small: AggregateMetrics, large: AggregateMetrics) -> None:
    print("=" * 88)
    print("AGGREGATE METRICS")
    print("=" * 88)
    print_metric_row("Metric", small.model_alias, large.model_alias)
    print_metric_row("Avg words", f"{small.avg_word_length:.1f}", f"{large.avg_word_length:.1f}")
    print_metric_row(
        "Avg repeated n-gram",
        f"{small.avg_repeated_ngram_ratio:.3f}",
        f"{large.avg_repeated_ngram_ratio:.3f}",
    )
    print_metric_row(
        "Exact paragraph rate",
        f"{small.exact_paragraph_match_rate:.0%}",
        f"{large.exact_paragraph_match_rate:.0%}",
    )
    print_metric_row(
        "Avg nearest similarity",
        f"{small.avg_nearest_similarity:.3f}",
        f"{large.avg_nearest_similarity:.3f}",
    )
    print_metric_row(
        "Avg domain consistency",
        f"{small.avg_domain_consistency:.3f}",
        f"{large.avg_domain_consistency:.3f}",
    )
    print_metric_row(
        "Memorization risk",
        small.memorization_risk,
        large.memorization_risk,
    )
    print()
    print("CONCLUSION")
    print("-" * 88)
    for line in build_conclusion(small, large):
        print(f"- {line}")
    print()


def save_report(
    *,
    small_samples: list[SampleMetrics],
    large_samples: list[SampleMetrics],
    small_agg: AggregateMetrics,
    large_agg: AggregateMetrics,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "evaluation_report.json"
    payload = {
        "models": {
            small_agg.config_name: small_agg.to_dict(),
            large_agg.config_name: large_agg.to_dict(),
        },
        "samples": {
            "default": [sample.to_dict() for sample in small_samples],
            "large_50m": [sample.to_dict() for sample in large_samples],
        },
        "conclusion": build_conclusion(small_agg, large_agg),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Marshmello-8M vs Marshmello-45M for memorization and coherence."
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--min-new-tokens", type=int, default=20)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--no-stop-on-sentence-end",
        action="store_true",
        help="Disable stop-on-sentence-end during generation",
    )
    args = parser.parse_args()

    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_on_sentence_end=not args.no_stop_on_sentence_end,
    )

    device = pick_device(force_cpu=args.cpu)
    corpus_index = build_corpus_index(load_corpus_text())

    print("Phase 16 — Evaluation Suite")
    print("=" * 88)
    print(f"Device: {device}")
    print(f"Corpus: {CORPUS_PATH} ({len(corpus_index.paragraphs)} paragraphs)")
    print(f"Prompts: {len(PROMPT_SUITE)}")
    print()

    small_samples, small_agg = run_model_eval("default", corpus_index, settings, device)
    large_samples, large_agg = run_model_eval("large_50m", corpus_index, settings, device)

    paired = zip(PROMPT_SUITE, small_samples, large_samples, strict=True)
    for item, left, right in paired:
        print_side_by_side(item.label, left, right)

    print_summary(small_agg, large_agg)
    report_path = save_report(
        small_samples=small_samples,
        large_samples=large_samples,
        small_agg=small_agg,
        large_agg=large_agg,
    )
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
