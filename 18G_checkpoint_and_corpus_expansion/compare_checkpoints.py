#!/usr/bin/env python3
"""Compare Marshmello base checkpoints on raw and chat-format prompts."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE13_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from config import resolve_config  # noqa: E402
from generate import (  # noqa: E402
    decode_ids_pretty,
    encode_prompt,
    generate,
    generated_suffix,
    load_model_and_tokenizer,
)
from training.trainer import pick_device  # noqa: E402

DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT / "13_gpt_pretraining" / "checkpoints" / "large_50m" / "step_001000.pt",
    PROJECT_ROOT / "13_gpt_pretraining" / "checkpoints" / "large_50m" / "latest.pt",
)
DEFAULT_REPORT = PHASE_ROOT / "reports" / "checkpoint_comparison.json"
DEFAULT_REFERENCE_CORPUS = PHASE13_ROOT / "data" / "corpus_chat_mixed.txt"

PROMPTS = (
    "Artificial intelligence",
    "Machine learning",
    "Database systems",
    "<USER> What is AI? <ASSISTANT>",
    "<USER> Explain database indexes. <ASSISTANT>",
    "<USER> Who are you? <ASSISTANT>",
)

TOPIC_KEYWORDS = {
    "Artificial intelligence": {
        "artificial",
        "intelligence",
        "computer",
        "machine",
        "reasoning",
        "tasks",
        "data",
        "patterns",
    },
    "Machine learning": {
        "machine",
        "learning",
        "data",
        "model",
        "training",
        "patterns",
        "prediction",
        "examples",
    },
    "Database systems": {
        "database",
        "data",
        "tables",
        "queries",
        "storage",
        "transactions",
        "index",
        "rows",
    },
    "<USER> What is AI? <ASSISTANT>": {
        "ai",
        "artificial",
        "intelligence",
        "computer",
        "machine",
        "thinking",
        "tasks",
        "patterns",
    },
    "<USER> Explain database indexes. <ASSISTANT>": {
        "database",
        "index",
        "indexes",
        "lookup",
        "rows",
        "queries",
        "faster",
        "writes",
    },
    "<USER> Who are you? <ASSISTANT>": {
        "marshmello",
        "assistant",
        "model",
        "answer",
        "help",
        "questions",
    },
}

UNRELATED_TOPICS = {
    "politics": {"president", "government", "election", "united states", "congress"},
    "literature": {"romeo", "juliet", "king", "shakespeare", "poem"},
    "finance": {"money", "stock", "market", "investment", "profit"},
    "health": {"diet", "exercise", "sleep", "doctor", "disease"},
}

RAW_CORPUS_STARTS = (
    "teams should write",
    "example note",
    "database systems need",
    "this text is about",
    "topic:",
)

KNOWN_ACRONYMS = {
    "ai",
    "api",
    "bpe",
    "cpu",
    "gpu",
    "gpt",
    "json",
    "llm",
    "sql",
    "ui",
}


@dataclass(frozen=True)
class PromptMetrics:
    prompt: str
    generated_text: str
    repeated_3gram: bool
    repeated_phrase: str | None
    topic_keywords_found: list[str]
    topic_keyword_score: float
    gibberish_looking_word_count: int
    gibberish_words: list[str]
    drift_to_unrelated_topics: bool
    unrelated_topics_found: list[str]
    chat_format_response_ok: bool


@dataclass(frozen=True)
class CheckpointSummary:
    checkpoint: str
    step: int
    train_loss: float | None
    val_loss: float | None
    repetition_count: int
    average_topic_score: float
    gibberish_count: int
    drift_count: int
    chat_format_successes: int
    recommendation_score: float


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def repeated_ngram(text: str, *, n: int = 3) -> tuple[str, ...] | None:
    words = re.findall(r"\b[\w'-]+\b", text.casefold())
    if len(words) < n * 2:
        return None
    seen: set[tuple[str, ...]] = set()
    for start in range(len(words) - n + 1):
        ngram = tuple(words[start : start + n])
        if ngram in seen:
            return ngram
        seen.add(ngram)
    return None


def topic_keyword_score(prompt: str, text: str) -> tuple[list[str], float]:
    lower = text.casefold()
    keywords = TOPIC_KEYWORDS[prompt]
    found = sorted(
        keyword
        for keyword in keywords
        if re.search(rf"\b{re.escape(keyword)}s?\b", lower)
    )
    return found, len(found) / len(keywords)


def load_reference_words(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8").casefold()
    return set(re.findall(r"\b[a-z][a-z'-]*\b", text))


def gibberish_looking_words(text: str, bpe=None, known_words: set[str] | None = None) -> list[str]:
    words = re.findall(r"\b[\w'-]+\b", text)
    suspicious: list[str] = []
    for word in words:
        lower = word.casefold().strip("-'")
        if not lower or lower in KNOWN_ACRONYMS:
            continue
        if len(lower) > 24:
            suspicious.append(word)
            continue
        if re.search(r"(.)\1\1", lower):
            suspicious.append(word)
            continue
        if lower.isalpha() and len(lower) >= 7 and not re.search(r"[aeiouy]", lower):
            suspicious.append(word)
            continue
        if re.search(r"[a-z]\d|\d[a-z]", lower):
            suspicious.append(word)
            continue
        if (
            bpe is not None
            and known_words is not None
            and lower.isalpha()
            and lower not in known_words
            and 5 <= len(lower) <= 20
        ):
            pieces = bpe.encode_word(lower)
            if len(pieces) >= 3 and len(pieces) / len(lower) >= 0.3:
                suspicious.append(word)
    return suspicious


def unrelated_topic_drift(prompt: str, text: str, topic_score: float) -> tuple[bool, list[str]]:
    lower = text.casefold()
    found: list[str] = []
    for topic, keywords in UNRELATED_TOPICS.items():
        hits = sum(
            1
            for keyword in keywords
            if re.search(rf"\b{re.escape(keyword)}\b", lower)
        )
        threshold = 2 if topic == "finance" else 1
        if hits >= threshold:
            found.append(topic)
    return bool(found and topic_score < 0.4), found


def is_chat_prompt(prompt: str) -> bool:
    return prompt.startswith("<USER>")


def chat_format_response_ok(text: str, topic_score: float, repeated: bool) -> bool:
    lower = normalize(text).casefold()
    if not lower or repeated:
        return False
    if "<user>" in lower or lower.startswith(("<assistant>", "<end>")):
        return False
    if any(lower.startswith(phrase) for phrase in RAW_CORPUS_STARTS):
        return False
    return topic_score > 0.0


def extract_generated_suffix(bpe, prompt: str, full_text: str, raw_text: str) -> str:
    prompt_ids = encode_prompt(bpe, prompt)
    prompt_decoded = decode_ids_pretty(bpe, prompt_ids)
    suffix = normalize(generated_suffix(full_text, prompt_decoded))
    if not suffix:
        suffix = normalize(generated_suffix(raw_text, prompt_decoded))
    return suffix


def evaluate_text(
    prompt: str,
    text: str,
    bpe=None,
    known_words: set[str] | None = None,
) -> PromptMetrics:
    repeated = repeated_ngram(text, n=3)
    found, score = topic_keyword_score(prompt, text)
    gibberish = gibberish_looking_words(text, bpe=bpe, known_words=known_words)
    drift, unrelated = unrelated_topic_drift(prompt, text, score)
    chat_ok = (
        chat_format_response_ok(text, score, repeated is not None)
        if is_chat_prompt(prompt)
        else False
    )
    return PromptMetrics(
        prompt=prompt,
        generated_text=text,
        repeated_3gram=repeated is not None,
        repeated_phrase=" ".join(repeated) if repeated else None,
        topic_keywords_found=found,
        topic_keyword_score=round(score, 4),
        gibberish_looking_word_count=len(gibberish),
        gibberish_words=gibberish,
        drift_to_unrelated_topics=drift,
        unrelated_topics_found=unrelated,
        chat_format_response_ok=chat_ok,
    )


def checkpoint_metadata(path: Path) -> tuple[int, float | None, float | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        int(payload.get("step", 0)),
        float(payload["train_loss"]) if payload.get("train_loss") is not None else None,
        float(payload["val_loss"]) if payload.get("val_loss") is not None else None,
    )


def summarize_checkpoint(
    checkpoint: Path,
    prompt_results: list[PromptMetrics],
    *,
    step: int,
    train_loss: float | None,
    val_loss: float | None,
) -> CheckpointSummary:
    repetitions = sum(result.repeated_3gram for result in prompt_results)
    avg_topic = sum(result.topic_keyword_score for result in prompt_results) / len(prompt_results)
    gibberish = sum(result.gibberish_looking_word_count for result in prompt_results)
    drifts = sum(result.drift_to_unrelated_topics for result in prompt_results)
    chat_successes = sum(
        result.chat_format_response_ok for result in prompt_results if is_chat_prompt(result.prompt)
    )
    score = (
        avg_topic * 100.0
        + chat_successes * 12.0
        - repetitions * 18.0
        - gibberish * 2.0
        - drifts * 10.0
    )
    return CheckpointSummary(
        checkpoint=str(checkpoint),
        step=step,
        train_loss=train_loss,
        val_loss=val_loss,
        repetition_count=repetitions,
        average_topic_score=round(avg_topic, 4),
        gibberish_count=gibberish,
        drift_count=drifts,
        chat_format_successes=chat_successes,
        recommendation_score=round(score, 4),
    )


def recommend(summaries: list[CheckpointSummary]) -> CheckpointSummary:
    return max(
        summaries,
        key=lambda item: (
            item.recommendation_score,
            -item.repetition_count,
            item.average_topic_score,
            item.chat_format_successes,
            -item.gibberish_count,
            -(item.val_loss if item.val_loss is not None else float("inf")),
        ),
    )


def print_prompt_result(result: PromptMetrics) -> None:
    print(f"  Prompt: {result.prompt}")
    print(f"  Generated: {result.generated_text}")
    print(f"  repeated_3gram: {result.repeated_3gram}")
    if result.repeated_phrase:
        print(f"  repeated_phrase: {result.repeated_phrase!r}")
    print(
        f"  topic_keyword_score: {result.topic_keyword_score:.4f} "
        f"{result.topic_keywords_found}"
    )
    print(
        f"  gibberish_looking_word_count: {result.gibberish_looking_word_count} "
        f"{result.gibberish_words}"
    )
    print(
        f"  drift_to_unrelated_topics: {result.drift_to_unrelated_topics} "
        f"{result.unrelated_topics_found}"
    )
    if is_chat_prompt(result.prompt):
        print(f"  chat_format_response_ok: {result.chat_format_response_ok}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Marshmello base checkpoints.")
    parser.add_argument("--config", default="large_50m")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--reference-corpus",
        type=Path,
        default=DEFAULT_REFERENCE_CORPUS,
        help="Known-word reference used by the gibberish heuristic.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    for checkpoint in args.checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not args.reference_corpus.exists():
        raise FileNotFoundError(f"Reference corpus not found: {args.reference_corpus}")

    cfg = resolve_config(args.config)
    device = pick_device(force_cpu=args.cpu)
    known_words = load_reference_words(args.reference_corpus)
    report_results: list[dict[str, object]] = []
    summaries: list[CheckpointSummary] = []

    print("Phase 18G: checkpoint comparison")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Tokenizer: {PHASE13_ROOT / 'tokenizer' / 'tokenizer.json'}")
    print()

    for checkpoint in args.checkpoints:
        step, train_loss, val_loss = checkpoint_metadata(checkpoint)
        print(f"Checkpoint: {checkpoint}")
        print(f"Metadata: step={step}, train_loss={train_loss}, val_loss={val_loss}")
        model, bpe, _ = load_model_and_tokenizer(checkpoint, device, cfg)

        prompt_results: list[PromptMetrics] = []
        for prompt_index, prompt in enumerate(PROMPTS):
            torch.manual_seed(10_000 + prompt_index)
            result = generate(
                model,
                bpe,
                prompt,
                max_new_tokens=args.max_new_tokens,
                greedy=True,
                repetition_penalty=1.2,
                presence_penalty=0.0,
                stop_sequence="<END>" if is_chat_prompt(prompt) else None,
                stop_on_eos_token=True,
                stop_on_sentence_end=False,
                device=device,
            )
            suffix = extract_generated_suffix(bpe, prompt, result.text, result.raw_text)
            metrics = evaluate_text(
                prompt,
                suffix,
                bpe=bpe,
                known_words=known_words,
            )
            prompt_results.append(metrics)
            print_prompt_result(metrics)

        summary = summarize_checkpoint(
            checkpoint,
            prompt_results,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        summaries.append(summary)
        report_results.append(
            {
                "summary": asdict(summary),
                "prompts": [asdict(result) for result in prompt_results],
            }
        )
        print("  Summary:")
        print(f"    repetition_count: {summary.repetition_count}")
        print(f"    average_topic_score: {summary.average_topic_score:.4f}")
        print(f"    gibberish_count: {summary.gibberish_count}")
        print(f"    drift_count: {summary.drift_count}")
        print(f"    chat_format_successes: {summary.chat_format_successes}/3")
        print(f"    recommendation_score: {summary.recommendation_score:.4f}")
        print("=" * 72)

        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    best = recommend(summaries)
    print()
    print("Automatic recommendation")
    print("=" * 72)
    print(f"BEST_CHECKPOINT_FROM_COMPARE={best.checkpoint}")
    print(
        "Reason: best combined score from lower repetition, higher topic relevance, "
        "lower gibberish, lower drift, and better chat-format responses."
    )
    print(f"Recommendation score: {best.recommendation_score:.4f}")
    print(f"Checkpoint val loss: {best.val_loss}")

    payload = {
        "device": str(device),
        "config": args.config,
        "results": report_results,
        "recommended_checkpoint": asdict(best),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {args.report_output}")


if __name__ == "__main__":
    main()
