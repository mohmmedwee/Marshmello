"""Memorization and quality metrics for Phase 16 evaluation."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from prompts import DOMAIN_KEYWORDS  # noqa: I001 — local phase module


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def char_count(text: str) -> int:
    return len(text.strip())


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


@dataclass
class CorpusIndex:
    paragraphs: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    normalized_paragraphs: list[str] = field(default_factory=list)
    normalized_lines: list[str] = field(default_factory=list)


def build_corpus_index(corpus_text: str) -> CorpusIndex:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", corpus_text) if p.strip()]
    lines = [line.strip() for line in corpus_text.splitlines() if len(line.strip()) >= 20]
    return CorpusIndex(
        paragraphs=paragraphs,
        lines=lines,
        normalized_paragraphs=[normalize_text(p) for p in paragraphs],
        normalized_lines=[normalize_text(line) for line in lines],
    )


def exact_paragraph_match(text: str, corpus: CorpusIndex) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if normalized in corpus.normalized_paragraphs:
        return True
    return any(normalized in paragraph for paragraph in corpus.normalized_paragraphs)


def exact_line_match(text: str, corpus: CorpusIndex) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return normalized in corpus.normalized_lines


def nearest_training_similarity(text: str, corpus: CorpusIndex) -> tuple[float, str]:
    normalized = normalize_text(text)
    if not normalized:
        return 0.0, ""

    best_ratio = 0.0
    best_snippet = ""
    candidates = corpus.lines + corpus.paragraphs
    for snippet in candidates:
        candidate = normalize_text(snippet)
        if len(candidate) < 20:
            continue
        ratio = SequenceMatcher(None, normalized, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_snippet = snippet.strip()[:120]
    return best_ratio, best_snippet


def domain_consistency(text: str, domain: str) -> float:
    keywords = DOMAIN_KEYWORDS.get(domain, ())
    if not keywords:
        return 0.0
    lowered = text.lower()
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return hits / len(keywords)


@dataclass
class SampleMetrics:
    prompt: str
    domain: str
    config_name: str
    model_alias: str
    output: str
    char_length: int
    word_length: int
    repeated_ngram_ratio: float
    exact_paragraph_match: bool
    exact_line_match: bool
    nearest_similarity: float
    nearest_snippet: str
    domain_consistency: float

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "domain": self.domain,
            "config_name": self.config_name,
            "model_alias": self.model_alias,
            "output": self.output,
            "char_length": self.char_length,
            "word_length": self.word_length,
            "repeated_ngram_ratio": round(self.repeated_ngram_ratio, 4),
            "exact_paragraph_match": self.exact_paragraph_match,
            "exact_line_match": self.exact_line_match,
            "nearest_similarity": round(self.nearest_similarity, 4),
            "nearest_snippet": self.nearest_snippet,
            "domain_consistency": round(self.domain_consistency, 4),
        }


def score_output(
    *,
    prompt: str,
    domain: str,
    config_name: str,
    model_alias: str,
    output: str,
    corpus: CorpusIndex,
    ngram_size: int = 4,
) -> SampleMetrics:
    nearest_ratio, nearest_snippet = nearest_training_similarity(output, corpus)
    return SampleMetrics(
        prompt=prompt,
        domain=domain,
        config_name=config_name,
        model_alias=model_alias,
        output=output,
        char_length=char_count(output),
        word_length=word_count(output),
        repeated_ngram_ratio=repeated_ngram_ratio(output, n=ngram_size),
        exact_paragraph_match=exact_paragraph_match(output, corpus),
        exact_line_match=exact_line_match(output, corpus),
        nearest_similarity=nearest_ratio,
        nearest_snippet=nearest_snippet,
        domain_consistency=domain_consistency(output, domain),
    )


@dataclass
class AggregateMetrics:
    model_alias: str
    config_name: str
    avg_char_length: float
    avg_word_length: float
    avg_repeated_ngram_ratio: float
    exact_paragraph_match_rate: float
    exact_line_match_rate: float
    avg_nearest_similarity: float
    avg_domain_consistency: float
    memorization_risk: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_alias": self.model_alias,
            "config_name": self.config_name,
            "avg_char_length": round(self.avg_char_length, 1),
            "avg_word_length": round(self.avg_word_length, 1),
            "avg_repeated_ngram_ratio": round(self.avg_repeated_ngram_ratio, 4),
            "exact_paragraph_match_rate": round(self.exact_paragraph_match_rate, 4),
            "exact_line_match_rate": round(self.exact_line_match_rate, 4),
            "avg_nearest_similarity": round(self.avg_nearest_similarity, 4),
            "avg_domain_consistency": round(self.avg_domain_consistency, 4),
            "memorization_risk": self.memorization_risk,
        }


def aggregate(samples: list[SampleMetrics]) -> AggregateMetrics:
    if not samples:
        raise ValueError("No samples to aggregate")

    count = len(samples)
    avg_char = sum(s.char_length for s in samples) / count
    avg_words = sum(s.word_length for s in samples) / count
    avg_ngram = sum(s.repeated_ngram_ratio for s in samples) / count
    paragraph_rate = sum(1 for s in samples if s.exact_paragraph_match) / count
    line_rate = sum(1 for s in samples if s.exact_line_match) / count
    avg_nearest = sum(s.nearest_similarity for s in samples) / count
    avg_domain = sum(s.domain_consistency for s in samples) / count

    risk_score = 0
    if paragraph_rate >= 0.34:
        risk_score += 2
    elif paragraph_rate > 0:
        risk_score += 1
    if line_rate >= 0.34:
        risk_score += 1
    if avg_nearest >= 0.90:
        risk_score += 2
    elif avg_nearest >= 0.80:
        risk_score += 1
    if avg_ngram >= 0.20:
        risk_score += 1

    if risk_score >= 4:
        risk = "high"
    elif risk_score >= 2:
        risk = "moderate"
    else:
        risk = "low"

    first = samples[0]
    return AggregateMetrics(
        model_alias=first.model_alias,
        config_name=first.config_name,
        avg_char_length=avg_char,
        avg_word_length=avg_words,
        avg_repeated_ngram_ratio=avg_ngram,
        exact_paragraph_match_rate=paragraph_rate,
        exact_line_match_rate=line_rate,
        avg_nearest_similarity=avg_nearest,
        avg_domain_consistency=avg_domain,
        memorization_risk=risk,
    )


def build_conclusion(
    small: AggregateMetrics,
    large: AggregateMetrics,
) -> list[str]:
    lines: list[str] = []

    coherence_delta = large.avg_domain_consistency - small.avg_domain_consistency
    repetition_delta = small.avg_repeated_ngram_ratio - large.avg_repeated_ngram_ratio

    if coherence_delta > 0.05 and repetition_delta > 0.02:
        lines.append(
            f"{large.model_alias} shows modestly better coherence "
            f"(domain consistency +{coherence_delta:.2f}, repetition −{repetition_delta:.2f})."
        )
    elif coherence_delta > 0.02:
        lines.append(
            f"{large.model_alias} improves domain consistency slightly (+{coherence_delta:.2f}), "
            "but gains are limited on the same small corpus."
        )
    else:
        lines.append(
            "Scaling parameter count did not produce a clear coherence jump — "
            "both models stay near training-distribution prose."
        )

    memorized_models = [
        metrics.model_alias
        for metrics in (small, large)
        if metrics.memorization_risk in {"moderate", "high"}
    ]
    if memorized_models:
        lines.append(
            f"Memorization detected ({', '.join(memorized_models)}): "
            "outputs closely match training paragraphs/lines. "
            "This is expected when val loss is low and the corpus repeats."
        )
    else:
        lines.append(
            "No strong verbatim memorization signal — outputs paraphrase or drift rather than copy entire paragraphs."
        )

    if small.avg_nearest_similarity > 0.85 or large.avg_nearest_similarity > 0.85:
        lines.append(
            "High nearest-line similarity to the corpus confirms the models are retrieving "
            "memorized training snippets, not generalizing to new facts."
        )

    lines.append(
        "Takeaway: bigger models on the same data mostly memorize faster and interpolate smoother — "
        "they do not replace better data or fine-tuning."
    )
    return lines
