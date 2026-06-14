"""Shared helpers for the Phase 14 dataset pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

PHASE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PHASE_ROOT / "raw"
CLEANED_DIR = PHASE_ROOT / "cleaned"
DEDUPED_DIR = PHASE_ROOT / "deduped"
SHARDS_DIR = PHASE_ROOT / "shards"
REPORTS_DIR = PHASE_ROOT / "reports"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".jsonl", ".html", ".htm", ".pdf"}

DOMAIN_LABELS = ("ai", "databases", "software_engineering", "general")

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ai": (
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural network",
        "transformer",
        "attention",
        "embedding",
        "pretraining",
        "llm",
        "model training",
    ),
    "databases": (
        "database",
        "sql",
        "query",
        "index",
        "transaction",
        "postgres",
        "mysql",
        "mongodb",
        "replication",
        "sharding",
        "schema",
        "table",
    ),
    "software_engineering": (
        "software",
        "engineering",
        "deployment",
        "testing",
        "refactor",
        "api",
        "git",
        "docker",
        "kubernetes",
        "ci/cd",
        "code review",
        "repository",
    ),
}


@dataclass
class Document:
    source: str
    text: str
    language: str = "en"
    domain: str = "general"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(
            source=str(data.get("source", "unknown")),
            text=str(data.get("text", "")),
            language=str(data.get("language", "en")),
            domain=str(data.get("domain", "general")),
            meta=dict(data.get("meta") or {}),
        )


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc


def write_jsonl(path: Path, records: Iterator[dict[str, Any] | Document]) -> int:
    ensure_dirs(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_dict() if isinstance(record, Document) else record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def hash_text(text: str) -> str:
    normalized = normalize_whitespace(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def infer_domain(text: str) -> str:
    lowered = text.lower()
    scores = {
        domain: sum(lowered.count(keyword) for keyword in keywords)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return "general"
    return best_domain


def detect_language(text: str) -> tuple[str, float]:
    """Lightweight English detector based on common words and ASCII ratio."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return "unknown", 0.0

    english_markers = {
        "the",
        "and",
        "is",
        "to",
        "of",
        "in",
        "for",
        "with",
        "that",
        "this",
        "are",
        "on",
        "as",
        "by",
        "from",
    }
    marker_hits = sum(1 for word in words if word in english_markers)
    marker_ratio = marker_hits / len(words)
    ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)
    confidence = min(1.0, 0.55 * marker_ratio + 0.45 * ascii_ratio)
    language = "en" if confidence >= 0.35 else "unknown"
    return language, confidence
