"""Step 2 — clean ingested documents."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from scripts.common import (  # noqa: E402
    CLEANED_DIR,
    Document,
    ensure_dirs,
    iter_jsonl,
    normalize_whitespace,
    write_jsonl,
)

NAV_LINE = re.compile(
    r"^(?:home|contact|login|sign in|privacy|terms|menu|navigation)\s*(?:\||/|-)\s*",
    re.IGNORECASE,
)
SEPARATOR_LINE = re.compile(r"^[\s\-_|=*#]{3,}$")
FOOTER_LINE = re.compile(
    r"^(?:copyright|all rights reserved|page \d+ of \d+|last updated).*$",
    re.IGNORECASE,
)
LOREM = re.compile(r"\blorem ipsum\b", re.IGNORECASE)


def strip_navigation_lines(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if NAV_LINE.match(stripped):
            continue
        if SEPARATOR_LINE.match(stripped):
            continue
        if FOOTER_LINE.match(stripped):
            continue
        if re.fullmatch(r"(?:\|\s*)+(?:home|contact|login)(?:\s*\|)*", stripped, re.IGNORECASE):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def clean_text(text: str) -> str:
    text = strip_navigation_lines(text)
    text = normalize_whitespace(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n\n".join(lines)


def clean_document(doc: Document) -> Document | None:
    cleaned = clean_text(doc.text)
    if not cleaned or LOREM.search(cleaned):
        return None
    return Document(
        source=doc.source,
        text=cleaned,
        language=doc.language,
        domain=doc.domain,
        meta={**doc.meta, "cleaned": True},
    )


def run_clean(input_path: Path, output_path: Path) -> dict[str, int]:
    ensure_dirs(output_path.parent)
    stats = {"input_documents": 0, "output_documents": 0, "removed": 0}

    def records():
        for payload in iter_jsonl(input_path):
            stats["input_documents"] += 1
            doc = clean_document(Document.from_dict(payload))
            if doc is None:
                stats["removed"] += 1
                continue
            stats["output_documents"] += 1
            yield doc.to_dict()

    write_jsonl(output_path, records())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean ingested JSONL documents.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CLEANED_DIR / "documents.jsonl")
    args = parser.parse_args()

    stats = run_clean(args.input, args.output)
    print(
        f"Cleaned {stats['output_documents']} documents "
        f"(removed {stats['removed']} of {stats['input_documents']}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
