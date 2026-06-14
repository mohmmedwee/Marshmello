"""Step 1 — ingest raw files into normalized JSONL documents."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from scripts.common import (  # noqa: E402
    CLEANED_DIR,
    Document,
    RAW_DIR,
    SUPPORTED_EXTENSIONS,
    detect_language,
    ensure_dirs,
    infer_domain,
    write_jsonl,
)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"script", "style", "nav", "header", "footer"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "header", "footer"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return html.unescape("".join(self._chunks))


def html_to_text(raw_html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    return parser.text()


def pdf_to_text(path: Path) -> str | None:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return None

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    return text or None


def source_from_path(path: Path, raw_root: Path) -> str:
    rel = path.relative_to(raw_root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return "unknown"


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_jsonl(path: Path, source: str) -> list[Document]:
    docs: list[Document] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        text = str(payload.get("text") or payload.get("content") or "").strip()
        if not text:
            continue
        language = str(payload.get("language") or "en")
        domain = str(payload.get("domain") or infer_domain(text))
        docs.append(
            Document(
                source=str(payload.get("source") or source),
                text=text,
                language=language,
                domain=domain,
                meta={"path": str(path), "line": line_no},
            )
        )
    return docs


def ingest_file(path: Path, raw_root: Path) -> list[Document]:
    source = source_from_path(path, raw_root)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return ingest_jsonl(path, source=source)

    if suffix in {".html", ".htm"}:
        text = html_to_text(read_text_file(path))
    elif suffix == ".pdf":
        text = pdf_to_text(path)
        if text is None:
            print(f"  skip pdf (install pypdf): {path.name}")
            return []
    else:
        text = read_text_file(path)

    text = text.strip()
    if not text:
        return []

    language, confidence = detect_language(text)
    return [
        Document(
            source=source,
            text=text,
            language=language,
            domain=infer_domain(text),
            meta={"path": str(path), "language_confidence": round(confidence, 4)},
        )
    ]


def discover_raw_files(raw_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def run_ingest(raw_root: Path, output_path: Path) -> dict[str, int]:
    ensure_dirs(output_path.parent)
    files = discover_raw_files(raw_root)
    counts: dict[str, int] = {"files": len(files), "documents": 0}

    def records():
        nonlocal counts
        for path in files:
            for doc in ingest_file(path, raw_root):
                counts["documents"] += 1
                yield doc.to_dict()

    write_jsonl(output_path, records())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw files into normalized JSONL.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=CLEANED_DIR.parent / "ingested" / "documents.jsonl",
        help="Intermediate ingested JSONL (default: ingested/documents.jsonl)",
    )
    args = parser.parse_args()

    stats = run_ingest(args.raw_dir, args.output)
    print(f"Ingested {stats['documents']} documents from {stats['files']} files -> {args.output}")


if __name__ == "__main__":
    main()
