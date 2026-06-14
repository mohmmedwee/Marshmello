"""Step 3 — exact hash deduplication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from scripts.common import (  # noqa: E402
    DEDUPED_DIR,
    Document,
    ensure_dirs,
    hash_text,
    iter_jsonl,
    write_jsonl,
)


def run_dedupe(input_path: Path, output_path: Path) -> dict[str, int | float]:
    ensure_dirs(output_path.parent)
    seen: set[str] = set()
    stats = {
        "input_documents": 0,
        "output_documents": 0,
        "duplicates_removed": 0,
    }

    def records():
        for payload in iter_jsonl(input_path):
            stats["input_documents"] += 1
            doc = Document.from_dict(payload)
            digest = hash_text(doc.text)
            if digest in seen:
                stats["duplicates_removed"] += 1
                continue
            seen.add(digest)
            doc.meta["content_hash"] = digest
            stats["output_documents"] += 1
            yield doc.to_dict()

    write_jsonl(output_path, records())
    total = int(stats["input_documents"])
    removed = int(stats["duplicates_removed"])
    stats["duplicate_ratio"] = round(removed / total, 4) if total else 0.0
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact deduplication via SHA-256.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEDUPED_DIR / "documents.jsonl")
    args = parser.parse_args()

    stats = run_dedupe(args.input, args.output)
    pct = float(stats["duplicate_ratio"]) * 100
    print(
        f"Deduped {stats['output_documents']} documents "
        f"(removed {stats['duplicates_removed']} duplicates, {pct:.1f}%) -> {args.output}"
    )


if __name__ == "__main__":
    main()
