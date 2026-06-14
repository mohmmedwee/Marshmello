"""Step 6 — split deduped documents into fixed-size JSONL shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from scripts.common import SHARDS_DIR, ensure_dirs, iter_jsonl  # noqa: E402


def run_shard(
    input_path: Path,
    output_dir: Path,
    *,
    max_bytes: int,
    prefix: str = "shard",
) -> dict[str, int | list[str]]:
    ensure_dirs(output_dir)
    for old in output_dir.glob(f"{prefix}_*.jsonl"):
        old.unlink()

    shard_paths: list[str] = []
    shard_idx = 0
    current_size = 0
    handle = None
    docs_in_shard = 0
    total_docs = 0

    def open_shard() -> None:
        nonlocal handle, shard_idx, current_size, docs_in_shard
        if handle:
            handle.close()
        path = output_dir / f"{prefix}_{shard_idx:03d}.jsonl"
        handle = path.open("w", encoding="utf-8")
        shard_paths.append(str(path))
        shard_idx += 1
        current_size = 0
        docs_in_shard = 0

    open_shard()

    for payload in iter_jsonl(input_path):
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        if current_size + len(encoded) > max_bytes and docs_in_shard > 0:
            open_shard()
        handle.write(line)
        current_size += len(encoded)
        docs_in_shard += 1
        total_docs += 1

    if handle:
        handle.close()

    if total_docs == 0:
        empty = output_dir / f"{prefix}_000.jsonl"
        empty.write_text("", encoding="utf-8")
        shard_paths = [str(empty)]

    return {
        "documents": total_docs,
        "shards": len(shard_paths),
        "shard_paths": shard_paths,
        "max_bytes": max_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard JSONL output by byte size.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SHARDS_DIR)
    parser.add_argument("--max-mb", type=float, default=100.0, help="Max shard size in MB")
    args = parser.parse_args()

    max_bytes = int(args.max_mb * 1024 * 1024)
    stats = run_shard(args.input, args.output_dir, max_bytes=max_bytes)
    print(
        f"Sharded {stats['documents']} documents into {stats['shards']} files "
        f"(max {args.max_mb} MB each) -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
