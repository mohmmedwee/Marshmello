"""Export sharded JSONL into a single corpus.txt for Phase 13 training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_ROOT))

from scripts.common import SHARDS_DIR, iter_jsonl  # noqa: E402


def export_corpus(
    shards_dir: Path,
    output_path: Path,
    *,
    separator: str = "\n\n",
) -> dict[str, int]:
    texts: list[str] = []
    for shard_path in sorted(shards_dir.glob("shard_*.jsonl")):
        for payload in iter_jsonl(shard_path):
            text = str(payload.get("text", "")).strip()
            if text:
                texts.append(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(separator.join(texts) + ("\n" if texts else ""), encoding="utf-8")
    return {"documents": len(texts), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Phase 14 shards into corpus.txt for GPT pretraining."
    )
    parser.add_argument("--shards-dir", type=Path, default=SHARDS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=PHASE_ROOT.parent / "13_gpt_pretraining" / "data" / "corpus.txt",
    )
    args = parser.parse_args()

    stats = export_corpus(args.shards_dir, args.output)
    print(f"Exported {stats['documents']} documents -> {stats['output']}")


if __name__ == "__main__":
    main()
