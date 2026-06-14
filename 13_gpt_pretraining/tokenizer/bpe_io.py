"""Save/load BPE tokenizer trained with phase 09 BPETokenizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from bpe_demo import BPETokenizer  # noqa: E402


def save_tokenizer(bpe: BPETokenizer, path: Path) -> None:
    if not hasattr(bpe, "stoi"):
        bpe.build_index()
    payload: dict[str, Any] = {
        "end_marker": bpe.END,
        "merges": [list(pair) for pair in bpe.merges],
        "vocab": sorted(bpe.vocab),
        "stoi": bpe.stoi,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tokenizer(path: Path) -> BPETokenizer:
    data = json.loads(path.read_text(encoding="utf-8"))
    bpe = BPETokenizer()
    bpe.END = data.get("end_marker", bpe.END)
    bpe.merges = [tuple(pair) for pair in data["merges"]]
    bpe.vocab = set(data["vocab"])
    bpe.stoi = {k: int(v) for k, v in data["stoi"].items()}
    bpe.itos = {int(i): tok for tok, i in bpe.stoi.items()}
    bpe._initial_vocab_size = len(
        {tok for tok in bpe.vocab if len(tok) == 1 or tok in (bpe.END, "\n")}
    )
    # Phase 13: readable spacing (</w> → word boundary, punctuation cleanup)
    from tokenizer.decode import decode_ids_pretty as _decode_ids_pretty

    bpe.decode_ids_pretty = lambda ids: _decode_ids_pretty(bpe, ids)  # type: ignore[attr-defined]
    return bpe
