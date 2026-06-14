#!/usr/bin/env python3
"""Download Marshmello weights from Hugging Face Hub into project checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from config import GPTConfig, checkpoint_dir_for, latest_checkpoint_for, resolve_config  # noqa: E402
from training.trainer import save_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Marshmello from Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, default="ostah-1010/Marshmello")
    parser.add_argument("--config", type=str, default="large_50m")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    local_dir = Path(snapshot_download(args.repo_id))
    config_path = local_dir / "config.json"
    weights_path = local_dir / "model.safetensors"
    tokenizer_src = local_dir / "tokenizer.json"
    training_meta_path = local_dir / "training_meta.json"

    if not weights_path.exists():
        weights_path = Path(hf_hub_download(args.repo_id, "model.safetensors"))

    hub_config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = resolve_config(args.config)
    if hub_config.get("config_name"):
        saved = GPTConfig(**hub_config.get("training_config", {})) if (local_dir / "training_meta.json").exists() else cfg
        if training_meta_path.exists():
            meta = json.loads(training_meta_path.read_text(encoding="utf-8"))
            saved = GPTConfig(**meta["training_config"])
        cfg = saved

    from safetensors.torch import load_file

    from model.gpt import GPT  # noqa: E402
    from training.trainer import build_model  # noqa: E402

    device = torch.device("cpu")
    model = build_model(hub_config["vocab_size"], cfg, device)
    state = load_file(str(weights_path))
    model.load_state_dict(state)

    ckpt_dir = checkpoint_dir_for(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = latest_checkpoint_for(cfg)
    meta = json.loads(training_meta_path.read_text(encoding="utf-8")) if training_meta_path.exists() else {}
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        out,
        model,
        optimizer,
        int(meta.get("step", 0)),
        cfg,
        float(meta.get("train_loss", 0.0)),
        float(meta.get("val_loss", 0.0)),
    )

    tokenizer_dst = PHASE_ROOT / "tokenizer" / "tokenizer.json"
    if tokenizer_src.exists():
        shutil.copy2(tokenizer_src, tokenizer_dst)

    print(f"Downloaded {args.repo_id}")
    print(f"Checkpoint → {out}")
    print(f"Tokenizer → {tokenizer_dst}")


if __name__ == "__main__":
    main()
