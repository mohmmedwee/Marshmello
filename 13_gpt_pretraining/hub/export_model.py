"""Export Marshmello GPT checkpoint + BPE tokenizer for Hugging Face Hub."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))
sys.path.insert(0, str(PROJECT_ROOT / "11_scale_model"))

from config import GPTConfig, TOKENIZER_PATH, latest_checkpoint_for, resolve_config  # noqa: E402
from estimate_params import estimate_for_gpt_model, gpt_parameter_breakdown  # noqa: E402
from model.gpt import GPT, count_parameters  # noqa: E402
from tokenizer.bpe_io import load_tokenizer  # noqa: E402
from training.trainer import build_model, load_checkpoint  # noqa: E402

MODEL_ALIASES = {
    "default": "Marshmello-8M",
    "large_50m": "Marshmello-45M",
}


def build_model_card(
    *,
    repo_id: str,
    alias: str,
    cfg: GPTConfig,
    vocab_size: int,
    param_count: int,
    checkpoint_step: int,
    train_loss: float | None,
    val_loss: float | None,
) -> str:
    train_note = ""
    if train_loss is not None and val_loss is not None:
        train_note = f"\nTraining snapshot: step {checkpoint_step}, train loss {train_loss:.4f}, val loss {val_loss:.4f}.\n"

    return f"""---
license: apache-2.0
language:
- en
tags:
- pytorch
- gpt
- causal-lm
- decoder-only
- from-scratch
- marshmello
library_name: pytorch
---

# {alias}

**{alias}** is a decoder-only GPT language model trained from scratch in the
[mini-transformer-from-scratch](https://github.com/) educational project.

| | |
|--|--|
| Parameters | ~{param_count / 1e6:.1f}M ({param_count:,}) |
| Architecture | GPT (causal self-attention, learned positional embeddings) |
| `d_model` | {cfg.d_model} |
| Layers | {cfg.num_layers} |
| Heads | {cfg.num_heads} |
| FFN dim | {cfg.d_ff} |
| Context | {cfg.block_size} tokens |
| Tokenizer | BPE (~{vocab_size:,} vocab) |
| Config key | `{cfg.config_name}` |
{train_note}
## Load in this project

```bash
git clone <your-repo> mini-transformer-from-scratch
cd mini-transformer-from-scratch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt huggingface_hub safetensors

# Download weights into checkpoints/
python 13_gpt_pretraining/hub/download_from_hub.py --repo-id {repo_id}

# Generate
python 13_gpt_pretraining/generate.py --config {cfg.config_name} --prompt "Database systems"
```

## Load with Python (Hub)

```python
from pathlib import Path
from huggingface_hub import snapshot_download

local = Path(snapshot_download("{repo_id}"))
# Use files: config.json, model.safetensors, tokenizer.json
```

## Marshmello family

| Model | Hub repo | Params |
|-------|----------|--------|
| Marshmello-8M | `ostah-1010/Marshmello-8M` | ~8M |
| Marshmello-45M | `ostah-1010/Marshmello` | ~45M |

Both models share the same BPE tokenizer and demo corpus. See Phase 16 evaluation for memorization analysis on small data.

## Limitations

- Trained on a **small educational corpus** (not web-scale pretraining)
- Outputs may **memorize** training paragraphs
- Not instruction-tuned — use Phase 17+ for chat behavior
- Custom PyTorch implementation (not `transformers` AutoModel)

## Citation

Built with the mini-transformer-from-scratch learning path (Phases 01–17).
"""


def export_model(
    *,
    config_name: str,
    output_dir: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    cfg = resolve_config(config_name)
    alias = MODEL_ALIASES.get(config_name, config_name)
    checkpoint = checkpoint_path or latest_checkpoint_for(cfg)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint for {alias} at {checkpoint}. "
            f"Train with: python 13_gpt_pretraining/training/trainer.py --config {config_name}"
        )
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config")
    if saved_cfg:
        cfg = GPTConfig(**saved_cfg)

    bpe = load_tokenizer(TOKENIZER_PATH)
    device = torch.device("cpu")
    model = build_model(bpe.vocab_size, cfg, device)
    step = load_checkpoint(checkpoint, model, optimizer=None, device=device)
    model.eval()

    params = count_parameters(model)
    breakdown = gpt_parameter_breakdown(model)
    estimate = estimate_for_gpt_model(model, cfg.__dict__)

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()

    from safetensors.torch import save_file

    save_file(model.state_dict(), output_dir / "model.safetensors")

    hub_config = {
        "model_type": "marshmello_gpt",
        "architectures": ["MarshmelloGPT"],
        "config_name": cfg.config_name,
        "model_alias": alias,
        "vocab_size": bpe.vocab_size,
        "d_model": cfg.d_model,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "d_ff": cfg.d_ff,
        "block_size": cfg.block_size,
        "dropout": cfg.dropout,
        "parameter_count": params,
        "parameter_estimate": estimate.total,
        "parameter_breakdown": breakdown.as_dict(),
        "torch_dtype": "float32",
    }
    (output_dir / "config.json").write_text(
        json.dumps(hub_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    generation_config = {
        "max_new_tokens": 120,
        "temperature": 0.7,
        "top_k": 30,
        "repetition_penalty": 1.15,
        "presence_penalty": 0.6,
        "stop_on_sentence_end": True,
    }
    (output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2) + "\n",
        encoding="utf-8",
    )

    training_meta = {
        "checkpoint_path": str(checkpoint),
        "step": step,
        "train_loss": ckpt.get("train_loss"),
        "val_loss": ckpt.get("val_loss"),
        "training_config": asdict(cfg),
    }
    (output_dir / "training_meta.json").write_text(
        json.dumps(training_meta, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    shutil.copy2(TOKENIZER_PATH, output_dir / "tokenizer.json")

    repo_id_placeholder = "ostah-1010/Marshmello"
    if config_name == "default":
        repo_id_placeholder = "ostah-1010/Marshmello-8M"

    readme = build_model_card(
        repo_id=repo_id_placeholder,
        alias=alias,
        cfg=cfg,
        vocab_size=bpe.vocab_size,
        param_count=params,
        checkpoint_step=step,
        train_loss=ckpt.get("train_loss"),
        val_loss=ckpt.get("val_loss"),
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    return {
        "alias": alias,
        "config_name": config_name,
        "params": params,
        "step": step,
        "output_dir": str(output_dir),
    }
