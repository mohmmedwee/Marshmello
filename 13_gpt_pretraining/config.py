"""Hyperparameters for Phase 13/15 GPT pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PHASE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PHASE_ROOT / "data"
TOKENIZER_DIR = PHASE_ROOT / "tokenizer"
MODEL_DIR = PHASE_ROOT / "model"
TRAINING_DIR = PHASE_ROOT / "training"
CHECKPOINT_DIR = PHASE_ROOT / "checkpoints"

CORPUS_PATH = DATA_DIR / "corpus.txt"
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"


@dataclass(frozen=True)
class GPTConfig:
    """Model + training defaults."""

    vocab_size: int = 8000
    d_model: int = 384
    num_layers: int = 4
    num_heads: int = 6
    d_ff: int = 1536
    block_size: int = 256
    dropout: float = 0.1

    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 5000
    eval_every: int = 100
    log_every: int = 10
    checkpoint_every: int = 1000
    val_ratio: float = 0.1
    grad_clip: float = 1.0

    target_vocab_size: int = 8000
    corpus_min_words: int = 100_000

    seed: int = 42
    config_name: str = "default"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.effective_batch_size * self.block_size


DEFAULT_CONFIG = GPTConfig()

# Phase 15 — ~50M params on M4 Max (vocab from trained BPE, typically ~45–55M)
LARGE_50M_CONFIG = GPTConfig(
    config_name="large_50m",
    d_model=768,
    num_layers=6,
    num_heads=12,
    d_ff=3072,
    block_size=512,
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.1,
    max_steps=10_000,
    eval_every=200,
    log_every=10,
    checkpoint_every=500,
    grad_clip=1.0,
    target_vocab_size=8000,
)

CONFIGS: dict[str, GPTConfig] = {
    "default": DEFAULT_CONFIG,
    "v1": DEFAULT_CONFIG,
    "large_50m": LARGE_50M_CONFIG,
}


def resolve_config(name: str) -> GPTConfig:
    key = name.strip().lower().replace("-", "_")
    if key not in CONFIGS:
        choices = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown config {name!r}. Choose from: {choices}")
    return CONFIGS[key]


def checkpoint_dir_for(cfg: GPTConfig) -> Path:
    if cfg.config_name == "default":
        return CHECKPOINT_DIR
    return CHECKPOINT_DIR / cfg.config_name


def latest_checkpoint_for(cfg: GPTConfig) -> Path:
    return checkpoint_dir_for(cfg) / "latest.pt"
