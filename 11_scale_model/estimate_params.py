"""
Phase 11: How Transformer parameter count scales

This script does NOT train anything. It counts parameters analytically so you
can see how d_model, num_layers, d_ff, and vocab_size affect model size.

Architecture assumed (same family as phases 07–10):
  token embedding → positional encoding → N transformer blocks → layer norm → LM head

Each transformer block:
  LayerNorm → Multi-Head Self-Attention → residual
  LayerNorm → Feed-Forward Network (d_model → d_ff → d_model) → residual

Phases 07–10 use FIXED sinusoidal positional encoding (0 learned params).
This estimator also shows learned positional embeddings for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParamBreakdown:
    """Parameter count by component (all counts are trainable weights + biases)."""

    token_embeddings: int
    positional_embeddings: int
    attention_qkv: int
    attention_output: int
    feed_forward: int
    layer_norms: int
    lm_head: int

    @property
    def attention_blocks(self) -> int:
        return self.attention_qkv + self.attention_output

    @property
    def ffn_blocks(self) -> int:
        return self.feed_forward

    @property
    def attention_total(self) -> int:
        return self.attention_blocks

    @property
    def transformer_blocks(self) -> int:
        return self.attention_total + self.feed_forward + (
            self.layer_norms - 2
        )  # block norms only (exclude final norm)

    @property
    def total(self) -> int:
        return (
            self.token_embeddings
            + self.positional_embeddings
            + self.attention_qkv
            + self.attention_output
            + self.feed_forward
            + self.layer_norms
            + self.lm_head
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "token_embeddings": self.token_embeddings,
            "positional_embeddings": self.positional_embeddings,
            "attention_qkv": self.attention_qkv,
            "attention_output": self.attention_output,
            "attention_total": self.attention_total,
            "feed_forward": self.feed_forward,
            "layer_norms": self.layer_norms,
            "lm_head": self.lm_head,
            "total": self.total,
        }


def estimate_transformer_params(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    block_size: int,
    *,
    learned_positional: bool = False,
    tie_embeddings: bool = False,
    lm_head_bias: bool = False,
) -> ParamBreakdown:
    """
    Estimate trainable parameters for a decoder-only transformer LM.

    Matches Phase 13 GPT when:
      learned_positional=True, tie_embeddings=False, lm_head_bias=False

    Attention uses one fused QKV projection (Linear(d_model, 3*d_model)),
    matching CausalSelfAttention in 13_gpt_pretraining/model/attention.py.

    Parameters
    ----------
    vocab_size:
        Number of tokens in the vocabulary (embedding rows + LM head columns).
    d_model:
        Hidden dimension. THIS is the biggest lever — many layers scale as d_model².
    num_layers:
        Number of stacked transformer blocks. Linear multiplier on block params.
    num_heads:
        Number of attention heads. Must divide d_model evenly when training,
        but head count does NOT change total attention parameter count in the
        standard fused QKV projection (included for completeness / validation).
    d_ff:
        Feed-forward inner dimension. Usually 4× d_model in large models; scales
        linearly with d_ff inside each block.
    block_size:
        Context length. Only affects positional embeddings if learned_positional=True.
    learned_positional:
        If True, count block_size × d_model learned position vectors.
        If False, count 0 (sinusoidal PE used in phases 07–10 — not trained).
    tie_embeddings:
        If True, LM head shares token embedding weights (counted once in token_embeddings).
    lm_head_bias:
        If True, include vocab bias terms on the LM head (phases 07–10 use bias=True).

    Returns
    -------
    ParamBreakdown with per-component counts.
    """
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

    # --- Token embedding table: one d_model vector per vocab token ---
    token_embeddings = vocab_size * d_model

    # --- Positional embeddings ---
    # Phases 07–10: fixed sin/cos buffer → 0 trainable params.
    # GPT-style models sometimes learn this table instead.
    positional_embeddings = block_size * d_model if learned_positional else 0

    # --- Self-attention (per layer): fused QKV Linear(d_model → 3*d_model) ---
    attention_qkv = num_layers * (d_model * (3 * d_model) + (3 * d_model))

    # Output projection after attention: Linear(d_model → d_model)
    attention_output = num_layers * (d_model * d_model + d_model)

    # --- Feed-forward (per layer): d_model → d_ff → d_model ---
    # Linear1: d_model×d_ff + d_ff ; Linear2: d_ff×d_model + d_model
    ffn_per_layer = d_model * d_ff + d_ff + d_ff * d_model + d_model
    feed_forward = num_layers * ffn_per_layer

    # --- LayerNorm (per layer × 2) + final pre-head norm ---
    # Each LayerNorm: scale + bias → 2 × d_model
    layer_norms = (2 * num_layers + 1) * (2 * d_model)

    # --- Language model head ---
    # Phase 13 GPT: separate untied head, bias=False.
    # Tied weights (GPT-2 style): counted only in token_embeddings.
    if tie_embeddings:
        lm_head = 0
    else:
        lm_head = d_model * vocab_size
        if lm_head_bias:
            lm_head += vocab_size

    return ParamBreakdown(
        token_embeddings=token_embeddings,
        positional_embeddings=positional_embeddings,
        attention_qkv=attention_qkv,
        attention_output=attention_output,
        feed_forward=feed_forward,
        layer_norms=layer_norms,
        lm_head=lm_head,
    )


def _module_param_count(module) -> int:  # noqa: ANN001
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def gpt_parameter_breakdown(model) -> ParamBreakdown:  # noqa: ANN001
    """
    Count trainable parameters from a Phase 13 GPT module tree.

    Expects: token_emb, pos_emb, blocks[].attn.qkv/out_proj/ffn/norm1/norm2, norm, lm_head.
    """
    attention_qkv = sum(_module_param_count(block.attn.qkv) for block in model.blocks)
    attention_output = sum(_module_param_count(block.attn.out_proj) for block in model.blocks)
    feed_forward = sum(_module_param_count(block.ffn) for block in model.blocks)
    block_norms = sum(
        _module_param_count(block.norm1) + _module_param_count(block.norm2)
        for block in model.blocks
    )
    layer_norms = block_norms + _module_param_count(model.norm)

    token_embeddings = _module_param_count(model.token_emb)
    positional_embeddings = _module_param_count(model.pos_emb)
    lm_head = _module_param_count(model.lm_head)

    return ParamBreakdown(
        token_embeddings=token_embeddings,
        positional_embeddings=positional_embeddings,
        attention_qkv=attention_qkv,
        attention_output=attention_output,
        feed_forward=feed_forward,
        layer_norms=layer_norms,
        lm_head=lm_head,
    )


def embeddings_are_tied(model) -> bool:  # noqa: ANN001
    """True when LM head shares the token embedding weight tensor."""
    if not hasattr(model, "token_emb") or not hasattr(model, "lm_head"):
        return False
    return model.lm_head.weight is model.token_emb.weight


def estimate_for_gpt_model(model, cfg: dict) -> ParamBreakdown:  # noqa: ANN001
    """Analytic estimate aligned with a built Phase 13 GPT instance."""
    return estimate_transformer_params(
        vocab_size=int(getattr(model, "vocab_size", cfg["vocab_size"])),
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        block_size=cfg["block_size"],
        learned_positional=True,
        tie_embeddings=embeddings_are_tied(model),
        lm_head_bias=getattr(model.lm_head, "bias", None) is not None,
    )


def print_parameter_comparison(
    actual: ParamBreakdown,
    estimated: ParamBreakdown,
    *,
    title: str = "Parameter breakdown (actual vs analytic)",
    weight_tied: bool | None = None,
    lm_head_bias: bool | None = None,
) -> None:
    print(title)
    print("=" * 72)
    print(f"{'Component':<24} {'Actual':>12} {'Estimate':>12} {'Delta':>10}")
    print("-" * 72)

    rows: list[tuple[str, int, int]] = [
        ("Token embeddings", actual.token_embeddings, estimated.token_embeddings),
        ("Positional embeddings", actual.positional_embeddings, estimated.positional_embeddings),
        ("Attention blocks", actual.attention_blocks, estimated.attention_blocks),
        ("FFN blocks", actual.ffn_blocks, estimated.ffn_blocks),
        ("Layer norms", actual.layer_norms, estimated.layer_norms),
        ("LM head", actual.lm_head, estimated.lm_head),
    ]
    for name, act, est in rows:
        delta_pct = 0.0 if act == 0 else (est - act) / act * 100
        print(f"{name:<24} {act:>12,} {est:>12,} {delta_pct:>9.2f}%")

    print("-" * 72)
    total_delta = 0.0 if actual.total == 0 else (estimated.total - actual.total) / actual.total * 100
    print(
        f"{'TOTAL':<24} {actual.total:>12,} {estimated.total:>12,} {total_delta:>9.2f}%"
    )
    if weight_tied is not None:
        print(f"Weight tying: {'yes' if weight_tied else 'no'}")
    if lm_head_bias is not None:
        print(f"LM head bias: {'yes' if lm_head_bias else 'no'}")


def format_count(n: int) -> str:
    """Human-readable parameter count."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M ({n:,})"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K ({n:,})"
    return f"{n:,}"


def print_breakdown(name: str, cfg: dict, breakdown: ParamBreakdown) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(
        f"  config: vocab={cfg['vocab_size']}, d_model={cfg['d_model']}, "
        f"layers={cfg['num_layers']}, heads={cfg['num_heads']}, "
        f"d_ff={cfg['d_ff']}, block_size={cfg['block_size']}"
    )
    print()
    print(f"  token embeddings:      {breakdown.token_embeddings:>12,}")
    print(f"  positional embeddings: {breakdown.positional_embeddings:>12,}")
    print(f"  attention QKV:         {breakdown.attention_qkv:>12,}")
    print(f"  attention output proj: {breakdown.attention_output:>12,}")
    print(f"  feed-forward (FFN):    {breakdown.feed_forward:>12,}")
    print(f"  layer norms:           {breakdown.layer_norms:>12,}")
    print(f"  LM head:               {breakdown.lm_head:>12,}")
    print(f"  {'—' * 28}")
    print(f"  TOTAL:                 {breakdown.total:>12,}  ≈ {format_count(breakdown.total)}")


def main() -> None:
    print("Phase 11: Transformer parameter scaling (estimate only — no training)")
    print("=" * 60)
    print()
    print("What increases parameters the MOST:")
    print("  1. d_model        → attention scales as O(d_model²) per layer")
    print("  2. num_layers     → linear multiplier on every block")
    print("  3. d_ff           → FFN scales as O(d_model × d_ff) per layer")
    print("  4. vocab_size     → embedding + LM head scale with vocab")
    print("  5. block_size     → only matters for LEARNED positional embeddings")
    print("  num_heads         → splits d_model; same total QKV params in standard impl")
    print()
    print("Phases 07–10 use sinusoidal positional encoding → 0 pos params.")
    print("Counts below use learned_positional=False unless noted.")

    # Hand-tuned configs targeting ~500K, ~2M, ~10M, ~50M totals
    configs: list[tuple[str, str, dict]] = [
        (
            "tiny",
            "~500K — similar scale to phase 10",
            {
                "vocab_size": 864,
                "d_model": 112,
                "num_layers": 2,
                "num_heads": 4,
                "d_ff": 448,
                "block_size": 64,
            },
        ),
        (
            "small",
            "~2M",
            {
                "vocab_size": 1000,
                "d_model": 208,
                "num_layers": 3,
                "num_heads": 8,
                "d_ff": 832,
                "block_size": 128,
            },
        ),
        (
            "medium",
            "~10M",
            {
                "vocab_size": 1500,
                "d_model": 432,
                "num_layers": 4,
                "num_heads": 8,
                "d_ff": 1728,
                "block_size": 256,
            },
        ),
        (
            "large",
            "~50M",
            {
                "vocab_size": 5000,
                "d_model": 768,
                "num_layers": 6,
                "num_heads": 12,
                "d_ff": 3072,
                "block_size": 512,
            },
        ),
    ]

    rows: list[tuple[str, int, int, int, int, int]] = []

    for name, label, cfg in configs:
        bd = estimate_transformer_params(**cfg, lm_head_bias=True)
        print_breakdown(f"{name.upper()} ({label})", cfg, bd)
        rows.append(
            (
                name,
                bd.total,
                bd.attention_total,
                bd.feed_forward,
                bd.token_embeddings + bd.lm_head,
                bd.layer_norms,
            )
        )

    # --- Summary table ---
    print(f"\n{'=' * 60}")
    print("SUMMARY TABLE")
    print("=" * 60)
    header = (
        f"{'config':<8} {'total':>10} {'attention':>10} "
        f"{'FFN':>10} {'emb+head':>10} {'norms':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, total, attn, ffn, emb_head, norms in rows:
        print(
            f"{name:<8} {total:>10,} {attn:>10,} {ffn:>10,} "
            f"{emb_head:>10,} {norms:>8,}"
        )

    # --- Show positional embedding impact on medium config ---
    print(f"\n{'─' * 60}")
    print("  BONUS: learned vs sinusoidal positional embeddings (medium config)")
    print(f"{'─' * 60}")
    medium_cfg = configs[2][2]
    fixed = estimate_transformer_params(**medium_cfg, learned_positional=False)
    learned = estimate_transformer_params(**medium_cfg, learned_positional=True)
    print(f"  Sinusoidal (phases 07–10): {fixed.total:,} params")
    print(f"  Learned positions:         {learned.total:,} params")
    print(
        f"  Difference:                +{learned.total - fixed.total:,} "
        f"(block_size × d_model = {medium_cfg['block_size']} × {medium_cfg['d_model']})"
    )

    # --- Scaling intuition ---
    print(f"\n{'=' * 60}")
    print("SCALING INTUITION")
    print("=" * 60)
    print("Doubling d_model (layers fixed) roughly QUADRUPLES attention params")
    print("because QKV matrices are d_model × d_model.")
    print()
    print("Example — tiny config but d_model 112 → 224 (2×):")
    base = estimate_transformer_params(**configs[0][2])
    doubled = estimate_transformer_params(**{**configs[0][2], "d_model": 224, "d_ff": 896})
    print(f"  d_model=112:  {base.total:,} params")
    print(f"  d_model=224:  {doubled.total:,} params")
    print(f"  ratio:        {doubled.total / base.total:.1f}×")
    print()
    print("Real GPT-2 small is ~124M params. Same formulas — bigger numbers.")


if __name__ == "__main__":
    main()
