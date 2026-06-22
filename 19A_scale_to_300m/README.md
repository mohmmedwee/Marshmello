# Phase 19A — Marshmello-300M

Scale the Marshmello model from ~45M/55M up to a **~269M-parameter** config
(`large_300m`) while reusing the existing v2 BPE tokenizer (`vocab_size=8000`).

This phase deliberately ships **config + benchmark + a 20-step smoke test only**.
No long training is run here.

## Why we are scaling

The 45M / 55M models hit a practical plateau:

- Best checkpoint: `18B_marshmello_instruct/checkpoints/best_18j_routing.pt`
- Best **18J routing**: **18%**
- Best **18K domain score**: **21.8%**

Every attempt to push quality at this size by *adding data* made things worse, not better:

- Core SFT **reduced** 18J routing.
- Micro-patching **reduced** 18J routing.
- Broad SFT (500 examples) **reduced** both 18K domain score and 18J routing.

When more/better fine-tuning data consistently *regresses* the benchmarks, the
bottleneck is most likely **model capacity**, not the dataset. So the next lever
is parameters, not more SFT on the small model.

## Why 300M before 500M

We jump to ~300M (not straight to 500M) on purpose:

- **De-risk the scale-up cheaply.** 300M is large enough to test whether extra
  capacity actually moves 18J/18K, but small enough to still iterate on a
  single Mac.
- **Memory headroom.** A 500M model with optimizer state (AdamW = params + 2
  moments) plus activations can exceed unified memory on MPS. 300M leaves room
  to find a workable batch size / accumulation schedule first.
- **Fail fast.** If 300M does *not* improve the benchmarks, 500M almost
  certainly won't be worth the cost — we learn that for a fraction of the
  compute.

## Config: `large_300m`

Defined in `13_gpt_pretraining/config.py`. Reuses the existing tokenizer
(`vocab_size=8000`); the tokenizer and `large_50m` are **unchanged**.

| Field      | large_50m | large_300m |
|------------|-----------|------------|
| d_model    | 768       | 1024       |
| num_layers | 6         | 20         |
| num_heads  | 12        | 16         |
| d_ff       | 3072      | 4096       |
| block_size | 512       | 512        |
| dropout    | 0.1       | 0.1 (same) |
| params     | ~45–55M   | **~269M**  |

> **Note on layer count.** The original Phase 19A suggestion was 18 layers, but
> at `d_model=1024 / d_ff=4096` that lands at ~244M — just under the 250M floor.
> Bumping to **20 layers** reaches ~269M, comfortably inside the 250M–350M
> target. All other suggested values (d_model, n_heads, d_ff, block_size,
> dropout) are kept as-is.

Checkpoints for this config are written to:

```
13_gpt_pretraining/checkpoints/large_300m/
```

## Expected: slower

This config is ~5–6× the parameters of `large_50m`. Expect:

- Much lower **tokens/sec** for both forward and training passes.
- Smaller usable **micro-batch size** — `large_300m` defaults to
  `batch_size=2` with `gradient_accumulation_steps=16` (effective batch 32) to
  fit in memory while keeping a reasonable optimizer-step token count.
- Larger checkpoints (~1GB+ for full train state including optimizer moments).

Run the benchmark first to get real numbers for your machine.

## ⚠️ Memory warning (Mac / MPS)

A 269M model is memory-hungry on Apple Silicon:

- AdamW keeps **two optimizer moments per parameter**, so optimizer state alone
  is ~2× the model weights (fp32 weights ≈ 1.1GB → ~2.2GB of moments).
- Activations scale with `batch_size × block_size`. On MPS, the unified memory
  is shared with the OS and GPU, so OOM shows up as a hard crash, not a slow
  swap.
- If you hit OOM: keep `batch_size` small (2 or 1), lean on
  `gradient_accumulation_steps`, and let the benchmark's auto-batch fallback
  find a size. Close other GPU/heavy apps before long runs.
- Prefer running real training overnight; expect it to be slow.

## Commands

**Benchmark** (throughput + memory + checkpoint size, no training):

```bash
python 13_gpt_pretraining/benchmark.py --config large_300m
```

**Smoke test** (20 optimizer steps only — proves it trains end-to-end):

```bash
./19A_scale_to_300m/train_smoke.sh
```

which runs:

```bash
python 13_gpt_pretraining/training/trainer.py \
  --config large_300m \
  --corpus 13_gpt_pretraining/data/corpus_chat_only.txt \
  --steps 20 \
  --lr 1e-4
```

## Full training plan (later — not run in this phase)

Only after the benchmark + smoke test look healthy:

1. **Chat-only base** — pretrain `large_300m` on
   `13_gpt_pretraining/data/corpus_chat_only.txt` to establish a solid base.
2. **Short teacher** — a brief teacher/SFT pass (the Phase 18I routing-fixed
   recipe), kept short to avoid the regressions seen at 45M.
3. **18J / 18K eval** — measure routing (18J) and domain score (18K). This is
   the gate: the whole point of scaling is to beat 18% / 21.8%.
4. **Broad SFT — only if stable.** Broad SFT (500) regressed the 45M model, so
   apply it **only if** steps 1–3 are stable and not regressing. If it hurts
   the benchmarks at 300M too, stop and keep the pre-broad-SFT checkpoint.

The guiding rule from the 45M plateau: **if added data regresses 18J/18K, roll
back** rather than pushing through.
