# Phase 15 — Scale to ~50M GPT

Phase 13 proved the stack on **~8M parameters**. Phase 15 scales the **same codebase** to **~50M** on Apple Silicon MPS — the point where training engineering matters as much as architecture.

```text
Phase 14 dataset pipeline
        ↓
Phase 13 BPE + corpus
        ↓
large_50m config (~45–55M params, vocab-dependent)
        ↓
Gradient accumulation + OOM batch fallback
        ↓
Longer training + checkpoints
```

---

## Architecture: `large_50m`

| Hyperparameter | Value |
|----------------|-------|
| `d_model` | 768 |
| `num_layers` | 6 |
| `num_heads` | 12 |
| `d_ff` | 3072 |
| `block_size` | 512 |
| `vocab_size` | from trained BPE (not hard-coded 8000) |
| `batch_size` | 4 (auto-reduced on MPS OOM) |
| `gradient_accumulation_steps` | 8 |
| **Effective batch** | 32 sequences × 512 tokens = **16,384 tokens/step** |

With a typical BPE vocab (~1100 tokens), parameter count is **~45M**. At the 8000 vocab target it reaches **~55M**. Both are in the “~50M class” used for scaling studies.

---

## Quick start

From `mini-transformer-from-scratch/` with venv active:

```bash
# 1. Ensure corpus + BPE exist (Phase 13 / 14)
python 14_dataset_pipeline/run_pipeline.py
python 14_dataset_pipeline/export_corpus.py
python 13_gpt_pretraining/tokenizer/train_bpe.py

# 2. Benchmark 50M model (throughput + memory)
python 15_scale_to_50m/benchmark.py
python 13_gpt_pretraining/benchmark.py --config large_50m

# 3. Compare with 8M baseline
python 13_gpt_pretraining/benchmark.py --config default

# 4. Train 50M (300-step smoke test)
python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 300

# Full quick script
bash 15_scale_to_50m/train_quick.sh
```

Checkpoints are saved under `13_gpt_pretraining/checkpoints/large_50m/`.

---

## What Phase 15 adds

### 1. Config registry

```python
# 13_gpt_pretraining/config.py
CONFIGS = {"default": ..., "large_50m": ...}
```

```bash
python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 300
```

### 2. Gradient accumulation

Large models do not fit a big batch on MPS. Instead:

```text
micro_batch = 4
accum_steps = 8
effective_batch = 32
```

Each optimizer step processes `4 × 8 × 512 = 16,384` tokens — same learning signal as batch 32, lower peak memory.

### 3. Automatic batch-size fallback

Before training, the trainer probes forward+backward on MPS. If memory runs out, it halves `batch_size` until a step succeeds:

```text
OOM at batch_size=4 → retrying with batch_size=2
```

This also applies mid-run if memory pressure spikes.

### 4. 50M benchmark

Same metrics as Phase 13 baseline:

```text
Forward tokens/sec
Training tokens/sec
Generation tokens/sec
Peak memory
Checkpoint size
Parameters (actual vs analytic)
```

---

## 8M vs 50M — what to expect

| | ~8M (`default`) | ~50M (`large_50m`) |
|--|-----------------|---------------------|
| Params | ~8M | ~45–55M |
| Peak memory | ~350 MB | ~2–6 GB (batch-dependent) |
| Training tok/s | ~100K+ | ~10–30K (rough) |
| Generation tok/s | ~20–35 | ~5–15 |
| Checkpoint | ~90 MB | ~500 MB+ |

Run both benchmarks on the same machine and save the numbers — Phase 16 will use them for evaluation.

---

## Why training 2B from scratch is not practical on a single Mac

Rough orders of magnitude for **full pretraining** (not inference):

| Model | Parameters | Weight memory (fp32) | Adam states | Activations (training) |
|-------|------------|----------------------|-------------|------------------------|
| 50M | 5×10⁷ | ~200 MB | ~400 MB | GB-scale |
| 1B | 10⁹ | ~4 GB | ~8 GB | 10–100+ GB |
| 2B | 2×10⁹ | ~8 GB | ~16 GB | often **> unified memory** |

A MacBook M4 Max has **36–128 GB unified memory** shared by CPU, GPU, and OS. That is enough to **train** a carefully tuned 50M model. It is **not** enough to pretrain 2B with standard Adam + activations + dataloader overhead.

Training 2B from scratch also needs:

- **Massive data** — billions of tokens, not a demo corpus
- **Weeks of GPU time** — thousands of device-hours on clusters
- **Distributed training** — data parallel across many GPUs
- **Mixed precision + checkpointing** — even then, single-node 2B full fine-tune is rare

So: **50M on MPS = educational scaling. 2B from scratch on one Mac = not realistic.**

---

## Why LoRA / QLoRA is the right path for 1B–2B local experiments

Instead of training all 2B parameters, modern local workflows:

```text
Download pretrained 1B–2B base model (already trained on big data)
        ↓
Freeze most weights
        ↓
Train small adapter matrices (LoRA) or quantized adapters (QLoRA)
        ↓
Fine-tune on your task with GB-scale memory, not TB-scale
```

| Approach | Trainable params | Memory | Use case |
|----------|------------------|--------|----------|
| Full pretrain 2B | 2B | Cluster | Build foundation model |
| Full fine-tune 2B | 2B | Multi-GPU server | Rare on laptop |
| LoRA 2B | ~1–50M | Single Mac / GPU | Chat, RAG, domain adapt |
| QLoRA 2B | ~1–50M | **4-bit base + adapters** | Same, tighter memory |

**LoRA** adds low-rank matrices to attention/FFN layers — you learn a delta, not the full weight tensor.

**QLoRA** stores the frozen base in 4-bit quantization and trains adapters in higher precision — the standard path for **local 7B–70B experimentation** on one GPU.

Phase 15 teaches **how scaling feels at 50M**. Phase 17 (chat fine-tuning) will move toward **adapter-style** training on larger bases — the production pattern for 1B–2B on a Mac.

---

## Folder layout

```text
15_scale_to_50m/
├── README.md           # this file
├── benchmark.py        # wrapper → 13_gpt_pretraining/benchmark.py --config large_50m
└── train_quick.sh      # benchmark + 300-step smoke train
```

Implementation lives in `13_gpt_pretraining/` (shared model, trainer, config) so Phase 13 and 15 stay one codebase.

---

## Success criteria

```text
✓ large_50m config (~50M class)
✓ Gradient accumulation
✓ MPS OOM batch fallback
✓ 50M benchmark (forward / train / generate / memory / checkpoint)
✓ python trainer.py --config large_50m --steps 300
✓ Checkpoints under checkpoints/large_50m/
✓ Documented: why 2B scratch fails on Mac, why LoRA/QLoRA wins locally
```
