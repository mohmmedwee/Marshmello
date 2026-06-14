# Phase 13 — GPT Pretraining

Build a **decoder-only GPT** from scratch and pretrain it on raw text with **next-token prediction** — the same objective used to train GPT-2, GPT-3, and LLaMA **before** instruction tuning (Phase 12).

Phase 12 taught: `Question → Answer` (SFT).  
Phase 13 teaches: `The cat sat on the ___` → predict `mat`.

---

## What this phase adds

| Piece | Phase 10 (old LM) | Phase 13 (GPT) |
|-------|-------------------|----------------|
| Attention | Bidirectional | **Causal** (no future peeking) |
| Positional | Fixed sin/cos | **Learned** embeddings |
| Objective | Next token | Next token (same) |
| Format | Raw text | Raw text (no chat tags) |
| Scale | ~200K params | **~13M params** |
| Checkpoints | Best epoch only | Every 1000 steps + resume |
| Device | CPU | **MPS** (Apple Silicon) |

---

## Folder layout

```
13_gpt_pretraining/
├── data/
│   ├── corpus.txt          # raw training text
│   └── prepare_corpus.py   # build corpus from phase 08 passages
├── tokenizer/
│   ├── train_bpe.py        # train BPE (~8000 vocab)
│   ├── bpe_io.py           # save/load tokenizer.json
│   └── tokenizer.json      # created by train_bpe.py
├── model/
│   ├── attention.py        # causal mask
│   ├── transformer_block.py
│   └── gpt.py              # decoder-only LM
├── training/
│   ├── dataset.py          # random token windows
│   └── trainer.py          # pretraining loop
├── checkpoints/            # step_001000.pt, latest.pt, …
├── generate.py             # sampling CLI
├── benchmark.py            # throughput + memory baseline
├── config.py               # V1 hyperparameters
└── README.md
```

---

## Quick start

From `mini-transformer-from-scratch/` with venv active:

```bash
# 1. Prepare raw text corpus (mixed = literature + technical)
python 13_gpt_pretraining/data/prepare_corpus.py --mix mixed --force

# Other mixes: classic | tech | mixed
python 13_gpt_pretraining/data/prepare_corpus.py --mix tech --force

# 2. Train BPE tokenizer (vocab ≈ 8000 target)
python 13_gpt_pretraining/tokenizer/train_bpe.py

# 3. Pretrain GPT (~10M params, MPS if available)
python 13_gpt_pretraining/training/trainer.py

# Smoke test pipeline (corpus + BPE + 300 training steps):
bash 13_gpt_pretraining/train_quick.sh

# Resume training
python 13_gpt_pretraining/training/trainer.py --resume 13_gpt_pretraining/checkpoints/latest.pt

# 4. Generate text
python 13_gpt_pretraining/generate.py \
  --prompt "Artificial intelligence" \
  --temperature 0.8 \
  --top-k 40 \
  --max-new-tokens 120

python 13_gpt_pretraining/generate.py --test-prompts
python 13_gpt_pretraining/generate.py --prompt "To be" --greedy
python 13_gpt_pretraining/generate.py --prompt "Database systems" --domain-hint databases
python 13_gpt_pretraining/generate.py --prompt "Database" --style paragraph --stop-on-sentence-end

# 5. Benchmark throughput + memory (baseline before Phase 15 scaling)
python 13_gpt_pretraining/benchmark.py
python 13_gpt_pretraining/benchmark.py --steps 30 --warmup 10
```

---

## Config V1 (M4 Max friendly)

| Hyperparameter | Value |
|----------------|-------|
| `vocab_size` | ~8000 target (actual ~3700 on this corpus) |
| `d_model` | 384 |
| `num_layers` | 4 |
| `num_heads` | 6 |
| `d_ff` | 1536 |
| `block_size` | 256 |
| `dropout` | 0.1 |
| `batch_size` | 16 |
| `learning_rate` | 3e-4 |
| **Parameters** | **~13.3M** |

Edit `config.py` to change defaults.

---

## Architecture

```
token IDs
  → token embedding
  → learned positional embedding
  → 4 × causal transformer blocks
      (LayerNorm → Multi-Head Causal Attention → residual)
      (LayerNorm → FFN GELU → residual)
  → final LayerNorm
  → LM head → vocab logits
```

**Loss:** standard cross-entropy on every next-token target (no SFT mask).

**Causal mask:** upper triangle set to `-inf` so position `t` cannot attend to `t+1, t+2, …`

---

## Dataset windowing

Given token stream `[10, 11, 12, 13, 14, 15]`:

```
input  x = [10, 11, 12, 13, 14]
target y = [11, 12, 13, 14, 15]
```

Random windows are sampled from the BPE-encoded corpus with a 90/10 train/val split.

---

## Training metrics

Each log line prints:

- step, train loss, learning rate, tokens/sec
- periodic eval: train loss, val loss, avg tokens/sec, estimated memory
- checkpoints: file size in MB

Checkpoints saved every **1000 steps** to `checkpoints/step_XXXXXX.pt` and `checkpoints/latest.pt`.

---

## Generation modes

| Flag | Behavior |
|------|----------|
| `--max-new-tokens 120` | Default generation length |
| `--min-new-tokens 20` | Used with `--stop-on-sentence-end` |
| `--greedy` | argmax each step |
| `--temperature 0.8` | scale logits before softmax |
| `--top-k 40` | sample from top 40 tokens |
| `--top-p 0.9` | nucleus sampling |
| `--repetition-penalty 1.15` | down-weight recently sampled tokens |
| `--presence-penalty 0.6` | down-weight tokens already in context |
| `--stop-sequence "=== Topic:"` | truncate when header leaks (default) |
| `--no-stop-sequence` | disable stop-sequence guard |
| `--style paragraph` | prepend *Write one coherent paragraph about* |
| `--domain-hint databases` | prepend *This text is about databases and data systems.* |
| `--stop-on-sentence-end` | stop after first complete sentence (after min_new_tokens) |
| `--stop-on-word-boundary` | legacy only — **not recommended** |
| `--no-eos-stop` | do not stop on tokenizer EOS (if vocab has one) |
| `--min-sentences` / `--max-sentences` | post-process to complete sentences |

**BPE `</w>` is NOT EOS.** It marks end-of-word inside a subword token. Generation no longer stops on `</w>` by default.

Each run prints **why it stopped** — usually `max_new_tokens` or `sentence_end`.

Run decode spacing tests:

```bash
python -m unittest 13_gpt_pretraining.tests.test_bpe_decode -v
```

---

## Training data mixes

`prepare_corpus.py --mix` controls what the model learns to imitate:

| Mix | Content |
|-----|---------|
| `classic` | Phase 08 Shakespeare-adjacent prose only |
| `tech` | AI, ML, databases, software engineering only (no Shakespeare) |
| `mixed` | ~50/50 both domains (default) |

Before writing `corpus.txt`, the script:

- repeats each source paragraph `--repeat-per-domain` times (default **20**)
- **shuffles** paragraphs (reduces block memorization)
- removes section markers and normalizes whitespace
- optional `--dedupe-lines` removes duplicate paragraphs

Prints **domain distribution** and a **data quality report** (unique lines, duplicate ratio, top repeats).

```bash
python 13_gpt_pretraining/data/prepare_corpus.py --mix tech --repeat-per-domain 20 --force
python 13_gpt_pretraining/data/prepare_corpus.py --mix mixed --dedupe-lines --force
```

**If you train on mixed domains, the model will mix styles** — e.g. a database prompt may jump into AI prose because paragraphs from different topics are interleaved without hard boundaries.

**Very low val loss often means memorization**, not intelligence — especially when the corpus repeats the same paragraphs many times. Check the duplicate ratio in the quality report.

**Important:** model output style follows the **training data distribution**.

---

## Success criteria

Before Phase 14:

- [x] BPE tokenizer trained (~8000 vocab)
- [x] GPT decoder-only model (~10M+ params)
- [x] Causal self-attention
- [x] Train + val loss logged
- [x] Checkpoint save/load/resume
- [x] Generation CLI (greedy, temperature, top-k, top-p)
- [x] MPS support on Apple Silicon

**Note:** Coherent long-form text needs more data and training steps than this educational corpus provides. The goal is to understand the **GPT pretraining pipeline**, not to match ChatGPT quality.

---

## Data source

`data/corpus.txt` is built locally — not downloaded. Choose a mix:

- **classic** — phase 08 embedded prose (Shakespeare-adjacent)
- **tech** — AI, machine learning, databases, software engineering (no classic text)
- **mixed** — both (~50/50 words)

In production you would use Wikipedia, books, code, etc. Same pipeline, bigger data.

**Mixed training → mixed generation.** If the corpus contains both literature and technical prose, the model learns both styles and may switch between them in one completion.

**The model speaks like its corpus.** Literature-heavy data → literary replies. Technical-heavy data → technical replies.

**Topic headers are stripped from training text.** Older checkpoints may still emit header-like fragments; use `--stop-sequence` at generation time if needed.

---

## Benchmark (`benchmark.py`)

Before scaling to Phase 15 (~50M), measure your **training stack** on the current model:

```bash
python 13_gpt_pretraining/benchmark.py
```

Example output on M4 Max (MPS):

```text
Model:                 ~8M
Parameters:            8,045,568
Analytic estimate:     8,045,568
Estimate error:        0.000%

Parameter breakdown (actual vs analytic)
Token embeddings       424,320    424,320    0.00%
Attention blocks     2,365,440  2,365,440    0.00%
FFN blocks           4,726,272  4,726,272    0.00%
LM head                424,320    424,320    0.00%
TOTAL                8,045,568  8,045,568    0.00%
Weight tying: no | LM head bias: no
```

The estimate uses the **trained BPE vocab size** (e.g. 1105), not the 8000 target. It also matches Phase 13 architecture: fused QKV projection, no LM head bias, untied embeddings.

| Metric | What it measures |
|--------|------------------|
| Forward tokens/sec | Inference-only throughput (batch × block_size per step) |
| Training tokens/sec | Forward + backward + AdamW step |
| Generation tokens/sec | Autoregressive greedy decode (1 token/step) |
| Peak memory | Device allocator peak during benchmark |
| Checkpoint size | Full checkpoint (model + optimizer) vs weights only |

Save this baseline — Phase 15 will compare **10M vs 50M** on the same corpus and device.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Very low val loss (~0.1) | Repetitive corpus → memorization; check duplicate ratio |
| Short one-word answers | Old `</w>` stop behavior — update `generate.py` (default no longer stops on `</w>`) |
| Database → AI topic jump | Mixed shuffled corpus; use `--mix tech` or `--domain-hint databases` |
| `KeyError` on punctuation | Prompt char missing from BPE vocab — use `encode_prompt` safe path (built in) |

---

## Relation to other phases

```
Phase 09  → BPE tokenizer
Phase 10  → LM on BPE (bidirectional — educational baseline)
Phase 12  → Instruction tuning (chat format, masked loss)
Phase 13  → Real GPT pretraining (causal, raw text, larger model)
```