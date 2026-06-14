# Marshmello

**Train a GPT from one weight to a ~45M decoder-only language model — entirely in readable Python.**

This repository is both a **hands-on transformer course** (Phases 01–18) and the home of the **Marshmello** model family: small GPT models built from scratch, trained on Apple Silicon (MPS), and published on Hugging Face.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Models-Marshmello-yellow)](https://huggingface.co/ostah-1010/Marshmello)

---

## Table of contents

- [Marshmello models](#marshmello-models)
- [Quick start](#quick-start)
- [What you will learn](#what-you-will-learn)
- [Learning path](#learning-path)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Full training pipeline](#full-training-pipeline)
- [How the pieces connect](#how-the-pieces-connect)
- [Requirements](#requirements)
- [Limitations](#limitations)
- [License](#license)

---

## Marshmello models

Two decoder-only GPT checkpoints trained in this repo. **Weights are not stored in Git** (~2 GB); download them from Hugging Face after cloning.

| Model | Params | Context | Hugging Face | Config key |
|-------|--------|---------|--------------|------------|
| **Marshmello-8M** | ~8M | 256 tokens | [ostah-1010/Marshmello-8M](https://huggingface.co/ostah-1010/Marshmello-8M) | `default` |
| **Marshmello-45M** | ~46M | 512 tokens | [ostah-1010/Marshmello](https://huggingface.co/ostah-1010/Marshmello) | `large_50m` |

**Marshmello-45M (latest base)** was pretrained on a ~1M-word local corpus (Phase 18A) covering AI/ML, databases, software engineering, cybersecurity, and Python APIs. It is the recommended base for instruction fine-tuning (Phase 18B).

Both models share the same BPE tokenizer (`13_gpt_pretraining/tokenizer/tokenizer.json`) and a custom PyTorch GPT implementation — not `transformers` AutoModel.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/mohmmedwee/Marshmello.git
cd Marshmello
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download weights from Hugging Face

```bash
# Marshmello-45M (recommended)
python 13_gpt_pretraining/hub/download_from_hub.py --repo-id ostah-1010/Marshmello

# Marshmello-8M (smaller, faster)
python 13_gpt_pretraining/hub/download_from_hub.py --repo-id ostah-1010/Marshmello-8M --config default
```

### 3. Generate text

```bash
python 13_gpt_pretraining/generate.py \
  --config large_50m \
  --prompt "Database systems" \
  --temperature 0.7 \
  --top-k 30 \
  --max-new-tokens 100
```

### 4. Instruction-tuned chat (Phase 18B)

After running SFT (see [Full training pipeline](#full-training-pipeline)):

```bash
python 18B_marshmello_instruct/chat.py --prompt "Explain database indexes"
```

### 5. Start from lesson 1 (no GPU required)

```bash
python 01_linear_model/train.py
```

---

## What you will learn

Each phase folder is a self-contained lesson. Read the code top to bottom — comments explain every step.

| Topic | Phases |
|-------|--------|
| Gradients and backprop | 01–03 |
| Embeddings and attention | 04–05 |
| Transformer blocks | 06 |
| Language modeling (char → word → BPE) | 07–10 |
| Parameter scaling | 11 |
| Instruction tuning | 12, 17–18B |
| GPT pretraining (causal LM) | 13, 15, 18A |
| Data engineering | 14 |
| Evaluation and memorization | 16 |

**Design principle:** clarity over performance. No hidden abstractions, no production shortcuts — just the ideas behind modern LLMs.

---

## Learning path

| Phase | What you learn | Libraries |
|-------|----------------|-----------|
| [01_linear_model](01_linear_model/) | One parameter `w`, loss, gradient descent | Pure Python |
| [02_neuron_layer](02_neuron_layer/) | Weighted sum, bias, ReLU, Sigmoid | Pure Python |
| [03_tiny_neural_network](03_tiny_neural_network/) | Forward pass, MSE loss, backprop | NumPy |
| [04_embeddings](04_embeddings/) | Tokenizer, token IDs, embedding vectors | NumPy |
| [05_attention](05_attention/) | Q, K, V, attention scores, softmax | NumPy |
| [06_mini_transformer](06_mini_transformer/) | Transformer block (embed, pos, attn, FFN, residual, norm) | PyTorch |
| [07_mini_language_model](07_mini_language_model/) | Next-character prediction | PyTorch |
| [08_word_level_language_model](08_word_level_language_model/) | Next-word prediction | PyTorch |
| [09_bpe_tokenizer_demo](09_bpe_tokenizer_demo/) | BPE subword tokenization | Pure Python |
| [10_bpe_language_model](10_bpe_language_model/) | LM on BPE tokens | PyTorch |
| [11_scale_model](11_scale_model/) | Parameter count scaling (~500K → 50M) | Pure Python |
| [12_instruction_tuning_demo](12_instruction_tuning_demo/) | Base LM → simple assistant | PyTorch |
| [13_gpt_pretraining](13_gpt_pretraining/) | Causal decoder-only GPT pretraining | PyTorch |
| [14_dataset_pipeline](14_dataset_pipeline/) | Ingest → clean → dedupe → quality → shards | Pure Python |
| [15_scale_to_50m](15_scale_to_50m/) | ~50M training on MPS (grad accum, OOM fallback) | PyTorch |
| [16_evaluation_suite](16_evaluation_suite/) | Compare 8M vs 45M; detect memorization | PyTorch |
| [17_instruction_dataset](17_instruction_dataset/) | Build instruction JSONL for SFT | Pure Python |
| [18A_large_pretraining_corpus](18A_large_pretraining_corpus/) | Larger corpus for Marshmello-45M-Base-v2 | Pure Python |
| [18B_marshmello_instruct](18B_marshmello_instruct/) | Fine-tune base → Marshmello-45M-Instruct | PyTorch |

Work through the folders in order. Each phase reuses ideas from the previous one.

---

## Architecture

Marshmello-45M (`large_50m` config):

| Component | Value |
|-----------|-------|
| Type | Decoder-only GPT (causal self-attention) |
| Layers | 6 |
| Hidden size (`d_model`) | 768 |
| Attention heads | 12 |
| FFN dimension | 3072 |
| Context length | 512 tokens |
| Positional embeddings | Learned |
| Tokenizer | BPE (~1,100 vocab) |
| LM head | Tied to token embeddings |

Marshmello-8M (`default` config): 4 layers, `d_model=384`, context 256 — same codebase, smaller scale.

---

## Project layout

```text
Marshmello/
├── 01_linear_model/ … 12_instruction_tuning_demo/   # Foundations → SFT demo
├── 13_gpt_pretraining/                              # GPT core (model, trainer, generate)
│   ├── model/                                       # Causal attention + GPT
│   ├── training/trainer.py                          # Pretraining loop
│   ├── generate.py                                  # Sampling CLI
│   ├── hub/                                         # Hugging Face upload/download
│   └── checkpoints/                                 # Weights (gitignored — use Hub)
├── 14_dataset_pipeline/                             # Data engineering
├── 15_scale_to_50m/                                 # 50M scaling notes + scripts
├── 16_evaluation_suite/                             # 8M vs 45M eval + memorization
├── 17_instruction_dataset/                          # Alpaca-style JSONL builder
├── 18A_large_pretraining_corpus/                    # ~1M-word base corpus
├── 18B_marshmello_instruct/                         # SFT + chat CLI
└── requirements.txt
```

---

## Full training pipeline

End-to-end path from raw text to an instruct model:

```bash
# Phase 18A — build larger base corpus (~1M words)
python 18A_large_pretraining_corpus/build_corpus.py --target-words 1000000

# Retrain BPE on expanded corpus
python 13_gpt_pretraining/tokenizer/train_bpe.py

# Pretrain Marshmello-45M base
python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 1000

# Phase 17 — prepare instruction data (requires `datasets` + network)
python 17_instruction_dataset/import_hf_datasets.py --max-examples 50000
python 17_instruction_dataset/process_instructions.py

# Phase 18B — instruction fine-tuning
python 18B_marshmello_instruct/train_instruct.py \
  --config large_50m \
  --base-checkpoint 13_gpt_pretraining/checkpoints/large_50m/step_001000.pt \
  --steps 1000

# Chat with the instruct checkpoint
python 18B_marshmello_instruct/chat.py --prompt "What is a database index?"
```

**Smoke test** (1 step, skip eval/save — useful on first run):

```bash
python 18B_marshmello_instruct/train_instruct.py \
  --config large_50m \
  --base-checkpoint 13_gpt_pretraining/checkpoints/large_50m/step_001000.pt \
  --steps 1 --no-eval --no-save
```

**Evaluate scaling vs memorization:**

```bash
python 16_evaluation_suite/evaluate.py
```

**Push updated weights to Hugging Face** (requires write token):

```bash
hf auth login
python 13_gpt_pretraining/hub/push_to_hub.py \
  --config large_50m \
  --repo-id ostah-1010/Marshmello \
  --checkpoint 13_gpt_pretraining/checkpoints/large_50m/step_001000.pt
```

---

## How the pieces connect

```
1 weight (w)          →  learns a line: y = w·x
     ↓
1 neuron              →  many inputs + bias + activation
     ↓
tiny network          →  layers + loss + gradients
     ↓
embeddings            →  discrete tokens become continuous vectors
     ↓
attention             →  tokens look at each other (Q, K, V)
     ↓
transformer block     →  attention + FFN + residuals + layer norm
     ↓
language model        →  predict next token (char → word → BPE)
     ↓
scale model           →  count params: 500K → 8M → 45M
     ↓
GPT pretraining       →  causal decoder-only LM on raw text
     ↓
dataset pipeline      →  ingest, clean, dedupe, quality, shard
     ↓
evaluation suite      →  8M vs 45M, memorization metrics
     ↓
instruction dataset   →  instruction/response JSONL for SFT
     ↓
larger base corpus    →  Marshmello-45M-Base-v2
     ↓
instruct fine-tuning  →  Marshmello-45M-Instruct
```

---

## Character-level vs word-level (Phases 07 vs 08)

Both phases train the same idea: predict the next token with a tiny transformer. The difference is **what a token is**.

| | Phase 07 (character) | Phase 08 (word) |
|--|----------------------|-----------------|
| Token | One letter or punctuation mark | One word, punctuation mark, or newline |
| Vocab size | ~50 characters | ~500–800 words |
| Failure mode | Letter loops (`rrrr`) | Word loops (`the the`) |
| Readability | Hard to read | More readable broken sentences |

Phase 08 uses a regex tokenizer and decoding with temperature, top-k, and repetition penalty — the same sampling ideas used in later phases at BPE granularity.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Python | 3.11+ recommended |
| PyTorch | Phases 06–18; MPS on Apple Silicon, CUDA optional |
| NumPy | Phases 03–05 |
| `huggingface_hub` | Download/upload Marshmello weights |
| `datasets` | Phase 17 HF dataset import only |
| Disk | ~3 GB free for checkpoints + venv |
| RAM | 16 GB+ comfortable for 45M training on MPS |

Phases 01–02 need no extra packages. Phases 09 and 11 are pure Python.

---

## Limitations

Marshmello is an **educational model**, not a production LLM.

- Trained on a **small local corpus**, not web-scale data
- Outputs may **memorize** training paragraphs (see Phase 16 evaluation)
- BPE vocab is limited — some punctuation in instruction data is stripped during encoding
- Custom PyTorch implementation — not compatible with `transformers` pipelines out of the box
- Not safety-aligned or RLHF-tuned

For learning how transformers work, this is a feature. For production chat, use larger open models.

---

## License

Model weights and code are published under **Apache 2.0** on Hugging Face. See individual model cards for details.

---

## Links

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/mohmmedwee/Marshmello |
| Marshmello-45M | https://huggingface.co/ostah-1010/Marshmello |
| Marshmello-8M | https://huggingface.co/ostah-1010/Marshmello-8M |
| Hub scripts | [13_gpt_pretraining/hub/](13_gpt_pretraining/hub/) |
