# mini-transformer-from-scratch

A step-by-step learning project that grows from **one weight** to a **tiny transformer**.

Each folder is a self-contained lesson. Read the code top to bottom — comments explain what happens at every step.

**Python:** 3.11  
**Goal:** clarity over performance. No production tricks.

---

## Learning path

| Phase | What you learn | Libraries |
|-------|----------------|-----------|
| [01_linear_model](01_linear_model/) | One parameter `w`, loss, gradient descent | Pure Python |
| [02_neuron_layer](02_neuron_layer/) | Weighted sum, bias, ReLU, Sigmoid | Pure Python |
| [03_tiny_neural_network](03_tiny_neural_network/) | Forward pass, MSE loss, backprop, updates | NumPy |
| [04_embeddings](04_embeddings/) | Tokenizer, token IDs, embedding vectors | NumPy |
| [05_attention](05_attention/) | Q, K, V, attention scores, softmax | NumPy |
| [06_mini_transformer](06_mini_transformer/) | Full transformer block (embed, pos, attn, FFN, residual, norm) | PyTorch |
| [07_mini_language_model](07_mini_language_model/) | Train a tiny next-character predictor | PyTorch |
| [08_word_level_language_model](08_word_level_language_model/) | Train a tiny next-word predictor (better readability) | PyTorch |
| [09_bpe_tokenizer_demo](09_bpe_tokenizer_demo/) | BPE subword tokenization — why real LLMs use it | Pure Python |
| [10_bpe_language_model](10_bpe_language_model/) | Train LM on BPE tokens (production-style granularity) | PyTorch |
| [11_scale_model](11_scale_model/) | How parameter count scales (~500K → 50M) | Pure Python |
| [12_instruction_tuning_demo](12_instruction_tuning_demo/) | Instruction tuning: base LM → simple assistant | PyTorch |
| [13_gpt_pretraining](13_gpt_pretraining/) | GPT pretraining: causal decoder-only LM on raw text (~13M params) | PyTorch |
| [14_dataset_pipeline](14_dataset_pipeline/) | Production-style data pipeline: ingest → clean → dedupe → quality → shards | Pure Python |
| [15_scale_to_50m](15_scale_to_50m/) | Scale GPT pretraining to ~50M on MPS (grad accum, OOM fallback) | PyTorch |
| [16_evaluation_suite](16_evaluation_suite/) | Compare Marshmello-8M vs 45M; detect memorization | PyTorch |
| [17_instruction_dataset](17_instruction_dataset/) | Build instruction JSONL for Marshmello-45M-Instruct | Pure Python |

---

## Setup

```bash
cd mini-transformer-from-scratch
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Phases 01–02 need no extra packages. Phases 03–05 need NumPy. Phases 06–10, 12–13 need PyTorch. Phases 09 and 11 are pure Python.

---

## Run each phase

```bash
# Phase 1 — one weight learns y = 5x
python 01_linear_model/train.py

# Phase 2 — single neuron with activations
python 02_neuron_layer/neuron.py

# Phase 3 — 2-layer network with backprop
python 03_tiny_neural_network/network.py

# Phase 4 — words → IDs → vectors
python 04_embeddings/embeddings.py

# Phase 5 — self-attention on "I love AI"
python 05_attention/attention.py

# Phase 6 — one transformer block (forward pass demo)
python 06_mini_transformer/transformer_block.py

# Phase 7 — train a tiny character-level language model
python 07_mini_language_model/train.py

# Phase 8 — train a tiny word-level language model
python 08_word_level_language_model/train.py

# Phase 9 — BPE tokenizer demo (char vs word vs subword)
python 09_bpe_tokenizer_demo/bpe_demo.py

# Phase 10 — train LM on BPE subword tokens
python 10_bpe_language_model/train.py

# Phase 11 — estimate transformer size at different scales (no training)
python 11_scale_model/estimate_params.py

# Phase 12 — instruction tuning demo (chat format)
python 12_instruction_tuning_demo/train.py

# Phase 13 — GPT pretraining (causal decoder-only, ~13M params)
python 13_gpt_pretraining/data/prepare_corpus.py
python 13_gpt_pretraining/tokenizer/train_bpe.py
python 13_gpt_pretraining/training/trainer.py
python 13_gpt_pretraining/benchmark.py
python 13_gpt_pretraining/generate.py --prompt "To be" --greedy

# Phase 14 — dataset pipeline (ingest → clean → dedupe → quality → shards)
python 14_dataset_pipeline/run_pipeline.py
python 14_dataset_pipeline/export_corpus.py

# Phase 15 — scale GPT to ~50M on MPS
python 15_scale_to_50m/benchmark.py
python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 300

# Phase 16 — compare 8M vs 45M, detect memorization
python 16_evaluation_suite/evaluate.py

# Phase 17 — prepare instruction tuning JSONL
python 17_instruction_dataset/import_hf_datasets.py --max-examples 50000
python 17_instruction_dataset/process_instructions.py
```

---

## Character-level vs word-level (phases 07 vs 08)

Both phases train the same idea: predict the next token with a tiny transformer. The difference is **what a token is**.

| | Phase 07 (character) | Phase 08 (word) |
|--|----------------------|-----------------|
| Token | One letter or punctuation mark | One word, punctuation mark, or newline |
| Vocab size | ~50 characters | ~500–800 words (depends on corpus) |
| When it overfits | Repeats **letters**: `rrrr`, `eeee` | Repeats **words/phrases**: `the the`, `to be to be` |
| Readability | Hard to read (keyboard mash) | More readable (broken but word-like sentences) |
| Data needed | Less (alphabet is tiny) | More (vocabulary is larger) |

Phase 08 uses a **regex tokenizer** (`words`, `punctuation`, `\n`) and a **~10,000-word** embedded corpus. Generation uses **temperature**, **top-k**, and a **repetition penalty** on recent tokens — same decoding ideas as phase 07, but at word granularity.

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
language model (char) →  predict next character, one token at a time
     ↓
language model (word) →  predict next word — more readable generation
     ↓
BPE tokenizer         →  subwords: no <UNK>, controlled vocab (real LLMs)
     ↓
language model (BPE)  →  readable generation + open vocabulary
     ↓
scale model           →  count params: 500K → 2M → 10M → 50M
     ↓
instruction tuning    →  chat format teaches helpful assistant behavior
     ↓
GPT pretraining       →  causal decoder-only LM on raw text (real GPT objective)
     ↓
dataset pipeline      →  ingest, clean, dedupe, quality, shard (real LLM data engineering)
     ↓
50M GPT scaling       →  grad accumulation, MPS memory, training stack engineering
     ↓
evaluation suite      →  8M vs 45M, memorization metrics, side-by-side outputs
     ↓
instruction dataset   →  instruction/response/domain JSONL for SFT
```

Work through the folders in order. Each phase reuses ideas from the previous one.

---

## Marshmello models

Decoder-only GPT models trained from scratch in this repo.

| Model | Parameters | Hugging Face | Config |
|-------|------------|--------------|--------|
| **Marshmello-8M** | ~8M | [ostah-1010/Marshmello-8M](https://huggingface.co/ostah-1010/Marshmello-8M) | `default` |
| **Marshmello-45M** | ~45M | [ostah-1010/Marshmello](https://huggingface.co/ostah-1010/Marshmello) | `large_50m` |

**GitHub:** https://github.com/mohmmedwee/Marshmello

### Download weights from Hugging Face

```bash
git clone https://github.com/mohmmedwee/Marshmello.git
cd Marshmello
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python 13_gpt_pretraining/hub/download_from_hub.py --repo-id ostah-1010/Marshmello
python 13_gpt_pretraining/generate.py --config large_50m --prompt "Database systems"
```

### Push updated weights to Hub

```bash
hf auth login --token hf_xxx   # write token for ostah-1010
python 13_gpt_pretraining/hub/push_to_hub.py --config large_50m --repo-id ostah-1010/Marshmello
```