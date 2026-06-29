---
license: apache-2.0
language:
- en
library_name: pytorch
tags:
- language-model
- educational
- decoder-only-transformer
- from-scratch
- marshmello
pipeline_tag: text-generation
---

# Marshmello

Marshmello is an educational open-source language model family built from scratch in PyTorch. The project is designed to show the full path from basic neural network components to GPT pretraining, instruction tuning, internal evaluation, and scaling experiments.

This model card is a draft for Hugging Face publication. It should be updated with the exact checkpoint name, commit, and files before publishing.

## Model Description

| Field | Value |
| ----- | ----- |
| Model family | Marshmello |
| Model types | Marshmello-8M, Marshmello-55M, Marshmello-300M smoke/benchmark |
| Architecture | Decoder-only GPT |
| Framework | Custom PyTorch implementation |
| Primary language | English |
| License | Apache 2.0 |
| Production status | Not production-critical |

## Intended Use

Marshmello is intended for:

- Learning how transformer language models are built.
- Studying BPE tokenization, decoder-only GPT pretraining, SFT, and evaluation.
- Comparing small educational checkpoints with internal benchmarks.
- Demonstrating honest model cards and dataset cards for small open models.

Marshmello is not intended for:

- Production-critical decisions.
- Medical, legal, financial, or safety-critical advice.
- High-accuracy factual answering.
- Deployment as a general-purpose assistant without additional safeguards.

## Architecture

| Component | Description |
| --------- | ----------- |
| Model | Decoder-only GPT |
| Attention | Causal self-attention |
| Positional embeddings | Learned |
| Objective | Next-token prediction for pretraining; masked assistant response loss for SFT |
| Implementation | Custom PyTorch modules in `13_gpt_pretraining/model/` |
| Hub compatibility | Not a `transformers` AutoModel by default |

Current family configs:

| Model | Parameters | Layers | Hidden size | Heads | Context |
| ----- | ---------: | -----: | ----------: | ----: | ------: |
| Marshmello-8M | 8M | 4 | 384 | 6 | 256 |
| Marshmello-55M | 55M | 6 | 768 | 12 | 512 |
| Marshmello-300M | 268.8M | 20 | 1024 | 16 | 512 |

## Tokenizer

Marshmello uses a project BPE tokenizer trained inside the repository. Tokenizer experiments include:

- Default BPE tokenizer for existing 8M and 55M checkpoints.
- Tokenizer v2 experiments with improved punctuation and chat marker coverage.
- 8000-token vocabulary target for the 300M Phase 19A config.

Changing tokenizer files changes token IDs and can make older checkpoints incompatible.

## Training Pipeline

The training path is educational and phase-based:

1. GPT pretraining on local raw text corpora.
2. Larger technical corpus construction.
3. Chat-only or mixed chat-boundary adaptation.
4. Teacher SFT with short direct answers.
5. Routing-focused experiments.
6. Internal evaluation with 18J and 18K.
7. 300M smoke/benchmark scaling experiments.

## Evaluation Results

Current honest internal results for the Marshmello-55M line:

| Benchmark | Result |
| --------- | -----: |
| 18J Core Routing | 18% |
| 18K General Domain Score | 22.5% |
| 18K Hallucination | 64.2% |

These are educational internal benchmarks for comparing Marshmello checkpoints, not commercial LLM benchmarks.

## Limitations

- Small educational model family, not a production LLM.
- Trained on limited local and curated data.
- Can hallucinate, repeat, or answer with incorrect domain routing.
- Limited factual coverage.
- Primarily English.
- No RLHF or formal safety tuning.
- Not compatible with standard `transformers` text-generation loading unless an adapter/export path is added.

## Ethical Limitations

Marshmello can generate plausible but incorrect text. It may reproduce limitations from its small training corpus and generated/curated SFT data. Users should not rely on it for decisions that affect health, law, finance, security, or personal safety.

The model is provided for education and experimentation. It should be evaluated, monitored, and constrained before any public interactive deployment.

## Not For Production-Critical Use

Do not use Marshmello for production-critical workflows. The model family is meant for learning and research into small-model training behavior.
