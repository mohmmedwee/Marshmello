# Marshmello

An educational open-source language model family built from scratch.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-model-yellow.svg)](https://huggingface.co/ostah-1010/Marshmello)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-dataset-orange.svg)](https://huggingface.co/datasets/ostah-1010/Marshmello-SFT)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)

Marshmello is a from-scratch transformer learning project and a small open model family. The repository walks from a single trainable weight to decoder-only GPT pretraining, instruction tuning, evaluation, and early 300M-scale experiments.

The code favors readable PyTorch over framework magic. The models are useful for education, benchmarking project checkpoints, and understanding how language model training pieces connect. They are not production LLMs.

## Model Family

| Model           | Parameters | Status                    |
| --------------- | ---------: | ------------------------- |
| Marshmello-8M   |         8M | Stable                    |
| Marshmello-55M  |        55M | Stable baseline           |
| Marshmello-300M |     268.8M | Phase 19A smoke/benchmark |

More detail: [MODEL_FAMILY.md](MODEL_FAMILY.md)

## Benchmarks

Current honest internal results for the Marshmello-55M line:

| Benchmark                | Marshmello-55M |
| ------------------------ | -------------: |
| 18J Core Routing         |            18% |
| 18K General Domain Score |          22.5% |
| 18K Hallucination        |          64.2% |

These are educational internal benchmarks for comparing Marshmello checkpoints, not commercial LLM benchmarks.

Evaluation details: [EVALUATION.md](EVALUATION.md)

## Project Timeline

| Phase | Milestone |
| ----- | --------- |
| Phase 13 | GPT pretraining |
| Phase 18E | Teacher SFT |
| Phase 18H | Chat-only adaptation |
| Phase 18I | Routing experiments |
| Phase 18J | Core routing benchmark |
| Phase 18K | General assistant benchmark |
| Phase 19A | 300M scaling |

Milestone history: [CHANGELOG.md](CHANGELOG.md)

## Quick Start

```bash
git clone https://github.com/mohmmedwee/Marshmello.git
cd Marshmello
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download published weights:

```bash
# Marshmello-55M baseline
python 13_gpt_pretraining/hub/download_from_hub.py \
  --repo-id ostah-1010/Marshmello

# Marshmello-8M small baseline
python 13_gpt_pretraining/hub/download_from_hub.py \
  --repo-id ostah-1010/Marshmello-8M \
  --config default
```

Generate with the 55M-class config:

```bash
python 13_gpt_pretraining/generate.py \
  --config large_50m \
  --prompt "Database systems" \
  --temperature 0.7 \
  --top-k 30 \
  --max-new-tokens 100
```

Chat with a local instruct/routing checkpoint after Phase 18E/18J training:

```bash
python 18B_marshmello_instruct/chat.py \
  --checkpoint 18B_marshmello_instruct/checkpoints/best_18j_routing.pt \
  --prompt "Explain database indexes" \
  --greedy
```

Start the educational path from the beginning:

```bash
python 01_linear_model/train.py
```

## Recommended Training Path

```text
18A  larger base corpus
  ↓
13   GPT pretraining
  ↓
18H  chat-only adaptation
  ↓
18E  tiny teacher SFT
  ↓
18I  routing experiments
  ↓
18J  core routing benchmark
  ↓
18K  general assistant benchmark
  ↓
19A  300M smoke/benchmark
```

The 55M line currently plateaus on internal evaluation: more broad SFT data improved some behaviors but regressed key 18J/18K gates. Phase 19A exists to test whether extra model capacity is the right next lever.

## Learning Path

| Phase | What you learn | Libraries |
| ----- | -------------- | --------- |
| [01_linear_model](01_linear_model/) | One parameter, loss, gradient descent | Pure Python |
| [02_neuron_layer](02_neuron_layer/) | Weighted sum, bias, activations | Pure Python |
| [03_tiny_neural_network](03_tiny_neural_network/) | Forward pass, MSE loss, backprop | NumPy |
| [04_embeddings](04_embeddings/) | Token IDs and embedding vectors | NumPy |
| [05_attention](05_attention/) | Q, K, V attention | NumPy |
| [06_mini_transformer](06_mini_transformer/) | Transformer block components | PyTorch |
| [07_mini_language_model](07_mini_language_model/) | Character-level next-token prediction | PyTorch |
| [08_word_level_language_model](08_word_level_language_model/) | Word-level language modeling | PyTorch |
| [09_bpe_tokenizer_demo](09_bpe_tokenizer_demo/) | BPE tokenization | Pure Python |
| [10_bpe_language_model](10_bpe_language_model/) | BPE language modeling | PyTorch |
| [11_scale_model](11_scale_model/) | Parameter count scaling | Pure Python |
| [12_instruction_tuning_demo](12_instruction_tuning_demo/) | Simple instruction tuning | PyTorch |
| [13_gpt_pretraining](13_gpt_pretraining/) | Decoder-only GPT pretraining | PyTorch |
| [14_dataset_pipeline](14_dataset_pipeline/) | Ingest, clean, dedupe, quality, shards | Pure Python |
| [15_scale_to_50m](15_scale_to_50m/) | 50M-class scaling on MPS | PyTorch |
| [16_evaluation_suite](16_evaluation_suite/) | Scaling and memorization evaluation | PyTorch |
| [17_instruction_dataset](17_instruction_dataset/) | Instruction JSONL processing | Pure Python |
| [18A_large_pretraining_corpus](18A_large_pretraining_corpus/) | Larger local corpus | Pure Python |
| [18B_marshmello_instruct](18B_marshmello_instruct/) | SFT, chat CLI, eval | PyTorch |
| [18C_base_chat_adaptation](18C_base_chat_adaptation/) | Mixed raw/chat continued pretraining | PyTorch |
| [18D_tokenizer_v2](18D_tokenizer_v2/) | Tokenizer v2 experiments | Pure Python |
| [18E_tiny_teacher_sft](18E_tiny_teacher_sft/) | Teacher SFT data and eval | Pure Python |
| [18G_checkpoint_and_corpus_expansion](18G_checkpoint_and_corpus_expansion/) | Checkpoint comparison and corpus expansion | Pure Python |
| [18H_chat_only_pretraining](18H_chat_only_pretraining/) | Chat-only corpus adaptation | Pure Python |
| [18I_routing_teacher_fix](18I_routing_teacher_fix/) | Routing data experiments | Pure Python |
| [18J_marshmello_core_sft](18J_marshmello_core_sft/) | Core routing benchmark | PyTorch |
| [18K_general_benchmark](18K_general_benchmark/) | General assistant benchmark | PyTorch |
| [19A_scale_to_300m](19A_scale_to_300m/) | 300M scaling smoke test | PyTorch |

## Architecture

Marshmello models use the custom decoder-only GPT implementation in [13_gpt_pretraining](13_gpt_pretraining/), not `transformers` AutoModel.

| Component | Marshmello-8M | Marshmello-55M | Marshmello-300M |
| --------- | ------------- | -------------- | --------------- |
| Config key | `default` | `large_50m` | `large_300m` |
| Layers | 4 | 6 | 20 |
| Hidden size | 384 | 768 | 1024 |
| Attention heads | 6 | 12 | 16 |
| FFN dimension | 1536 | 3072 | 4096 |
| Context length | 256 | 512 | 512 |
| Tokenizer | BPE | BPE | BPE v2 / 8000 vocab target |
| Positional embeddings | Learned | Learned | Learned |

Parameter counts are vocabulary-dependent for the smaller configurations. The 300M Phase 19A config has an exact count of 268,834,816 parameters with the current 8000-token vocabulary target.

## Evaluation

Run the core routing benchmark:

```bash
python 18J_marshmello_core_sft/evaluate_core_routing.py \
  --checkpoint 18B_marshmello_instruct/checkpoints/best_18j_routing.pt \
  --no-baseline
```

Run the general assistant comparison:

```bash
python 18K_general_benchmark/compare_checkpoints.py
```

Important reports:

| Report | Purpose |
| ------ | ------- |
| [reports/latest_eval_summary.md](reports/latest_eval_summary.md) | Latest broad SFT comparison |
| [18J_marshmello_core_sft/data/reports/marshmello_core_eval_comparison.md](18J_marshmello_core_sft/data/reports/marshmello_core_eval_comparison.md) | 18J routing comparison |
| [18K_general_benchmark/reports/comparison.md](18K_general_benchmark/reports/comparison.md) | 18K general benchmark comparison |
| [PHASE_19A.md](PHASE_19A.md) | 300M benchmark snapshot |

## Hugging Face Drafts

Draft cards are included for publishing polish:

| Draft | File |
| ----- | ---- |
| Model card | [huggingface/MODEL_CARD.md](huggingface/MODEL_CARD.md) |
| Dataset card | [huggingface/DATASET_CARD.md](huggingface/DATASET_CARD.md) |

These are drafts for Hugging Face repository pages. They do not upload or change any weights or datasets.

## Repository Layout

```text
Marshmello/
├── 01_linear_model/ ... 12_instruction_tuning_demo/
├── 13_gpt_pretraining/                  # GPT model, tokenizer, trainer, generation, Hub helpers
├── 14_dataset_pipeline/                 # Local data processing pipeline
├── 15_scale_to_50m/                     # 50M-class scaling notes
├── 16_evaluation_suite/                 # Scaling and memorization checks
├── 17_instruction_dataset/              # Instruction dataset processing
├── 18A_large_pretraining_corpus/        # Larger local pretraining corpus
├── 18B_marshmello_instruct/             # SFT and chat CLI
├── 18C_base_chat_adaptation/            # Mixed raw/chat continued pretraining
├── 18D_tokenizer_v2/                    # Tokenizer v2 experiments
├── 18E_tiny_teacher_sft/                # Teacher SFT data
├── 18G_checkpoint_and_corpus_expansion/ # Checkpoint comparison and corpus expansion
├── 18H_chat_only_pretraining/           # Chat-only adaptation
├── 18I_routing_teacher_fix/             # Routing experiment data
├── 18J_marshmello_core_sft/             # Core routing benchmark
├── 18K_general_benchmark/               # General assistant benchmark
├── 19A_scale_to_300m/                   # 300M scaling smoke test
├── huggingface/                         # Draft Hub cards
├── reports/                             # Evaluation reports and archives
└── requirements.txt
```

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Limitations

- Educational model family, not a production assistant.
- Trained on small local and synthetic/curated educational datasets, not web-scale corpora.
- Outputs can be repetitive, incorrect, or hallucinated.
- No RLHF, safety tuning, tool use, retrieval, or production serving stack.
- Custom PyTorch model code is not directly compatible with `transformers` pipelines.
- Benchmarks are internal project checks, not public leaderboard scores.

## License

Code and published model artifacts are documented as Apache 2.0. See Hugging Face model cards and repository metadata before redistribution.

## Links

| Resource | URL |
| -------- | --- |
| GitHub | https://github.com/mohmmedwee/Marshmello |
| Marshmello-55M | https://huggingface.co/ostah-1010/Marshmello |
| Marshmello-8M | https://huggingface.co/ostah-1010/Marshmello-8M |
| Marshmello-SFT draft dataset page | https://huggingface.co/datasets/ostah-1010/Marshmello-SFT |
