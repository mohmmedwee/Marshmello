# Phase 19A: Marshmello-300M Scaling

Phase 19A is the first 300M-class scaling pass for Marshmello. It documents the `large_300m` configuration, measured smoke/benchmark results, and the next gates before any stable release claim.

This phase does not change training code, datasets, or checkpoints. It documents the current 300M scaling target.

## Config: `large_300m`

Defined in [13_gpt_pretraining/config.py](13_gpt_pretraining/config.py).

| Field | Value |
| ----- | ----: |
| `d_model` | 1024 |
| `num_layers` | 20 |
| `num_heads` | 16 |
| `d_ff` | 4096 |
| `block_size` | 512 |
| `dropout` | 0.1 |
| `batch_size` | 2 |
| `gradient_accumulation_steps` | 16 |
| `learning_rate` | 1.5e-4 |
| `target_vocab_size` | 8000 |

## Parameter Count

| Model | Parameters |
| ----- | ---------: |
| Marshmello-300M / `large_300m` | 268,834,816 |

The exact count assumes the current 8000-token vocabulary target, learned positional embeddings, an untied LM head, and no LM head bias.

## Benchmark Result

| Metric | Result |
| ------ | -----: |
| Training tokens/sec | 3,552 |
| Forward tokens/sec | 14,813 |
| Generation tokens/sec | 11 |
| Peak memory | 5.5GB |
| Checkpoint size | 3.0GB |
| Model weights | 1.0GB |

These are smoke/throughput numbers, not quality numbers. Phase 19A has not established that Marshmello-300M is a better assistant than the 55M baseline.

## Smoke Test Command

Run from the repository root:

```bash
./19A_scale_to_300m/train_smoke.sh
```

The script runs:

```bash
python 13_gpt_pretraining/training/trainer.py \
  --config large_300m \
  --corpus 13_gpt_pretraining/data/corpus_chat_only.txt \
  --steps 20 \
  --lr 1e-4
```

A benchmark-only run is:

```bash
python 13_gpt_pretraining/benchmark.py --config large_300m
```

## Next Steps

- Finish 300M smoke testing.
- Run chat-only pretraining for 300M.
- Apply Teacher SFT v2 only after the base run is stable.
- Evaluate with both 18J and 18K.
- Compare against the 55M plateau: 18% 18J routing, 22.5% 18K domain score, and 64.2% 18K hallucination.
- Do not publish a stable 300M model card until quality benchmarks improve honestly.

More context: [19A_scale_to_300m/README.md](19A_scale_to_300m/README.md)
