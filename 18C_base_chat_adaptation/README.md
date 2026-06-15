# Phase 18C - Base Chat Adaptation

Phase 18C continues base-model pretraining on a mixed corpus that includes
normal technical paragraphs and chat-formatted text.

This is still next-token prediction on all tokens. It is not SFT, and it does
not mask user tokens or train only assistant answers.

## Why

Base pretraining teaches language completion: the model learns how technical
text continues.

Chat-format continued pretraining teaches the base model that `<USER>`,
`<ASSISTANT>`, and `<END>` are meaningful boundary tokens in text.

SFT comes after this step. SFT teaches instruction following by weighting the
assistant answer targets.

## Build Mixed Corpus

```bash
python 18C_base_chat_adaptation/build_chat_mixed_corpus.py \
  --chat-ratio 0.2 \
  --max-chat-examples 10000
```

Output:

```text
13_gpt_pretraining/data/corpus_chat_mixed.txt
```

The default mix targets about 80% raw technical text and 20% chat-formatted
examples by word count.

## Continue Base Pretraining

Recommended: pass the mixed corpus explicitly so `corpus.txt` stays unchanged.

```bash
python 13_gpt_pretraining/training/trainer.py \
  --config large_50m \
  --resume 13_gpt_pretraining/checkpoints/large_50m/latest.pt \
  --corpus 13_gpt_pretraining/data/corpus_chat_mixed.txt \
  --steps 1000 \
  --lr 5e-5
```

The trainer resumes from the existing base checkpoint and trains the same
causal next-token objective on every token in the mixed corpus.
With `--resume`, `--steps 1000` runs 1,000 additional optimizer steps.

If you instead copy `corpus_chat_mixed.txt` over `corpus.txt`, the safe training
command is:

```bash
python 13_gpt_pretraining/training/trainer.py \
  --config large_50m \
  --resume 13_gpt_pretraining/checkpoints/large_50m/latest.pt \
  --steps 1000 \
  --lr 5e-5
```

## Evaluate Chat Boundary

```bash
python 18C_base_chat_adaptation/eval_chat_format.py
```

The eval prompts:

```text
<USER> What is AI? <ASSISTANT>
```

It checks whether generation begins like an answer instead of continuing as a
random raw-corpus paragraph.
