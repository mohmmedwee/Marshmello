# Phase 18G - Checkpoint Selection and Corpus Expansion

Tokenizer v2 fixes coverage, but tokenizer coverage does not create language
ability by itself. The 55M base model still needs stronger and more varied
pretraining before broad SFT can reliably teach assistant behavior.

## Why More SFT Is Not Enough Yet

SFT changes how a capable base model responds. It cannot reliably replace
missing language, reasoning, or technical knowledge in the base checkpoint.
When the base is weak, broad SFT can amplify narrow phrases and cause repetitive
answers instead of improving general instruction following.

The safer order is:

1. select the strongest available base checkpoint
2. continue base pretraining on a larger clean corpus
3. verify raw and chat-format generation
4. run tiny teacher or curated SFT
5. expand to broader SFT only after generation is stable

## Compare Checkpoints

```bash
python 18G_checkpoint_and_corpus_expansion/compare_checkpoints.py
```

The comparison tests raw technical prompts and exact chat-format prompts. It
reports repetition, topic relevance, gibberish-looking words, unrelated-topic
drift, and chat-boundary behavior. It prints:

```text
BEST_CHECKPOINT_FROM_COMPARE=...
```

Lower validation loss is useful, but generation quality matters too. A later
checkpoint can have lower training loss while becoming more repetitive or less
useful on prompts.

## Build Expanded Corpus

```bash
python 18G_checkpoint_and_corpus_expansion/build_expanded_corpus.py \
  --target-words 5000000 \
  --chat-ratio 0.2
```

For a larger run:

```bash
python 18G_checkpoint_and_corpus_expansion/build_expanded_corpus.py \
  --target-words 10000000 \
  --chat-ratio 0.2
```

Output:

```text
13_gpt_pretraining/data/corpus_chat_mixed_expanded.txt
```

Report:

```text
18G_checkpoint_and_corpus_expansion/reports/expanded_corpus_report.json
```

The builder keeps the current base corpus, adds deterministic clean lessons
across AI, machine learning, databases, software engineering, Python, APIs,
Docker, cybersecurity, and transformers/LLMs, then mixes chat examples at the
requested ratio.

## Recommended Fresh Training

```bash
python 13_gpt_pretraining/training/trainer.py \
  --config large_50m \
  --corpus 13_gpt_pretraining/data/corpus_chat_mixed_expanded.txt \
  --steps 5000 \
  --lr 1e-4
```

## Safer Resume

Replace the placeholder with the recommendation printed by
`compare_checkpoints.py`:

```bash
python 13_gpt_pretraining/training/trainer.py \
  --config large_50m \
  --resume <BEST_CHECKPOINT_FROM_COMPARE> \
  --corpus 13_gpt_pretraining/data/corpus_chat_mixed_expanded.txt \
  --steps 3000 \
  --lr 5e-5
```

## Tokenizer

Phase 18G does not modify the tokenizer. It uses:

```text
13_gpt_pretraining/tokenizer/tokenizer.json
```

That file is the current tokenizer v2 with an 8,000-token vocabulary.
