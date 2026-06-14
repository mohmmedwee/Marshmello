# Phase 18A — Larger Base Pretraining Corpus

This phase builds a larger local pretraining corpus for
**Marshmello-45M-Base-v2**.

Do not skip this step before instruction tuning. Base pretraining teaches the
model language, technical vocabulary, syntax, and broad domain patterns.
Instruction tuning later teaches the model how to follow prompts. SFT without a
good base model mostly memorizes answers instead of learning robust behavior.

---

## Sources

The corpus builder uses local data only:

- existing Phase 13 technical corpus
- generated software engineering notes
- generated database notes
- generated AI/ML notes
- generated cybersecurity notes
- generated Python/API notes
- a small instruction-format note containing `<USER>`, `<ASSISTANT>`, and
  `<END>` so the tokenizer can encode later chat-formatted data

No internet is required by default.

---

## Run

```bash
python 18A_large_pretraining_corpus/build_corpus.py --target-words 1000000
```

For a quick smoke test:

```bash
python 18A_large_pretraining_corpus/build_corpus.py --target-words 10000
```

---

## Outputs

Corpus:

```text
13_gpt_pretraining/data/corpus.txt
```

Report:

```text
18A_large_pretraining_corpus/reports/corpus_report.json
```

The report includes:

- total words
- estimated BPE tokens
- domain distribution
- duplicate ratio
- unique paragraph count

---

## Next Step

After building the corpus:

```bash
python 13_gpt_pretraining/tokenizer/train_bpe.py
python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 3000
```
