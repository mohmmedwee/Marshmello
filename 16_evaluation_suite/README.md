# Phase 16 — Evaluation Suite

Compare **Marshmello-8M** (`default`) vs **Marshmello-45M** (`large_50m`) on the same prompts and corpus — and measure whether bigger means better, or just **memorizes faster**.

```text
Same corpus
Same BPE
Same prompts
        ↓
Generate with 8M + 45M
        ↓
Memorization + coherence metrics
        ↓
Side-by-side report + conclusion
```

---

## Why this phase exists

Phase 15 scaled parameters. Phase 16 asks the question real LLM teams ask after scaling:

> Did quality improve — or did we just fit the training set tighter?

On a **small, repetitive corpus**, a 45M model often:

- Copies paragraphs verbatim (low val loss ≈ memorization)
- Scores higher on **nearest-line similarity** to training text
- Shows only **modest** gains in domain consistency

That is the core lesson: **scaling without more data has diminishing returns.**

---

## Prompt suite

| Prompt | Expected domain |
|--------|-----------------|
| Database systems | databases |
| Artificial intelligence | artificial_intelligence |
| Machine learning | machine_learning |
| Software engineering | software_engineering |
| Python APIs | software_engineering |
| Distributed systems | distributed_systems |

---

## Metrics

| Metric | What it detects |
|--------|-----------------|
| **Avg generation length** | Too short = collapsed; very long on tiny corpus = looping |
| **Repeated n-gram ratio** | Internal repetition (`the the`, phrase loops) |
| **Exact paragraph match** | Verbatim copy of a training paragraph |
| **Nearest training line similarity** | Close paraphrase / retrieval of memorized snippet |
| **Domain consistency** | Keyword overlap with expected topic |

---

## Quick start

From `mini-transformer-from-scratch/` with venv active:

```bash
# Train both models first (if needed)
python 13_gpt_pretraining/training/trainer.py --quick
python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 300

# Run evaluation
python 16_evaluation_suite/evaluate.py

# Faster / deterministic
python 16_evaluation_suite/evaluate.py --greedy --max-new-tokens 80

# Tests
python -m unittest 16_evaluation_suite.tests.test_metrics -v
```

Output:

- Side-by-side generations per prompt
- Aggregate metrics for Marshmello-8M vs Marshmello-45M
- **Conclusion**: coherence change + memorization verdict
- JSON report → `16_evaluation_suite/reports/evaluation_report.json`

---

## Example conclusion (typical on demo corpus)

```text
- Marshmello-45M improves domain consistency slightly, but gains are limited on the same small corpus.
- Memorization detected (Marshmello-8M, Marshmello-45M): outputs closely match training paragraphs/lines.
- High nearest-line similarity confirms retrieval of memorized snippets, not new facts.
- Takeaway: bigger models on the same data mostly memorize faster — they do not replace better data.
```

---

## Why scaling without more data gives limited improvement

| Factor | 8M on small corpus | 45M on same corpus |
|--------|--------------------|--------------------|
| Capacity vs data | Can already memorize passages | Extra capacity → **tighter memorization**, not new knowledge |
| Val loss | Drops quickly | Drops **lower** (often overfitting signal) |
| Generalization | Weak (expected) | Still weak — parameters ≠ information |
| Coherence | Template-like prose | Smoother interpolation between **seen** paragraphs |

**Chinchilla scaling laws** say optimal training uses ~20 tokens per parameter. Our demo corpus has thousands of tokens, not millions — both models are severely **under-trained on data** but **over-capacity for memorization**.

So:

```text
10M → 45M on same 50KB corpus ≠ smarter model
                                  = better compression of memorized text
```

What actually helps next:

1. **More diverse data** (Phase 14 pipeline at scale)
2. **Longer training** with regularization
3. **Fine-tuning / LoRA** for task behavior (Phase 17)
4. **Not** 10× params on the same repeated paragraphs

---

## Folder layout

```text
16_evaluation_suite/
├── evaluate.py       # main CLI
├── metrics.py        # memorization + quality metrics
├── prompts.py        # prompt suite + domain keywords
├── reports/          # evaluation_report.json
├── tests/
└── README.md
```

---

## Success criteria

```text
✓ 6-prompt evaluation suite
✓ Generate from default + large_50m
✓ Memorization metrics vs training corpus
✓ Side-by-side outputs
✓ Automated conclusion
✓ README: scaling limits without more data
```
