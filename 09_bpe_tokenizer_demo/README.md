# Phase 09: Why BPE?

Real LLMs (GPT, LLaMA, Mistral, etc.) almost never use raw **character** or **word** tokenization. They use **subword** methods — most often **Byte Pair Encoding (BPE)** or a close variant (SentencePiece, WordPiece).

Run the demo:

```bash
python 09_bpe_tokenizer_demo/bpe_demo.py
```

---

## Three ways to cut text into tokens

| Method | Example: `"tokenization"` | Vocab size | New word `"ChatGPT"` |
|--------|---------------------------|------------|----------------------|
| **Character** | `t`, `o`, `k`, `e`, `n`, ... (15 tokens) | ~100 | Works (spell letter by letter) but sequences are **very long** |
| **Word** | `tokenization` (1 token if seen before) | ~10k–100k+ | **`<UNK>`** if word was never in training data |
| **BPE (subword)** | `token`, `ization` or `tok`, `en`, `ization` | ~32k (you choose) | `Chat`, `G`, `PT` — **no UNK**, built from pieces |

---

## Problems with character-level (phase 07)

- A sentence becomes hundreds of tokens → slow, hard to learn long-range patterns.
- The model must learn spelling **and** meaning at the same time.
- Tiny alphabet (~50 chars) but very long sequences.

**Good for:** learning, tiny demos.  
**Bad for:** real language modeling.

---

## Problems with word-level (phase 08)

- Vocabulary explodes: every unique word needs its own ID.
- `"running"` and `"run"` are unrelated tokens — no shared structure.
- Any word not in training data → **`<UNK>`** (model has no idea what it means).
- Languages with compounding (German) or rich morphology make this worse.

**Good for:** readable toy output, simple demos.  
**Bad for:** open vocabulary, production systems.

---

## Why BPE wins (what this demo shows)

BPE starts with **characters** and repeatedly merges the most common pairs:

```
't' + 'h'  →  'th'
'th' + 'e' →  'the'
...
```

After thousands of merges you get:

- **Common words** as single tokens: `the`, `and`, `fortune`
- **Rare words** split into pieces: `super` + `cal` + `ifi` + ...
- **New words** handled the same way — split into known subwords, **never `<UNK>`**
- **Vocab size** is a hyperparameter: `num_merges` (GPT-2 uses ~50,000)

This is the sweet spot real LLMs use:

```
sequence length  ≈  word-level  (much shorter than char-level)
vocabulary       ≈  controlled  (unlike unbounded word vocab)
unknown words    ≈  handled     (unlike word-level <UNK>)
```

---

## What the demo prints

1. **Vocab size** before and after BPE merges on the phase 08 corpus  
2. **Same sentence** tokenized three ways (char, word, BPE)  
3. **Unknown sentences** showing `<UNK>` vs subword splits  
4. **First merges learned** — usually common English pairs like `'t'+'h'→'th'`, `'e'+' '</w>'→'e</w>'`

---

## Connection to the learning path

| Phase | Token unit |
|-------|------------|
| 07 | Character |
| 08 | Word |
| **09** | **Subword (BPE)** ← what production LLMs use |

Phase 09 is **tokenizer-only** — no transformer training. It explains *why* the token step matters before you scale to billion-parameter models.

---

## Not production-grade

This BPE implementation is simplified for teaching:

- Trains on whitespace-split words (real BPE often uses bytes or GPT-2 pre-tokenizer regex)
- No byte-level fallback for unicode edge cases
- No special tokens (`<|endoftext|>`, etc.)

For production, libraries like `tiktoken`, `sentencepiece`, or Hugging Face `tokenizers` handle these details. The **algorithm** is the same idea you see here.
