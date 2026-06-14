"""
Phase 04: Tokenizer + embeddings

Transformers don't read words directly. The pipeline is:

  text → tokens → token IDs → embedding vectors (rows in a matrix)

Each token ID picks one row from an embedding table.
That row is a learned (here: random) vector in R^embed_dim.
"""

import numpy as np


class SimpleTokenizer:
    """
    Maps words ↔ integers.

    Special tokens:
      <PAD> — padding (index 0)
      <UNK> — unknown word (index 1)

    Real tokenizers (BPE, SentencePiece) are more complex;
    this one splits on whitespace for clarity.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self) -> None:
        self.word_to_id: dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self.id_to_word: dict[int, str] = {0: self.PAD, 1: self.UNK}

    def _ensure_word(self, word: str) -> int:
        """Add word to vocabulary if new; return its ID."""
        if word not in self.word_to_id:
            new_id = len(self.word_to_id)
            self.word_to_id[word] = new_id
            self.id_to_word[new_id] = word
        return self.word_to_id[word]

    def encode(self, text: str) -> list[int]:
        """Convert a sentence to a list of token IDs."""
        words = text.strip().split()
        return [self._ensure_word(w) for w in words]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to a string (for debugging)."""
        return " ".join(self.id_to_word[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)


class EmbeddingTable:
    """
    A lookup table: vocab_size rows × embed_dim columns.

    embedding[token_id] returns the vector for that token.
    In training, these rows are updated by backprop (not random forever).
    """

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        # Small random values — real models learn meaningful directions
        self.weights = rng.normal(0, 0.1, size=(vocab_size, embed_dim))

    def lookup(self, token_ids: list[int]) -> np.ndarray:
        """
        Fetch one embedding vector per token ID.
        Returns shape (num_tokens, embed_dim).
        """
        return self.weights[token_ids]

    def show_vector(self, token_id: int, id_to_word: dict[int, str]) -> None:
        """Pretty-print one embedding row."""
        word = id_to_word.get(token_id, "?")
        vec = self.weights[token_id]
        preview = np.array2string(vec, precision=3, separator=", ")
        print(f"  ID {token_id:2d} '{word:8s}' → {preview}")


def main() -> None:
    print("Phase 04: Tokenizer and embeddings")
    print("=" * 60)

    sentence = "I love AI and I love learning"
    embed_dim = 8

    # --- Step 1: tokenize ---
    tokenizer = SimpleTokenizer()
    token_ids = tokenizer.encode(sentence)

    print(f"\nSentence: \"{sentence}\"")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded:   \"{tokenizer.decode(token_ids)}\"")
    print(f"Vocab size: {tokenizer.vocab_size} (includes <PAD> and <UNK>)")

    # --- Step 2: build embedding table ---
    table = EmbeddingTable(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim)

    # --- Step 3: IDs → vectors ---
    vectors = table.lookup(token_ids)

    print(f"\nEmbedding matrix shape: {table.weights.shape}  (vocab × dim)")
    print(f"Sequence embedding shape: {vectors.shape}  (tokens × dim)")

    print("\nEach token becomes a vector (first 8 dimensions shown):")
    for tid in token_ids:
        table.show_vector(tid, tokenizer.id_to_word)

    # --- Step 4: show that the same word gets the same vector ---
    print("\n--- Same word, same vector ---")
    id_love = tokenizer.word_to_id["love"]
    v1 = table.lookup([id_love])[0]
    v2 = table.lookup([id_love])[0]
    print(f"  'love' appears twice in the sentence but always maps to ID {id_love}")
    print(f"  Vectors equal: {np.allclose(v1, v2)}")

    print("\n" + "=" * 60)
    print("Next step (Phase 05): these vectors will 'talk' via attention.")


if __name__ == "__main__":
    main()
