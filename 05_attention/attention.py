"""
Phase 05: Self-attention from scratch (NumPy)

BIG PICTURE
-----------
Each token in a sentence gets a new vector that mixes information from ALL tokens.
The mix is controlled by "attention weights" (numbers between 0 and 1 that sum to 1).

THREE MATRICES (Q, K, V)
------------------------
  Q (Query)  = what this token is LOOKING FOR
  K (Key)    = what each token OFFERS to be matched against
  V (Value)  = the actual INFORMATION each token contributes to the output

FROM EMBEDDINGS (real models do this step first)
-----------------------------------------------
  Q = X @ W_q     multiply embeddings X by learned matrix W_q
  K = X @ W_k     same for keys
  V = X @ W_v     same for values

ATTENTION FORMULA (the heart of the transformer)
------------------------------------------------
  1. scores     = Q @ K.T / sqrt(d_k)     compare every query to every key
  2. weights    = softmax(scores)         turn scores into probabilities per row
  3. output     = weights @ V              blend value vectors using those weights

For one token "it":
  output(it) = w(it→it)*V(it) + w(it→animal)*V(animal) + w(it→street)*V(street)

Run this file to see TWO demos:
  1. manual_attention_demo() — hand-picked Q, K, V (easy to follow)
  2. random_attention_demo() — random embeddings (like real training init)
"""

import numpy as np  # NumPy gives us fast matrix math (dot products, softmax, etc.)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax turns a row of numbers into probabilities that sum to 1.

    Example: [2.0, 1.0, 0.1] → [0.659, 0.242, 0.099]
    The largest number gets the biggest probability.
    """
    # Subtract the row max so exp() never overflows on large positive numbers
    shifted = x - np.max(x, axis=axis, keepdims=True)

    # exp turns each score into a positive number (bigger score → bigger number)
    exp_x = np.exp(shifted)

    # Divide each by the row sum so all values in the row add up to 1.0
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def print_qkv(Q: np.ndarray, K: np.ndarray, V: np.ndarray, token_names: list[str]) -> None:
    """Pretty-print the three matrices so you can see one row per token."""

    # Q: each row is one token's "search query" — what it wants to find
    print("\n--- Q (queries): what each token is LOOKING FOR ---")
    for i, name in enumerate(token_names):  # loop over tokens in order
        print(f"  {name:8s}: {np.array2string(Q[i], precision=3, separator=', ')}")

    # K: each row is one token's "label" — what it advertises to match queries
    print("\n--- K (keys): what each token OFFERS to match against ---")
    for i, name in enumerate(token_names):
        print(f"  {name:8s}: {np.array2string(K[i], precision=3, separator=', ')}")

    # V: each row is the actual content/info that token will share if attended to
    print("\n--- V (values): the INFORMATION each token contributes ---")
    for i, name in enumerate(token_names):
        print(f"  {name:8s}: {np.array2string(V[i], precision=3, separator=', ')}")


def print_attention_weights(
    scaled_scores: np.ndarray,
    attention_weights: np.ndarray,
    token_names: list[str],
) -> None:
    """Print score matrix and weight matrix as readable grids."""

    seq_len = len(token_names)  # number of tokens = size of the square matrix

    # Column headers: one label per token (it, animal, street, ...)
    header = "         " + "  ".join(f"{n:>8s}" for n in token_names)

    # --- Raw compatibility scores (before softmax) ---
    print("\n--- Attention scores (Q @ K^T / sqrt(d_k)) ---")
    print("Higher score = query finds that key more relevant.")
    print(header)
    for i, name in enumerate(token_names):  # i = row = "query token"
        # One number per column j = "key token" — how much i attends to j
        row = "  ".join(f"{scaled_scores[i, j]:8.3f}" for j in range(seq_len))
        print(f"  {name:8s} {row}")

    # --- Probabilities after softmax (these are the weights used on V) ---
    print("\n--- Attention weights (softmax of each row; each row sums to 1.0) ---")
    print("Each weight = how much of that token's V goes into the output.")
    print(header)
    for i, name in enumerate(token_names):
        row = "  ".join(f"{attention_weights[i, j]:8.3f}" for j in range(seq_len))
        print(f"  {name:8s} {row}")
        print(f"           row sum = {attention_weights[i].sum():.3f}")  # should always be 1.000


def compute_attention_from_qkv(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    token_names: list[str],
    *,
    print_details: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full attention formula when Q, K, V are already built.

    Shapes (for 3 tokens, d_k=3):
      Q, K, V  → (3, 3)   one row per token, d_k numbers per row
      scores   → (3, 3)   scores[i,j] = how much token i likes token j
      output   → (3, 3)   new vector for each token
    """
    d_k = Q.shape[1]  # length of each Q/K/V vector (used for scaling)

    # STEP 1: Compare every query to every key via dot product
    # Q @ K.T means: matrix multiply Q (3×3) by K transposed (3×3) → (3×3)
    # scores[i, j] = dot(Q[i], K[j]) — compatibility between token i and token j
    scores = Q @ K.T

    # STEP 2: Scale down so dot products don't get too large (stabilizes softmax)
    # Dividing by sqrt(d_k) is standard in the "scaled dot-product attention" paper
    scaled_scores = scores / np.sqrt(d_k)

    # STEP 3: Softmax each ROW → turn scores into weights that sum to 1
    # axis=1 means "softmax across columns" (each row independently)
    attention_weights = softmax(scaled_scores, axis=1)

    # STEP 4: Blend all V rows using the weights
    # (3×3) @ (3×3) → (3×3): each output row is a weighted average of all V rows
    output = attention_weights @ V

    if print_details:  # only print when caller wants verbose output
        print_qkv(Q, K, V, token_names)
        print_attention_weights(scaled_scores, attention_weights, token_names)

    return scaled_scores, attention_weights, output


def self_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    token_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline used in real models:
      embeddings X  →  project to Q, K, V  →  run attention
    """
    # X shape: (seq_len, d_model) — e.g. 3 tokens × 6-dim embedding each
    # W_q shape: (d_model, d_k) — projects 6-dim embedding down to d_k-dim query
    Q = X @ W_q  # each token's embedding becomes a query vector

    # Same idea: embedding → key vector
    K = X @ W_k

    # Same idea: embedding → value vector (the info we'll blend)
    V = X @ W_v

    # Now run the attention math and print everything
    scaled_scores, attention_weights, output = compute_attention_from_qkv(
        Q, K, V, token_names, print_details=True
    )
    return Q, K, V, scaled_scores, output


def manual_attention_demo() -> None:
    """
    Demo 1: We SKIP embeddings and projection matrices.
    We write Q, K, V directly so you can see exactly why "it" looks at "animal".
    """
    print("\n" + "=" * 60)
    print("DEMO 1 (manual): \"it animal street\" — hand-crafted Q, K, V")
    print("=" * 60)

    tokens = ["it", "animal", "street"]  # our tiny 3-word sentence
    # Row indices: it=0, animal=1, street=2 (used when indexing matrices)

    # ------------------------------------------------------------------
    # Q — what each token is LOOKING FOR (one row per token, 3 numbers each)
    # ------------------------------------------------------------------
    Q = np.array([
        [2.0, 0.0, 0.0],   # it:     strong signal on dim 0 → "find a noun referent"
        [1.5, 0.0, 0.0],   # animal: also looks along dim 0 (for its own attention row)
        [0.0, 0.0, 1.0],   # street: looks along dim 2 → "find place/location info"
    ])

    # ------------------------------------------------------------------
    # K — what each token OFFERS when another token's query searches for a match
    # Dot product Q(it) · K(animal) will be LARGE because both point along dim 0
    # Dot product Q(it) · K(street) will be TINY because they point different ways
    # ------------------------------------------------------------------
    K = np.array([
        [0.3, 0.0, 0.0],   # it:     weak self-match (pronoun doesn't refer to itself much)
        [2.0, 0.0, 0.0],   # animal: strong offer on dim 0 → perfect match for Q(it)
        [0.0, 0.0, 0.1],   # street: tiny offer → Q(it) barely notices street
    ])

    # ------------------------------------------------------------------
    # V — the INFORMATION each token contributes to the final blended output
    # These are what get mixed together using attention weights
    # ------------------------------------------------------------------
    V = np.array([
        [0.0, 1.0, 0.0],   # it:     dim 1 lit up → "pronoun / pointer" flavor
        [1.0, 0.0, 0.0],   # animal: dim 0 lit up → "living creature" flavor
        [0.0, 0.0, 1.0],   # street: dim 2 lit up → "road / place" flavor
    ])

    print("\nScenario: the word \"it\" often refers to a nearby noun.")
    print("We designed Q and K so Q(it) · K(animal) is large,")
    print("and Q(it) · K(street) is tiny.\n")

    # Run attention; we ignore scaled_scores here (_) but keep weights and output
    _, attention_weights, output = compute_attention_from_qkv(Q, K, V, tokens)

    # ------------------------------------------------------------------
    # Pull out the "it" row of the weight matrix — where does "it" look?
    # attention_weights[0] = [w(it→it), w(it→animal), w(it→street)]
    # ------------------------------------------------------------------
    w_it = attention_weights[0]
    idx_it, idx_animal, idx_street = 0, 1, 2  # readable names for row indices

    print("\n--- Where does \"it\" look? (row 0 of the weight matrix) ---")
    print(f"  weight(it → it):      {w_it[idx_it]:.3f}")       # self-attention (usually small here)
    print(f"  weight(it → animal):  {w_it[idx_animal]:.3f}  ← strongest (refers to animal)")
    print(f"  weight(it → street):  {w_it[idx_street]:.3f}  ← weak (not the referent)")

    # ------------------------------------------------------------------
    # Show the output for "it" as an explicit weighted sum (the core formula):
    #
    #   output(it) = weight_to_it     * V(it)
    #              + weight_to_animal  * V(animal)
    #              + weight_to_street  * V(street)
    #
    # This is exactly what  output = attention_weights @ V  does in one line.
    # ------------------------------------------------------------------
    v_it = V[idx_it]           # value vector for "it"
    v_animal = V[idx_animal]   # value vector for "animal"
    v_street = V[idx_street]   # value vector for "street"

    manual_output = (
        w_it[idx_it] * v_it           # small slice of pronoun info
        + w_it[idx_animal] * v_animal # big slice of animal info  ← dominates
        + w_it[idx_street] * v_street # tiny slice of street info
    )

    print("\n--- Building output(\"it\") as a weighted sum of V vectors ---")
    print("Formula:")
    print("  output(it) = weight_to_it * V(it)")
    print("             + weight_to_animal * V(animal)")
    print("             + weight_to_street * V(street)")
    print()
    print(f"  = {w_it[idx_it]:.3f} * {np.array2string(v_it, precision=2, separator=', ')}")
    print(f"  + {w_it[idx_animal]:.3f} * {np.array2string(v_animal, precision=2, separator=', ')}")
    print(f"  + {w_it[idx_street]:.3f} * {np.array2string(v_street, precision=2, separator=', ')}")
    print(f"  = {np.array2string(manual_output, precision=3, separator=', ')}")
    # np.allclose checks the manual math matches the matrix multiply (should be True)
    print(f"  (matches matrix output: {np.allclose(manual_output, output[0])})")

    print("\n--- Final output vectors for all tokens ---")
    for i, name in enumerate(tokens):
        print(f"  {name:8s}: {np.array2string(output[i], precision=3, separator=', ')}")


def random_attention_demo() -> None:
    """
    Demo 2: Same math as Demo 1, but Q/K/V come from random embeddings.
    This is closer to how a model looks BEFORE training.
    """
    print("\n" + "=" * 60)
    print("DEMO 2 (random): \"I love AI\" — random embeddings & projections")
    print("=" * 60)

    tokens = ["I", "love", "AI"]
    seq_len = len(tokens)   # 3 tokens in the sentence
    d_model = 6               # each token embedding has 6 numbers
    d_k = 4                   # each Q/K/V vector has 4 numbers after projection

    rng = np.random.default_rng(7)  # fixed seed → same random numbers every run

    # X: (3, 6) matrix — 3 token embeddings, 6 dimensions each (fake, not trained)
    X = rng.normal(0, 0.5, size=(seq_len, d_model))

    print("\nInput embeddings X (one row per token):")
    for i, name in enumerate(tokens):
        print(f"  {name:8s}: {np.array2string(X[i], precision=3, separator=', ')}")

    # W_q, W_k, W_v: (6, 4) projection matrices — learned in real models, random here
    W_q = rng.normal(0, 0.3, size=(d_model, d_k))  # embedding → query
    W_k = rng.normal(0, 0.3, size=(d_model, d_k))  # embedding → key
    W_v = rng.normal(0, 0.3, size=(d_model, d_k))  # embedding → value

    # Project X into Q, K, V then run attention (we discard Q,K,V,scores with _)
    _, _, _, _, output = self_attention(X, W_q, W_k, W_v, tokens)

    print("\n--- Attention output (new representation per token) ---")
    for i, name in enumerate(tokens):
        print(f"  {name:8s}: {np.array2string(output[i], precision=3, separator=', ')}")

    print("\nWith random weights, attention is roughly uniform.")
    print("Training learns Q, K, V so weights become meaningful (like Demo 1).")


def main() -> None:
    print("Phase 05: Self-attention from scratch")
    manual_attention_demo()   # start with the easy, hand-crafted example
    random_attention_demo()   # then show the same math with random numbers
    print("\n" + "=" * 60)
    print("Key idea: each token's output is a blend of ALL value vectors,")
    print("weighted by how well its query matches each key.")


# Only run main() when you execute this file directly (not when importing it)
if __name__ == "__main__":
    main()
