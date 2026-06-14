"""
Phase 03: Tiny 2-layer neural network from scratch (NumPy)

Architecture:
  input (2) → hidden (4, ReLU) → output (1, linear)

We train on y = x1 + x2 (a simple sum) to keep the task easy to follow.

Pipeline each epoch:
  1. Forward pass  — compute predictions
  2. Loss (MSE)      — measure error
  3. Backward pass   — compute gradients via chain rule
  4. Weight update   — gradient descent step
"""

import numpy as np

# ---------------------------------------------------------------------------
# Data: learn f(x1, x2) = x1 + x2
# ---------------------------------------------------------------------------
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [0.5, 0.5]])
Y = np.array([[3.0], [5.0], [4.0], [1.0]])  # targets as column vectors

input_size = 2
hidden_size = 4
output_size = 1

# Xavier-style small random init
rng = np.random.default_rng(42)
W1 = rng.normal(0, 0.5, size=(input_size, hidden_size))
b1 = np.zeros((1, hidden_size))
W2 = rng.normal(0, 0.5, size=(hidden_size, output_size))
b2 = np.zeros((1, output_size))

learning_rate = 0.1
epochs = 500


def relu(z: np.ndarray) -> np.ndarray:
    """Element-wise ReLU."""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """ReLU derivative: 1 where z > 0, else 0."""
    return (z > 0).astype(float)


def forward(X_batch: np.ndarray) -> tuple:
    """
    Forward pass through both layers.

    Returns all intermediate values — backprop needs them.
    """
    # Layer 1: linear → ReLU
    z1 = X_batch @ W1 + b1          # (batch, hidden)
    a1 = relu(z1)                   # activations after ReLU

    # Layer 2: linear (no activation on output for regression)
    z2 = a1 @ W2 + b2               # (batch, output)
    predictions = z2

    cache = (X_batch, z1, a1, z2)
    return predictions, cache


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((predictions - targets) ** 2))


def backward(cache: tuple, targets: np.ndarray) -> tuple:
    """
    Backpropagation: chain rule from loss back to W1, b1, W2, b2.

    Loss L = mean((y_hat - y)^2)
    dL/d(y_hat) = 2 * (y_hat - y) / n   (for each sample, averaged in batch)
    """
    global W1, b1, W2, b2

    X_batch, z1, a1, z2 = cache
    batch_size = X_batch.shape[0]

    # Gradient at output
    dL_dz2 = (2.0 / batch_size) * (z2 - targets)  # (batch, output)

    # Layer 2 gradients
    dL_dW2 = a1.T @ dL_dz2                          # (hidden, output)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

    # Propagate into hidden layer
    dL_da1 = dL_dz2 @ W2.T                          # (batch, hidden)
    dL_dz1 = dL_da1 * relu_derivative(z1)           # through ReLU

    # Layer 1 gradients
    dL_dW1 = X_batch.T @ dL_dz1                     # (input, hidden)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    return dL_dW1, dL_db1, dL_dW2, dL_db2


def update_weights(grads: tuple) -> None:
    """Gradient descent: move each parameter opposite its gradient."""
    global W1, b1, W2, b2

    dL_dW1, dL_db1, dL_dW2, dL_db2 = grads
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2


def main() -> None:
    print("Phase 03: Tiny neural network — forward, loss, backprop, update")
    print("=" * 60)
    print("Task: learn f(x1, x2) = x1 + x2")
    print(f"Architecture: {input_size} → {hidden_size} (ReLU) → {output_size}\n")

    for epoch in range(1, epochs + 1):
        predictions, cache = forward(X)
        loss = mse_loss(predictions, Y)
        grads = backward(cache, Y)
        update_weights(grads)

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d} | loss = {loss:.6f}")

    print("\n--- Predictions after training ---")
    for i in range(len(X)):
        pred, _ = forward(X[i : i + 1])
        print(f"  input {X[i]} → predicted {pred[0, 0]:.4f}  (target {Y[i, 0]:.1f})")

    print("\n" + "=" * 60)
    print("You just ran a full training loop: forward → loss → backward → update.")


if __name__ == "__main__":
    main()
