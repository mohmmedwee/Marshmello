"""
Phase 01: One weight learns y = 5x

This is the smallest possible "model":
  - One input x
  - One learnable parameter w
  - Prediction: y_hat = w * x

We minimize mean squared error (MSE) with plain gradient descent.
No PyTorch, no NumPy — just Python floats.
"""

# ---------------------------------------------------------------------------
# Training data: y = 5 * x
# The model does NOT know that 5 is the answer — it must discover w.
# ---------------------------------------------------------------------------
X = [1.0, 2.0, 3.0, 4.0]
Y = [5.0, 10.0, 15.0, 20.0]

# Start with a wrong guess so we can watch learning happen.
w = 0.5

# Learning rate: how big each weight update step is.
learning_rate = 0.01

# Number of passes over the full dataset.
epochs = 100


def predict(x: float, weight: float) -> float:
    """Forward pass: multiply input by our one weight."""
    return weight * x


def mse_loss(predictions: list[float], targets: list[float]) -> float:
    """
    Mean Squared Error:
      loss = average of (prediction - target)^2

    Squaring penalizes large errors more than small ones.
    """
    squared_errors = [(p - t) ** 2 for p, t in zip(predictions, targets)]
    return sum(squared_errors) / len(squared_errors)


def compute_gradient(x_values: list[float], y_values: list[float], weight: float) -> float:
    """
    Gradient of MSE with respect to w.

    For one sample: loss = (w*x - y)^2
    d(loss)/d(w) = 2 * (w*x - y) * x

    We average the gradient over all training samples.
    """
    gradients = []
    for x, y in zip(x_values, y_values):
        prediction = weight * x
        error = prediction - y
        grad = 2 * error * x
        gradients.append(grad)
    return sum(gradients) / len(gradients)


def main() -> None:
    global w

    print("Phase 01: Linear model — learning y = 5x with one weight w")
    print("=" * 60)
    print(f"Initial w = {w:.4f}")
    print(f"True relationship: y = 5x (we want w → 5.0)\n")

    for epoch in range(1, epochs + 1):
        # --- Forward pass: compute predictions for all x ---
        predictions = [predict(x, w) for x in X]

        # --- Loss: how wrong are we? ---
        loss = mse_loss(predictions, Y)

        # --- Backward pass: which direction should w move? ---
        gradient = compute_gradient(X, Y, w)

        # --- Update: move w opposite to the gradient (downhill on the loss) ---
        w = w - learning_rate * gradient

        # Print every epoch so you can see w crawl toward 5.0
        print(f"Epoch {epoch:3d} | w = {w:8.4f} | loss = {loss:.6f}")

    print("\n" + "=" * 60)
    print(f"Final w = {w:.4f}  (target was 5.0)")
    print("Try changing learning_rate or initial w and run again!")


if __name__ == "__main__":
    main()
