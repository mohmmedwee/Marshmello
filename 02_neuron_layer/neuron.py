"""
Phase 02: A single neuron with multiple inputs

A neuron is still a weighted sum + optional bias + optional activation:

    z = w1*x1 + w2*x2 + ... + b        (linear part)
    output = activation(z)              (non-linearity)

Activations let networks learn curves, not just straight lines.
"""


def dot_product(weights: list[float], inputs: list[float]) -> float:
    """
    Multiply matching pairs and add them up.
    Example: weights=[0.5, -1.0], inputs=[2.0, 3.0] → 0.5*2 + (-1)*3 = -2.0
    """
    total = 0.0
    for w, x in zip(weights, inputs):
        total += w * x
    return total


def relu(z: float) -> float:
    """
    ReLU (Rectified Linear Unit): max(0, z)

    - Negative values become 0 (neuron "turns off")
    - Positive values pass through unchanged
    Used heavily in hidden layers of modern networks.
    """
    return max(0.0, z)


def sigmoid(z: float) -> float:
    """
    Sigmoid: squashes any number into (0, 1)

    Useful when you want a probability-like output.
    Formula: 1 / (1 + e^(-z))
    """
    import math

    return 1.0 / (1.0 + math.exp(-z))


class Neuron:
    """
    One neuron = weights + bias + chosen activation function.

    Think of weights as "how important each input is"
    and bias as "how eager the neuron is to fire by default."
    """

    def __init__(
        self,
        num_inputs: int,
        weights: list[float] | None = None,
        bias: float = 0.0,
        activation: str = "relu",
    ) -> None:
        # If no weights given, start small random-ish values for demo
        if weights is None:
            self.weights = [0.5 * (i + 1) for i in range(num_inputs)]
        else:
            self.weights = weights

        self.bias = bias
        self.activation_name = activation

    def linear(self, inputs: list[float]) -> float:
        """Weighted sum plus bias — before activation."""
        return dot_product(self.weights, inputs) + self.bias

    def activate(self, z: float) -> float:
        """Apply the chosen activation function to z."""
        if self.activation_name == "relu":
            return relu(z)
        if self.activation_name == "sigmoid":
            return sigmoid(z)
        # "linear" = no activation (identity)
        return z

    def forward(self, inputs: list[float]) -> tuple[float, float]:
        """
        Full forward pass.
        Returns (pre_activation z, final output).
        """
        z = self.linear(inputs)
        output = self.activate(z)
        return z, output


def demo_neuron(name: str, neuron: Neuron, inputs: list[float]) -> None:
    """Print a readable trace of one forward pass."""
    z, output = neuron.forward(inputs)

    print(f"\n--- {name} ---")
    print(f"  inputs:     {inputs}")
    print(f"  weights:    {neuron.weights}")
    print(f"  bias:       {neuron.bias}")
    print(f"  activation: {neuron.activation_name}")
    print(f"  z (linear): {z:.4f}")
    print(f"  output:     {output:.4f}")


def main() -> None:
    print("Phase 02: Single neuron — weighted sum, bias, activations")
    print("=" * 60)

    # Same inputs for all demos so you can compare activations
    inputs = [1.0, 2.0, -0.5]

    # Neuron with ReLU — common in hidden layers
    relu_neuron = Neuron(
        num_inputs=3,
        weights=[0.3, -0.8, 1.2],
        bias=0.1,
        activation="relu",
    )
    demo_neuron("ReLU neuron", relu_neuron, inputs)

    # Neuron with Sigmoid — outputs between 0 and 1
    sigmoid_neuron = Neuron(
        num_inputs=3,
        weights=[0.3, -0.8, 1.2],
        bias=0.1,
        activation="sigmoid",
    )
    demo_neuron("Sigmoid neuron", sigmoid_neuron, inputs)

    # No activation — pure linear combination (like Phase 01, but multi-input)
    linear_neuron = Neuron(
        num_inputs=3,
        weights=[0.3, -0.8, 1.2],
        bias=0.1,
        activation="linear",
    )
    demo_neuron("Linear neuron (no activation)", linear_neuron, inputs)

    print("\n" + "=" * 60)
    print("Key idea: stacking many neurons + non-linear activations")
    print("lets a network approximate complex functions.")


if __name__ == "__main__":
    main()
