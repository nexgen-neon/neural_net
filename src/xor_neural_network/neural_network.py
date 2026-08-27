import math
import random


class XORNeuralNetwork:
    """
    Neural network architecture:

        Input Layer     Hidden Layer     Output Layer

            x1 ────────┐
                       ├── h1 ──────┐
            x2 ────────┘            │
                                    ├── output
            x1 ────────┐            │
                       ├── h2 ──────┘
            x2 ────────┘

    Architecture: 2 -> 2 -> 1
    Activation: Sigmoid
    """

    def __init__(self, learning_rate=0.5, seed=42):
        self.learning_rate = learning_rate

        # Fixed seed makes training reproducible.
        random.seed(seed)

        # Input -> Hidden weights
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)
        self.w3 = random.uniform(-1, 1)
        self.w4 = random.uniform(-1, 1)

        # Hidden -> Output weights
        self.w5 = random.uniform(-1, 1)
        self.w6 = random.uniform(-1, 1)

        # Biases
        self.b1 = random.uniform(-1, 1)
        self.b2 = random.uniform(-1, 1)
        self.b3 = random.uniform(-1, 1)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function:

            sigmoid(x) = 1 / (1 + e^(-x))
        """

        if x < -700:
            return 0.0

        if x > 700:
            return 1.0

        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(output):
        """
        Derivative of sigmoid:

            sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

        Since 'output' is already sigmoid(x):

            sigmoid'(x) = output * (1 - output)
        """

        return output * (1 - output)

    def forward(self, x1, x2):
        """
        Perform forward propagation.
        """

        # -----------------------------
        # Hidden neuron 1
        # -----------------------------

        z1 = (
            x1 * self.w1
            + x2 * self.w2
            + self.b1
        )

        h1 = self.sigmoid(z1)

        # -----------------------------
        # Hidden neuron 2
        # -----------------------------

        z2 = (
            x1 * self.w3
            + x2 * self.w4
            + self.b2
        )

        h2 = self.sigmoid(z2)

        # -----------------------------
        # Output neuron
        # -----------------------------

        z3 = (
            h1 * self.w5
            + h2 * self.w6
            + self.b3
        )

        output = self.sigmoid(z3)

        return {
            "z1": z1,
            "z2": z2,
            "h1": h1,
            "h2": h2,
            "z3": z3,
            "output": output,
        }

    def predict(self, x1, x2):
        """
        Return the network prediction.
        """

        result = self.forward(x1, x2)

        return result["output"]

    def train_one(self, x1, x2, target):
        """
        Train the network using one training example.

        Steps:

        1. Forward propagation
        2. Calculate error
        3. Backpropagation
        4. Update weights and biases
        """

        # ==========================================
        # 1. FORWARD PROPAGATION
        # ==========================================

        result = self.forward(x1, x2)

        h1 = result["h1"]
        h2 = result["h2"]

        output = result["output"]

        # ==========================================
        # 2. CALCULATE LOSS
        # ==========================================

        error = output - target

        loss = error ** 2

        # ==========================================
        # 3. BACKPROPAGATION
        # ==========================================

        # dL/d(output)
        d_loss_output = 2 * (output - target)

        # d(output)/d(z3)
        d_output_z3 = self.sigmoid_derivative(output)

        # dL/d(z3)
        delta_output = (
            d_loss_output
            * d_output_z3
        )

        # ------------------------------------------
        # Output layer gradients
        # ------------------------------------------

        dw5 = delta_output * h1
        dw6 = delta_output * h2

        db3 = delta_output

        # ------------------------------------------
        # Hidden layer gradients
        # ------------------------------------------

        delta_h1 = delta_output * self.w5
        delta_h2 = delta_output * self.w6

        delta_z1 = (
            delta_h1
            * self.sigmoid_derivative(h1)
        )

        delta_z2 = (
            delta_h2
            * self.sigmoid_derivative(h2)
        )

        # ------------------------------------------
        # Input -> Hidden gradients
        # ------------------------------------------

        dw1 = delta_z1 * x1
        dw2 = delta_z1 * x2

        dw3 = delta_z2 * x1
        dw4 = delta_z2 * x2

        db1 = delta_z1
        db2 = delta_z2

        # ==========================================
        # 4. GRADIENT DESCENT
        # ==========================================

        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2

        self.w3 -= self.learning_rate * dw3
        self.w4 -= self.learning_rate * dw4

        self.w5 -= self.learning_rate * dw5
        self.w6 -= self.learning_rate * dw6

        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2
        self.b3 -= self.learning_rate * db3

        return loss

    def get_parameters(self):
        """
        Return all learned weights and biases.
        """

        return {
            "weights": {
                "w1": self.w1,
                "w2": self.w2,
                "w3": self.w3,
                "w4": self.w4,
                "w5": self.w5,
                "w6": self.w6,
            },
            "biases": {
                "b1": self.b1,
                "b2": self.b2,
                "b3": self.b3,
            },
        }