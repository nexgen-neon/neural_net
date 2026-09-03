import random
import math


class XORNeuralNetwork:

    def __init__(
        self,
        learning_rate=0.5,
        seed=42,
    ):

        self.learning_rate = learning_rate

        random.seed(seed)

        # Input -> Hidden 1
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)

        # Input -> Hidden 2
        self.w3 = random.uniform(-1, 1)
        self.w4 = random.uniform(-1, 1)

        # Hidden -> Output
        self.w5 = random.uniform(-1, 1)
        self.w6 = random.uniform(-1, 1)

        # Biases
        self.bias1 = random.uniform(-1, 1)
        self.bias2 = random.uniform(-1, 1)
        self.bias3 = random.uniform(-1, 1)

    @staticmethod
    def sigmoid(value):

        if value < -700:
            return 0.0

        if value > 700:
            return 1.0

        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def sigmoid_derive(value):

        return value * (1.0 - value)

    def forward(self, x1, x2):

        # Hidden neuron 1
        z1 = (
            x1 * self.w1
            + x2 * self.w2
            + self.bias1
        )

        h1 = self.sigmoid(z1)

        # Hidden neuron 2
        z2 = (
            x1 * self.w3
            + x2 * self.w4
            + self.bias2
        )

        h2 = self.sigmoid(z2)

        # Output neuron
        z3 = (
            h1 * self.w5
            + h2 * self.w6
            + self.bias3
        )

        output = self.sigmoid(z3)

        return {
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "h1": h1,
            "h2": h2,
            "output": output,
        }

    def prediction(self, x1, x2):

        result = self.forward(
            x1,
            x2,
        )

        return result["output"]

    def train(
        self,
        x1,
        x2,
        target,
    ):

       
        # Forward pass

        result = self.forward(
            x1,
            x2,
        )

        h1 = result["h1"]
        h2 = result["h2"]
        output = result["output"]

      
        # Loss
       
        error = output - target

        loss = error ** 2

        # dL/d(output)
        loss_grad = 2 * (
            output - target
        )

     
        # Output layer

        d_z3 = self.sigmoid_derive(
            output
        )

        delta_output = (
            loss_grad * d_z3
        )

        # Gradients for w5 and w6
        dw5 = delta_output * h1
        dw6 = delta_output * h2

        db3 = delta_output

        
        # Hidden layer

        delta_h1 = (
            delta_output * self.w5
        )

        delta_h2 = (
            delta_output * self.w6
        )

        sigmoid_h1_deriv = (
            self.sigmoid_derive(h1)
        )

        sigmoid_h2_deriv = (
            self.sigmoid_derive(h2)
        )

        delta_z1 = (
            delta_h1
            * sigmoid_h1_deriv
        )

        delta_z2 = (
            delta_h2
            * sigmoid_h2_deriv
        )

        
        # Input -> Hidden gradients

        dw1 = delta_z1 * x1
        dw2 = delta_z1 * x2

        dw3 = delta_z2 * x1
        dw4 = delta_z2 * x2

        db1 = delta_z1
        db2 = delta_z2

        # Gradient descent

        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2

        self.w3 -= self.learning_rate * dw3
        self.w4 -= self.learning_rate * dw4

        self.w5 -= self.learning_rate * dw5
        self.w6 -= self.learning_rate * dw6

        self.bias1 -= self.learning_rate * db1
        self.bias2 -= self.learning_rate * db2
        self.bias3 -= self.learning_rate * db3

        return loss

    def parameters(self):

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
                "b1": self.bias1,
                "b2": self.bias2,
                "b3": self.bias3,
            },
        }