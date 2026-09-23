import math
import json
import random

from new_students.neuron import Neuron


class StudentsNet:
    def __init__(self, learning_rate=0.01, seed=42):
        self.learning_rate = learning_rate
        self.seed = seed

        rng = random.Random(seed)

        self.hidden_neuron1 = Neuron(2, rng)
        self.hidden_neuron2 = Neuron(2, rng)
        self.output_neuron_a = Neuron(2, rng)
        self.output_neuron_b = Neuron(2, rng)
        self.output_neuron_c = Neuron(2, rng)

    @staticmethod
    def sigmoid(value):
        if value < -700:
            return 0.0

        if value > 700:
            return 1.0

        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def sigmoid_derivative(value):
       
        return value * (1.0 - value)

    @staticmethod
    def softmax(z_a, z_b, z_c):

        maximum = max(z_a, z_b, z_c)

        exp_a = math.exp(z_a - maximum)
        exp_b = math.exp(z_b - maximum)
        exp_c = math.exp(z_c - maximum)

        total = exp_a + exp_b + exp_c

        probability_a = exp_a / total
        probability_b = exp_b / total
        probability_c = exp_c / total

        return probability_a, probability_b, probability_c

    @staticmethod
    def softmax_derivative(z_a, z_b, z_c):
        probability_a, probability_b, probability_c = StudentsNet.softmax(
            z_a,
            z_b,
            z_c
        )

        return [
            [
                probability_a * (1.0 - probability_a),
                -probability_a * probability_b,
                -probability_a * probability_c
            ],
            [
                -probability_b * probability_a,
                probability_b * (1.0 - probability_b),
                -probability_b * probability_c
            ],
            [
                -probability_c * probability_a,
                -probability_c * probability_b,
                probability_c * (1.0 - probability_c)
            ]
        ]

    def forward(self, x1, x2):


        inputs = [x1, x2]

        # Neuron 1:
        # z1 = x1*w1 + x2*w2 + b1
        z1 = self.hidden_neuron1.calculate(inputs)

        # h1 = sigmoid(z1)
        h1 = self.sigmoid(z1)

        # Neuron 2:
        # z2 = x1*w3 + x2*w4 + b2
        z2 = self.hidden_neuron2.calculate(inputs)

        # h2 = sigmoid(z2)
        h2 = self.sigmoid(z2)


        hidden_outputs = [h1, h2]

        # Output neuron A:
        # z3 = h1*w5 + h2*w6 + b3
        z3 = self.output_neuron_a.calculate(hidden_outputs)

        # Output neuron B:
        # z4 = h1*w7 + h2*w8 + b4
        z4 = self.output_neuron_b.calculate(hidden_outputs)

        # Output neuron C:
        # z5 = h1*w9 + h2*w10 + b5
        z5 = self.output_neuron_c.calculate(hidden_outputs)

      
        probability_a, probability_b, probability_c = self.softmax(
            z3,
            z4,
            z5
        )

        return {
            "z1": z1,
            "z2": z2,
            "h1": h1,
            "h2": h2,
            "z3": z3,
            "z4": z4,
            "z5": z5,
            "A": probability_a,
            "B": probability_b,
            "C": probability_c
        }

    def predict(self, x1, x2):
        result = self.forward(x1, x2)

        return {
            "A": result["A"],
            "B": result["B"],
            "C": result["C"]
        }

    def train(
        self,
        x1,
        x2,
        grade_a_target,
        grade_b_target,
        grade_c_target
    ):
       

        result = self.forward(x1, x2)

        h1 = result["h1"]
        h2 = result["h2"]

        probability_a = result["A"]
        probability_b = result["B"]
        probability_c = result["C"]


        error_a = probability_a - grade_a_target
        error_b = probability_b - grade_b_target
        error_c = probability_c - grade_c_target

        loss_a = error_a ** 2
        loss_b = error_b ** 2
        loss_c = error_c ** 2

      
        total_loss = (loss_a + loss_b + loss_c)/3



        d_loss_a = 2.0/3.0 * error_a
        d_loss_b = 2.0/3.0 * error_b
        d_loss_c = 2.0/3.0 * error_c


        softmax_jacobian = self.softmax_derivative(
            result["z3"],
            result["z4"],
            result["z5"]
        )


        # dz3
        dz3 = (
            d_loss_a * softmax_jacobian[0][0] 
            + d_loss_b * softmax_jacobian[1][0]
            + d_loss_c * softmax_jacobian[2][0]
        )

        # dz4
        dz4 = (
            d_loss_a * softmax_jacobian[0][1]
            + d_loss_b * softmax_jacobian[1][1]
            + d_loss_c * softmax_jacobian[2][1]
        )

        # dz5
        dz5 = (
            d_loss_a * softmax_jacobian[0][2]
            + d_loss_b * softmax_jacobian[1][2]
            + d_loss_c * softmax_jacobian[2][2]
        )

        dw5 = dz3 * h1
        dw6 = dz3 * h2
        db3 = dz3


        dw7 = dz4 * h1
        dw8 = dz4 * h2
        db4 = dz4


        dw9 = dz5 * h1
        dw10 = dz5 * h2
        db5 = dz5


        self.output_neuron_a.update(
            [dw5, dw6],
            db3,
            self.learning_rate
        )

        self.output_neuron_b.update(
            [dw7, dw8],
            db4,
            self.learning_rate
        )

        self.output_neuron_c.update(
            [dw9, dw10],
            db5,
            self.learning_rate
        )

  

        dh1 = (
            dz3 * self.output_neuron_a.weights[0]
            + dz4 * self.output_neuron_b.weights[0]
            + dz5 * self.output_neuron_c.weights[0]
        )

        dh2 = (
            dz3 * self.output_neuron_a.weights[1]
            + dz4 * self.output_neuron_b.weights[1]
            + dz5 * self.output_neuron_c.weights[1]
        )

        # z1 -> sigmoid -> h1
        dz1 = dh1 * self.sigmoid_derivative(h1)

        # z2 -> sigmoid -> h2
        dz2 = dh2 * self.sigmoid_derivative(h2)

        # Hidden neuron 1
        dw1 = dz1 * x1
        dw2 = dz1 * x2
        db1 = dz1

        # Hidden neuron 2
        dw3 = dz2 * x1
        dw4 = dz2 * x2
        db2 = dz2

        self.hidden_neuron1.update(
            [dw1, dw2],
            db1,
            self.learning_rate
        )

        self.hidden_neuron2.update(
            [dw3, dw4],
            db2,
            self.learning_rate
        )

        return total_loss

    def save(self, filename):
        model = {
            "learning_rate": self.learning_rate,
            "seed": self.seed,

            "hidden_neuron1": {
                "weights": self.hidden_neuron1.weights,
                "bias": self.hidden_neuron1.bias
            },

            "hidden_neuron2": {
                "weights": self.hidden_neuron2.weights,
                "bias": self.hidden_neuron2.bias
            },

            "output_neuron_a": {
                "weights": self.output_neuron_a.weights,
                "bias": self.output_neuron_a.bias
            },

            "output_neuron_b": {
                "weights": self.output_neuron_b.weights,
                "bias": self.output_neuron_b.bias
            },

            "output_neuron_c": {
                "weights": self.output_neuron_c.weights,
                "bias": self.output_neuron_c.bias
            }
        }

        with open(filename, "w") as file:
            json.dump(model, file, indent=4)

    def load(self, filename):
        with open(filename, "r") as file:
            model = json.load(file)

        self.learning_rate = model["learning_rate"]
        self.seed = model.get("seed", 42)

        self.hidden_neuron1.weights = model["hidden_neuron1"]["weights"]
        self.hidden_neuron1.bias = model["hidden_neuron1"]["bias"]

        self.hidden_neuron2.weights = model["hidden_neuron2"]["weights"]
        self.hidden_neuron2.bias = model["hidden_neuron2"]["bias"]

        self.output_neuron_a.weights = model["output_neuron_a"]["weights"]
        self.output_neuron_a.bias = model["output_neuron_a"]["bias"]

        self.output_neuron_b.weights = model["output_neuron_b"]["weights"]
        self.output_neuron_b.bias = model["output_neuron_b"]["bias"]

        self.output_neuron_c.weights = model["output_neuron_c"]["weights"]
        self.output_neuron_c.bias = model["output_neuron_c"]["bias"]