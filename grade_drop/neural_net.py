import math
import json
import random


class StudentsNet:

    def __init__(self, learning_rate=0.01, seed=42):

        self.learning_rate = learning_rate
        self.seed = seed

        random.seed(seed)

        # w1, w2, w3, w4 ~ random values
        self.w1 = random.uniform(-0.5, 0.5)
        self.w2 = random.uniform(-0.5, 0.5)
        self.w3 = random.uniform(-0.5, 0.5)
        self.w4 = random.uniform(-0.5, 0.5)

        # b1, b2 ~ random values
        self.b1 = random.uniform(-0.5, 0.5)
        self.b2 = random.uniform(-0.5, 0.5)

        # w5, w6 ~ random values
        self.w5 = random.uniform(-0.5, 0.5)
        self.w6 = random.uniform(-0.5, 0.5)

        # b3 ~ random value
        self.b3 = random.uniform(-0.5, 0.5)

        # w7, w8 ~ random values
        self.w7 = random.uniform(-0.5, 0.5)
        self.w8 = random.uniform(-0.5, 0.5)

        # b4 ~ random value
        self.b4 = random.uniform(-0.5, 0.5)

        # w9, w10 ~ random values
        self.w9 = random.uniform(-0.5, 0.5)
        self.w10 = random.uniform(-0.5, 0.5)

        # b5 ~ random value
        self.b5 = random.uniform(-0.5, 0.5)

    @staticmethod
    def sigmoid(value):

        # sigmoid(z) = 1 / (1 + e^(-z))

        if value < -700:
            return 0.0

        if value > 700:
            return 1.0

        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def sigmoid_derivative(value):

        # d(sigmoid(z))/dz = sigmoid(z) * (1 - sigmoid(z))

        return value * (1.0 - value)

    @staticmethod
    def softmax(z_a, z_b, z_c):

        # P(A) = e^z_a / (e^z_a + e^z_b + e^z_c)
        # P(B) = e^z_b / (e^z_a + e^z_b + e^z_c)
        # P(C) = e^z_c / (e^z_a + e^z_b + e^z_c)

        maximum = max(z_a, z_b, z_c)

        exp_a = math.exp(z_a - maximum)
        exp_b = math.exp(z_b - maximum)
        exp_c = math.exp(z_c - maximum)

        total = (
            exp_a
            + exp_b
            + exp_c
        )

        return (
            exp_a / total,
            exp_b / total,
            exp_c / total
        )

    @staticmethod
    def softmax_derivative(z_a, z_b, z_c):

        probability_a, probability_b, probability_c = (
            StudentsNet.softmax(
                z_a,
                z_b,
                z_c
            )
        )

        # dA/dz_a = A(1-A)
        # dA/dz_b = -A*B
        # dA/dz_c = -A*C

        # dB/dz_a = -B*A
        # dB/dz_b = B(1-B)
        # dB/dz_c = -B*C

        # dC/dz_a = -C*A
        # dC/dz_b = -C*B
        # dC/dz_c = C(1-C)

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

        # z1 = x1*w1 + x2*w2 + b1

        z1 = (
            x1 * self.w1
            + x2 * self.w2
            + self.b1
        )

        # h1 = sigmoid(z1)

        h1 = self.sigmoid(z1)

        # z2 = x1*w3 + x2*w4 + b2

        z2 = (
            x1 * self.w3
            + x2 * self.w4
            + self.b2
        )

        # h2 = sigmoid(z2)

        h2 = self.sigmoid(z2)

        # z3 = h1*w5 + h2*w6 + b3

        z3 = (
            h1 * self.w5
            + h2 * self.w6
            + self.b3
        )

        # z4 = h1*w7 + h2*w8 + b4

        z4 = (
            h1 * self.w7
            + h2 * self.w8
            + self.b4
        )

        # z5 = h1*w9 + h2*w10 + b5

        z5 = (
            h1 * self.w9
            + h2 * self.w10
            + self.b5
        )

        # A, B, C = softmax(z3, z4, z5)

        probability_a, probability_b, probability_c = (
            self.softmax(
                z3,
                z4,
                z5
            )
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

        # error_A = A - target_A
        # error_B = B - target_B
        # error_C = C - target_C

        error_a = (
            probability_a
            - grade_a_target
        )

        error_b = (
            probability_b
            - grade_b_target
        )

        error_c = (
            probability_c
            - grade_c_target
        )

        # MSE-style loss:
        #
        # total_loss =
        # (A-target_A)^2
        # + (B-target_B)^2
        # + (C-target_C)^2

        loss_a = error_a ** 2
        loss_b = error_b ** 2
        loss_c = error_c ** 2

        total_loss = (
            loss_a
            + loss_b
            + loss_c
        )/3

        # dL/dA = 2(A-target_A)
        # dL/dB = 2(B-target_B)
        # dL/dC = 2(C-target_C)

        d_loss_a = 2.0/3.0 * error_a
        d_loss_b = 2.0/3.0 * error_b
        d_loss_c = 2.0/3.0 * error_c

        # J_softmax =
        # d(A,B,C) / d(z3,z4,z5)

        softmax_jacobian = self.softmax_derivative(
            result["z3"],
            result["z4"],
            result["z5"]
        )

        # dL/dz3 =
        #
        # dL/dA*dA/dz3
        # + dL/dB*dB/dz3
        # + dL/dC*dC/dz3

        dz3 = (
            d_loss_a * softmax_jacobian[0][0]
            + d_loss_b * softmax_jacobian[1][0]
            + d_loss_c * softmax_jacobian[2][0]
        )

        # dL/dz4 =
        #
        # dL/dA*dA/dz4
        # + dL/dB*dB/dz4
        # + dL/dC*dC/dz4

        dz4 = (
            d_loss_a * softmax_jacobian[0][1]
            + d_loss_b * softmax_jacobian[1][1]
            + d_loss_c * softmax_jacobian[2][1]
        )

        # dL/dz5 =
        #
        # dL/dA*dA/dz5
        # + dL/dB*dB/dz5
        # + dL/dC*dC/dz5

        dz5 = (
            d_loss_a * softmax_jacobian[0][2]
            + d_loss_b * softmax_jacobian[1][2]
            + d_loss_c * softmax_jacobian[2][2]
        )

        # dL/dw5 = dL/dz3 * dz3/dw5
        # dz3/dw5 = h1

        dw5 = dz3 * h1

        # dL/dw6 = dL/dz3 * dz3/dw6
        # dz3/dw6 = h2

        dw6 = dz3 * h2

        # dL/db3 = dL/dz3

        db3 = dz3

        # dL/dw7 = dL/dz4 * dz4/dw7
        # dz4/dw7 = h1

        dw7 = dz4 * h1

        # dL/dw8 = dL/dz4 * dz4/dw8
        # dz4/dw8 = h2

        dw8 = dz4 * h2

        # dL/db4 = dL/dz4

        db4 = dz4

        # dL/dw9 = dL/dz5 * dz5/dw9
        # dz5/dw9 = h1

        dw9 = dz5 * h1

        # dL/dw10 = dL/dz5 * dz5/dw10
        # dz5/dw10 = h2

        dw10 = dz5 * h2

        # dL/db5 = dL/dz5

        db5 = dz5

        # dL/dh1 =
        #
        # dL/dz3*w5
        # + dL/dz4*w7
        # + dL/dz5*w9

        dh1 = (
            dz3 * self.w5
            + dz4 * self.w7
            + dz5 * self.w9
        )

        # dL/dh2 =
        #
        # dL/dz3*w6
        # + dL/dz4*w8
        # + dL/dz5*w10

        dh2 = (
            dz3 * self.w6
            + dz4 * self.w8
            + dz5 * self.w10
        )

        # dL/dz1 =
        # dL/dh1 * dh1/dz1
        #
        # dh1/dz1 = h1(1-h1)

        dz1 = (
            dh1
            * self.sigmoid_derivative(h1)
        )

        # dL/dz2 =
        # dL/dh2 * dh2/dz2
        #
        # dh2/dz2 = h2(1-h2)

        dz2 = (
            dh2
            * self.sigmoid_derivative(h2)
        )

        # dL/dw1 = dL/dz1 * x1

        dw1 = dz1 * x1

        # dL/dw2 = dL/dz1 * x2

        dw2 = dz1 * x2

        # dL/db1 = dL/dz1

        db1 = dz1

        # dL/dw3 = dL/dz2 * x1

        dw3 = dz2 * x1

        # dL/dw4 = dL/dz2 * x2

        dw4 = dz2 * x2

        # dL/db2 = dL/dz2

        db2 = dz2

        # w_new = w_old - learning_rate * gradient

        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2

        self.w3 -= self.learning_rate * dw3
        self.w4 -= self.learning_rate * dw4

        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2

        self.w5 -= self.learning_rate * dw5
        self.w6 -= self.learning_rate * dw6
        self.b3 -= self.learning_rate * db3

        self.w7 -= self.learning_rate * dw7
        self.w8 -= self.learning_rate * dw8
        self.b4 -= self.learning_rate * db4

        self.w9 -= self.learning_rate * dw9
        self.w10 -= self.learning_rate * dw10
        self.b5 -= self.learning_rate * db5

        return total_loss

    def save(self, filename):

        model = {
            "learning_rate": self.learning_rate,
            "seed": self.seed,

            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
            "w4": self.w4,

            "b1": self.b1,
            "b2": self.b2,

            "w5": self.w5,
            "w6": self.w6,
            "b3": self.b3,

            "w7": self.w7,
            "w8": self.w8,
            "b4": self.b4,

            "w9": self.w9,
            "w10": self.w10,
            "b5": self.b5
        }

        with open(filename, "w") as file:
            json.dump(
                model,
                file,
                indent=4
            )

    def load(self, filename):

        with open(filename, "r") as file:
            model = json.load(file)

        self.learning_rate = model["learning_rate"]
        self.seed = model.get("seed", 42)

        self.w1 = model["w1"]
        self.w2 = model["w2"]

        self.w3 = model["w3"]
        self.w4 = model["w4"]

        self.b1 = model["b1"]
        self.b2 = model["b2"]

        self.w5 = model["w5"]
        self.w6 = model["w6"]
        self.b3 = model["b3"]

        self.w7 = model["w7"]
        self.w8 = model["w8"]
        self.b4 = model["b4"]

        self.w9 = model["w9"]
        self.w10 = model["w10"]
        self.b5 = model["b5"]