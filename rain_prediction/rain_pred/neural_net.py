import math
import random


class Neuron:

    def __init__(self, input_count):
        self.weights = [
            random.uniform(-0.5, 0.5)
            for _ in range(input_count)
        ]

        self.bias = random.uniform(-0.5, 0.5)

        self.z = 0.0
        self.output = 0.0
        self.delta = 0.0

    @staticmethod
    def sigmoid(z):

        if z > 700:
            z = 700

        if z < -700:
            z = -700

        # sigmoid(z) = 1 / (1 + e^(-z))
        return 1 / (1 + math.exp(-z))

    def forward(self, inputs):

        # z = sum(x_i * w_i) + b
        self.z = self.bias

        for i in range(len(inputs)):
            self.z += inputs[i] * self.weights[i]

        # output = sigmoid(z)
        self.output = self.sigmoid(self.z)

        return self.output


class NeuralNetwork:

    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate

        self.hidden_neurons = [
            Neuron(5)
            for _ in range(5)
        ]

        self.output_neuron = Neuron(5)

    def forward(self, inputs):

        hidden_outputs = []

        for neuron in self.hidden_neurons:
            output = neuron.forward(inputs)
            hidden_outputs.append(output)

        prediction = self.output_neuron.forward(
            hidden_outputs
        )

        self.hidden_outputs = hidden_outputs

        return prediction

    def backward(self, inputs, target):

        prediction = self.output_neuron.output

        # L = 1/2 * (prediction - target)^2
        # dL/d_prediction = prediction - target
        d_loss_prediction = prediction - target

        # sigmoid'(z) = output * (1 - output)
        sigmoid_derivative = (
            prediction * (1 - prediction)
        )

        # delta_output = dL/d_prediction * sigmoid'(z)
        self.output_neuron.delta = (
            d_loss_prediction
            * sigmoid_derivative
        )

        old_output_weights = (
            self.output_neuron.weights.copy()
        )

        for i in range(5):

            # dL/dw = delta_output * hidden_output
            gradient = (
                self.output_neuron.delta
                * self.hidden_outputs[i]
            )

            # w = w - learning_rate * gradient
            self.output_neuron.weights[i] -= (
                self.learning_rate * gradient
            )

        # dL/db = delta_output
        # b = b - learning_rate * gradient
        self.output_neuron.bias -= (
            self.learning_rate
            * self.output_neuron.delta
        )

        for hidden_index, hidden_neuron in enumerate(
            self.hidden_neurons
        ):

            # dL/d_hidden = delta_output * output_weight
            d_loss_hidden = (
                self.output_neuron.delta
                * old_output_weights[hidden_index]
            )

            # sigmoid'(z) = output * (1 - output)
            hidden_sigmoid_derivative = (
                hidden_neuron.output
                * (1 - hidden_neuron.output)
            )

            # delta_hidden = dL/d_hidden * sigmoid'(z)
            hidden_neuron.delta = (
                d_loss_hidden
                * hidden_sigmoid_derivative
            )

            for input_index in range(len(inputs)):

                # dL/dw = delta_hidden * input
                gradient = (
                    hidden_neuron.delta
                    * inputs[input_index]
                )

                # w = w - learning_rate * gradient
                hidden_neuron.weights[input_index] -= (
                    self.learning_rate * gradient
                )

            # dL/db = delta_hidden
            # b = b - learning_rate * gradient
            hidden_neuron.bias -= (
                self.learning_rate
                * hidden_neuron.delta
            )

    def train(self, inputs, target):

        prediction = self.forward(inputs)

        # L = 1/2 * (prediction - target)^2
        error = prediction - target
        loss = 0.5 * (error ** 2)

        self.backward(inputs, target)

        return loss

    def predict(self, inputs):

        return self.forward(inputs)