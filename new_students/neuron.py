import random


class Neuron:
    def __init__(self, number_of_inputs, rng):
        self.weights = []

        for _ in range(number_of_inputs):
            self.weights.append(rng.uniform(-0.5, 0.5))

        self.bias = rng.uniform(-0.5, 0.5)

    def calculate(self, inputs):
        z = self.bias

        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]

        return z

    def update(self, weight_gradients, bias_gradient, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]

        self.bias -= learning_rate * bias_gradient