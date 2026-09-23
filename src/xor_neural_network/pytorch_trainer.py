from .loss import mean_squared_error
from .pytorch_neural_net import (
    PyTorchXORNeuralNetwork,
)

class PyTorchXORTrainer:

    def __init__(
        self,
        learning_rate=0.5,
        epochs=10000,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, rows):

        network = PyTorchXORNeuralNetwork(
            learning_rate=self.learning_rate,
            seed=42,
        )

        loss_history = []

        for epoch in range(self.epochs):

            predictions = []
            expected_values = []

            for row in rows:

                x1 = row["x1"]
                x2 = row["x2"]
                expected = row["expected"]

                network.train_step(
                    x1,
                    x2,
                    expected,
                )

                prediction = network.prediction(
                    x1,
                    x2,
                )

                predictions.append(
                    prediction
                )

                expected_values.append(
                    expected
                )

            # Calculate the loss for this epoch
            epoch_loss = mean_squared_error(
                predictions,
                expected_values,
            )

            loss_history.append(
                epoch_loss
            )

        return network, loss_history