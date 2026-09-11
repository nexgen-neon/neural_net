import torch
import torch.nn as nn


class PyTorchXORNeuralNetwork(nn.Module):

    def __init__(
        self,
        learning_rate=0.5,
        seed=42,
    ):
        super().__init__()

        torch.manual_seed(seed)

        self.learning_rate = learning_rate

        # Input -> Hidden layer
        self.hidden_layer = nn.Linear(2, 2)

        # Hidden -> Output layer
        self.output_layer = nn.Linear(2, 1)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # Mean Squared Error
        self.loss_function = nn.MSELoss()

        # Gradient Descent optimizer
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=learning_rate,
        )

    def forward(self, inputs):

        # Input -> Hidden
        hidden = self.hidden_layer(inputs)

        # Sigmoid activation
        hidden = self.sigmoid(hidden)

        # Hidden -> Output
        output = self.output_layer(hidden)

        # Sigmoid activation
        output = self.sigmoid(output)

        return output

    def prediction(self, x1, x2):

        inputs = torch.tensor(
            [[x1, x2]],
            dtype=torch.float32,
        )

        # No gradients are needed for prediction
        with torch.no_grad():

            output = self.forward(inputs)

        return output.item()

    def train_step(
        self,
        x1,
        x2,
        target,
    ):

        inputs = torch.tensor(
            [[x1, x2]],
            dtype=torch.float32,
        )

        target = torch.tensor(
            [[target]],
            dtype=torch.float32,
        )

        # Remove gradients from previous step
        self.optimizer.zero_grad()

        # Forward pass
        prediction = self.forward(inputs)

        # Calculate loss
        loss = self.loss_function(
            prediction,
            target,
        )

        # Backpropagation
        loss.backward()

        # Update weights and biases
        self.optimizer.step()

        return loss.item()

    def parameters_dict(self):

        result = {
            "weights": {},
            "biases": {},
        }

        result["weights"]["hidden"] = (
            self.hidden_layer.weight.detach().tolist()
        )

        result["biases"]["hidden"] = (
            self.hidden_layer.bias.detach().tolist()
        )

        result["weights"]["output"] = (
            self.output_layer.weight.detach().tolist()
        )

        result["biases"]["output"] = (
            self.output_layer.bias.detach().tolist()
        )

        return result