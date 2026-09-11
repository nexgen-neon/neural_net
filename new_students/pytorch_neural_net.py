import torch
import torch.nn as nn


class PyTorchStudentsNet(nn.Module):

    def __init__(
        self,
        learning_rate=0.01,
        seed=42,
    ):
        super().__init__()

        torch.manual_seed(seed)

        self.learning_rate = learning_rate

        # 2 inputs -> 2 hidden neurons
        self.hidden_layer = nn.Linear(2, 2)

        # 2 hidden neurons -> 3 output neurons
        self.output_layer = nn.Linear(2, 3)

        # Sigmoid activation for hidden layer
        self.sigmoid = nn.Sigmoid()

        # Convert output logits into A/B/C probabilities
        self.softmax = nn.Softmax(dim=1)

        # Mean Squared Error
        self.loss_function = nn.MSELoss()

        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
        )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, inputs):

        # Input -> hidden layer
        hidden = self.hidden_layer(inputs)

        # Sigmoid activation
        hidden = self.sigmoid(hidden)

        # Hidden -> output layer
        output = self.output_layer(hidden)

        # Softmax -> probabilities
        output = self.softmax(output)

        return output

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(
        self,
        hours_studied,
        hours_slept,
    ):

        inputs = torch.tensor(
            [
                [
                    hours_studied,
                    hours_slept,
                ]
            ],
            dtype=torch.float32,
        )

        with torch.no_grad():

            output = self.forward(inputs)

        return {
            "A": output[0][0].item(),
            "B": output[0][1].item(),
            "C": output[0][2].item(),
        }

    # =====================================================
    # BATCH TRAINING
    # =====================================================

    def train_batch(
        self,
        inputs,
        targets,
    ):

        # Clear gradients from previous batch
        self.optimizer.zero_grad()

        # Forward pass for the complete batch
        predictions = self.forward(inputs)

        # Calculate average MSE for the batch
        loss = self.loss_function(
            predictions,
            targets,
        )

        # Calculate gradients
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss.item()

    # =====================================================
    # PARAMETERS
    # =====================================================

    def parameters_dict(self):

        return {
            "weights": {
                "hidden": (
                    self.hidden_layer
                    .weight
                    .detach()
                    .tolist()
                ),

                "output": (
                    self.output_layer
                    .weight
                    .detach()
                    .tolist()
                ),
            },

            "biases": {
                "hidden": (
                    self.hidden_layer
                    .bias
                    .detach()
                    .tolist()
                ),

                "output": (
                    self.output_layer
                    .bias
                    .detach()
                    .tolist()
                ),
            },
        }