import torch
import torch.nn as nn


class PyTorchRainNet(nn.Module):
    """A PyTorch version of the five-feature rain probability network."""

    def __init__(self, learning_rate=0.01, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.learning_rate = learning_rate
        self.hidden_layer = nn.Linear(5, 5)
        self.output_layer = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        hidden = self.sigmoid(self.hidden_layer(inputs))
        return self.sigmoid(self.output_layer(hidden))

    def predict(self, inputs):
        inputs = torch.tensor([inputs], dtype=torch.float32)
        with torch.no_grad():
            return self.forward(inputs)[0][0].item()

    def train_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        loss = self.loss_function(self.forward(inputs), targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
