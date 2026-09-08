from abc import ABC, abstractmethod


class BaseNeuralNetwork(ABC):

    @abstractmethod
    def forward(self, inputs):
        """Run forward propagation."""
        pass

    @abstractmethod
    def predict(self, inputs):
        """Make a prediction."""
        pass

    @abstractmethod
    def train(self, inputs, targets):
        """Train the network on one sample or batch."""
        pass

    @abstractmethod
    def get_parameters(self):
        """Return the model parameters."""
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        """Load model parameters."""
        pass

    @abstractmethod
    def get_loss_history(self):
        """Return training loss history."""
        pass