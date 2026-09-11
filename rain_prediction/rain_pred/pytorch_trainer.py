import torch

from .pytorch_neural_net import PyTorchRainNet


class PyTorchRainTrainer:

    def __init__(self, learning_rate=0.01, epochs=10, batch_size=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train, y_train):
        network = PyTorchRainNet(learning_rate=self.learning_rate, seed=42)
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        dataset_size = len(inputs)
        loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            batch_count = 0
            permutation = torch.randperm(dataset_size)
            shuffled_inputs = inputs[permutation]
            shuffled_targets = targets[permutation]

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                total_loss += network.train_batch(
                    shuffled_inputs[start:end],
                    shuffled_targets[start:end],
                )
                batch_count += 1

            epoch_loss = total_loss / batch_count
            loss_history.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.6f}")

        return network, loss_history
