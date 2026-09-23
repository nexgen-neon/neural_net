import torch

from .pytorch_neural_net import (
    PyTorchStudentsNet,
)


class PyTorchStudentsTrainer:

    def __init__(
        self,
        learning_rate=0.01,
        epochs=50,
        batch_size=1000,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    # =====================================================
    # PREPARE DATA
    # =====================================================

    def prepare_data(self, rows):

        inputs = []
        targets = []

        for row in rows:

            inputs.append(
                [
                    row["hours_studied"],
                    row["hours_slept"],
                ]
            )

            targets.append(
                [
                    row["A"],
                    row["B"],
                    row["C"],
                ]
            )

        inputs = torch.tensor(
            inputs,
            dtype=torch.float32,
        )

        targets = torch.tensor(
            targets,
            dtype=torch.float32,
        )

        return inputs, targets

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, rows):

        network = PyTorchStudentsNet(
            learning_rate=self.learning_rate,
            seed=42,
        )

        inputs, targets = self.prepare_data(rows)

        dataset_size = len(inputs)

        loss_history = []

        for epoch in range(self.epochs):

            total_loss = 0.0
            batch_count = 0

            # ---------------------------------------------
            # Shuffle the complete dataset
            # ---------------------------------------------

            permutation = torch.randperm(
                dataset_size
            )

            shuffled_inputs = inputs[
                permutation
            ]

            shuffled_targets = targets[
                permutation
            ]

            # ---------------------------------------------
            # Create mini-batches
            # ---------------------------------------------

            for start in range(
                0,
                dataset_size,
                self.batch_size,
            ):

                # Starting position of this batch
                end = min(
                    start + self.batch_size,
                    dataset_size,
                )

                # Select batch inputs
                batch_inputs = shuffled_inputs[
                    start:end
                ]

                # Select matching batch targets
                batch_targets = shuffled_targets[
                    start:end
                ]

                # -----------------------------------------
                # Train on one complete batch
                # -----------------------------------------

                loss = network.train_batch(
                    batch_inputs,
                    batch_targets,
                )

                # Add this batch's loss
                total_loss += loss

                # Count this batch
                batch_count += 1

            # ---------------------------------------------
            # Average loss for the epoch
            # ---------------------------------------------

            epoch_loss = (
                total_loss
                / batch_count
            )

            loss_history.append(
                epoch_loss
            )

            # ---------------------------------------------
            # Check predictions while training
            # ---------------------------------------------

            prediction_7_5 = network.predict(
                7,
                5,
            )

            prediction_10_5 = network.predict(
                10,
                5,
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{self.epochs} "
                f"- Loss: "
                f"{epoch_loss:.6f}"
            )

            print(
                f"  7h studied, 5h slept -> "
                f"A: {prediction_7_5['A']:.3f}, "
                f"B: {prediction_7_5['B']:.3f}, "
                f"C: {prediction_7_5['C']:.3f}"
            )

            print(
                f"  10h studied, 5h slept -> "
                f"A: {prediction_10_5['A']:.3f}, "
                f"B: {prediction_10_5['B']:.3f}, "
                f"C: {prediction_10_5['C']:.3f}"
            )

        return network, loss_history