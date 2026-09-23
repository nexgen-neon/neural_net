from grade_drop.csv_reader import read_batches


class Trainer:

    def __init__(self, network):

        self.network = network

    def train(
        self,
        filename,
        epochs=10,
        batch_size=1000
    ):

        for epoch in range(epochs):

            total_loss = 0.0
            total_records = 0

            for batch in read_batches(
                filename,
                batch_size
            ):

                for (
                    inputs,
                    grade_a_target,
                    grade_b_target,
                    grade_c_target

                ) in batch:

                    x1 = inputs[0]
                    x2 = inputs[1]

                    loss = self.network.train(
                        x1,
                        x2,
                        grade_a_target,
                        grade_b_target,
                        grade_c_target
                    )

                    total_loss += loss
                    total_records += 1

            average_loss = (
                total_loss
                / total_records
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {average_loss:.6f}"
            )