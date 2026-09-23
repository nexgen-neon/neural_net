class Trainer:

    def __init__(self, network, epochs=10):
        self.network = network
        self.epochs = epochs
        self.loss_history = []

    def train(self, X_train, y_train):

        for epoch in range(self.epochs):

            total_loss = 0.0

            for i in range(len(X_train)):

                loss = self.network.train(
                    X_train[i],
                    y_train[i]
                )

                total_loss += loss

            average_loss = (
                total_loss / len(X_train)
            )

            self.loss_history.append(average_loss)

            print(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"Loss: {average_loss}"
            )

        return self.loss_history