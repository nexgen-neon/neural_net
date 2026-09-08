from rain_pred.neural_net import NeuralNetwork
from rain_pred.trainer import Trainer


def train_model(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=10
):

    network = NeuralNetwork(
        learning_rate=learning_rate
    )

    trainer = Trainer(
        network=network,
        epochs=epochs
    )

    loss_history = trainer.train(
        X_train,
        y_train
    )

    return network, loss_history