from xor_neural_network.neural_network import (
    XORNeuralNetwork,
)


def test_network_prediction_is_between_zero_and_one():
    network = XORNeuralNetwork(
        learning_rate=0.5,
        seed=42,
    )

    prediction = network.predict(
        0,
        1,
    )

    assert 0.0 <= prediction <= 1.0


def test_network_has_parameters():
    network = XORNeuralNetwork(
        learning_rate=0.5,
        seed=42,
    )

    parameters = network.get_parameters()

    assert isinstance(
        parameters,
        dict,
    )

    assert len(parameters) > 0


def test_network_prediction_is_deterministic():
    network1 = XORNeuralNetwork(
        learning_rate=0.5,
        seed=42,
    )

    network2 = XORNeuralNetwork(
        learning_rate=0.5,
        seed=42,
    )

    prediction1 = network1.predict(
        0,
        1,
    )

    prediction2 = network2.predict(
        0,
        1,
    )

    assert prediction1 == prediction2