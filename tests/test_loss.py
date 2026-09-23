import pytest

from xor_neural_network.loss import (
    calculate_sample_losses,
    mean_squared_error,
)


def test_mean_squared_error():
    predictions = [0.0, 1.0, 0.5, 0.2]
    expected = [0.0, 1.0, 1.0, 0.0]

    mse = mean_squared_error(
        predictions,
        expected,
    )

    expected_mse = (
        (0.0 - 0.0) ** 2
        + (1.0 - 1.0) ** 2
        + (0.5 - 1.0) ** 2
        + (0.2 - 0.0) ** 2
    ) / 4

    assert mse == expected_mse


def test_sample_losses():
    predictions = [0.0, 1.0, 0.5, 0.2]
    expected = [0.0, 1.0, 1.0, 0.0]

    losses = calculate_sample_losses(
        predictions,
        expected,
    )

    expected_losses = [
        0.0,
        0.0,
        0.25,
        0.04,
    ]

    for actual, expected_loss in zip(
        losses,
        expected_losses,
    ):
        assert actual == pytest.approx(
            expected_loss
        )