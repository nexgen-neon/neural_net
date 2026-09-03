def mean_squared_error(predictions, expected_values):
    """
    Calculate the mean squared error (MSE).

    MSE = average of (prediction - expected)^2
    """

    if len(predictions) != len(expected_values):
        raise ValueError(
            "Predictions and expected values must have the same length."
        )

    if len(predictions) == 0:
        return 0.0

    total_loss = 0.0

    for prediction, expected in zip(
        predictions,
        expected_values,
    ):
        total_loss += (
            prediction - expected
        ) ** 2

    return total_loss / len(predictions)


def calculate_sample_losses(
    predictions,
    expected_values,
):
    """
    Calculate the individual squared loss
    for every sample.
    """

    if len(predictions) != len(expected_values):
        raise ValueError(
            "Predictions and expected values must have the same length."
        )

    losses = []

    for prediction, expected in zip(
        predictions,
        expected_values,
    ):
        loss = (
            prediction - expected
        ) ** 2

        losses.append(loss)

    return losses