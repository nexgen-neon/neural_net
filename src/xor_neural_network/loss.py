def squared_error(predicted, expected):
    """
    Calculate squared error for one prediction.

    Formula:

        Loss = (predicted - expected)^2
    """

    return (predicted - expected) ** 2


def mean_squared_error(predictions, expected):
    """
    Calculate Mean Squared Error.

    Formula:

        MSE = sum((prediction - expected)^2) / n
    """

    if not predictions:
        return 0.0

    total_loss = 0.0

    for predicted, target in zip(
        predictions,
        expected,
    ):
        total_loss += squared_error(
            predicted,
            target,
        )

    return total_loss / len(predictions)


def calculate_sample_losses(predictions, expected):
    """
    Calculate the loss for each individual sample.

    Returns a list containing one loss value
    for each prediction.
    """

    losses = []

    for predicted, target in zip(
        predictions,
        expected,
    ):
        loss = squared_error(
            predicted,
            target,
        )

        losses.append(loss)

    return losses