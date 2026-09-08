import json


def create_result_json(
    truth_table,
    predictions,
    losses,
    overall_loss,
    network,
    epochs,
    learning_rate,
):
    """
    Create a JSON representation of the
    neural-network training results.
    """

    results = []

    for row, prediction, loss in zip(
        truth_table,
        predictions,
        losses,
    ):

        predicted_class = (
            1 if prediction >= 0.5 else 0
        )

        results.append(
            {
                "input": {
                    "x1": row["x1"],
                    "x2": row["x2"],
                },
                "expected_output": row["expected"],
                "predicted_output": prediction,
                "predicted_class": predicted_class,
                "loss": loss,
            }
        )

    data = {
        "model": {
            "type": "feedforward_neural_network",
            "architecture": "2-2-1",
            "activation": "sigmoid",
            "optimizer": "gradient_descent",
        },
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "truth_table": truth_table,
        "results": results,
        "overall_mse": overall_loss,
        "parameters": network.parameters(),
    }

    return json.dumps(
        data,
        indent=4,
    )