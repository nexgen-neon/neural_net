import json


def save_model(network, file_path):

    model_data = {
        "learning_rate": network.learning_rate,
        "hidden_neurons": [],
        "output_neuron": {
            "weights": network.output_neuron.weights,
            "bias": network.output_neuron.bias
        }
    }

    for neuron in network.hidden_neurons:

        model_data["hidden_neurons"].append({
            "weights": neuron.weights,
            "bias": neuron.bias
        })

    with open(file_path, "w") as file:
        json.dump(model_data, file, indent=4)


def load_model(network, file_path):

    with open(file_path, "r") as file:
        model_data = json.load(file)

    network.learning_rate = model_data["learning_rate"]

    for i in range(len(network.hidden_neurons)):

        network.hidden_neurons[i].weights = (
            model_data["hidden_neurons"][i]["weights"]
        )

        network.hidden_neurons[i].bias = (
            model_data["hidden_neurons"][i]["bias"]
        )

    network.output_neuron.weights = (
        model_data["output_neuron"]["weights"]
    )

    network.output_neuron.bias = (
        model_data["output_neuron"]["bias"]
    )

    return network


def save_preprocessing(
    features,
    imputer,
    scaler,
    file_path
):

    preprocessing_data = {
        "features": features,
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }

    with open(file_path, "w") as file:
        json.dump(
            preprocessing_data,
            file,
            indent=4
        )


def load_preprocessing(file_path):

    with open(file_path, "r") as file:
        preprocessing_data = json.load(file)

    return preprocessing_data