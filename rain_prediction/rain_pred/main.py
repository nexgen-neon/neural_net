import os

from rain_pred.data_loader import load_dataset
from rain_pred.data_audit import audit_dataset
from rain_pred.cleaning import clean_dataset
from rain_pred.preprocessing import (
    prepare_data,
    preprocess_input
)
from rain_pred.training import train_model
from rain_pred.inference import predict
from rain_pred.serializer import (
    save_model,
    load_model,
    save_preprocessing,
    load_preprocessing
)
from rain_pred.neural_net import NeuralNetwork


MODEL_PATH = "models/rain_prediction_model.json"
PREPROCESSING_PATH = "models/preprocessing.json"


def main():

    print("RAIN PREDICTION")

    model_exists = os.path.exists(MODEL_PATH)
    preprocessing_exists = os.path.exists(
        PREPROCESSING_PATH
    )

    if not model_exists or not preprocessing_exists:

        print()
        print("No saved model found.")
        print("Training model...")

        file_path = "data/raw/weatherAUS.csv"

        df = load_dataset(file_path)

        audit_dataset(df)

        cleaned_df = clean_dataset(df)

        print()
        print(
            f"Cleaned shape: {cleaned_df.shape}"
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
            imputer,
            scaler
        ) = prepare_data(cleaned_df)

        print()
        print("Data shapes:")
        print("X_train :", X_train.shape)
        print("y_train :", y_train.shape)
        print("X_test  :", X_test.shape)
        print("y_test  :", y_test.shape)

        network, loss_history = train_model(
            X_train,
            y_train,
            learning_rate=0.01,
            epochs=10
        )

        save_model(
            network,
            MODEL_PATH
        )

        save_preprocessing(
            features=[
                "MinTemp",
                "MaxTemp",
                "Rainfall",
                "Humidity3pm",
                "Pressure3pm"
            ],
            imputer=imputer,
            scaler=scaler,
            file_path=PREPROCESSING_PATH
        )

        preprocessing = {
            "features": [
                "MinTemp",
                "MaxTemp",
                "Rainfall",
                "Humidity3pm",
                "Pressure3pm"
            ],
            "imputer_statistics": (
                imputer.statistics_.tolist()
            ),
            "scaler_mean": (
                scaler.mean_.tolist()
            ),
            "scaler_scale": (
                scaler.scale_.tolist()
            )
        }

        print()
        print("Model trained and saved.")

    else:

        print()
        print("Saved model found.")
        print("Loading model...")

        network = NeuralNetwork()

        load_model(
            network,
            MODEL_PATH
        )

        preprocessing = load_preprocessing(
            PREPROCESSING_PATH
        )

        print("Model loaded successfully.")

    print()
    print("Enter weather information:")
    print()

    min_temp = float(
        input("MinTemp: ")
    )

    max_temp = float(
        input("MaxTemp: ")
    )

    rainfall = float(
        input("Rainfall: ")
    )

    humidity_3pm = float(
        input("Humidity3pm: ")
    )

    pressure_3pm = float(
        input("Pressure3pm: ")
    )

    inputs = [
        min_temp,
        max_temp,
        rainfall,
        humidity_3pm,
        pressure_3pm
    ]

    processed_inputs = preprocess_input(
        inputs,
        preprocessing
    )

    probability = predict(
        network,
        processed_inputs
    )

    print()
    print("RAIN PREDICTION")
    print("----------------")

    print(
        f"Rain probability: {probability:.2%}"
    )


if __name__ == "__main__":
    main()