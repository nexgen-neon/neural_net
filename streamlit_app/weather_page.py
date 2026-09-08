import os
from neural_network.backend import Backend, get_backend
import streamlit as st
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from rain_prediction.rain_pred.data_loader import load_dataset
from rain_prediction.rain_pred.data_audit import audit_dataset
from rain_prediction.rain_pred.cleaning import clean_dataset
from rain_prediction.rain_pred.training import train_model
from rain_prediction.rain_pred.neural_net import NeuralNetwork
from rain_prediction.rain_pred.inference import predict
from rain_prediction.rain_pred.serializer import (
    save_model,
    load_model,
    save_preprocessing,
    load_preprocessing,
)


MODEL_PATH = "rain_prediction/models/rain_prediction_model.json"
PREPROCESSING_PATH = "rain_prediction/models/preprocessing.json"
DATA_PATH = "rain_prediction/data/raw/weatherAUS.csv"


FEATURE_COLUMNS = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Humidity3pm",
    "Pressure3pm",
]

TARGET_COLUMN = "RainTomorrow"


def train_weather_model():
    """Load, clean, preprocess and train the weather model."""

    st.info("Loading weather dataset...")

    # Load dataset
    df = load_dataset(DATA_PATH)

    st.write(
        f"Original dataset: {df.shape[0]:,} rows × "
        f"{df.shape[1]} columns"
    )

    # Run dataset audit
    with st.expander("Dataset Audit"):
        audit_dataset(df)

    # Clean dataset
    cleaned_df = clean_dataset(df)

    st.write(
        f"Cleaned dataset: {cleaned_df.shape[0]:,} rows × "
        f"{cleaned_df.shape[1]} columns"
    )

    # ---------------------------------------------------------
    # Separate features and target
    # ---------------------------------------------------------

    X = cleaned_df[FEATURE_COLUMNS].values
    y = cleaned_df[TARGET_COLUMN].values

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y,
    )

    # ---------------------------------------------------------
    # Imputation
    # ---------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # ---------------------------------------------------------
    # Standardization
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------------------------------------
    # Train neural network
    # ---------------------------------------------------------

    st.info("Training neural network...")

    network, loss_history = train_model(
        X_train.tolist(),
        y_train.tolist(),
        learning_rate=0.01,
        epochs=10,
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    save_model(
        network,
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # Save preprocessing information
    # ---------------------------------------------------------

    save_preprocessing(
        FEATURE_COLUMNS,
        imputer,
        scaler,
        PREPROCESSING_PATH,
    )

    return network, loss_history


def preprocess_input(inputs, preprocessing):
    """Apply the saved training preprocessing to user input."""

    processed = []

    for i, value in enumerate(inputs):

        # Replace missing value with training median
        if value is None:
            value = preprocessing["imputer_statistics"][i]

        # Standardize using training mean and scale
        value = (
            value - preprocessing["scaler_mean"][i]
        ) / preprocessing["scaler_scale"][i]

        processed.append(value)

    return processed


def show_weather_page():

    st.header("🌦️ Weather Prediction")

    st.write(
        "Predict the probability of rain tomorrow using "
        "a neural network built from scratch."
    )

    st.divider()

    # =========================================================
    # MODEL
    # =========================================================

    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(PREPROCESSING_PATH)
    ):

        st.success("Saved weather model found.")

        network = NeuralNetwork(
            learning_rate=0.01
        )

        network = load_model(
            network,
            MODEL_PATH,
        )

        preprocessing = load_preprocessing(
            PREPROCESSING_PATH,
        )

    else:

        st.warning(
            "No trained weather model was found."
        )

        if st.button("Train Weather Model"):

            network, loss_history = train_weather_model()

            preprocessing = load_preprocessing(
                PREPROCESSING_PATH
            )

            st.session_state["weather_network"] = network

            st.session_state[
                "weather_preprocessing"
            ] = preprocessing

            st.session_state[
                "weather_loss_history"
            ] = loss_history

            st.success(
                "Weather model trained successfully!"
            )

        else:
            st.info(
                "Click 'Train Weather Model' to create "
                "the model."
            )

            return

    # =========================================================
    # SESSION STATE
    # =========================================================

    if "weather_network" in st.session_state:

        network = st.session_state[
            "weather_network"
        ]

    if "weather_preprocessing" in st.session_state:

        preprocessing = st.session_state[
            "weather_preprocessing"
        ]

    # =========================================================
    # INPUTS
    # =========================================================

    st.subheader("Enter Weather Information")

    col1, col2 = st.columns(2)

    with col1:

        min_temp = st.number_input(
            "Minimum Temperature (°C)",
            value=15.0,
        )

        max_temp = st.number_input(
            "Maximum Temperature (°C)",
            value=25.0,
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=0.0,
        )

    with col2:

        humidity_3pm = st.number_input(
            "Humidity at 3 PM (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
        )

        pressure_3pm = st.number_input(
            "Pressure at 3 PM (hPa)",
            value=1010.0,
        )

    # =========================================================
    # PREDICTION
    # =========================================================

    if st.button("Predict Rain Probability"):

        inputs = [
            min_temp,
            max_temp,
            rainfall,
            humidity_3pm,
            pressure_3pm,
        ]

        # Apply the SAME preprocessing used during training
        processed_inputs = preprocess_input(
            inputs,
            preprocessing,
        )

        # Get prediction from neural network
        probability = predict(
            network,
            processed_inputs,
        )

        # =====================================================
        # RESULT
        # =====================================================

        st.subheader("Prediction")

        st.metric(
            "Rain Probability",
            f"{probability:.2%}",
        )

        st.progress(
            min(
                max(probability, 0.0),
                1.0,
            )
        )

    implementation = st.sidebar.selectbox(
        "Implementation",
        [
            Backend.FROM_SCRATCH.value,
            Backend.PYTORCH.value,
        ],
    )
    if implementation == "From Scratch":
    
            st.sidebar.success(
                "Using the from-scratch neural network."
            )
    
    else:
    
            st.sidebar.info(
                "PyTorch implementation will be added "
                "in a future version."
            )