import sys
from pathlib import Path

import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from neural_network.backend import Backend, get_backend
from rain_prediction.rain_pred.cleaning import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_dataset,
)
from rain_prediction.rain_pred.data_audit import audit_dataset
from rain_prediction.rain_pred.data_loader import load_dataset
from rain_prediction.rain_pred.inference import predict
from rain_prediction.rain_pred.neural_net import NeuralNetwork
from rain_prediction.rain_pred.pytorch_neural_net import PyTorchRainNet
from rain_prediction.rain_pred.pytorch_trainer import PyTorchRainTrainer
from rain_prediction.rain_pred.serializer import (
    load_model,
    load_preprocessing,
    load_pytorch_model,
    save_model,
    save_preprocessing,
    save_pytorch_model,
)
from rain_prediction.rain_pred.training import train_model


MODEL_PATH = PROJECT_ROOT / "rain_prediction" / "models" / "rain_prediction_model.json"
PYTORCH_MODEL_PATH = PROJECT_ROOT / "rain_prediction" / "models" / "rain_prediction_pytorch_model.pt"
PREPROCESSING_PATH = PROJECT_ROOT / "rain_prediction" / "models" / "preprocessing.json"
DATA_PATH = PROJECT_ROOT / "rain_prediction" / "data" / "raw" / "weatherAUS.csv"


def prepare_weather_data():
    """Preserve the established split, imputation, and scaling behavior."""
    df = load_dataset(DATA_PATH)
    cleaned_df = clean_dataset(df)
    X = cleaned_df[FEATURE_COLUMNS].values
    y = cleaned_df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # UI values are the raw, imputed training values—not standardized values.
    feature_ranges = {
        feature: {
            "min": float(X_train[:, index].min()),
            "max": float(X_train[:, index].max()),
        }
        for index, feature in enumerate(FEATURE_COLUMNS)
    }

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        df,
        cleaned_df,
        X_train,
        X_test,
        y_train,
        y_test,
        imputer,
        scaler,
        feature_ranges,
    )


def train_weather_model(backend, learning_rate, epochs, batch_size):
    st.info("Loading and preprocessing the weather dataset...")
    (
        df,
        cleaned_df,
        X_train,
        _X_test,
        y_train,
        _y_test,
        imputer,
        scaler,
        feature_ranges,
    ) = prepare_weather_data()

    st.write(f"Original dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    with st.expander("Dataset Audit"):
        audit_dataset(df)
    st.write(f"Cleaned dataset: {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns")
    st.info("Training neural network...")

    if backend == "scratch":
        network, loss_history = train_model(
            X_train.tolist(),
            y_train.tolist(),
            learning_rate=learning_rate,
            epochs=epochs,
        )
        save_model(network, MODEL_PATH)
    else:
        trainer = PyTorchRainTrainer(
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )
        network, loss_history = trainer.train(X_train, y_train)
        save_pytorch_model(network, PYTORCH_MODEL_PATH)

    save_preprocessing(
        FEATURE_COLUMNS,
        imputer,
        scaler,
        PREPROCESSING_PATH,
        feature_ranges,
    )
    return network, load_preprocessing(PREPROCESSING_PATH), loss_history


def get_feature_ranges(preprocessing):
    if "feature_ranges" in preprocessing:
        return preprocessing["feature_ranges"]

    # Old scratch preprocessing files did not persist ranges. Recreate the
    # identical training preprocessing so their UI remains data-constrained.
    return prepare_weather_data()[-1]


def preprocess_input(inputs, preprocessing):
    processed = []
    for index, value in enumerate(inputs):
        if value is None:
            value = preprocessing["imputer_statistics"][index]
        processed.append(
            (value - preprocessing["scaler_mean"][index])
            / preprocessing["scaler_scale"][index]
        )
    return processed


def load_weather_model(backend):
    if backend == "scratch":
        if not MODEL_PATH.exists():
            return None
        return load_model(NeuralNetwork(), MODEL_PATH)

    if not PYTORCH_MODEL_PATH.exists():
        return None
    return load_pytorch_model(PyTorchRainNet(), PYTORCH_MODEL_PATH)


def show_weather_page():
    st.header("🌦️ Weather Prediction")
    st.write("Predict the probability of rain tomorrow from five weather measurements.")
    st.divider()

    implementation = st.sidebar.selectbox(
        "Implementation",
        [Backend.FROM_SCRATCH.value, Backend.PYTORCH.value],
    )
    backend = get_backend(implementation)
    if backend == "scratch":
        st.sidebar.success("Using the from-scratch neural network.")
    else:
        st.sidebar.info("Using the PyTorch neural network.")

    st.sidebar.subheader("Weather Model Settings")
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=10000, value=10, step=1)
    learning_rate = st.sidebar.number_input(
        "Learning Rate", min_value=0.0001, max_value=5.0,
        value=0.01, step=0.01, format="%.4f",
    )
    batch_size = 1000
    if backend == "pytorch":
        batch_size = st.sidebar.number_input(
            "Batch Size", min_value=32, max_value=10000,
            value=1000, step=32,
        )

    model_path = MODEL_PATH if backend == "scratch" else PYTORCH_MODEL_PATH
    if model_path.exists() and PREPROCESSING_PATH.exists():
        st.success("Saved weather model found.")
    else:
        st.warning("No trained weather model was found for this implementation.")

    st.subheader("Model Training")
    if st.button("Train Weather Prediction Model"):
        with st.spinner("Training weather prediction model..."):
            network, preprocessing, loss_history = train_weather_model(
                backend, learning_rate, epochs, batch_size,
            )
        st.session_state["weather_network"] = network
        st.session_state["weather_backend"] = backend
        st.session_state["weather_preprocessing"] = preprocessing
        st.session_state["weather_loss_history"] = loss_history
        st.success("Weather prediction model trained successfully!")

    network = None
    if st.session_state.get("weather_backend") == backend:
        network = st.session_state.get("weather_network")
    if network is None:
        network = load_weather_model(backend)

    preprocessing = st.session_state.get("weather_preprocessing")
    if preprocessing is None and PREPROCESSING_PATH.exists():
        preprocessing = load_preprocessing(PREPROCESSING_PATH)
    if preprocessing is None:
        st.info("Train a weather model to enter weather measurements.")
        return

    feature_ranges = get_feature_ranges(preprocessing)
    st.divider()
    st.subheader("Enter Weather Information")
    labels = [
        "Minimum Temperature (°C)", "Maximum Temperature (°C)",
        "Rainfall (mm)", "Humidity at 3 PM (%)", "Pressure at 3 PM (hPa)",
    ]
    defaults = [15.0, 25.0, 0.0, 50.0, 1010.0]
    values = []
    columns = st.columns(2)
    for index, (feature, label, default) in enumerate(zip(FEATURE_COLUMNS, labels, defaults)):
        bounds = feature_ranges[feature]
        value = min(max(default, bounds["min"]), bounds["max"])
        with columns[index % 2]:
            values.append(st.number_input(
                label,
                min_value=float(bounds["min"]),
                max_value=float(bounds["max"]),
                value=float(value),
            ))

    if st.button("Predict Rain Probability"):
        if network is None:
            st.error("No trained model is available. Please train the model first.")
            return
        probability = predict(network, preprocess_input(values, preprocessing))
        st.subheader("Prediction")
        st.metric("Rain Probability", f"{probability:.2%}")
        st.progress(min(max(probability, 0.0), 1.0))
        if backend == "pytorch" and "weather_loss_history" in st.session_state:
            st.subheader("Training Loss")
            st.line_chart(st.session_state["weather_loss_history"])
