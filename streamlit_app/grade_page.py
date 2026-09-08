import os
from pathlib import Path

import streamlit as st

from new_students.neural_net import StudentsNet
from new_students.trainer import Trainer


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "new_students"
    / "student_dataset.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "new_students"
    / "students3_model.json"
)


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

def train_grade_model(learning_rate, epochs):
    """Train the grade prediction neural network."""

    network = StudentsNet(
        learning_rate=learning_rate,
        seed=42,
    )

    trainer = Trainer(network)

    trainer.train(
        filename=DATASET_FILE,
        epochs=epochs,
        batch_size=1000,
    )

    network.save(MODEL_FILE)

    return network


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

def load_grade_model():
    """Load the saved grade prediction model."""

    network = StudentsNet()

    network.load(MODEL_FILE)

    return network


# ---------------------------------------------------------
# Grade Prediction Page
# ---------------------------------------------------------

def show_grade_page():

    st.header("🎓 Grade Prediction")

    st.write(
        "Predict the probability of a student receiving "
        "grade A, B, or C based on hours studied and "
        "hours slept."
    )

    st.divider()

    # =====================================================
    # TRAINING SETTINGS
    # =====================================================

    st.sidebar.subheader("Grade Model Settings")

    epochs = st.sidebar.number_input(
        "Epochs",
        min_value=1,
        max_value=10000,
        value=10,
        step=1,
    )

    learning_rate = st.sidebar.number_input(
        "Learning Rate",
        min_value=0.0001,
        max_value=5.0,
        value=0.01,
        step=0.01,
        format="%.4f",
    )

    # =====================================================
    # MODEL STATUS
    # =====================================================

    if os.path.exists(MODEL_FILE):

        st.success(
            "Trained grade prediction model found."
        )

    else:

        st.warning(
            "No trained grade prediction model found."
        )

    # =====================================================
    # TRAINING
    # =====================================================

    st.subheader("Model Training")

    if st.button("Train Grade Prediction Model"):

        with st.spinner(
            "Training grade prediction model..."
        ):

            network = train_grade_model(
                learning_rate=learning_rate,
                epochs=epochs,
            )

        # Store trained model in session state
        st.session_state[
            "grade_network"
        ] = network

        st.session_state[
            "grade_epochs"
        ] = epochs

        st.session_state[
            "grade_learning_rate"
        ] = learning_rate

        st.success(
            "Grade prediction model trained successfully!"
        )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    if "grade_network" in st.session_state:

        network = st.session_state[
            "grade_network"
        ]

    elif os.path.exists(MODEL_FILE):

        network = load_grade_model()

    else:

        network = None

    # =====================================================
    # STUDENT INPUT
    # =====================================================

    st.divider()

    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:

        hours_studied = st.number_input(
            "Hours Studied",
            min_value=0.0,
            value=5.0,
            step=0.5,
        )

    with col2:

        hours_slept = st.number_input(
            "Hours Slept",
            min_value=0.0,
            value=7.0,
            step=0.5,
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button("Predict Grade"):

        if network is None:

            st.error(
                "No trained model is available. "
                "Please train the model first."
            )

            return

        probabilities = network.predict(
            hours_studied,
            hours_slept,
        )

        # -------------------------------------------------
        # Extract probabilities
        # -------------------------------------------------

        probability_a = probabilities["A"]
        probability_b = probabilities["B"]
        probability_c = probabilities["C"]

        # -------------------------------------------------
        # Find grade with highest probability
        # -------------------------------------------------

        predicted_grade = max(
            probabilities,
            key=probabilities.get,
        )

        # =================================================
        # RESULTS
        # =================================================

        st.divider()

        st.subheader("Grade Prediction")

        st.metric(
            "Predicted Grade",
            predicted_grade,
        )

        # =================================================
        # PROBABILITY DISTRIBUTION
        # =================================================

        st.subheader(
            "Grade Probability Distribution"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Grade A",
                f"{probability_a:.2%}",
            )

        with col2:

            st.metric(
                "Grade B",
                f"{probability_b:.2%}",
            )

        with col3:

            st.metric(
                "Grade C",
                f"{probability_c:.2%}",
            )

        # =================================================
        # BAR CHART
        # =================================================

        chart_data = {
            "Grade": ["A", "B", "C"],
            "Probability": [
                probability_a,
                probability_b,
                probability_c,
            ],
        }

        st.bar_chart(
            chart_data,
            x="Grade",
            y="Probability",
        )

        # =================================================
        # PROBABILITY SUM
        # =================================================

        probability_sum = (
            probability_a
            + probability_b
            + probability_c
        )

        st.write(
            f"Probability Sum: "
            f"{probability_sum:.4f}"
        )