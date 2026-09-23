import os
import sys
import csv
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Backend
# ---------------------------------------------------------

from neural_network.backend import (
    Backend,
    get_backend,
)


# ---------------------------------------------------------
# From Scratch
# ---------------------------------------------------------

from new_students.neural_net import (
    StudentsNet,
)

from new_students.trainer import (
    Trainer,
)


# ---------------------------------------------------------
# PyTorch
# ---------------------------------------------------------

from new_students.pytorch_trainer import (
    PyTorchStudentsTrainer,
)


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

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


# =========================================================
# FROM SCRATCH - TRAIN
# =========================================================

def train_grade_model(
    learning_rate,
    epochs,
):
    """Train the from-scratch grade model."""

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


# =========================================================
# FROM SCRATCH - LOAD
# =========================================================

def load_grade_model():

    network = StudentsNet()

    network.load(MODEL_FILE)

    return network


# =========================================================
# PYTORCH - READ DATA
# =========================================================

def read_grade_dataset():

    rows = []

    with open(
        DATASET_FILE,
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            grade = row["grade"]

            rows.append(
                {
                    "hours_studied": float(
                        row["hours_studied"]
                    ),

                    "hours_slept": float(
                        row["hours_slept"]
                    ),

                    # One-hot encoding
                    "A": (
                        1.0
                        if grade == "A"
                        else 0.0
                    ),

                    "B": (
                        1.0
                        if grade == "B"
                        else 0.0
                    ),

                    "C": (
                        1.0
                        if grade == "C"
                        else 0.0
                    ),
                }
            )

    return rows


# =========================================================
# PYTORCH - TRAIN
# =========================================================

def train_pytorch_grade_model(
    learning_rate,
    epochs,
    batch_size,
):

    trainer = PyTorchStudentsTrainer(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
    )

    rows = read_grade_dataset()

    network, loss_history = trainer.train(
        rows
    )

    return network, loss_history


# =========================================================
# GRADE PAGE
# =========================================================

def show_grade_page():

    st.header("🎓 Grade Prediction")

    st.write(
        "Predict the probability of a student receiving "
        "grade A, B, or C based on hours studied and "
        "hours slept."
    )

    st.divider()

    # =====================================================
    # IMPLEMENTATION
    # =====================================================

    implementation = st.sidebar.selectbox(
        "Implementation",
        [
            Backend.FROM_SCRATCH.value,
            Backend.PYTORCH.value,
        ],
    )

    backend = get_backend(
        implementation
    )

    if backend == "scratch":

        st.sidebar.success(
            "Using the from-scratch neural network."
        )

    else:

        st.sidebar.info(
            "Using the PyTorch neural network."
        )

    # =====================================================
    # TRAINING SETTINGS
    # =====================================================

    st.sidebar.subheader(
        "Grade Model Settings"
    )

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

    # -----------------------------------------------------
    # Batch size only for PyTorch
    # -----------------------------------------------------

    if backend == "pytorch":

        batch_size = st.sidebar.number_input(
            "Batch Size",
            min_value=32,
            max_value=10000,
            value=1000,
            step=32,
        )

    else:

        batch_size = 1000

    # =====================================================
    # MODEL STATUS
    # =====================================================

    if backend == "scratch":

        if os.path.exists(MODEL_FILE):

            st.success(
                "Trained grade prediction model found."
            )

        else:

            st.warning(
                "No trained grade prediction model found."
            )

    else:

        st.info(
            "PyTorch model will be trained "
            "during this session."
        )

    # =====================================================
    # TRAINING
    # =====================================================

    st.subheader(
        "Model Training"
    )

    if st.button(
        "Train Grade Prediction Model"
    ):

        with st.spinner(
            "Training grade prediction model..."
        ):

            # ---------------------------------------------
            # From Scratch
            # ---------------------------------------------

            if backend == "scratch":

                network = train_grade_model(
                    learning_rate=learning_rate,
                    epochs=epochs,
                )

                loss_history = None

            # ---------------------------------------------
            # PyTorch
            # ---------------------------------------------

            else:

                (
                    network,
                    loss_history,
                ) = train_pytorch_grade_model(
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                )

        # -------------------------------------------------
        # Store model
        # -------------------------------------------------

        st.session_state[
            "grade_network"
        ] = network

        st.session_state[
            "grade_backend"
        ] = backend

        st.session_state[
            "grade_epochs"
        ] = epochs

        st.session_state[
            "grade_learning_rate"
        ] = learning_rate

        # -------------------------------------------------
        # Store loss history
        # -------------------------------------------------

        if loss_history is not None:

            st.session_state[
                "grade_loss_history"
            ] = loss_history

        st.success(
            "Grade prediction model trained successfully!"
        )

    # =====================================================
    # GET MODEL
    # =====================================================

    network = None

    if (
        "grade_network" in st.session_state
        and st.session_state.get(
            "grade_backend"
        ) == backend
    ):

        network = st.session_state[
            "grade_network"
        ]

    elif (
        backend == "scratch"
        and os.path.exists(MODEL_FILE)
    ):

        network = load_grade_model()

    # =====================================================
    # STUDENT INPUT
    # =====================================================

    st.divider()

    st.subheader(
        "Student Information"
    )

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

    if st.button(
        "Predict Grade"
    ):

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

        probability_a = probabilities[
            "A"
        ]

        probability_b = probabilities[
            "B"
        ]

        probability_c = probabilities[
            "C"
        ]

        # -------------------------------------------------
        # Highest probability
        # -------------------------------------------------

        predicted_grade = max(
            probabilities,
            key=probabilities.get,
        )

        # =================================================
        # RESULTS
        # =================================================

        st.divider()

        st.subheader(
            "Grade Prediction"
        )

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
            "Grade": [
                "A",
                "B",
                "C",
            ],

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

        # =================================================
        # PYTORCH LOSS
        # =================================================

        if (
            backend == "pytorch"
            and "grade_loss_history"
            in st.session_state
        ):

            st.subheader(
                "Training Loss"
            )

            st.line_chart(
                st.session_state[
                    "grade_loss_history"
                ]
            )