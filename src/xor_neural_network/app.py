import streamlit as st

from xor_neural_network.loss import (
    calculate_sample_losses,
    mean_squared_error,
)

from xor_neural_network.serializer import (
    create_result_json,
)

from xor_neural_network.trainer import (
    XORTrainer,
)

from xor_neural_network.truth_table import (
    DEFAULT_XOR_TABLE,
    is_xor_table,
    validate_truth_table,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="XOR Neural Network",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# Title
# ============================================================

st.title("🧠 XOR Gate Using Neural Network")

st.write(
    "A neural network implemented completely "
    "from scratch using Python."
)

st.info(
    "No PyTorch, NumPy, TensorFlow, "
    "or scikit-learn is used."
)


# ============================================================
# Sidebar - Training Settings
# ============================================================

st.sidebar.header("Training Settings")

epochs = st.sidebar.number_input(
    "Number of Epochs",
    min_value=1,
    max_value=100000,
    value=10000,
    step=1000,
)

learning_rate = st.sidebar.number_input(
    "Learning Rate",
    min_value=0.001,
    max_value=5.0,
    value=0.5,
    step=0.1,
)


# ============================================================
# Truth Table
# ============================================================

st.header("1. Input Truth Table")

st.write(
    "Modify the expected outputs below and train "
    "the neural network."
)

edited_table = st.data_editor(
    DEFAULT_XOR_TABLE,
    column_config={
        "x1": st.column_config.NumberColumn(
            "Input X1",
            min_value=0,
            max_value=1,
            step=1,
        ),
        "x2": st.column_config.NumberColumn(
            "Input X2",
            min_value=0,
            max_value=1,
            step=1,
        ),
        "expected": st.column_config.NumberColumn(
            "Expected Output",
            min_value=0,
            max_value=1,
            step=1,
        ),
    },
    num_rows="fixed",
    use_container_width=True,
)


# ============================================================
# Convert Edited Table to Normal Python List
# ============================================================

if hasattr(edited_table, "to_dict"):
    # Handles DataFrame-like results
    rows = edited_table.to_dict("records")
else:
    # Handles list-like results
    rows = edited_table


rows = [
    {
        "x1": int(row["x1"]),
        "x2": int(row["x2"]),
        "expected": int(row["expected"]),
    }
    for row in rows
]


# ============================================================
# Validate Truth Table
# ============================================================

valid, message = validate_truth_table(rows)

if valid:

    st.success(message)

    if is_xor_table(rows):

        st.success(
            "✅ This is the standard XOR truth table."
        )

    else:

        st.warning(
            "⚠️ The table is valid, but it is not "
            "the standard XOR truth table."
        )

else:

    st.error(message)


# ============================================================
# Train Neural Network
# ============================================================

st.header("2. Train Neural Network")

train_button = st.button(
    "🚀 Train Network",
    type="primary",
)


if train_button:

    if not valid:

        st.error(
            "Please correct the truth table before training."
        )

    else:

        trainer = XORTrainer(
            learning_rate=learning_rate,
            epochs=epochs,
        )

        with st.spinner(
            "Training neural network..."
        ):

            network, loss_history = trainer.train(
                rows
            )

        # Store trained model and results
        # inside Streamlit session state.

        st.session_state.network = network

        st.session_state.loss_history = (
            loss_history
        )

        st.session_state.rows = rows

        st.session_state.epochs = epochs

        st.session_state.learning_rate = (
            learning_rate
        )

        st.success(
            "✅ Training completed successfully!"
        )


# ============================================================
# Display Results
# ============================================================

if "network" in st.session_state:

    network = st.session_state.network

    rows = st.session_state.rows

    loss_history = (
        st.session_state.loss_history
    )


    # ========================================================
    # Predictions
    # ========================================================

    st.header("3. Expected vs Predicted Output")

    predictions = []

    expected_values = []

    for row in rows:

        prediction = network.predict(
            row["x1"],
            row["x2"],
        )

        predictions.append(prediction)

        expected_values.append(
            row["expected"]
        )


    # ========================================================
    # Calculate Individual Losses
    # ========================================================

    losses = calculate_sample_losses(
        predictions,
        expected_values,
    )


    # ========================================================
    # Calculate Overall MSE
    # ========================================================

    overall_loss = mean_squared_error(
        predictions,
        expected_values,
    )


    # ========================================================
    # Result Table
    # ========================================================

    result_rows = []

    for row, prediction, loss in zip(
        rows,
        predictions,
        losses,
    ):

        predicted_class = (
            1 if prediction >= 0.5 else 0
        )

        result_rows.append(
            {
                "X1": row["x1"],
                "X2": row["x2"],
                "Expected": row["expected"],
                "Prediction": round(
                    prediction,
                    6,
                ),
                "Predicted Class": predicted_class,
                "Loss": round(
                    loss,
                    6,
                ),
            }
        )

    st.dataframe(
        result_rows,
        use_container_width=True,
    )


    # ========================================================
    # Loss Calculator
    # ========================================================

    st.header("4. Loss Calculator")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Overall MSE",
            f"{overall_loss:.8f}",
        )

    with col2:

        st.metric(
            "Final Training Loss",
            f"{loss_history[-1]:.8f}",
        )


    st.subheader("Loss for Each Input")

    for index, loss in enumerate(losses):

        row = rows[index]

        st.write(
            f"Input ({row['x1']}, {row['x2']}) "
            f"→ Expected = {row['expected']} "
            f"→ Prediction = {predictions[index]:.6f} "
            f"→ Loss = {loss:.8f}"
        )


    # ========================================================
    # Training Loss Graph
    # ========================================================

    st.header("5. Training Loss")

    st.line_chart(
        loss_history
    )


    # ========================================================
    # Expected vs Predicted Graph
    # ========================================================

    st.header("6. Expected vs Predicted")

    chart_data = {
        "Expected": expected_values,
        "Prediction": predictions,
    }

    st.bar_chart(
        chart_data
    )


    # ========================================================
    # Learned Parameters
    # ========================================================

    st.header("7. Learned Weights and Biases")

    parameters = network.get_parameters()

    st.json(
        parameters
    )


    # ========================================================
    # JSON Download
    # ========================================================

    st.header("8. Download Results")

    json_data = create_result_json(
        truth_table=rows,
        predictions=predictions,
        losses=losses,
        overall_loss=overall_loss,
        network=network,
        epochs=st.session_state.epochs,
        learning_rate=st.session_state.learning_rate,
    )

    st.download_button(
        label="⬇️ Download Results as JSON",
        data=json_data,
        file_name="xor_neural_network_results.json",
        mime="application/json",
    )