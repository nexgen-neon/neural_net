import streamlit as st

from src.xor_neural_network.pytorch_trainer import (
    PyTorchXORTrainer,
)

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


def show_xor_page():

    # ========================================================
    # Title
    # ========================================================

    st.header("🧠 XOR Gate Using Neural Network")

    st.write(
        "A neural network implementation for learning "
        "the XOR logic gate."
    )

    st.info(
        "You can choose between the from-scratch "
        "implementation and the PyTorch implementation."
    )

    # ========================================================
    # Training Settings
    # ========================================================

    st.sidebar.header("XOR Training Settings")

    implementation = st.sidebar.selectbox(
        "Neural Network Implementation",
        [
            "From Scratch",
            "PyTorch",
        ],
    )

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

    # ========================================================
    # Implementation Information
    # ========================================================

    if implementation == "From Scratch":

        st.sidebar.success(
            "Using the from-scratch neural network."
        )

    else:

        st.sidebar.success(
            "Using the PyTorch neural network."
        )

    # ========================================================
    # Truth Table
    # ========================================================

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

    # ========================================================
    # Convert Table
    # ========================================================

    if hasattr(edited_table, "to_dict"):

        rows = edited_table.to_dict(
            "records"
        )

    else:

        rows = edited_table

    rows = [
        {
            "x1": int(row["x1"]),
            "x2": int(row["x2"]),
            "expected": int(row["expected"]),
        }
        for row in rows
    ]

    # ========================================================
    # Validate Truth Table
    # ========================================================

    valid, message = validate_truth_table(
        rows
    )

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

    # ========================================================
    # Train Neural Network
    # ========================================================

    st.header("2. Train Neural Network")

    train_button = st.button(
        "🚀 Train Network",
        type="primary",
    )

    if train_button:

        # ----------------------------------------------------
        # Validate truth table
        # ----------------------------------------------------

        if not valid:

            st.error(
                "Please correct the truth table before training."
            )

            st.stop()

        # ----------------------------------------------------
        # Select Implementation
        # ----------------------------------------------------

        if implementation == "From Scratch":

            trainer = XORTrainer(
                learning_rate=learning_rate,
                epochs=epochs,
            )

        else:

            trainer = PyTorchXORTrainer(
                learning_rate=learning_rate,
                epochs=epochs,
            )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        with st.spinner(
            "Training neural network..."
        ):

            network, loss_history = trainer.train(
                rows
            )

        # ----------------------------------------------------
        # Store Results
        # ----------------------------------------------------

        st.session_state.xor_network = network

        st.session_state.xor_loss_history = (
            loss_history
        )

        st.session_state.xor_rows = rows

        st.session_state.xor_epochs = epochs

        st.session_state.xor_learning_rate = (
            learning_rate
        )

        st.session_state.xor_implementation = (
            implementation
        )

        st.success(
            "✅ Training completed successfully!"
        )

    # ========================================================
    # Display Results
    # ========================================================

    if "xor_network" not in st.session_state:

        return

    network = st.session_state.xor_network

    rows = st.session_state.xor_rows

    loss_history = (
        st.session_state.xor_loss_history
    )

    # ========================================================
    # Predictions
    # ========================================================

    st.header(
        "3. Expected vs Predicted Output"
    )

    predictions = []

    expected_values = []

    for row in rows:

        prediction = network.prediction(
            row["x1"],
            row["x2"],
        )

        predictions.append(
            prediction
        )

        expected_values.append(
            row["expected"]
        )

    # ========================================================
    # Individual Losses
    # ========================================================

    losses = calculate_sample_losses(
        predictions,
        expected_values,
    )

    # ========================================================
    # Overall MSE
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
                "Predicted Class": (
                    predicted_class
                ),
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

    st.header(
        "4. Loss Calculator"
    )

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

    st.subheader(
        "Loss for Each Input"
    )

    for index, loss in enumerate(losses):

        row = rows[index]

        st.write(
            f"Input ({row['x1']}, {row['x2']}) "
            f"→ Expected = {row['expected']} "
            f"→ Prediction = "
            f"{predictions[index]:.6f} "
            f"→ Loss = {loss:.8f}"
        )

    # ========================================================
    # Training Loss Graph
    # ========================================================

    st.header(
        "5. Training Loss"
    )

    st.line_chart(
        loss_history
    )

    # ========================================================
    # Expected vs Predicted Graph
    # ========================================================

    st.header(
        "6. Expected vs Predicted"
    )

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

    st.header(
        "7. Learned Weights and Biases"
    )

    if hasattr(network, "parameters_dict"):

        parameters = network.parameters_dict()

    else:

        parameters = network.parameters()

    st.json(
        parameters
    )

    # ========================================================
    # JSON Download
    # ========================================================

    st.header(
        "8. Download Results"
    )

    json_data = create_result_json(
        truth_table=rows,
        predictions=predictions,
        losses=losses,
        overall_loss=overall_loss,
        network=network,
        epochs=st.session_state.xor_epochs,
        learning_rate=(
            st.session_state.xor_learning_rate
        ),
    )

    st.download_button(
        label="⬇️ Download Results as JSON",
        data=json_data,
        file_name=(
            "xor_neural_network_results.json"
        ),
        mime="application/json",
    )