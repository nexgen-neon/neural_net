# XOR Neural Network From Scratch

A simple **XOR gate implemented using a neural network from scratch in Python**.

The project demonstrates how a neural network performs:

* Forward propagation
* Loss calculation
* Backpropagation
* Gradient descent
* Weight and bias updates
* XOR classification

The project also includes a **Streamlit web application** where users can:

* Enter and modify an XOR truth table
* Train the neural network
* View expected and predicted outputs
* Calculate individual and overall losses
* View the training-loss graph
* View learned weights and biases
* Download the results as a `.json` file

No PyTorch, TensorFlow, NumPy, or scikit-learn is used for the neural-network implementation.

---

## 1. Requirements

Before starting, make sure you have:

* Python 3.12 or newer
* Git
* `uv`

This project uses **uv** for Python environment and dependency management.

You do not need to install Python packages manually using `pip`.

---

## 2. Clone the Repository

Open PowerShell or a terminal and clone the project:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```powershell
cd xor-neural-network
```

For example:

```powershell
cd C:\Users\YourName\Documents\xor-neural-network
```

---

## 3. Create and Set Up the Virtual Environment

The project uses a `.venv` virtual environment.

Run:

```powershell
uv sync
```

`uv sync` will:

1. Create the `.venv` environment if it does not already exist.
2. Read the dependencies from `pyproject.toml`.
3. Use `uv.lock` to install the locked dependency versions.
4. Install the project itself.

You do not need to run:

```powershell
python -m venv .venv
```

because `uv sync` handles the environment setup.

---

## 4. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should then see something similar to:

```text
(xor-neural-network) PS C:\Users\YourName\Documents\xor-neural-network>
```

If PowerShell prevents the activation script from running, you can allow locally created scripts for your user account:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 5. Run the Streamlit Application

From the project root, run:

```powershell
uv run streamlit run src/xor_neural_network/app.py
```

Streamlit will start the application and provide a local address, usually similar to:

```text
Local URL: http://localhost:8501
```

Open that address in your browser.

### Important

The application is located under:

```text
src/xor_neural_network/app.py
```

Do not run:

```powershell
streamlit run app.py
```

from the project root because `app.py` is inside the `src/xor_neural_network` package.

---

# 6. Using the Application

The application provides the following sections.

### Input Truth Table

The default XOR truth table is:

| X1 | X2 | Expected Output |
| -: | -: | --------------: |
|  0 |  0 |               0 |
|  0 |  1 |               1 |
|  1 |  0 |               1 |
|  1 |  1 |               0 |

You can modify the expected outputs in the Streamlit interface.

### Training Settings

You can configure:

* Number of epochs
* Learning rate

For example:

```text
Epochs: 10000
Learning rate: 0.5
```

### Training

Click:

```text
Train Network
```

The neural network will train using the supplied truth table.

### Results

After training, the application displays:

* Expected output
* Network prediction
* Predicted class
* Individual loss
* Overall MSE
* Training loss graph
* Expected vs predicted graph
* Learned weights and biases

### JSON Download

The application provides a button to download the training results:

```text
xor_neural_network_results.json
```

---

# 7. Run the Tests

The project includes automated tests using `pytest`.

You can run all tests with:

```powershell
uv run pytest
```

A successful run should look similar to:

```text
============================== test session starts ==============================

tests\test_loss.py ........
tests\test_neural_network.py ...
tests\test_trainer.py ...
tests\test_truth_table.py ...

============================== XX passed ==============================
```

The exact number of tests may change as the project develops.

---

## 8. Run a Specific Test File

To test the loss functions:

```powershell
uv run pytest tests/test_loss.py
```

To test the neural network:

```powershell
uv run pytest tests/test_neural_network.py
```

To test training:

```powershell
uv run pytest tests/test_trainer.py
```

To test truth-table validation:

```powershell
uv run pytest tests/test_truth_table.py
```

---

# 9. Project Structure

The project is organized using a `src` layout:

```text
xor-neural-network/
│
├── src/
│   └── xor_neural_network/
│       ├── __init__.py
│       ├── app.py
│       ├── neural_network.py
│       ├── trainer.py
│       ├── loss.py
│       ├── truth_table.py
│       └── serializer.py
│
├── tests/
│   ├── __init__.py
│   ├── test_loss.py
│   ├── test_neural_network.py
│   ├── test_trainer.py
│   └── test_truth_table.py
│
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

### `app.py`

Contains the Streamlit user interface.

It handles:

* Truth-table input
* Training settings
* Training button
* Displaying predictions
* Displaying losses
* Charts
* JSON download

### `neural_network.py`

Contains the actual neural-network implementation.

It handles:

* Network initialization
* Weights
* Biases
* Sigmoid activation
* Forward propagation
* Backpropagation
* Gradient calculation
* Parameter updates
* Predictions

### `trainer.py`

Controls the training process.

It handles:

* Number of epochs
* Learning rate
* Training examples
* Calling the neural network
* Recording loss history

### `loss.py`

Contains loss calculations such as:

```text
Mean Squared Error
Individual sample loss
```

### `truth_table.py`

Handles:

* Default XOR truth table
* Truth-table validation
* XOR validation

### `serializer.py`

Converts the training results into JSON so they can be downloaded from the Streamlit application.

### `tests/`

Contains automated tests that verify that the individual components and the complete training process work correctly.

---

# 10. Understanding the XOR Network

The network learns the following mapping:

```text
0 0 → 0
0 1 → 1
1 0 → 1
1 1 → 0
```

The network contains an input layer, hidden layer, and output layer:

```text
       Input Layer       Hidden Layer       Output Layer

          X1 ──────────► h1 ──────────┐
                                      │
                                      ├──────► ŷ
          X2 ──────────► h2 ──────────┘
```

During training:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Calculation
  ↓
Backpropagation
  ↓
Gradient Calculation
  ↓
Weight/Bias Updates
  ↓
Next Epoch
```

This process is repeated for the configured number of epochs.

---

# 11. uv and Dependency Management

This project uses three important uv-related files/configurations.

### `pyproject.toml`

Defines the project and its dependencies.

### `uv.lock`

Locks the exact dependency versions used by the project.

Do not manually edit `uv.lock`.

If dependencies change, use uv commands such as:

```powershell
uv add <package>
```

or:

```powershell
uv add --dev <package>
```

Then synchronize the environment:

```powershell
uv sync
```

### `.venv`

Contains the project's virtual environment.

It should **not** be committed to Git.

---

# 12. Installing a New Dependency

If you need a normal project dependency:

```powershell
uv add package-name
```

For example:

```powershell
uv add streamlit
```

For a development dependency:

```powershell
uv add --dev pytest
```

Then:

```powershell
uv sync
```

`uv` updates `pyproject.toml` and `uv.lock`.

---

# 13. Git-Ignored Files

The project ignores generated and environment-specific files such as:

```text
.venv/
__pycache__/
.pytest_cache/
*.pyc
.vscode/
```

These files should not be committed to Git.

The source code, tests, `pyproject.toml`, `uv.lock`, and `README.md` should be committed.

---

# 14. Quick Start

For someone who just wants to get started:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd xor-neural-network

uv sync

.\.venv\Scripts\Activate.ps1

uv run streamlit run src/xor_neural_network/app.py
```

To run the tests:

```powershell
uv run pytest
```

That's it.

---

## 15. Troubleshooting

### `uv` is not recognized

Make sure uv is installed and available in your system PATH.

Check:

```powershell
uv --version
```

---

### Streamlit cannot be imported

Run:

```powershell
uv sync
```

Then start the application using:

```powershell
uv run streamlit run src/xor_neural_network/app.py
```

---

### Tests cannot find the package

Run the tests from the **project root**:

```text
xor-neural-network/
```

using:

```powershell
uv run pytest
```

Do not run pytest from inside the `src/xor_neural_network` directory.

---

### PowerShell blocks virtual-environment activation

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

Alternatively, you can use `uv run` commands without manually activating the environment:

```powershell
uv run pytest
```

and:

```powershell
uv run streamlit run src/xor_neural_network/app.py
```

---

# 16. License

Add your project's license information here.
