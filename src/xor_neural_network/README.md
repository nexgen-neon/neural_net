# 🧠 XOR Neural Network — Setup & Run Guide

This guide explains how to clone the **XOR Neural Network** project onto another computer, create the Python environment using `uv`, install dependencies, run the tests, and start the Streamlit application.

The instructions assume the new computer is using **Windows + PowerShell**, like the development environment for this project.

---

## 📋 Requirements

Before starting, install the following:

- Git
- Python
- `uv`

You can verify whether they are installed with:

```powershell
git --version
python --version
uv --version
```

If all three commands return a version number, you are ready to continue.

---

# 1. Clone the Repository

Open **PowerShell** and navigate to the directory where you want to keep the project.

For example:

```powershell
cd Documents
```

Then clone the GitHub repository:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

For example:

```powershell
git clone https://github.com/USERNAME/xor-neural-network.git
```

Move into the project:

```powershell
cd xor-neural-network
```

You should now be inside:

```text
xor-neural-network/
```

You can verify this with:

```powershell
pwd
```

---

# 2. Check the Project Files

Run:

```powershell
dir
```

You should see files/folders similar to:

```text
src/
streamlit_app/
tests/
.gitignore
pyproject.toml
uv.lock
```

The important files for the XOR project are:

```text
src/xor_neural_network/
streamlit_app/
tests/
pyproject.toml
uv.lock
```

---

# 3. Create the Virtual Environment

The project uses `uv` to manage the Python environment.

From the project root, run:

```powershell
uv venv
```

This creates:

```text
.venv/
```

inside the project.

Your project will now look approximately like:

```text
xor-neural-network/
│
├── .venv/
├── src/
├── streamlit_app/
├── tests/
├── pyproject.toml
└── uv.lock
```

---

# 4. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

After activation, your terminal should look similar to:

```text
(xor-neural-network) PS C:\...\xor-neural-network>
```

The:

```text
(xor-neural-network)
```

indicates that the virtual environment is active.

---

# 5. Install the Project Dependencies

Run:

```powershell
uv sync
```

`uv` reads the project's dependency information from:

```text
pyproject.toml
```

and uses:

```text
uv.lock
```

to install the locked dependency versions.

You do **not** need to manually install every package one by one.

---

# 6. Verify the Environment

Check which Python is being used:

```powershell
python --version
```

You can also check:

```powershell
python -c "import sys; print(sys.executable)"
```

It should point to the project's `.venv` directory.

For example:

```text
C:\...\xor-neural-network\.venv\Scripts\python.exe
```

---

# 7. Run the Tests

Before starting the application, it is a good idea to verify that the project works correctly.

Run:

```powershell
uv run pytest
```

If everything passes, you should see output similar to:

```text
==================== test session starts ====================
...
==================== passed ====================
```

If a test fails, fix the test/code issue before assuming the Streamlit application is working correctly.

---

# 8. Start the Streamlit Application

From the **project root**, run:

```powershell
uv run streamlit run streamlit_app/app.py
```

You should see something similar to:

```text
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

Open the displayed local URL in your browser.

Usually:

```text
http://localhost:8501
```

---

# 9. Use the XOR Neural Network

After opening the application, select:

```text
XOR Neural Network
```

The application allows you to configure:

- Implementation
- Number of epochs
- Learning rate
- XOR truth table

Then click the **Train** button.

---

# 10. Configure the Training

## Epochs

The number of epochs determines how many times the network trains over the XOR dataset.

For example:

```text
10000
```

means the training process runs for 10,000 epochs.

---

## Learning Rate

The learning rate controls how much the weights and biases change during each update.

For example:

```text
0.5
```

can be used as the learning rate.

The application allows you to change this value from the UI.

---

# 11. XOR Truth Table

The standard XOR table is:

| Input 1 | Input 2 | Expected Output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The neural network should learn these four relationships.

After training, predictions should be approximately:

```text
0 XOR 0 → close to 0
0 XOR 1 → close to 1
1 XOR 0 → close to 1
1 XOR 1 → close to 0
```

Because the output uses sigmoid, the network normally produces values between `0` and `1` rather than exactly `0` or `1`.

For example:

```text
0 XOR 0 → 0.02
0 XOR 1 → 0.97
1 XOR 0 → 0.96
1 XOR 1 → 0.03
```

---

# 12. What the Application Shows

After training, the Streamlit application can display:

### Predictions

The prediction produced for each XOR input.

### Sample Losses

The loss associated with individual training samples.

### Overall MSE

The overall Mean Squared Error of the predictions.

### Training Loss Graph

Shows how the loss changes across epochs.

Generally, you want to see the loss decrease during training.

### Expected vs Predicted

A comparison between:

```text
Expected output
```

and:

```text
Predicted output
```

### Learned Parameters

The trained:

- Weights
- Biases

of the neural network.

### JSON Result

The application can generate a JSON representation containing training information and network parameters.

---

# 13. How the Neural Network Works

The project implements a small neural network:

```text
Input Layer
    2 neurons
       ↓
Hidden Layer
    2 neurons
       ↓
Output Layer
    1 neuron
```

Architecture:

```text
2 → 2 → 1
```

The training process is:

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
Weight/Bias Update
  ↓
Next Training Step
```

The network repeats this process until it learns the XOR relationship.

---

# 14. Project Structure

After cloning, the important structure should look like:

```text
xor-neural-network/
│
├── src/
│   └── xor_neural_network/
│       ├── __init__.py
│       ├── backend.py
│       ├── loss.py
│       ├── neural_network.py
│       ├── serializer.py
│       ├── trainer.py
│       └── truth_table.py
│
├── streamlit_app/
│   ├── app.py
│   └── xor_page.py
│
├── tests/
│
├── .gitignore
├── pyproject.toml
└── uv.lock
```

---

# 15. Important: Run Commands From the Project Root

Make sure your PowerShell terminal is inside:

```text
xor-neural-network/
```

For example:

```text
PS C:\Users\YourName\Documents\xor-neural-network>
```

Then run:

```powershell
uv sync
```

and:

```powershell
uv run pytest
```

and:

```powershell
uv run streamlit run streamlit_app/app.py
```

Do **not** run the Streamlit page by directly opening:

```text
streamlit_app/xor_page.py
```

The main application is:

```text
streamlit_app/app.py
```

---

# 16. PowerShell Activation Problems

If you get an error similar to:

```text
running scripts is disabled on this system
```

PowerShell is preventing the virtual-environment activation script from running.

You can allow locally created scripts for your current Windows user with:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

You should then see:

```text
(xor-neural-network)
```

at the beginning of your PowerShell prompt.

---

# 17. You Can Also Run Without Activating `.venv`

Because the project uses `uv`, you can run commands through the project's environment using `uv run`.

For example:

```powershell
uv run pytest
```

and:

```powershell
uv run streamlit run streamlit_app/app.py
```

Therefore, activation is convenient but not strictly required when using `uv run`.

---

# 18. Updating the Project

If you already cloned the repository and later want the latest changes:

```powershell
git pull
```

Then synchronize the environment:

```powershell
uv sync
```

Run the tests again:

```powershell
uv run pytest
```

Then start Streamlit:

```powershell
uv run streamlit run streamlit_app/app.py
```

---

# 19. Complete Setup — Quick Version

If Git, Python, and `uv` are already installed, the entire setup can be reduced to:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd xor-neural-network
uv venv
.venv\Scripts\Activate.ps1
uv sync
uv run pytest
uv run streamlit run streamlit_app/app.py
```

Then open:

```text
http://localhost:8501
```

---

# 20. Troubleshooting

## `git` is not recognized

Install Git and restart PowerShell.

Verify:

```powershell
git --version
```

---

## `python` is not recognized

Install Python and make sure Python is available through PATH.

Verify:

```powershell
python --version
```

---

## `uv` is not recognized

Install `uv` and restart your terminal.

Verify:

```powershell
uv --version
```

---

## `ModuleNotFoundError`

Make sure you are in the project root:

```powershell
pwd
```

Then run:

```powershell
uv sync
```

and:

```powershell
uv run streamlit run streamlit_app/app.py
```

---

## Streamlit does not start

Try:

```powershell
uv run streamlit --version
```

If that works, start the application with:

```powershell
uv run streamlit run streamlit_app/app.py
```

---

## Tests fail

Run:

```powershell
uv run pytest
```

Read the traceback and identify which test failed.

Make sure dependencies are synchronized:

```powershell
uv sync
```

Then run the tests again.

---

# 21. Stopping the Streamlit Application

When Streamlit is running in PowerShell, press:

```text
Ctrl + C
```

This stops the Streamlit server.

---

# 22. Starting the Project Again Later

Once the project has already been set up, you do not need to clone it or recreate the environment.

Open PowerShell and go to the project:

```powershell
cd path\to\xor-neural-network
```

Then:

```powershell
uv sync
```

and:

```powershell
uv run streamlit run streamlit_app/app.py
```

---

# 🧠 Final Quick Start

For a completely new computer:

```text
Install Git
    ↓
Install Python
    ↓
Install uv
    ↓
Clone repository
    ↓
cd xor-neural-network
    ↓
uv venv
    ↓
Activate .venv
    ↓
uv sync
    ↓
uv run pytest
    ↓
uv run streamlit run streamlit_app/app.py
    ↓
Open http://localhost:8501
```

The project is now ready to train the **XOR neural network from scratch**.
```

### Replace this before committing

In the README, replace:

```text
<YOUR_GITHUB_REPOSITORY_URL>
```

with your actual GitHub repository URL.

For example:

```powershell
git clone https://github.com/your-username/xor-neural-network.git
```

This version is specifically a **new-machine setup/run README**: clone → environment → dependencies → tests → Streamlit → use the XOR network.