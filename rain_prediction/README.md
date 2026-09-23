# Weather Prediction — README

Below is a **two-level README** for the weather-prediction part of your project:

1. **High-Level Documentation** — for a non-technical reader: what problem are we solving, why are we solving it, what happens from beginning to end.
2. **Technical Documentation** — for a developer: dataset, cleaning, preprocessing, architecture, mathematics, training, serialization, inference, and project structure.

You can place this in:

```text
rain_prediction/README.md
```

---

# Weather Prediction Neural Network

A neural network built **from scratch in Python** to predict the probability of rain tomorrow using historical Australian weather observations.

The project uses the `weatherAUS.csv` dataset and implements the complete machine-learning pipeline:

```text
Raw Weather Data
       ↓
Data Loading
       ↓
Data Auditing
       ↓
Data Cleaning
       ↓
Train/Test Split
       ↓
Missing-Value Handling
       ↓
Feature Scaling
       ↓
Neural Network Training
       ↓
Model Evaluation
       ↓
Save Model as JSON
       ↓
Load Model
       ↓
New Weather Input
       ↓
Rain Probability
```

The neural network itself is implemented **without PyTorch, TensorFlow, Keras, or other neural-network frameworks**.

---

# Table of Contents

* [1. High-Level Documentation](#1-high-level-documentation)

  * [1.1 What is this project?](#11-what-is-this-project)
  * [1.2 The problem we are solving](#12-the-problem-we-are-solving)
  * [1.3 What does the model predict?](#13-what-does-the-model-predict)
  * [1.4 Why predict rain tomorrow?](#14-why-predict-rain-tomorrow)
  * [1.5 What information does the model use?](#15-what-information-does-the-model-use)
  * [1.6 Story of a prediction](#16-story-of-a-prediction)
  * [1.7 What does the final prediction mean?](#17-what-does-the-final-prediction-mean)
  * [1.8 Why don't we train every time?](#18-why-dont-we-train-every-time)
  * [1.9 Limitations](#19-limitations)
* [2. Technical Documentation](#2-technical-documentation)

  * [2.1 Dataset](#21-dataset)
  * [2.2 Dataset structure](#22-dataset-structure)
  * [2.3 Initial data audit](#23-initial-data-audit)
  * [2.4 Feature selection](#24-feature-selection)
  * [2.5 Target variable](#25-target-variable)
  * [2.6 Data cleaning](#26-data-cleaning)
  * [2.7 Train/test split](#27-traintest-split)
  * [2.8 Missing-value handling](#28-missing-value-handling)
  * [2.9 Feature scaling](#29-feature-scaling)
  * [2.10 Neural network architecture](#210-neural-network-architecture)
  * [2.11 Forward propagation](#211-forward-propagation)
  * [2.12 Sigmoid activation](#212-sigmoid-activation)
  * [2.13 Loss function](#213-loss-function)
  * [2.14 Backpropagation](#214-backpropagation)
  * [2.15 Weight updates](#215-weight-updates)
  * [2.16 Training process](#216-training-process)
  * [2.17 Model serialization](#217-model-serialization)
  * [2.18 Preprocessing serialization](#218-preprocessing-serialization)
  * [2.19 Model loading](#219-model-loading)
  * [2.20 Inference](#220-inference)
  * [2.21 Prediction pipeline](#221-prediction-pipeline)
  * [2.22 Project structure](#222-project-structure)
  * [2.23 Running the project](#223-running-the-project)
  * [2.24 Example](#224-example)
  * [2.25 Important implementation decisions](#225-important-implementation-decisions)
  * [2.26 Limitations and future improvements](#226-limitations-and-future-improvements)

---

# 1. High-Level Documentation

## 1.1 What is this project?

Imagine having many years of weather observations from different locations across Australia.

Every day, we know things such as:

* How cold the day was
* How hot the day was
* How much rain fell
* How humid the afternoon was
* What the atmospheric pressure was

We also know whether it **rained the following day**.

The purpose of this project is to teach a computer to look at those historical patterns and learn a relationship between today's weather conditions and the possibility of rain tomorrow.

The project therefore builds a small neural network that learns from historical weather data.

---

## 1.2 The problem we are solving

The question we want the model to answer is:

> **"Based on the available weather measurements, how likely is it to rain tomorrow?"**

For example, imagine the following weather conditions:

```text
Minimum temperature       15°C
Maximum temperature       25°C
Rainfall                   0 mm
Afternoon humidity        60%
Afternoon pressure       1012 hPa
```

These values are given to the trained neural network.

The network might produce:

```text
Rain probability: 73.42%
```

The number comes directly from the neural network's sigmoid output.

---

## 1.3 What does the model predict?

The original dataset contains a column called:

```text
RainTomorrow
```

It contains two possible values:

```text
Yes
No
```

For neural-network training, these values are converted into numbers:

```text
No  → 0
Yes → 1
```

The network therefore learns to produce a value between:

```text
0 and 1
```

The output is interpreted as a rain probability.

For example:

```text
0.10 → 10%
0.35 → 35%
0.50 → 50%
0.73 → 73%
0.91 → 91%
```

The project intentionally keeps the model output as a **probability** rather than forcing it into a Yes/No result.

---

## 1.4 Why predict rain tomorrow?

Rain prediction is a useful example of supervised machine learning because the historical data contains both:

**Input information**

and

**Known answers**

For every historical observation, we have weather measurements and, where available, whether it rained the next day.

This gives the neural network examples from which it can learn.

Conceptually:

```text
Weather conditions today
          ↓
     Neural Network
          ↓
Probability of rain tomorrow
```

---

## 1.5 What information does the model use?

Although the original dataset contains many weather-related columns, the neural network uses five selected features:

| Feature       | Meaning                               |
| ------------- | ------------------------------------- |
| `MinTemp`     | Minimum temperature                   |
| `MaxTemp`     | Maximum temperature                   |
| `Rainfall`    | Rainfall measurement                  |
| `Humidity3pm` | Humidity measured at 3 PM             |
| `Pressure3pm` | Atmospheric pressure measured at 3 PM |

Therefore, the model receives exactly **five inputs**.

```text
MinTemp
MaxTemp
Rainfall
Humidity3pm
Pressure3pm
       ↓
Neural Network
       ↓
Rain probability
```

---

## 1.6 Story of a prediction

The easiest way to understand the entire system is to follow one weather observation.

### Step 1 — Weather information arrives

We receive:

```text
MinTemp = 13
MaxTemp = 25
Rainfall = 15
Humidity3pm = 70
Pressure3pm = 1010
```

---

### Step 2 — The values are prepared

The model was trained using standardized data.

Therefore, the new weather values must go through the **same preprocessing procedure** used during training.

---

### Step 3 — The neural network receives the values

The five values enter the network.

```text
5 input values
      ↓
5 hidden neurons
      ↓
1 output neuron
```

---

### Step 4 — The hidden layer processes the information

Each hidden neuron looks at all five input values.

Each neuron has:

* five weights
* one bias

The neuron calculates a weighted sum and passes it through the sigmoid function.

---

### Step 5 — The output neuron produces a result

The five hidden-neuron outputs are passed to the final neuron.

The final neuron also applies sigmoid activation.

The result might be:

```text
0.7342
```

---

### Step 6 — The result is displayed

The application converts the number into a percentage for readability:

```text
Rain probability: 73.42%
```

The model has therefore completed its prediction.

---

## 1.7 What does the final prediction mean?

The output is a probability-like value between 0 and 1.

For example:

```text
Rain probability: 8.86%
```

means the model's output is approximately:

```text
0.0886
```

Another example:

```text
Rain probability: 73.42%
```

means the model produced approximately:

```text
0.7342
```

The project does **not** apply a `0.5` threshold to turn this into `"Yes"` or `"No"`.

The raw sigmoid output is preserved.

---

## 1.8 Why don't we train every time?

Training a neural network requires repeatedly processing the training dataset and updating its weights.

The weather dataset contains more than **140,000 usable observations**, so training every time the program starts would be unnecessary.

Instead, the project follows a **train once, save, and reuse** approach.

### First execution

```text
Dataset
   ↓
Training
   ↓
Learned weights and biases
   ↓
JSON files
```

The learned model is saved to:

```text
models/rain_prediction_model.json
```

The preprocessing information is saved to:

```text
models/preprocessing.json
```

### Later executions

```text
Model JSON
     ↓
Load
     ↓
New weather data
     ↓
Prediction
```

The model does not need to be retrained.

This makes the application much faster to start after the initial training.

---

## 1.9 Limitations

This project is primarily an educational implementation of a neural network.

It should not be considered a production-grade weather forecasting system.

The model:

* Uses only five weather features.
* Does not use external live weather services.
* Does not model time-series relationships explicitly.
* Does not use advanced weather forecasting techniques.
* Uses a relatively small neural-network architecture.
* Uses MSE as the training loss.
* Produces a sigmoid score interpreted as a probability.
* Does not guarantee that the probability is perfectly calibrated.

The goal is to understand the **complete machine-learning pipeline and neural-network implementation**, rather than build a state-of-the-art meteorological forecasting system.

---

# 2. Technical Documentation

## 2.1 Dataset

The project uses:

```text
weatherAUS.csv
```

The dataset contains historical Australian weather observations.

The original dataset contains:

```text
145,460 rows
23 columns
```

The raw dataset is kept untouched.

It is stored at:

```text
data/raw/weatherAUS.csv
```

---

# 2.2 Dataset structure

The original dataset contains:

```text
Date
Location
MinTemp
MaxTemp
Rainfall
Evaporation
Sunshine
WindGustDir
WindGustSpeed
WindDir9am
WindDir3pm
WindSpeed9am
WindSpeed3pm
Humidity9am
Humidity3pm
Pressure9am
Pressure3pm
Cloud9am
Cloud3pm
Temp9am
Temp3pm
RainToday
RainTomorrow
```

The project does not feed all 23 columns into the neural network.

---

# 2.3 Initial data audit

Before training, the dataset is inspected.

The audit checks important properties such as:

* Number of rows
* Number of columns
* Column names
* Data types
* Missing values
* Duplicate rows
* Target distribution

The original dataset has:

```text
Rows:     145460
Columns:  23
```

There are no duplicate rows.

The target column contains:

```text
No       110316
Yes       31877
Missing    3267
```

Therefore, the target itself contains missing values.

These rows cannot be used for supervised training because the model does not know the correct answer.

---

# 2.4 Feature selection

Five features were selected:

```python
FEATURE_COLUMNS = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Humidity3pm",
    "Pressure3pm",
]
```

The target is:

```python
TARGET_COLUMN = "RainTomorrow"
```

The resulting learning problem is:

```text
X = [
    MinTemp,
    MaxTemp,
    Rainfall,
    Humidity3pm,
    Pressure3pm
]

y = RainTomorrow
```

---

# 2.5 Target variable

The original target contains strings:

```text
No
Yes
```

The neural network requires numerical values.

Therefore:

```python
df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
    "No": 0,
    "Yes": 1,
})
```

This produces:

```text
No  → 0
Yes → 1
```

The target is therefore compatible with a single sigmoid output neuron.

---

# 2.6 Data cleaning

The cleaning process performs several operations.

### Step 1 — Select required columns

Only the five features and target are retained:

```python
df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]
```

---

### Step 2 — Remove rows without a target

```python
df = df.dropna(
    subset=[TARGET_COLUMN]
)
```

Rows where `RainTomorrow` is missing cannot be used for supervised learning.

After this operation:

```text
Cleaned shape:
142,193 rows × 6 columns
```

---

### Step 3 — Convert the target

```text
No  → 0
Yes → 1
```

---

### Step 4 — Ensure numerical feature values

Each selected feature is converted to numeric form:

```python
for column in FEATURE_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )
```

Invalid values become missing values and are handled later during preprocessing.

---

# 2.7 Train/test split

The cleaned data is divided into:

```text
90% training
10% testing
```

The split uses:

```python
train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y,
)
```

### Why `random_state=42`?

It makes the split reproducible.

Running the same code again produces the same train/test division.

### Why `stratify=y`?

The target is not evenly distributed.

There are substantially more `"No"` observations than `"Yes"` observations.

Stratification helps preserve approximately the same class distribution in both training and test sets.

Final shapes:

```text
X_train : (127973, 5)
y_train : (127973,)

X_test  : (14220, 5)
y_test  : (14220,)
```

---

# 2.8 Missing-value handling

Some of the selected features contain missing values.

For example:

```text
MinTemp
MaxTemp
Rainfall
Humidity3pm
Pressure3pm
```

may contain missing observations.

The project uses **median imputation**.

```python
imputer = SimpleImputer(
    strategy="median"
)
```

The imputer learns the median from the training data:

```python
X_train = imputer.fit_transform(X_train)
```

Then the same learned medians are applied to the test data:

```python
X_test = imputer.transform(X_test)
```

This distinction is important.

We do **not** calculate separate medians for the test set.

The training data determines the preprocessing parameters.

---

# 2.9 Feature scaling

The five features have very different numerical ranges.

For example:

```text
Temperature     ≈ tens
Rainfall        ≈ potentially hundreds
Humidity        ≈ 0–100
Pressure        ≈ around 1000
```

Feeding these raw values directly into a neural network can make training more difficult.

Therefore, StandardScaler is used:

```python
scaler = StandardScaler()
```

Training data:

```python
X_train = scaler.fit_transform(X_train)
```

Test data:

```python
X_test = scaler.transform(X_test)
```

Conceptually:

$$
x_{scaled} =
\frac{x-\mu}{\sigma}
$$

where:

* \(x\) = original value
* \(\mu\) = training-set mean
* \(\sigma\) = training-set standard deviation

Again, the scaler is fitted only on training data.

---

# 2.10 Neural network architecture

The neural network architecture is:

```text
5 → 5 → 1
```

Meaning:

```text
Input layer
    5 neurons
       ↓
Hidden layer
    5 neurons
       ↓
Output layer
    1 neuron
```

The five inputs are:

```text
MinTemp
MaxTemp
Rainfall
Humidity3pm
Pressure3pm
```

The hidden layer contains five neurons.

The output layer contains one neuron.

---

## Total trainable parameters

Input → hidden:

```text
5 × 5 weights = 25
5 biases       = 5
```

Total:

```text
30
```

Hidden → output:

```text
5 weights = 5
1 bias    = 1
```

Total:

```text
6
```

Overall:

```text
30 + 6 = 36 trainable parameters
```

---

# 2.11 Forward propagation

For a neuron, the first calculation is:

$$
z = \sum_i x_iw_i+b
$$

In code:

```python
self.z = self.bias

for i in range(len(inputs)):
    self.z += (
        inputs[i] * self.weights[i]
    )
```

The result is then passed through sigmoid:

$$
output =
\frac{1}{1+e^{-z}}
$$

In code:

```python
self.output = self.sigmoid(
    self.z
)
```

---

# 2.12 Sigmoid activation

The sigmoid function is:

$$
\sigma(z)=
\frac{1}{1+e^{-z}}
$$

It converts any real-valued number into a value between:

```text
0 and 1
```

This is useful for the output because the project wants a rain probability.

For example:

```text
z = -2
↓
sigmoid(z) ≈ 0.119
```

and:

```text
z = 2
↓
sigmoid(z) ≈ 0.881
```

The implementation also clamps extreme values before calculating `exp()`:

```python
if z > 700:
    z = 700

if z < -700:
    z = -700
```

This helps prevent numerical overflow.

---

# 2.13 Loss function

The project uses:

$$
L=
\frac{1}{2}
(\hat{y}-y)^2
$$

where:

* \(y\) = actual target
* \(\hat{y}\) = model prediction

In code:

```python
error = prediction - target

loss = 0.5 * (error ** 2)
```

The factor \(1/2\) is convenient when differentiating the loss because it cancels the `2` produced by the square.

The derivative is:

$$
\frac{\partial L}{\partial\hat{y}}
=
\hat{y}-y
$$

---

# 2.14 Backpropagation

Backpropagation calculates how much each weight and bias contributed to the error.

The output neuron's delta is:

$$
\delta_o =
(\hat{y}-y)
\hat{y}(1-\hat{y})
$$

The sigmoid derivative is:

$$
\sigma'(z)
=
\sigma(z)(1-\sigma(z))
$$

Because the neuron's output is already stored, the implementation uses:

```python
sigmoid_derivative = (
    prediction *
    (1 - prediction)
)
```

Then:

```python
self.output_neuron.delta = (
    d_loss_prediction
    * sigmoid_derivative
)
```

---

## Output-layer weight gradient

For an output weight connected to hidden output \(h_i\):

$$
\frac{\partial L}{\partial w_i}
=
\delta_o h_i
$$

Code:

```python
gradient = (
    self.output_neuron.delta
    * self.hidden_outputs[i]
)
```

The weight is then updated.

---

## Hidden-layer delta

For hidden neuron \(j\):

$$
\delta_j =
\delta_o
w_j
h_j(1-h_j)
$$

The implementation first stores the old output weights:

```python
old_output_weights = (
    self.output_neuron.weights.copy()
)
```

This is important because hidden-layer gradients should use the output weights from the current forward pass, before those weights are modified.

---

## Hidden-layer weight gradient

For an input \(x_i\) connected to hidden neuron \(j\):

$$
\frac{\partial L}{\partial w_{ij}}
=
\delta_jx_i
$$

Code:

```python
gradient = (
    hidden_neuron.delta
    * inputs[input_index]
)
```

---

# 2.15 Weight updates

The standard gradient-descent update is:

$$
w_{new}
=
w_{old}
-
\eta
\frac{\partial L}{\partial w}
$$

where \(\eta\) is the learning rate.

In code:

```python
weight -= (
    learning_rate * gradient
)
```

The bias is updated similarly:

$$
b_{new}
=
b_{old}
-
\eta
\frac{\partial L}{\partial b}
$$

For the output neuron:

```python
self.output_neuron.bias -= (
    self.learning_rate
    * self.output_neuron.delta
)
```

---

# 2.16 Training process

The `Trainer` class repeatedly passes training examples through the network.

For each sample:

```text
Input
 ↓
Forward propagation
 ↓
Prediction
 ↓
Loss calculation
 ↓
Backpropagation
 ↓
Weight updates
```

This happens for every training sample.

After all training samples have been processed, the average loss for the epoch is calculated:

$$
AverageLoss =
\frac{\sum Loss_i}{N}
$$

The current implementation trains for:

```text
10 epochs
```

with:

```text
learning rate = 0.01
```

An example training run produced decreasing loss:

```text
Epoch 1/10   Loss: 0.06499
Epoch 2/10   Loss: 0.06119
Epoch 3/10   Loss: 0.06089
...
Epoch 10/10  Loss: 0.06027
```

The decrease indicates that the model was reducing its training error during training.

The corresponding test loss was approximately:

```text
0.05968
```

---

# 2.17 Model serialization

Training is computationally more expensive than inference.

Therefore, after training, the learned parameters are stored in:

```text
models/rain_prediction_model.json
```

The file stores:

* Learning rate
* Hidden-neuron weights
* Hidden-neuron biases
* Output-neuron weights
* Output-neuron bias

Conceptually:

```json
{
    "learning_rate": 0.01,
    "hidden_neurons": [
        {
            "weights": [],
            "bias": 0.0
        }
    ],
    "output_neuron": {
        "weights": [],
        "bias": 0.0
    }
}
```

The actual JSON contains all five hidden neurons.

The JSON therefore represents the learned state of the neural network.

---

# 2.18 Preprocessing serialization

Saving the neural-network weights alone is not sufficient.

The model was trained on **imputed and standardized data**.

Therefore, the preprocessing parameters must also be saved.

They are stored in:

```text
models/preprocessing.json
```

The file contains:

```text
features
imputer_statistics
scaler_mean
scaler_scale
```

For example:

```json
{
    "features": [
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Humidity3pm",
        "Pressure3pm"
    ],
    "imputer_statistics": [],
    "scaler_mean": [],
    "scaler_scale": []
}
```

The actual numerical values are generated during training.

This ensures that a new weather observation is transformed in exactly the same way as the training data.

---

# 2.19 Model loading

On later executions, the program checks whether the JSON files exist.

Conceptually:

```python
os.path.exists(
    MODEL_PATH
)
```

and:

```python
os.path.exists(
    PREPROCESSING_PATH
)
```

If both exist:

```text
Train
```

is skipped.

The program instead:

```text
Create network
       ↓
Load saved weights
       ↓
Load preprocessing parameters
       ↓
Accept new input
       ↓
Predict
```

This is the project's **train-once, inference-many-times** design.

---

# 2.20 Inference

Inference means using the trained model to make a prediction.

The current `inference.py` contains:

```python
def predict(network, inputs):

    probability = network.predict(
        inputs
    )

    return probability
```

There is intentionally no:

```python
if probability >= 0.5:
```

and no thresholding.

The model output is returned directly.

If the network produces:

```text
0.7342
```

the application displays:

```text
Rain probability: 73.42%
```

---

# 2.21 Prediction pipeline

The complete inference process is:

```text
User enters weather values
            ↓
        Raw values
            ↓
   Saved preprocessing
            ↓
Missing-value handling
            ↓
     Standardization
            ↓
     Neural network
            ↓
      Sigmoid output
            ↓
       Probability
```

For example:

```text
MinTemp       = 13
MaxTemp       = 25
Rainfall      = 15
Humidity3pm   = 70
Pressure3pm   = 1010
```

becomes a five-value input vector.

The preprocessing parameters saved during training are applied.

The resulting standardized vector is passed to:

```text
5 → 5 → 1
```

The final sigmoid output is displayed as a percentage.

---

# 2.22 Project structure

The weather-prediction project currently follows this structure:

```text
rain_prediction/
│
├── data/
│   ├── raw/
│   │   └── weatherAUS.csv
│   │
│   └── processed/
│       ├── weather_cleaned.csv
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── rain_prediction_model.json
│   └── preprocessing.json
│
├── rain_pred/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── data_audit.py
│   ├── data_loader.py
│   ├── inference.py
│   ├── main.py
│   ├── neural_net.py
│   ├── preprocessing.py
│   ├── serializer.py
│   ├── trainer.py
│   └── training.py
│
└── notebook/
    └── rain_prediction.ipynb
```

### Responsibility of each file

| File               | Responsibility                                              |
| ------------------ | ----------------------------------------------------------- |
| `data_loader.py`   | Loads the raw dataset                                       |
| `data_audit.py`    | Inspects and reports dataset characteristics                |
| `cleaning.py`      | Selects required columns and cleans the target/features     |
| `preprocessing.py` | Splits, imputes and scales data                             |
| `neural_net.py`    | Implements neurons, forward propagation and backpropagation |
| `trainer.py`       | Controls the training loop                                  |
| `training.py`      | Creates the network and trainer                             |
| `inference.py`     | Performs predictions and calculates loss                    |
| `serializer.py`    | Saves and loads model/preprocessing JSON                    |
| `main.py`          | Orchestrates the complete workflow                          |
| `notebook/`        | Used for experimentation and analysis                       |

---

# 2.23 Running the project

Make sure you are inside:

```text
rain_prediction
```

Then run:

```powershell
uv run python -m rain_pred.main
```

---

## First run

If the model JSON does not exist, the program will:

```text
Load dataset
      ↓
Audit dataset
      ↓
Clean dataset
      ↓
Prepare data
      ↓
Train model
      ↓
Save model
      ↓
Save preprocessing
      ↓
Ask for weather information
      ↓
Predict
```

---

## Subsequent runs

If the JSON files already exist:

```text
Load model
      ↓
Load preprocessing
      ↓
Ask for weather information
      ↓
Predict
```

The model does **not** train again.

---

# 2.24 Example

Run:

```powershell
uv run python -m rain_pred.main
```

The program detects the saved model:

```text
RAIN PREDICTION

Saved model found.
Loading model...

Model loaded successfully.
```

It then asks:

```text
Enter weather information:

MinTemp: 13
MaxTemp: 25
Rainfall: 15
Humidity3pm: 70
Pressure3pm: 1010
```

The values are preprocessed and passed through the network.

The final output may look like:

```text
RAIN PREDICTION
----------------
Rain probability: 73.42%
```

The exact probability depends on the learned weights stored in the model JSON.

---

# 2.25 Important implementation decisions

## No PyTorch or TensorFlow

The neural network is manually implemented.

This means the project explicitly demonstrates:

* Neuron computation
* Weights
* Biases
* Activation functions
* Forward propagation
* Loss calculation
* Backpropagation
* Gradients
* Gradient descent
* Model serialization
* Inference

---

## Single output neuron

The network uses:

```text
5 → 5 → 1
```

rather than:

```text
5 → 5 → 2
```

because there is a single binary target:

```text
RainTomorrow
```

The output neuron uses sigmoid activation.

---

## No Softmax

Softmax is not required because the model has only one output neuron.

The output is a single sigmoid value.

---

## No threshold in the neural network

The neural network produces:

```text
0 → 1
```

directly.

The current application displays this as a percentage.

There is no:

```python
if probability >= 0.5:
```

inside the prediction pipeline.

---

## Training preprocessing is reused

The preprocessing parameters are learned from the training data and saved.

New inputs use those same parameters.

This prevents the model from being trained with one representation of the data and being given a differently scaled representation during inference.

---

## Raw data remains untouched

The original:

```text
data/raw/weatherAUS.csv
```

is not modified.

Processed datasets are stored separately under:

```text
data/processed/
```

This keeps the original source data reproducible.

---

# 2.26 Limitations and future improvements

The current implementation provides the complete fundamental machine-learning workflow, but several improvements could be made later.

### Better evaluation metrics

MSE is currently used as the training loss.

For a binary prediction problem, additional evaluation could include:

* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrix
* ROC-AUC
* Precision-Recall curve

---

### Better class-imbalance handling

The dataset contains significantly more `"No"` examples than `"Yes"` examples.

Future versions could investigate:

* Class weighting
* Resampling
* Alternative loss functions
* Threshold selection using validation data

---

### More features

The current network uses only:

```text
MinTemp
MaxTemp
Rainfall
Humidity3pm
Pressure3pm
```

Future versions could investigate additional weather measurements.

---

### Better architecture

The current architecture is deliberately small:

```text
5 → 5 → 1
```

Future experimentation could compare:

```text
5 → 8 → 1
```

```text
5 → 10 → 5 → 1
```

and other architectures.

---

### Hyperparameter tuning

The current model uses:

```text
Learning rate = 0.01
Epochs = 10
```

Future versions could experiment with different values.

---

### Streamlit interface

A graphical interface can be placed on top of the trained model so that a user can enter weather information through form fields rather than the terminal.

The Streamlit application can load:

```text
rain_prediction_model.json
```

and:

```text
preprocessing.json
```

without retraining the model.

---

# Final Architecture

At the highest level, the complete weather-prediction system is:

```text
                    TRAINING

             weatherAUS.csv
                    │
                    ▼
             load_dataset()
                    │
                    ▼
              audit_dataset()
                    │
                    ▼
              clean_dataset()
                    │
                    ▼
             train / test split
                    │
                    ▼
          median imputation
                    │
                    ▼
             standardization
                    │
                    ▼
              5 input values
                    │
                    ▼
              ┌───────────┐
              │ 5 Hidden  │
              │  Neurons  │
              └───────────┘
                    │
                    ▼
              ┌───────────┐
              │ 1 Output  │
              │  Neuron   │
              └───────────┘
                    │
                    ▼
                Sigmoid
                    │
                    ▼
             Rain probability
                    │
                    ▼
          Save learned parameters
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    model.json         preprocessing.json
```

Then:

```text
                    INFERENCE

          model.json
               +
     preprocessing.json
               │
               ▼
       Load trained model
               │
               ▼
       User enters weather
               │
               ▼
       Apply saved preprocessing
               │
               ▼
          5 input values
               │
               ▼
          5 → 5 → 1 network
               │
               ▼
            Sigmoid
               │
               ▼
       Rain probability
               │
               ▼
       "Rain probability:
            73.42%"
```

This separation between **training** and **inference** is the key design of the project: **the model learns once, its learned parameters are persisted, and subsequent runs use those saved parameters to make predictions without retraining.**
