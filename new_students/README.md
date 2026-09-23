# Student Grade Prediction Neural Network

A neural network built **from scratch in Python** to predict a student's probable grade based on:

* Hours studied
* Hours slept

The network predicts three grade probabilities:

* **A**
* **B**
* **C**

The project intentionally implements the neural-network mathematics manually rather than relying on frameworks such as PyTorch, TensorFlow, NumPy, or other machine-learning libraries.

---

# LEVEL 1 — HIGH-LEVEL DOCUMENTATION

## 1. What Is This Project?

Imagine that we have a very large collection of student records.

For every student, we know:

```text
How many hours did the student study?
How many hours did the student sleep?
What grade did the student receive?
```

For example:

| Hours Studied | Hours Slept | Grade |
| ------------: | ----------: | :---: |
|           8.0 |         7.0 |   A   |
|           5.0 |         6.0 |   B   |
|           2.0 |         5.0 |   C   |

The goal of this project is to teach a neural network to recognize the relationship between these inputs and the resulting grade.

Once training is complete, we can give the network a new student:

```text
Hours studied = 7
Hours slept   = 8
```

and ask:

```text
What is the probability of A?
What is the probability of B?
What is the probability of C?
```

The network might produce:

```text
A: 0.91
B: 0.07
C: 0.02
```

This means the network considers **A the most probable grade**.

---

# 2. The Story Behind the Neural Network

The neural network behaves somewhat like a student learning from examples.

Initially, the network does **not know** what relationship exists between study time, sleep, and grades.

It starts with randomly initialized weights and biases.

For example:

```text
Weight = 0.137
Bias   = -0.284
```

These numbers initially have no useful meaning.

The network then sees a student record.

It makes a prediction.

The prediction is compared with the actual grade.

The difference tells the network how wrong it was.

That error is propagated backward through the network.

The weights and biases are then slightly adjusted.

This process happens repeatedly across the dataset.

Over many examples, the network gradually changes its parameters so that its predictions become closer to the expected grades.

---

# 3. What Does the Network Receive?

The network receives two input values:

```text
x1 = hours studied
x2 = hours slept
```

Therefore, every student can be represented as:

```text
[x1, x2]
```

For example:

```text
[7.5, 8.0]
```

means:

```text
7.5 hours studied
8.0 hours slept
```

---

# 4. What Does the Network Predict?

The network has three outputs:

```text
A
B
C
```

However, it does not directly return the string `"A"` or `"B"`.

Instead, it returns probabilities.

For example:

```text
A = 0.80
B = 0.15
C = 0.05
```

The three probabilities add up to approximately:

```text
1.0
```

This is achieved using the **softmax activation function**.

---

# 5. Why Three Probabilities?

The problem has three possible classes:

```text
A
B
C
```

Therefore, the output layer contains three neurons.

The output layer produces three numerical values called **logits**.

Those logits are converted into probabilities using softmax.

Conceptually:

```text
              ┌── Output A
              │
Input → Hidden├── Output B
              │
              └── Output C
```

---

# 6. How Does the Network Learn?

The learning process can be described as:

```text
Student record
      ↓
Network makes prediction
      ↓
Prediction compared with actual grade
      ↓
Loss calculated
      ↓
Backpropagation calculates gradients
      ↓
Weights and biases updated
      ↓
Next student
```

This happens repeatedly.

The complete training process consists of:

```text
Forward propagation
        ↓
Loss calculation
        ↓
Backpropagation
        ↓
Gradient descent
```

---

# 7. The Complete Student Journey

Suppose the dataset contains:

```text
hours_studied = 8
hours_slept = 7
grade = A
```

The CSV reader converts the grade into a numerical target:

```text
A → [1, 0, 0]
```

The network receives:

```text
[8, 7]
```

The two hidden neurons process these values.

Their outputs are passed to the three output neurons.

The output neurons produce:

```text
z3
z4
z5
```

Softmax converts these into:

```text
P(A)
P(B)
P(C)
```

The probabilities are compared against:

```text
[1, 0, 0]
```

The loss is calculated.

Backpropagation determines how much each parameter contributed to the error.

The parameters are updated.

The network then moves to the next student.

---

# LEVEL 2 — TECHNICAL DOCUMENTATION

# 8. Project Architecture

The network uses the following architecture:

```text
Input Layer
    │
    │
    ├── x1 = hours_studied
    └── x2 = hours_slept
            │
            ▼
      Hidden Layer
       2 neurons
            │
       Sigmoid
            │
            ▼
      Output Layer
       3 neurons
            │
         Softmax
            │
            ▼
       A / B / C
```

In shorthand:

```text
2 → 2 → 3
```

Where:

```text
2 inputs
2 hidden neurons
3 output neurons
```

---

# 9. Project Structure

The current project contains the following important modules:

```text
new_students/
│
├── csv_reader.py
├── inference.py
├── neuron.py
├── neural_net.py
├── trainer.py
└── training.py
```

There is also a generated model file:

```text
students3_model.json
```

and the dataset:

```text
student_dataset.csv
```

---

# 10. Responsibility of Each File

## `csv_reader.py`

Responsible for:

* Opening the CSV file
* Reading student records
* Converting strings to numbers
* Converting grades into one-hot targets
* Producing records in batches

---

## `neuron.py`

Contains the reusable `Neuron` class.

Each neuron owns:

```text
weights
bias
```

and provides two main operations:

```python
calculate()
update()
```

---

## `neural_net.py`

Contains the `StudentsNet` class.

This is the main neural-network implementation.

It handles:

* Network construction
* Random initialization
* Sigmoid
* Sigmoid derivative
* Softmax
* Softmax derivative
* Forward propagation
* Prediction
* Loss calculation
* Backpropagation
* Gradient calculation
* Parameter updates
* Model saving
* Model loading

---

## `trainer.py`

Responsible for repeatedly feeding dataset batches to the neural network.

It handles:

```text
epochs
batches
records
loss accumulation
average loss
```

---

## `training.py`

This is the training entry point.

It creates the neural network and trainer and starts training.

---

## `inference.py`

This is the prediction entry point.

It:

1. Loads the trained model.
2. Accepts hours studied.
3. Accepts hours slept.
4. Runs prediction.
5. Displays A/B/C probabilities.

---

# 11. Dataset

The dataset contains three columns:

```text
hours_studied
hours_slept
grade
```

Example:

```csv
hours_studied,hours_slept,grade
0.0,4.0,C
0.7,5.1,C
1.4,6.2,C
...
```

The dataset contains:

```text
1,000,000 records
```

The data is generated deterministically.

---

# 12. Why the CSV Reader Uses Batches

The project does not load the entire million-record dataset into memory at once.

Instead, the reader uses:

```python
DEFAULT_BATCH_SIZE = 1000
```

The CSV file is processed approximately as:

```text
1000 records
      ↓
train
      ↓
1000 records
      ↓
train
      ↓
1000 records
      ↓
...
```

This is implemented using a Python generator:

```python
yield batch
```

The important advantage is that the entire dataset does not have to exist in memory as one giant list.

---

# 13. Grade Encoding

The network cannot directly perform mathematical operations on:

```text
"A"
"B"
"C"
```

Therefore, the grades are converted into one-hot representations.

### Grade A

```text
A → [1, 0, 0]
```

### Grade B

```text
B → [0, 1, 0]
```

### Grade C

```text
C → [0, 0, 1]
```

This is implemented in:

```python
grade_to_targets()
```

---

# 14. The `Neuron` Class

The neural network uses a reusable `Neuron` class.

```python
class Neuron:
    def __init__(self, number_of_inputs, rng):
        self.weights = []

        for _ in range(number_of_inputs):
            self.weights.append(rng.uniform(-0.5, 0.5))

        self.bias = rng.uniform(-0.5, 0.5)
```

Each neuron owns its own parameters.

For example:

```python
neuron = Neuron(2, rng)
```

creates:

```text
2 weights
1 bias
```

Conceptually:

```text
weight 1
weight 2
bias
```

---

# 15. What Is `self.weights`?

`self.weights` contains the weights belonging to that particular neuron.

For a neuron receiving two inputs:

```text
x1
x2
```

there are two weights:

```text
w1
w2
```

The neuron calculates:

$$
z = x_1w_1 + x_2w_2 + b
$$

---

# 16. What Is `self.bias`?

Every neuron has its own bias.

In the class:

```python
self.bias = rng.uniform(-0.5, 0.5)
```

`self.bias` is therefore not one global network bias.

It belongs to the specific neuron instance.

For example:

```python
self.hidden_neuron1
```

has its own:

```text
weights
bias
```

while:

```python
self.hidden_neuron2
```

has a different:

```text
weights
bias
```

---

# 17. Neuron Calculation

The `calculate()` method performs the weighted sum:

```python
def calculate(self, inputs):
    z = self.bias

    for i in range(len(inputs)):
        z += inputs[i] * self.weights[i]

    return z
```

Mathematically:

$$
z = b + \sum_i x_iw_i
$$

For two inputs:

$$
z = x_1w_1 + x_2w_2 + b
$$

The neuron itself does not apply sigmoid or softmax.

It only calculates its weighted sum.

The network decides what activation should be applied.

---

# 18. Neurons Used by the Network

`StudentsNet` creates five neurons:

```python
self.hidden_neuron1 = Neuron(2, rng)
self.hidden_neuron2 = Neuron(2, rng)

self.output_neuron_a = Neuron(2, rng)
self.output_neuron_b = Neuron(2, rng)
self.output_neuron_c = Neuron(2, rng)
```

Therefore:

```text
Hidden neuron 1 → 2 inputs
Hidden neuron 2 → 2 inputs

Output A → 2 hidden outputs
Output B → 2 hidden outputs
Output C → 2 hidden outputs
```

---

# 19. Random Initialization

The network creates:

```python
rng = random.Random(seed)
```

with:

```text
seed = 42
```

Each weight and bias is initialized using:

```python
rng.uniform(-0.5, 0.5)
```

This means parameters initially lie between:

```text
-0.5 and +0.5
```

Using a fixed seed makes the initialization reproducible.

---

# 20. Why Use a Separate Random Generator?

Instead of repeatedly calling:

```python
random.seed(...)
```

the network creates:

```python
rng = random.Random(seed)
```

and passes the same generator to every neuron.

This allows the five neurons to receive different random values while keeping the entire initialization reproducible.

---

# 21. Forward Propagation

The `forward()` method performs the complete forward pass.

The inputs are:

```python
inputs = [x1, x2]
```

where:

```text
x1 = hours studied
x2 = hours slept
```

---

# 22. Hidden Neuron 1

The first hidden neuron calculates:

$$
z_1 = x_1w_1 + x_2w_2 + b_1
$$

The actual calculation is delegated to:

```python
self.hidden_neuron1.calculate(inputs)
```

Then sigmoid is applied:

$$
h_1 = \sigma(z_1)
$$

---

# 23. Hidden Neuron 2

The second hidden neuron calculates:

$$
z_2 = x_1w_3 + x_2w_4 + b_2
$$

Then:

$$
h_2 = \sigma(z_2)
$$

Therefore, the hidden layer produces:

```text
h1
h2
```

These become the inputs to the output layer.

---

# 24. Sigmoid Activation

The network implements:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

in:

```python
def sigmoid(value):
    if value < -700:
        return 0.0

    if value > 700:
        return 1.0

    return 1.0 / (1.0 + math.exp(-value))
```

Sigmoid converts the hidden neuron's weighted sum into a value between:

```text
0 and 1
```

---

# 25. Why the `-700` and `700` Checks Exist

The exponential function:

```python
math.exp(-value)
```

can become numerically problematic for extremely large values.

Therefore:

```python
if value < -700:
    return 0.0
```

and:

```python
if value > 700:
    return 1.0
```

prevent extreme values from causing overflow problems.

---

# 26. Sigmoid Derivative

The derivative is:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

Since the forward pass already calculated:

```text
h = sigmoid(z)
```

the derivative can be calculated as:

```python
value * (1.0 - value)
```

Therefore:

```python
sigmoid_derivative(h1)
```

calculates:

$$
h_1(1-h_1)
$$

---

# 27. Output Layer

The two hidden outputs:

```text
h1
h2
```

are passed to three output neurons.

### Output A

$$
z_3=h_1w_5+h_2w_6+b_3
$$

### Output B

$$
z_4=h_1w_7+h_2w_8+b_4
$$

### Output C

$$
z_5=h_1w_9+h_2w_{10}+b_5
$$

The implementation calculates these using:

```python
self.output_neuron_a.calculate(hidden_outputs)
self.output_neuron_b.calculate(hidden_outputs)
self.output_neuron_c.calculate(hidden_outputs)
```

---

# 28. Softmax

The three logits:

```text
z3
z4
z5
```

are converted into probabilities using softmax.

The implementation first calculates:

```python
maximum = max(z_a, z_b, z_c)
```

and then:

```python
exp_a = math.exp(z_a - maximum)
exp_b = math.exp(z_b - maximum)
exp_c = math.exp(z_c - maximum)
```

This subtraction improves numerical stability.

Then:

$$
P(A)=\frac{e^{z_3}}{e^{z_3}+e^{z_4}+e^{z_5}}
$$

$$
P(B)=\frac{e^{z_4}}{e^{z_3}+e^{z_4}+e^{z_5}}
$$

$$
P(C)=\frac{e^{z_5}}{e^{z_3}+e^{z_4}+e^{z_5}}
$$

The implementation returns:

```python
probability_a
probability_b
probability_c
```

---

# 29. Why Softmax?

Softmax is appropriate because this is a **three-class classification problem**.

The outputs represent competing probabilities.

For example:

```text
A = 0.72
B = 0.21
C = 0.07
```

The probabilities sum to:

```text
1.00
```

---

# 30. Softmax Jacobian

The project explicitly implements the complete softmax derivative matrix.

For probabilities:

$$
p_A,p_B,p_C
$$

the Jacobian is:

$$
J =
\begin{bmatrix}
p_A(1-p_A) & -p_Ap_B & -p_Ap_C\\
-p_Bp_A & p_B(1-p_B) & -p_Bp_C\\
-p_Cp_A & -p_Cp_B & p_C(1-p_C)
\end{bmatrix}
$$

This is represented by:

```python
[
    [
        probability_a * (1.0 - probability_a),
        -probability_a * probability_b,
        -probability_a * probability_c
    ],
    [
        -probability_b * probability_a,
        probability_b * (1.0 - probability_b),
        -probability_b * probability_c
    ],
    [
        -probability_c * probability_a,
        -probability_c * probability_b,
        probability_c * (1.0 - probability_c)
    ]
]
```

The complete Jacobian is retained because each softmax probability depends on all three logits.

---

# 31. Loss Function

The project uses Mean Squared Error across the three output probabilities.

For A:

$$
E_A=P_A-y_A
$$

For B:

$$
E_B=P_B-y_B
$$

For C:

$$
E_C=P_C-y_C
$$

Individual squared losses are:

$$
L_A=(P_A-y_A)^2
$$

$$
L_B=(P_B-y_B)^2
$$

$$
L_C=(P_C-y_C)^2
$$

The total loss is:

$$
L=
\frac{L_A+L_B+L_C}{3}
$$

The implementation is:

```python
total_loss = (loss_a + loss_b + loss_c) / 3
```

---

# 32. Why Divide by 3?

There are three output values:

```text
A
B
C
```

The project calculates the **mean** of their squared errors rather than their sum.

Therefore:

$$
L=\frac{E_A^2+E_B^2+E_C^2}{3}
$$

Differentiating gives:

$$
\frac{\partial L}{\partial P_A}
=
\frac{2}{3}E_A
$$

and similarly:

$$
\frac{\partial L}{\partial P_B}
=
\frac{2}{3}E_B
$$

$$
\frac{\partial L}{\partial P_C}
=
\frac{2}{3}E_C
$$

This is why the code contains:

```python
d_loss_a = 2.0 / 3.0 * error_a
d_loss_b = 2.0 / 3.0 * error_b
d_loss_c = 2.0 / 3.0 * error_c
```

The division by 3 in the loss and derivative must remain consistent.

---

# 33. Backpropagation

After calculating the loss, the network determines how each parameter affected that loss.

The process goes backward:

```text
Loss
 ↓
Softmax
 ↓
Output neurons
 ↓
Hidden neurons
 ↓
Inputs
```

This is backpropagation.

---

# 34. From Loss to Output Logits

The loss derivatives are:

```text
d_loss_a
d_loss_b
d_loss_c
```

The softmax Jacobian is:

```text
softmax_jacobian
```

The network calculates:

```text
dz3
dz4
dz5
```

using the chain rule.

For example:

$$
\frac{\partial L}{\partial z_3}
=
\frac{\partial L}{\partial P_A}
\frac{\partial P_A}{\partial z_3}
+
\frac{\partial L}{\partial P_B}
\frac{\partial P_B}{\partial z_3}
+
\frac{\partial L}{\partial P_C}
\frac{\partial P_C}{\partial z_3}
$$

This is why the implementation contains three terms for `dz3`.

The same principle applies to `dz4` and `dz5`.

---

# 35. Output Weight Gradients

For output A:

$$
z_3=h_1w_5+h_2w_6+b_3
$$

Therefore:

$$
\frac{\partial L}{\partial w_5}=dz_3h_1
$$

$$
\frac{\partial L}{\partial w_6}=dz_3h_2
$$

$$
\frac{\partial L}{\partial b_3}=dz_3
$$

The same process is applied to output B and C.

---

# 36. Updating Output Neurons

The gradients are passed into:

```python
self.output_neuron_a.update(
    [dw5, dw6],
    db3,
    self.learning_rate
)
```

and similarly for B and C.

The `Neuron.update()` method performs gradient descent:

$$
w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}
$$

and:

$$
b_{new}=b_{old}-\eta\frac{\partial L}{\partial b}
$$

where:

$$
\eta=0.01
$$

in the current training configuration.

---

# 37. Backpropagation Into the Hidden Layer

The hidden neuron outputs influence all three output neurons.

Therefore, the derivative with respect to `h1` is:

$$
dh_1 =
dz_3w_5+
dz_4w_7+
dz_5w_9
$$

Similarly:

$$
dh_2 =
dz_3w_6+
dz_4w_8+
dz_5w_{10}
$$

This is an important part of the chain rule.

`h1` affects all three outputs, so its gradient receives contributions from all three output neurons.

---

# 38. Sigmoid Backpropagation

Since:

$$
h_1=\sigma(z_1)
$$

we have:

$$
\frac{\partial L}{\partial z_1}
=
\frac{\partial L}{\partial h_1}
\frac{\partial h_1}{\partial z_1}
$$

Therefore:

$$
dz_1=dh_1\sigma'(z_1)
$$

and because:

$$
\sigma'(z_1)=h_1(1-h_1)
$$

the implementation uses:

```python
dz1 = dh1 * self.sigmoid_derivative(h1)
```

Likewise:

```python
dz2 = dh2 * self.sigmoid_derivative(h2)
```

---

# 39. Hidden-Layer Gradients

For hidden neuron 1:

$$
z_1=x_1w_1+x_2w_2+b_1
$$

Therefore:

$$
dw_1=dz_1x_1
$$

$$
dw_2=dz_1x_2
$$

$$
db_1=dz_1
$$

For hidden neuron 2:

$$
dw_3=dz_2x_1
$$

$$
dw_4=dz_2x_2
$$

$$
db_2=dz_2
$$

These gradients are passed to:

```python
self.hidden_neuron1.update(...)
self.hidden_neuron2.update(...)
```

---

# 40. Complete Backpropagation Chain

The complete mathematical chain is:

```text
Loss
 │
 ▼
dL/dA, dL/dB, dL/dC
 │
 ▼
Softmax Jacobian
 │
 ▼
dz3, dz4, dz5
 │
 ├───────────────┐
 ▼               ▼
Output weights   Hidden gradients
 │               │
 ▼               ▼
Output update    dh1, dh2
                 │
                 ▼
              Sigmoid
                 │
                 ▼
              dz1, dz2
                 │
                 ▼
             Hidden weights
                 │
                 ▼
                Update
```

---

# 41. Gradient Descent

The network updates parameters using:

$$
parameter_{new}
=
parameter_{old}
-
learning\_rate\times gradient
$$

The current learning rate is:

```python
0.01
```

For example:

```python
self.weights[i] -= learning_rate * weight_gradients[i]
```

---

# 42. Training Configuration

The current training script creates:

```python
network = StudentsNet(
    learning_rate=0.01,
    seed=42
)
```

Training uses:

```python
epochs=10
batch_size=1000
```

Therefore:

```text
Learning rate = 0.01
Seed           = 42
Epochs         = 10
Batch size     = 1000
```

---

# 43. What Is an Epoch?

One epoch means the network has processed the entire dataset once.

The training loop:

```python
for epoch in range(epochs):
```

runs ten times.

Because the dataset contains one million records:

```text
1 epoch = 1,000,000 records
```

and:

```text
10 epochs = 10,000,000 record presentations
```

assuming the full dataset is processed during every epoch.

---

# 44. What Is a Batch?

The reader produces:

```python
batch_size=1000
```

Therefore, records are grouped into batches of up to 1000.

Conceptually:

```text
Dataset
│
├── Batch 1 → 1000 students
├── Batch 2 → 1000 students
├── Batch 3 → 1000 students
├── ...
└── Final batch
```

The current implementation processes individual records inside each batch and updates the network after each record.

So the batching primarily controls **how records are streamed/read**, rather than performing vectorized batch-gradient computation.

---

# 45. Trainer

The `Trainer` receives a network:

```python
class Trainer:
    def __init__(self, network):
        self.network = network
```

Then:

```python
trainer.train(...)
```

loops through epochs and batches.

For every record it extracts:

```text
x1
x2
target A
target B
target C
```

and calls:

```python
self.network.train(...)
```

The returned loss is accumulated.

At the end of an epoch:

```python
average_loss = total_loss / total_records
```

is calculated and displayed.

---

# 46. Model Saving

After training:

```python
network.save(model_file)
```

The model is saved as:

```text
students3_model.json
```

The JSON contains:

```text
learning_rate
seed

hidden_neuron1
hidden_neuron2

output_neuron_a
output_neuron_b
output_neuron_c
```

Each neuron contains:

```text
weights
bias
```

---

# 47. Why Save the Model?

Without saving the model, the trained parameters would disappear when the Python program terminates.

Saving allows the trained network to be loaded later.

The inference program therefore does not need to train the network again.

The workflow becomes:

```text
Training
   ↓
students3_model.json
   ↓
Inference
```

---

# 48. Model Loading

The inference program creates:

```python
network = StudentsNet()
```

and then:

```python
network.load(model_file)
```

The saved weights and biases are restored into the appropriate neuron objects.

For example:

```python
self.hidden_neuron1.weights = model["hidden_neuron1"]["weights"]
```

and:

```python
self.hidden_neuron1.bias = model["hidden_neuron1"]["bias"]
```

The same process occurs for all five neurons.

---

# 49. Inference

The inference program asks:

```text
Enter hours studied:
Enter hours slept:
```

For example:

```text
Enter hours studied: 8
Enter hours slept: 7
```

The values are passed to:

```python
network.predict(
    hours_studied,
    hours_slept
)
```

---

# 50. Prediction

`predict()` internally calls:

```python
result = self.forward(x1, x2)
```

but only returns:

```python
{
    "A": result["A"],
    "B": result["B"],
    "C": result["C"]
}
```

Therefore inference receives only the three final probabilities.

---

# 51. Why Doesn't `predict()` Use an `if/else`?

The network does not contain logic such as:

```python
if probability_a > probability_b:
    grade = "A"
```

Instead, it returns the complete probability distribution:

```text
A → probability
B → probability
C → probability
```

This keeps the neural network focused on producing model outputs.

The caller can interpret those probabilities separately.

---

# 52. Probability Sum

Inference also prints:

```python
sum(probabilities.values())
```

The result should be approximately:

```text
1.0000
```

because softmax produces a probability distribution.

For example:

```text
A: 0.8500
B: 0.1200
C: 0.0300

Probability Sum:
1.0000
```

---

# 53. Complete Training Flow

The entire application can be summarized as:

```text
student_dataset.csv
        │
        ▼
   csv_reader.py
        │
        ▼
     Batches
        │
        ▼
    trainer.py
        │
        ▼
   StudentsNet
        │
        ▼
 Forward propagation
        │
        ▼
    Softmax
        │
        ▼
 Probabilities
        │
        ▼
     MSE Loss
        │
        ▼
 Backpropagation
        │
        ▼
 Gradient Descent
        │
        ▼
 Updated Neurons
        │
        ▼
 Repeat
        │
        ▼
students3_model.json
```

---

# 54. Complete Inference Flow

```text
students3_model.json
        │
        ▼
    Load model
        │
        ▼
User enters:
hours studied
hours slept
        │
        ▼
     StudentsNet
        │
        ▼
 Forward propagation
        │
        ▼
     Softmax
        │
        ▼
 A / B / C probabilities
        │
        ▼
      Terminal
```

---

# 55. Running the Project

From the project root:

```powershell
python -m new_students.training
```

This starts training.

After training finishes, the model is saved as:

```text
students3_model.json
```

Then run:

```powershell
python -m new_students.inference
```

---

# 56. Expected Training Output

The exact values depend on the current dataset and implementation, but the program prints the loss after every epoch in this format:

```text
Starting neural network training...

Epoch 1/10 - Loss: ...
Epoch 2/10 - Loss: ...
Epoch 3/10 - Loss: ...
...
Epoch 10/10 - Loss: ...

Training completed.
Model saved to: ...\students3_model.json
```

The important purpose of watching the loss is to determine whether the network is generally learning.

---

# 57. Expected Inference Output

A typical run looks like:

```text
Trained model loaded successfully.

Enter hours studied: 8
Enter hours slept: 7

Grade Probability Distribution
--------------------------------
A: 0.xxxx
B: 0.xxxx
C: 0.xxxx

Probability Sum:
1.0000
```

The actual probability values depend on the trained model.

---

# 58. Why the Network Is Structured This Way

The project deliberately separates responsibilities.

### `Neuron`

Responsible for:

```text
weights
bias
weighted sum
parameter update
```

### `StudentsNet`

Responsible for:

```text
network architecture
activations
forward propagation
loss
backpropagation
serialization
```

### `Trainer`

Responsible for:

```text
epochs
dataset iteration
loss tracking
```

### `csv_reader`

Responsible for:

```text
CSV reading
data conversion
batch creation
```

### `training.py`

Responsible for:

```text
starting training
```

### `inference.py`

Responsible for:

```text
loading model
getting user input
displaying predictions
```

This separation makes the project easier to understand and maintain.

---

# 59. Why Create a `Neuron` Class?

Originally, every weight and bias could be represented directly inside the network.

That quickly becomes difficult to manage.

For example:

```text
w1
w2
w3
w4
w5
...
w10
b1
b2
...
b5
```

With the `Neuron` abstraction, each neuron owns its own parameters.

Instead of thinking:

```text
w1, w2, b1
```

we can think:

```text
hidden_neuron1
    ├── weights
    └── bias
```

This is conceptually closer to how a neural network is structured.

---

# 60. Important Design Principle

A neuron does **not** know about the other neurons.

For example:

```python
self.hidden_neuron1
```

does not calculate:

```text
hidden_neuron2
output_neuron_a
output_neuron_b
```

It only knows how to calculate its own weighted sum.

The `StudentsNet` class connects the neurons together.

This is an important separation of responsibility.

---

# 61. No Normalization

The current implementation does **not** normalize the input values.

The network directly receives:

```text
hours_studied
hours_slept
```

For example:

```text
8.5
7.0
```

No min-max scaling or standardization is performed.

This is an intentional characteristic of the current implementation.

---

# 62. No External Machine-Learning Framework

The neural network is implemented manually.

The project uses Python's standard library components such as:

```python
math
json
random
csv
pathlib
```

The neural-network calculations themselves are implemented directly.

There is no dependency on:

```text
PyTorch
TensorFlow
NumPy
scikit-learn
```

---

# 63. Mathematical Summary

## Hidden layer

$$
z_1=x_1w_1+x_2w_2+b_1
$$

$$
h_1=\sigma(z_1)
$$

$$
z_2=x_1w_3+x_2w_4+b_2
$$

$$
h_2=\sigma(z_2)
$$

---

## Output layer

$$
z_3=h_1w_5+h_2w_6+b_3
$$

$$
z_4=h_1w_7+h_2w_8+b_4
$$

$$
z_5=h_1w_9+h_2w_{10}+b_5
$$

---

## Softmax

$$
P_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

---

## MSE

$$
L=
\frac{
(P_A-y_A)^2+
(P_B-y_B)^2+
(P_C-y_C)^2
}{3}
$$

---

## Gradient descent

$$
w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}
$$

$$
b_{new}=b_{old}-\eta\frac{\partial L}{\partial b}
$$

---

# 64. Current Configuration

| Component                | Current implementation |
| ------------------------ | ---------------------- |
| Inputs                   | 2                      |
| Hidden neurons           | 2                      |
| Output neurons           | 3                      |
| Architecture             | 2 → 2 → 3              |
| Hidden activation        | Sigmoid                |
| Output activation        | Softmax                |
| Classes                  | A, B, C                |
| Target encoding          | One-hot                |
| Loss                     | Mean Squared Error     |
| Learning rate            | 0.01                   |
| Epochs                   | 10                     |
| Batch size               | 1000                   |
| Random seed              | 42                     |
| Initial parameter range  | -0.5 to 0.5            |
| Normalization            | None                   |
| Model format             | JSON                   |
| Model filename           | `students3_model.json` |
| Dataset size             | 1,000,000 records      |
| Neural-network framework | None                   |
| Numerical library        | None                   |

---

# 65. Current Limitations

This is an educational neural-network implementation, so it intentionally has several limitations.

### 1. Very small architecture

The network is:

```text
2 → 2 → 3
```

This is useful for learning the mathematics but is not a sophisticated predictive model.

### 2. Synthetic dataset

The student dataset is generated rather than collected from real student performance.

Therefore, the model is primarily demonstrating the neural-network pipeline rather than solving a real-world educational prediction problem.

### 3. No normalization

The current implementation directly uses the input values.

### 4. MSE for classification

The project intentionally uses MSE rather than cross-entropy.

This is useful educationally because it exposes the complete chain rule through softmax and its Jacobian.

### 5. Per-record parameter updates

Although records are read in batches, the network updates its parameters after each individual training record.

The batches therefore serve as a streaming/read mechanism rather than true vectorized mini-batch gradient descent.

---

# 66. Future Improvements

Possible future improvements include:

```text
Real student dataset
        ↓
Input normalization
        ↓
Larger neural network
        ↓
More useful features
        ↓
Better classification loss
        ↓
Validation dataset
        ↓
Test dataset
        ↓
Accuracy / precision / recall
        ↓
Model evaluation
```

Other improvements could include:

* Multiple hidden layers
* Configurable architecture
* Mini-batch gradient accumulation
* Learning-rate scheduling
* Early stopping
* Model evaluation
* Confusion matrix
* More student features
* Real-world data
* Cross-validation

These are **future possibilities**, not features of the current implementation.

---

# 67. Final Mental Model

If you need to understand the entire project in one picture, think of it this way:

```text
                 STUDENT
                    │
          ┌─────────┴─────────┐
          │                   │
     Hours Studied        Hours Slept
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
             Hidden Neuron 1
                    │
                 Sigmoid
                    │
                    ├──────────┐
                    │          │
                    ▼          ▼
             Hidden Neuron 2
                    │
                 Sigmoid
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
        Output A       Output B       Output C
             │             │             │
             └─────────────┼─────────────┘
                           │
                        Softmax
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          P(A)           P(B)           P(C)
```

During **training**, the direction then reverses:

```text
Prediction
    ↓
Loss
    ↓
Softmax derivative
    ↓
Output gradients
    ↓
Hidden gradients
    ↓
Weight/bias gradients
    ↓
Parameter updates
```

So the fundamental idea behind the entire project is:

> **The network takes study time and sleep time, transforms those values through two hidden neurons, converts the final three outputs into A/B/C probabilities, measures how far those probabilities are from the expected grade, and repeatedly adjusts its weights and biases to reduce that error.**

This README now reflects the **actual `Neuron` abstraction, actual `StudentsNet` implementation, actual MSE `/3` calculation, actual full softmax Jacobian, actual JSON structure, actual `students3_model.json` filename, actual batch reader, and actual training/inference entry points** you provided.
