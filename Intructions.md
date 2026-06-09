# Thermal Battery Digital Twin PINN - Development Specification

## Project Goal

Develop a Physics-Informed Neural Network (PINN) that acts as a thermal digital twin for a Li-ion battery pack.

The model will be trained using electrothermal datasets generated from MATLAB/Simulink.

The objective is to predict battery temperature evolution while enforcing known thermal physics.

---

# Dataset Description

Datasets are located in:

```text
SimuLink Datasets/
├── ambient/
│   ├── ambient_30C.csv
│   ├── ambient_40C.csv
│   ├── ambient_50C.csv
│   └── baseline_20C.csv
├── baseline/
│   └── baseline_20C.csv
├── combined/
└── drivecycles/
```

Each CSV contains:

```text
Time
Current
Voltage
SOC
PowerLoss
Temperature
```

Temperature is stored in Kelvin.

---

# Phase 1: Data Processing Pipeline

## Requirements

Create a data loader that:

1. Loads all CSV files.
2. Removes duplicate datasets.
3. Adds an AmbientTemp column.

Ambient temperatures:

| Dataset      | AmbientTemp |
| ------------ | ----------- |
| baseline_20C | 293.15      |
| ambient_30C  | 303.15      |
| ambient_40C  | 313.15      |
| ambient_50C  | 323.15      |

---

## Data Validation

Verify:

* No NaN values
* No Inf values
* Time monotonicity
* Temperature monotonic trends

Generate:

```python
summary_statistics.csv
```

containing:

* min
* max
* mean
* std

for every variable.

---

# Phase 2: Baseline Neural Network

Before implementing a PINN, train a conventional neural network.

Purpose:

* Verify dataset quality
* Establish benchmark performance

---

## Inputs

```python
[
    Current,
    Voltage,
    SOC,
    PowerLoss,
    AmbientTemp
]
```

---

## Output

```python
Temperature
```

---

## Architecture

PyTorch MLP:

```text
Input(5)

Dense(64)
ReLU

Dense(64)
ReLU

Dense(32)
ReLU

Dense(1)
```

---

## Training

Train/Test Split:

```python
80/20
```

Metrics:

```python
MAE
RMSE
R²
```

Save:

```text
results/
├── baseline_model.pt
├── baseline_metrics.json
└── baseline_predictions.csv
```

---

# Phase 3: State Space Formulation

Convert the dataset into a dynamic prediction problem.

---

## State Inputs

At time t:

```python
[
    Temperature_t,
    Current_t,
    Voltage_t,
    SOC_t,
    PowerLoss_t,
    AmbientTemp_t
]
```

---

## Target

```python
Temperature_(t+1)
```

Generate a new processed dataset:

```text
processed_state_space.csv
```

---

# Phase 4: Physics-Informed Neural Network

## Governing Thermal Equation

Use:

dT/dt = (Q_gen - Q_loss) / (m * Cp)

---

## Heat Generation

Use:

Q_gen = PowerLoss

PowerLoss is already provided by Simulink.

---

## Heat Loss

Use Newton Cooling:

Q_loss = hA * (T - T_amb)

Where:

```python
hA
```

is a trainable parameter.

---

## Initial Physical Parameters

Battery Pack Mass:

```python
m = 4.0
```

kg

Specific Heat:

```python
Cp = 1000
```

J/kg/K

Thermal Capacity:

```python
mCp = 4000
```

J/K

---

# PINN Architecture

Inputs:

```python
[
    Temperature,
    Current,
    Voltage,
    SOC,
    PowerLoss,
    AmbientTemp
]
```

Output:

```python
Temperature_pred
```

---

# Physics Loss

Use automatic differentiation.

Compute:

```python
dTdt_pred
```

Physics residual:

```python
Residual =
mCp * dTdt_pred
-
(
PowerLoss
-
hA * (T_pred - AmbientTemp)
)
```

Physics loss:

```python
MSE(Residual, 0)
```

---

# Data Loss

```python
MSE(
Temperature_pred,
Temperature_true
)
```

---

# Total Loss

```python
Loss =
DataLoss
+
lambda_physics * PhysicsLoss
```

Start with:

```python
lambda_physics = 0.1
```

and make configurable.

---

# Training Requirements

Framework:

```python
PyTorch
```

Preferred libraries:

```python
numpy
pandas
matplotlib
scikit-learn
torch
```

---

## Logging

Log:

* Data loss
* Physics loss
* Total loss

every epoch.

---

## Outputs

Save:

```text
results/
├── pinn_model.pt
├── pinn_metrics.json
├── loss_curve.png
├── prediction_vs_truth.png
└── residual_distribution.png
```

---

# Evaluation

Compute:

```python
MAE
RMSE
R²
```

for:

1. Baseline MLP
2. PINN

Generate comparison table.

---

# Visualizations

Create:

1. Temperature prediction vs ground truth
2. Training loss curve
3. Physics residual histogram
4. Temperature trajectory plots
5. Error distribution

Save all plots automatically.

---

# Code Structure

```text
project/
│
├── data/
├── datasets/
├── models/
│   ├── baseline_mlp.py
│   └── pinn.py
│
├── training/
│   ├── train_baseline.py
│   └── train_pinn.py
│
├── utils/
│   ├── preprocessing.py
│   ├── metrics.py
│   └── plotting.py
│
├── results/
│
├── config.yaml
│
└── main.py
```

---

# Deliverables

The final codebase must:

1. Train a baseline MLP.
2. Train a Physics-Informed Neural Network.
3. Compare both models.
4. Generate evaluation plots.
5. Save trained weights.
6. Be executable from a single command.

Example:

```bash
python main.py
```

The implementation should prioritize readability, modularity, reproducibility, and research-grade documentation.
