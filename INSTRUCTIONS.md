SYSTEM SPECIFICATION

Build a research-grade PyTorch training pipeline for an EV Battery Thermal Digital Twin.

Dataset:

SimuLink/PINN_Dataset/

Contains:

TRAIN_*.csv
TEST_*.csv

Train set:

All TRAIN_*.csv

Test set:

All TEST_*.csv (WLTP2 unseen drive cycle)

This split MUST NEVER be randomized.

The goal is drive-cycle generalization.

Dataset Columns

Each CSV contains:

Time_s
Current_A
Voltage_V
SOC_pct
BatteryPower_W
PowerLoss_W
Temperature_C
AmbientTemp_C
InitialSOC_pct
InitialBattTemp_C
MassScale
DriveCycle
State Space Formulation

Generate:

Temperature_next
Delta_T
dt
Trajectory_ID

For each CSV independently.

Never create transitions across files.

Features

Use:

FEATURES = [
    "Temperature_C",
    "Current_A",
    "Voltage_V",
    "SOC_pct",
    "PowerLoss_W",
    "AmbientTemp_C",
    "MassScale"
]

Target:

TARGET = "Delta_T"

Do NOT use:

BatteryPower_W
InitialSOC_pct
InitialBattTemp_C
DriveCycle
Time_s
Baseline MLP

Architecture:

7 → 64 → 64 → 32 → 1

ReLU activations.

Loss:

MSELoss

Optimizer:

Adam(lr=1e-3)

Epochs:

100

Batch size:

8192

Mixed precision training:

torch.cuda.amp
PINN

Same network.

Additional physics residual:

dt
ΔT
	​

=
mC
p
	​

P
loss
	​

−hA(T−T
amb
	​

)
	​


Use:

mCp = 900

Trainable:

hA

constrained positive via:

softplus

NOT clamp.

PINN loss:

total =
data_loss
+ λ * physics_loss

Start:

lambda = 0.01
Important Fixes

Implement:

1. Noise Injection

Add Gaussian noise only to:

Temperature_C

during training.

2. Rollout Evaluation

Implement autoregressive rollout:

T_pred[t+1]
=
T_pred[t]
+
ΔT_pred

Evaluate on ALL TEST trajectories.

Metrics:

Rollout MAE
Rollout RMSE
Maximum Error

This is your actual digital twin metric.

Single-step MAE is secondary.

This directly addresses the exposure-bias issue described in your analysis.

3. Save Scalers

Save:

datasets/scalers/

using joblib.

4. Save Outputs

Baseline:

results/baseline/
    model.pt
    metrics.json
    rollout_metrics.json
    predictions.csv
    rollout.csv
    loss.png

PINN:

results/pinn/
    model.pt
    metrics.json
    rollout_metrics.json
    hA_history.csv
    predictions.csv
    rollout.csv
    loss.png
    residual.png
5. Create Dataset Class

Create:

ThermalDataset(Dataset)

which:

lazily loads CSVs
preprocesses
builds state space
avoids loading entire 78M rows into RAM

Your dataset is roughly 70–80 million rows, so this is essential.

One-command scripts
run_baseline.py

Pipeline:

preprocess()
train()
evaluate()
rollout()
plots()
save()

Run:

python run_baseline.py
run_pinn.py

Pipeline:

preprocess()
train()
evaluate()
rollout()
plots()
save()

Run:

python run_pinn.py
Extra Research Features

Have the agent implement:

seed_everything(42)

Automatic CUDA detection:

device = cuda if available else cpu

Training logs:

Epoch
Loss
MAE
RMSE
hA
GPU memory

Early stopping:

patience = 10

Gradient clipping:

1.0

This will take your project from a class project to something much closer to a publishable battery digital twin pipeline. Your earlier analysis already identified the core limitations—especially exposure bias and incorrect thermal parameters—which this new structure directly addresses.