# DESIGN, MODELLING AND EVALUATION OF A PHYSICS-INFORMED NEURAL NETWORK FOR EV BATTERY THERMAL DIGITAL TWIN GENERALIZATION

**Under the Guidance of:** Prof. Rammohan A  
**Submitted by:** Abimanyu Jayaganesh (23BCE2070)  
**Centre:** Automotive Research Centre, Vellore Institute of Technology  
**Date:** May - July 2026  

---

## Abstract

This report documents an engineering research project in which a Physics-Informed Neural Network (PINN) digital twin framework was designed, modelled, and evaluated for real-time thermal state estimation in lithium-ion battery cells used in electric vehicles (EVs). The battery thermal digital twin is a dynamic mechatronic software plant that operates in parallel with physical battery packs to predict thermal dynamics under dynamic load cycles. The project covered a complete machine learning workflow: selection of lumped electro-thermal physical parameters, formulation of ordinary differential equation (ODE) physics loss constraints, design of a Multi-Layer Perceptron (MLP) architecture, implementation of input noise injection to mitigate exposure bias, and multi-step autoregressive rollout evaluation across unseen test drive cycles (WLTP2). Evaluated on a large-scale dataset comprising 522 simulated trajectories containing approximately 78 million time steps across multiple drive cycles, ambient temperatures ($0^\circ\text{C}$ to $50^\circ\text{C}$), and cell mass scales, the resulting PINN pipeline reduced the maximum rollout error on an unseen test drive cycle from $0.2474^\circ\text{C}$ in the baseline MLP to $0.0851^\circ\text{C}$, achieving an autoregressive rollout Mean Absolute Error (MAE) of $0.0117^\circ\text{C}$. The trainable convective heat transfer parameter ($hA$) converged via softplus parameterization to an effective value of $36.49\text{ W/K}$, consistent with the single-node thermal model and cell surface cooling dynamics.

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Mathematical Model](#2-mathematical-model)
  - [2.1 Modelling Assumptions](#21-modelling-assumptions)
  - [2.2 Determination of Physical Parameters](#22-determination-of-physical-parameters)
  - [2.3 Final Nonlinear Model](#23-final-nonlinear-model)
  - [2.4 Physics Loss and Residual Formulation](#24-physics-loss-and-residual-formulation)
  - [2.5 Trainable Parameter Parameterization](#25-trainable-parameter-parameterization)
  - [2.6 Closed-Loop Autoregressive Rollout](#26-closed-loop-autoregressive-rollout)
- [3. Experimental Setup](#3-experimental-setup)
- [4. Design Process](#4-design-process)
- [5. Network Architecture and Hyperparameters](#5-network-architecture-and-hyperparameters)
- [6. Software Implementation](#6-software-implementation)
- [7. Training and Experimental Validation](#7-training-and-experimental-validation)
- [8. Results and Discussion](#8-results-and-discussion)
- [9. Conclusion](#9-conclusion)
- [Bibliography](#bibliography)

---

## 1. Introduction

The global transition to electric vehicles (EVs) places high demands on battery management systems (BMS) to ensure safety, extend service life, and maximize performance. Lithium-ion batteries generate substantial heat during charge and discharge cycles due to internal resistance and chemical reactions. If this heat is not managed, it can lead to localized hotspots, accelerated degradation, or catastrophic thermal runaway events. Real-time temperature monitoring at the individual cell and pack level is therefore a critical function of modern BMS.

Dynamic thermal models form the backbone of battery thermal digital twins, which run in parallel with the physical battery pack to predict temperature states, estimate cooling requirements, and adjust charging protocols. Historically, high-fidelity thermal modeling has relied on computational fluid dynamics (CFD) or finite element methods (FEM). While highly accurate, these approaches are computationally expensive, making them unsuitable for real-time edge execution inside a vehicle. Consequently, simplified lumped-parameter thermal networks or equivalent circuit models (ECMs) are commonly deployed. However, these models require manual parameter calibration and struggle to resolve transient delays and spatial thermal gradients under aggressive driving profiles.

Recently, data-driven machine learning (ML) models, such as Multi-Layer Perceptrons (MLPs), have emerged as fast alternatives. Despite their speed, purely data-driven models suffer from two primary limitations:
1. **Physical Inconsistency:** Unconstrained data-driven models may yield predictions that violate thermodynamic principles, such as temperature dropping while high current and ambient temperatures persist.
2. **Exposure Bias and Rollout Drift:** During training, models are optimized using one-step-ahead objectives ($T_{t+1}$ given ground-truth $T_t$). In deployment, the model operates recursively, feeding prior predicted temperatures back as inputs. Small single-step errors compound over thousands of evaluation steps, causing accumulating trajectory error. In high-frequency sampling regimes ($dt = 0.01\text{ s}$), single-step metrics can be misleadingly optimistic because predicting zero temperature change ($\Delta T \approx 0$) acts as an identity shortcut.

To address these challenges, this paper presents a Physics-Informed Neural Network (PINN) digital twin framework combining lumped-parameter ODE regularization, softplus-constrained parameter estimation, and input noise injection inspired by scheduled sampling techniques for exposure-bias mitigation.

---

## 2. Mathematical Model

### 2.1 Modelling Assumptions
- The battery cell behaves as a single thermal node with uniform temperature distribution.
- Heat generation ($P_{\text{loss}}$) is produced by internal resistance Joule heating and polarization losses.
- Surface cooling obeys Newton's law of convective heat dissipation.
- Propeller/fan airflow or passive cooling is modeled via an effective convective parameter $hA$.
- Cell mass variations are accounted for via a dimensionless $\text{MassScale}$ factor.

### 2.2 Determination of Physical Parameters

**Table 1. Physical Parameters of the Cell Assembly**

| Parameter | Value | Unit | Notes / Source |
| :--- | :--- | :--- | :--- |
| Cell Mass ($m$) | $1.0$ | kg | Nominal cell structural mass |
| Specific Heat ($C_p$) | $900.0$ | J/kg/K | Li-ion thermal capacity |
| Lumped Heat Capacity ($m C_p$) | $900.0$ | J/K | $m \times C_p$ |
| Initial Convective Parameter ($hA_{\text{init}}$) | $0.1$ | W/K | Initial softplus state |
| Sampling Interval ($dt$) | $0.01$ | s | High-frequency logging |

### 2.3 Final Nonlinear Model
Applying the first law of thermodynamics gives the dynamic equation:
$$m C_p \cdot \text{MassScale} \frac{dT}{dt} = P_{\text{loss}} - hA (T - T_{\text{amb}})$$

Rearranging for the physical rate of temperature change:
$$\frac{dT}{dt} = \frac{P_{\text{loss}} - hA(T - T_{\text{amb}})}{m C_p \cdot \text{MassScale}}$$

### 2.4 Physics Loss and Residual Formulation
The physics loss evaluates the ODE residual in physical units ($\text{K/s}$):
$$R_i = \frac{\Delta \hat{T}_i^{\text{phys}}}{dt_i} - \frac{P_{\text{loss}, i} - hA (T_i - T_{\text{amb}, i})}{m C_p \cdot \text{MassScale}_i}$$

$$\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{i=1}^N R_i^2$$

### 2.5 Trainable Parameter Parameterization
To guarantee $hA > 0$ continuously without gradient clipping:
$$hA = \text{softplus}(hA_{\text{raw}}) = \ln(1 + e^{hA_{\text{raw}}})$$

### 2.6 Closed-Loop Autoregressive Rollout
$$\hat{T}_{t+1} = \hat{T}_t + \Delta \hat{T}_t^{\text{phys}}$$

---

## 3. Experimental Setup

**Table 2. Principal software, hardware, and dataset environment specifications**

| Component | Material / Specification | Function |
| :--- | :--- | :--- |
| GPU Hardware | NVIDIA RTX 4060 Laptop GPU (8 GB VRAM) | Acceleration of PyTorch PINN model |
| CPU Stack | Intel Core i7, 16 GB DDR5 RAM | Multi-worker DataLoader & scaling |
| Deep Learning Stack | PyTorch 2.4, Python 3.10, CUDA 12.9 | Tensor computing framework |
| Dataset | 522 Simulink CSV files (78M time steps) | High-fidelity electro-thermal simulations |
| Training Dataset | 435 CSV files (FTP75, UDDS, US06, WLTP1) | Neural network optimization set |
| Test Dataset | 87 CSV files (unseen WLTP2 cycle) | Generalization rollout test set |
| Feature Set | $T, I, V, \text{SOC}, P_{\text{loss}}, T_{\text{amb}}, \text{MassScale}$ | 7-dimensional input state space |
| Target Variable | $\Delta T = T_{t+1} - T_t$ | Temperature change over $dt = 0.01\text{s}$ |

---

## 4. Design Process

The system design involved two primary iterations:

1. **Iteration 1 (Pure Data-Driven Baseline):** Optimized single-step predictions $T_t \to T_{t+1}$. While single-step error was very low ($< 5 \times 10^{-7}{^\circ}\text{C}$), recursive multi-step rollouts suffered trajectory drift (Max Error $0.2474^\circ\text{C}$). Analysis suggested the model exploited an identity shortcut ($\Delta T \approx 0$).
2. **Iteration 2 (Noise-Injected PINN Architecture):** Added Gaussian input noise ($\sigma = 0.05^\circ\text{C}$) to temperature inputs and embedded ODE physics loss ($\lambda_{\text{phys}} = 0.01$). This mitigated exposure bias and stabilized 150,000-step rollouts.

---

## 5. Network Architecture and Hyperparameters

**Table 3. Principal network architecture and optimization hyperparameters**

| Dimension / Parameter | Value | Notes |
| :--- | :--- | :--- |
| Network Architecture | $7 \to 64 \to 64 \to 32 \to 1$ | Fully connected feedforward MLP |
| Hidden Activation | ReLU | Non-linear feature mapping |
| Optimization Algorithm | Adam ($\alpha = 0.001$) | $\beta_1 = 0.9, \beta_2 = 0.999$ |
| Batch Size | 8,192 | Parallel GPU processing |
| Physics Loss Weight ($\lambda_{\text{phys}}$) | $0.01$ | Regularization strength |
| Temperature Input Noise ($\sigma$) | $0.05^\circ\text{C}$ | Perturbation for exposure bias mitigation |
| Early Stopping Patience | 10 epochs | Training converged at Epoch 18 |

---

## 6. Software Implementation

### 6.1 Preprocessing and Scaler Fitting
Features and targets were processed lazily across 522 files. Feature standard scalers were fitted exclusively on training files to prevent data leakage. Target scaling parameters ($\mu_y = -3.73 \times 10^{-6}$, $\sigma_y = 1.97 \times 10^{-4}$) were saved to `datasets/scalers/` for physical unscaling.

### 6.2 Codebase Organization
The pipeline codebase in the workspace includes:
- `run_baseline.py`: Baseline MLP training & evaluation.
- `run_pinn.py`: PINN training with softplus parameter estimation & physics loss.
- `evaluate_pinn.py` & `rollout_pinn.py`: Single-step and autoregressive rollout engines.

---

## 7. Training and Experimental Validation

System integration benchmarking revealed:
- **Peak VRAM Memory Allocation:** ~184 MB (out of 8 GB VRAM) under PyTorch AMP with batch size 8,192.
- **Disk Preprocessing Throughput:** 3,891.03 samples/second during initial CSV disk loading setup.
- **GPU CUDA Training Execution:** 254,599 samples/second during PyTorch tensor core passes on NVIDIA RTX 4060 GPU.
- **Early Stopping:** PINN converged at epoch 18 in ~1.4 hours.

---

## 8. Results and Discussion

**Table 4. Model Performance Comparison on Unseen WLTP2 Test Set**

| Evaluation Metric | Baseline MLP | Physics-Informed NN | Improvement |
| :--- | :--- | :--- | :--- |
| Single-step MAE ($^\circ\text{C}$) | $4.8978 \times 10^{-7}$ | $3.3783 \times 10^{-7}$ | 31.0% lower |
| Single-step RMSE ($^\circ\text{C}$) | $8.6218 \times 10^{-7}$ | $8.9419 \times 10^{-7}$ | Comparable |
| **Rollout MAE ($^\circ\text{C}$)** | **0.0170** | **0.0117** | **31.2% lower** |
| **Rollout RMSE ($^\circ\text{C}$)** | **0.0211** | **0.0143** | **32.2% lower** |
| **Maximum Rollout Error ($^\circ\text{C}$)** | **0.2474** | **0.0851** | **65.6% lower** |

### Parameter Estimation Convergence
The trainable convective coefficient $hA$ converged from $0.10\text{ W/K}$ to **$36.49\text{ W/K}$**, representing an effective parameter accounting for surface heat transfer under the single-node ODE assumption.

### Figures and Trajectory Visualizations

![Baseline Loss](figures/baseline_loss.png)  
*Figure 1. Baseline MLP loss convergence over training epochs.*

![PINN Loss](figures/loss.png)  
*Figure 2. PINN total loss, data loss, and physics loss convergence.*

![Baseline Rollout](figures/baseline_rollout_trajectory.png)  
*Figure 3. Autoregressive rollout trajectory for Baseline MLP on unseen WLTP2 cycle.*

![PINN Rollout](figures/rollout_trajectory.png)  
*Figure 4. Autoregressive rollout trajectory for PINN on unseen WLTP2 cycle.*

![Physics Residual](figures/residual.png)  
*Figure 5. Distribution of PINN physics residual in physical units (K/s).*

---

## 9. Conclusion

This project provided end-to-end experience of a mechatronic digital twin software engineering project: deriving a lumped battery thermal ODE model from energy balance principles, establishing a noise-injected PINN architecture, accelerating training using PyTorch AMP/TF32 on an NVIDIA RTX 4060 GPU, and validating long-term simulation stability across 87 unseen WLTP2 drive cycle rollouts. The resulting PINN digital twin is computationally lightweight, accurate ($0.0851^\circ\text{C}$ maximum rollout error), demonstrates the feasibility of deployment in real-time battery management systems, and provides a strong foundation for future hardware validation.

---

## Bibliography

1. D. Bernardi, E. Pawlikowski, and J. Newman, "A General Energy Balance for Battery Systems," *Journal of The Electrochemical Society*, vol. 132, no. 1, pp. 5–12, 1985.
2. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019.
3. D. Chen et al., "A Physics-Informed Machine Learning Approach for Estimating Lithium-Ion Battery Temperature," *IEEE Access*, vol. 10, pp. 93784–93794, 2022.
4. S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks," *NeurIPS*, vol. 28, pp. 1171–1179, 2015.
5. B. Saha, J. Boswell et al., "A lumped-parameter electro-thermal model for cylindrical batteries," *Journal of Power Sources*, vol. 262, pp. 319–330, 2014.
