# Analysis of PINN and Baseline MLP Thermal Digital Twin Performance

This analysis investigates why the models exhibit extremely low single-step loss but fail to generalize to long-term simulations (rollouts), why the PINN's convective heat transfer coefficient ($hA$) is clamped to its minimum value, and what actions we can take to resolve these issues.

---

## 1. The Core Issue: Single-Step vs. Multi-Step Rollout (Exposure Bias)

The training and evaluation pipelines currently define the learning task as **one-step-ahead prediction**:
- **Input:** $X_t = [T_t, I_t, V_t, SOC_t, P_{loss, t}, T_{amb, t}]$
- **Target:** $\Delta T_t = T_{t+1} - T_t$
- **Prediction:** $T^{pred}_{t+1} = T_t + \Delta T^{pred}_t$

### Why the single-step loss is negligible:
1. **Small Time Steps ($dt = 0.01$s):** In such a short interval, the temperature changes by less than $10^{-4}$ Kelvin on average. The target $\Delta T_t$ is extremely small (standard deviation of $1.97 \times 10^{-4}$ K).
2. **Identity Shortcut:** Since the model is fed the true ground-truth $T_t$ at every single time step, it doesn't need to learn the system dynamics. A model that simply predicts $\Delta T \approx 0$ (meaning $T_{t+1} \approx T_t$) will achieve a training and test MAE of only $\approx 0.000019$ K!
3. **Misleading Metrics:** The extremely high $R^2 = 1.00000$ and tiny MAE/RMSE on the test set are artifacts of this single-step evaluation. Both models actually perform *worse* than the naive identity mapping baseline.

### What happens in a real Digital Twin (Rollout):
During actual inference/simulation, the model is run autoregressively:
$$T^{pred}_{t+1} = T^{pred}_t + \text{Model}(T^{pred}_t, I_t, V_t, SOC_t, P_{loss, t}, T_{amb, t})$$
Here, the model is fed its own *predicted* temperature from the previous step. Small systematic biases in the prediction of $\Delta T$ accumulate over time. Over a 1500-second simulation ($150,000$ steps), these errors compound dramatically:
- **Baseline MLP Rollout MAE:** **3.8461 K** (Max Error: **8.5555 K**)
- **PINN Rollout MAE:** **4.6997 K** (Max Error: **6.4600 K**)

This divergence due to training on ground-truth histories but testing on self-generated histories is known in machine learning as **exposure bias**.

---

## 2. Physics Inconsistency: Mismatched Thermal Capacity ($m C_p$)

In `models/pinn.py`, the physics loss is calculated as:
$$\text{Residual} = \frac{dT^{pred}}{dt} - \frac{P_{loss} - hA(T - T_{amb})}{m C_p}$$
The code hardcodes $m = 4.0$ kg and $C_p = 1000$ J/kg/K, resulting in a thermal capacity $m C_p = 4000$ J/K.

### Parameter Estimation from the Data:
Fitting a linear regression model directly to the physical governing equation across the dataset reveals:
- **Actual $m C_p$ from Simulink:** $\approx \mathbf{900}$ **J/K**
- **Assumed $m C_p$ in training code:** **4000 J/K** (over 4.4x too high!)

Because of this mismatch:
1. The physics loss forces the network to satisfy an incorrect physical relationship where the temperature rate of change is 4.4 times smaller than it should be for a given PowerLoss.
2. This creates a conflict between the **Data Loss** (which fits the actual temperature change) and the **Physics Loss** (which expects a much slower temperature rise).

---

## 3. Convective Heat Transfer Coefficient ($hA$) Clamping

During PINN training, the learned convective heat transfer coefficient ($hA$) is clamped to its lower bound of `1e-5` (0.00001 W/K) and never increases.

### Why this happens:
1. **Temperature Rising when PowerLoss is Zero:** In the dataset, when current/power loss drops to zero, the battery temperature *continues to rise* for a short period. For example, in the 20°C ambient run:
   - At $t=390$s, $P_{loss} = 0$, $T = 314.867$ K ($T > T_{amb}$)
   - At $t=400$s, $P_{loss} = 0$, $T = 314.894$ K (Temperature increases by 0.027 K!)
2. **Negative Convection Coefficient:** Fitting the lumped thermal equation to the data yields an estimated $hA$ of approximately **$-0.006$ to $-0.010$ W/K** (negative). A negative $hA$ is physically impossible for cooling, but mathematically necessary to explain how the battery continues to heat up when $P_{loss}=0$ and $T > T_{amb}$.
3. **Optimizer Behavior:** Because $hA$ is constrained to be positive (`torch.clamp(hA_param, min=1e-5)`), the optimizer pushes $hA$ to the lowest possible value to minimize the physics loss.

### Physical Explanation of the Thermal Delay:
The Simulink dataset was generated using a **two-node thermal model** (representing Cell Core and Cell Casing):
- The heat generation $P_{loss}$ occurs in the cell core.
- The temperature recorded in the dataset is likely the cell casing/surface temperature.
- When $P_{loss}$ stops, the core is still hotter than the casing, causing heat to continue flowing from the core to the casing. This creates a phase lag where the surface temperature keeps rising even after the heat source is turned off.
- The single-node lumped capacitance model used in the PINN cannot model this lag, leading to the physically incorrect negative $hA$ fit.

---

## 4. Actionable Solutions to Fix the Model

To build a reliable digital twin that generalizes well and respects the physics, we can implement the following strategies:

### A. Fix the Physical Parameters (Immediate)
- Update `config.yaml` to set the correct thermal capacity:
  ```yaml
  physics:
    m: 1.0         # 1.0 kg
    Cp: 900.0      # 900 J/kg/K (making mCp = 900 J/K)
  ```

### B. Implement Multi-Step Rollout Loss during Training
- Instead of training only on $T_t \to T_{t+1}$, train the model to predict trajectories over a window (e.g., $T_t \to T_{t+H}$ for $H = 10$ to $50$ steps) by recursively feeding predictions back.
- Backpropagate through the rollout steps (Truncated Backpropagation Through Time). This directly minimizes exposure bias and ensures long-term simulation stability.

### C. Train with Input Noise Injection
- Add small Gaussian noise to the input `Temperature_t` during training:
  $$T_t^{input} = T_t^{true} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
  This trains the neural network to recover from small errors and stabilizes the rollout.

### D. Formulate a Two-State Physics Model or Include Lags
- **Two-State PINN:** Predict both core temperature $T_c$ and casing temperature $T_s$, and enforce the dual differential equations.
- **Autoregressive Inputs (Lags):** Include historical temperatures (e.g., $T_{t}, T_{t-1}, T_{t-2}$) as inputs to the neural network. This allows the model to capture the thermal gradient and phase lag implicitly.
