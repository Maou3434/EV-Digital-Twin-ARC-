---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #f4f8fb
color: #334155
html: true
style: |
  section {
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    padding: 40px 60px;
    background-color: #f4f8fb;
    color: #334155;
  }
  h1 {
    color: #1a365d;
    font-size: 1.8em;
    margin-top: 0px;
    margin-bottom: 25px;
    border-bottom: 3px solid #3182ce;
    padding-bottom: 10px;
    font-weight: 700;
  }
  h2 {
    color: #2c5282;
    font-size: 1.15em;
    margin-top: 0px;
    margin-bottom: 10px;
    font-weight: 600;
  }
  p, li {
    font-size: 0.85em;
    line-height: 1.5;
    color: #4a5568;
  }
  ul {
    margin-top: 5px;
    margin-bottom: 10px;
    padding-left: 20px;
  }
  code {
    background-color: #edf2f7;
    color: #2b6cb0;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 0.85em;
  }
  pre {
    background-color: #1a202c;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px;
    margin: 5px 0px;
  }
  pre code {
    background-color: transparent;
    padding: 0;
    color: #48bb78;
    font-size: 0.75em;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 25px;
  }
  .card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
  }
  .highlight-blue { border-left: 5px solid #3182ce; }
  .highlight-green { border-left: 5px solid #38a169; }
  .highlight-red { border-left: 5px solid #e53e3e; }
  .highlight-amber { border-left: 5px solid #dd6b20; }
  .text-highlight { color: #3182ce; font-weight: bold; }
  .text-danger { color: #e53e3e; font-weight: bold; }
  .text-success { color: #38a169; font-weight: bold; }
  
  /* Title Slide styling */
  .title-circle {
    width: 560px;
    height: 560px;
    border-radius: 50%;
    background: rgba(26, 43, 73, 0.92);
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 70px;
    box-sizing: border-box;
    color: #ffffff;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
  }
  .title-circle h1 {
    color: #63b3ed;
    font-size: 2.2em;
    border: none;
    padding: 0;
    margin-bottom: 15px;
    line-height: 1.2;
    font-weight: 800;
  }
  .title-circle p {
    color: #e2e8f0;
    font-size: 0.95em;
    line-height: 1.4;
    margin-bottom: 30px;
  }
  .title-circle .meta {
    font-size: 0.75em;
    color: #a0aec0;
    border-top: 1px solid rgba(255,255,255,0.1);
    padding-top: 15px;
  }

  /* Timeline styling */
  .timeline {
    display: flex;
    justify-content: space-between;
    position: relative;
    margin-top: 30px;
    margin-bottom: 20px;
  }
  .timeline::before {
    content: '';
    position: absolute;
    top: 25px;
    left: 8%;
    right: 8%;
    height: 4px;
    background-color: #e2e8f0;
    z-index: 1;
  }
  .timeline-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    z-index: 2;
    width: 18%;
  }
  .timeline-circle {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background-color: #ffffff;
    border: 4px solid #3182ce;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    color: #1a365d;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    font-size: 1.1em;
  }
  .timeline-step.active .timeline-circle {
    background-color: #3182ce;
    color: #ffffff;
  }
  .timeline-title {
    margin-top: 12px;
    font-size: 0.75em;
    font-weight: 700;
    color: #2d3748;
    text-align: center;
  }
  
  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 0.8em;
  }
  th {
    background-color: #2b6cb0;
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 8px 12px;
    border-bottom: 1px solid #e2e8f0;
    background-color: #ffffff;
  }
  tr:nth-child(even) td {
    background-color: #f7fafc;
  }
---

<!-- _backgroundImage: url('slide_bg_clean.png') -->
<!-- _backgroundColor: #0b0f19 -->
<!-- _paginate: false -->
<div class="title-circle">
  <h1>EV Battery<br>Thermal Twin</h1>
  <p>Physics-Informed Neural Network (PINN) vs. Data-Driven MLP</p>
  <div class="meta">
    <strong>Technical Review & Pipeline Analysis</strong><br>
    Prepared by Antigravity AI Codebase Specialist
  </div>
</div>

---

# 1. The Problem We Are Solving

<div class="grid-2">
  <div class="card highlight-blue">
    <h2>The Target System</h2>
    <p>For electric vehicle safety and longevity, the Battery Management System (BMS) must estimate <b>cell core temperatures in real-time</b>.</p>
    <ul>
      <li>High temperatures trigger safety cutoffs.</li>
      <li>Accurate temperature tracking allows faster charging profiles.</li>
      <li>Requires tracking dynamic thermal behavior during driving cycles.</li>
    </ul>
  </div>
  <div class="card highlight-red">
    <h2>The Technical Challenge</h2>
    <p>We are caught between two unsuitable options:</p>
    <ol>
      <li><b>Finite Element Analysis (FEA)</b>: Physically accurate but requires minutes/hours of computation—impossible for real-time edge devices.</li>
      <li><b>Simplified Equations</b>: Fast but inaccurate under rapid transient loads.</li>
    </ol>
    <p style="margin-top: 15px;" class="text-highlight">Goal: Deep learning speed with physical convergence guarantees.</p>
  </div>
</div>

---

# 2. What Has Been Done Before

<div class="grid-2">
  <div class="card highlight-amber">
    <h2>Approach A: Pure Data NNs</h2>
    <p>Neural networks trained strictly on data signals (Voltage, Current, SOC) to predict temperature changes.</p>
    <p class="text-success">✔ High Flexibility:</p>
    <p style="margin-top: -10px; font-size: 0.8em;">Fits highly non-linear dynamics easily without physical parameters.</p>
    <p class="text-danger">✖ Unbounded Extrapolation:</p>
    <p style="margin-top: -10px; font-size: 0.8em;">Violates conservation laws and drifts completely when exposed to out-of-distribution cycles.</p>
  </div>
  <div class="card highlight-amber">
    <h2>Approach B: Simplified RC Models</h2>
    <p>Resistance-Capacitance thermal equivalent circuits representing heat transfer via ODEs.</p>
    <p class="text-success">✔ Physical Consistency:</p>
    <p style="margin-top: -10px; font-size: 0.8em;">Guarantees thermal bounds and respects energy conservation laws.</p>
    <p class="text-danger">✖ Calibration Bottleneck:</p>
    <p style="margin-top: -10px; font-size: 0.8em;">Extremely tedious to calibrate and fails to capture structural multi-node gradients.</p>
  </div>
</div>

---

# 3. What Has NOT Been Done (The Gap)

<div class="grid-2">
  <div class="card highlight-red">
    <h2>The Multi-Step Rollout Gap</h2>
    <p>Previous ML models are evaluated strictly on <b>single-step-ahead prediction</b> ($T_t \to T_{t+1}$).</p>
    <p>Because the step size is tiny ($dt=0.01$s), temperature changes very little. Models exploit this <b>"identity shortcut"</b> to report artificial $R^2 = 1.00000$ scores, but fail completely when run autoregressively for long simulations.</p>
  </div>
  <div class="card highlight-red">
    <h2>Physical Prior Mismatches</h2>
    <p>Prior attempts at regularizing networks with physics equations hardcode generalized parameters (like textbook mass and heat capacities).</p>
    <p>This creates a direct conflict with the empirical data generation environment, causing the optimizer to reject physical bounds.</p>
  </div>
</div>

---

# 4. The Idea Behind the Solution

<div class="card highlight-green" style="margin-bottom: 25px;">
  <h2>A Physics-Informed Neural Network (PINN) with Trainable Parameters</h2>
  <p>Our solution couples a deep neural network's representative power with the governing lumped-parameter thermal differential equation directly inside the optimizer loss graph.</p>
</div>

<div class="grid-2">
  <div class="card highlight-blue">
    <h2>1. Physics-Guided Loss Regularizer</h2>
    <p>Instead of hoping the network behaves physically, we enforce Newton's cooling law as a penalty term: <code>Loss = Data_Loss + Physics_Loss</code>.</p>
  </div>
  <div class="card highlight-blue">
    <h2>2. Dynamic Parameter Inference</h2>
    <p>Rather than hardcoding the heat transfer coefficient ($hA$), we define it as a <b>trainable model variable</b>. The model dynamically learns the real physical parameters from data stream behaviors.</p>
  </div>
</div>

---

# 5. Step-by-Step Implementation Timeline

We solved this challenge systematically across five distinct development phases:

<div class="timeline">
  <div class="timeline-step active">
    <div class="timeline-circle">1</div>
    <div class="timeline-title">Data Sweep & Cycle Isolation</div>
  </div>
  <div class="timeline-step">
    <div class="timeline-circle">2</div>
    <div class="timeline-title">Lazy Loading Stream</div>
  </div>
  <div class="timeline-step">
    <div class="timeline-circle">3</div>
    <div class="timeline-title">Dual-Loss Formulation</div>
  </div>
  <div class="timeline-step">
    <div class="timeline-circle">4</div>
    <div class="timeline-title">Physical Discrepancy Fix</div>
  </div>
  <div class="timeline-step">
    <div class="timeline-circle">5</div>
    <div class="timeline-title">Hardware Tuning</div>
  </div>
</div>

<div class="card highlight-blue" style="margin-top: 10px;">
  <p style="margin: 0; font-size: 0.85em; text-align: center;">
    Let's walk through the technical and engineering details of each step.
  </p>
</div>

---

# Step 1: Data Sweep & Split Isolation

To train the models, we executed a parameterized grid simulation in MATLAB/Simulink:

<div class="grid-2">
  <div class="card highlight-blue">
    <h2>The Grid Configurations</h2>
    <ul>
      <li><b>Model</b>: <code>Ebike_Thermal_DT_v1.slx</code> (electrothermal)</li>
      <li><b>Drive Cycles</b>: FTP75, US06, UDDS, HUDDS, WLTP1, WLTP2</li>
      <li><b>Initial Temperatures</b>: 0°C, 10°C, 25°C, 40°C, 50°C</li>
      <li><b>Mass Scales</b>: 0.8, 1.0, 1.2 (affects thermal inertia)</li>
    </ul>
    <p style="margin-top: 10px;" class="text-highlight">Total Missions Simulated: 540 files</p>
  </div>
  <div class="card highlight-green">
    <h2>Preventing Data Leakage</h2>
    <p>Standard random splits leak temporal correlations, inflating model accuracy.</p>
    <p><b>Decision</b>: We reserved the entire <code>cycleWLTP2</code> drive cycle exclusively as the <b>TEST SET</b>. The model has never seen this dynamic drive pattern during training, verifying real-world generalization.</p>
  </div>
</div>

---

# Step 2: Stream Preprocessing

To handle large-scale dataset loading without crashing RAM, we designed a custom stream pipeline:

<div class="grid-2">
  <div class="card highlight-blue">
    <h2>1. Incremental Preprocessing</h2>
    <p>We designed standardizing scalers to fit incrementally via <code>joblib</code>, avoiding loading the entire dataset into memory simultaneously.</p>
    <p>Saved features: <code>scaler_X.joblib</code>, <code>scaler_y.joblib</code>.</p>
  </div>
  <div class="card highlight-blue">
    <h2>2. Sequential Batch Sampler</h2>
    <p>Global shuffling destroys time-series continuity, preventing correct calculations of $\frac{dT}{dt}$.</p>
    <p>Our custom sampler ensures that each batch only draws indices from a <b>single simulation file sequentially</b>, maintaining temporal causality.</p>
  </div>
</div>

<div class="card highlight-green" style="margin-top: 15px; padding: 12px;">
  <h2>3. On-Demand Dataset Loading</h2>
  <p style="margin: 0; font-size: 0.8em;">The <code>ThermalDataset</code> dynamically loads file contents on-demand, caching files as needed. Pinned memory buffers facilitate direct transfers to CUDA page-locked memory.</p>
</div>

---

# Step 3: Dual-Loss Model Formulations

<div class="grid-2">
  <div class="card highlight-blue">
    <h2>Baseline MLP</h2>
    <p>Standard Feedforward network mapping 7 features to temperature delta ($\Delta T$).</p>
    <pre><code class="language-python">self.network = nn.Sequential(
    nn.Linear(7, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1)
)</code></pre>
  </div>
  <div class="card highlight-green">
    <h2>Physics-Informed NN</h2>
    <p>Identical architecture, but convective parameter $hA$ is trainable. We map it via <code>Softplus</code> to avoid vanishing gradients during optimization:</p>
    <pre><code class="language-python"># Trainable parameter initialized at 0.1
self.hA_raw = nn.Parameter(
    torch.tensor(raw_init)
)
def get_hA(self):
    return F.softplus(self.hA_raw)</code></pre>
  </div>
</div>

---

# Step 3: PINN Governing Equation Residuals

The PINN checks physical validity at every epoch using the lumped capacity equation:

<div style="text-align: center; margin: 15px 0;">
  <span style="font-size: 1.25em; color: #1a365d; font-weight: bold; background: #edf2f7; padding: 8px 20px; border-radius: 6px; border: 1px dashed #cbd5e1;">
    $$\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}$$
  </span>
</div>

<div class="grid-2">
  <div class="card highlight-blue">
    <h2>Governing Physical Law</h2>
    <p>$$\frac{dT}{dt} = \frac{P_{loss} - hA(T - T_{amb})}{m C_p \cdot \text{MassScale}}$$</p>
    <ul>
      <li>$P_{loss}$ is internal heat generation.</li>
      <li>$hA(T-T_{amb})$ is convection cooling.</li>
      <li>$m C_p$ is thermal capacity.</li>
    </ul>
  </div>
  <div class="card highlight-blue">
    <h2>Residual Loss Computation</h2>
    <pre style="margin:0;"><code class="language-python"># Inside forward / loss pass:
dTdt_pred = Delta_T_pred_phys / dt
hA = self.get_hA()
Q_loss = hA * (T_t - AmbientTemp)
rhs = (PowerLoss - Q_loss) / (mCp * Mass)
residual = dTdt_pred - rhs
loss_phys = torch.mean(residual ** 2)</code></pre>
  </div>
</div>

---

# Step 4: Diagnosing Physical Conflicts

During testing, the PINN failed to converge and $hA$ clamped to its minimum. We audited the dataset parameters:

<div class="grid-2">
  <div class="card highlight-red">
    <h2>1. The $mC_p$ Mismatch</h2>
    <p>The code assumed a thermal capacity of $m C_p = 4000$ J/K.</p>
    <p>We ran a linear regression directly on the Simulink data stream and found the actual capacity was **$900$ J/K** (4.4x lower). Enforcing a prior that was so far off created conflicting gradients. We corrected this in <code>config.yaml</code>.</p>
  </div>
  <div class="card highlight-red">
    <h2>2. Casing-Core Phase Lag</h2>
    <p>Simulink utilized a <b>two-node</b> core-surface model. When current drops to 0, casing temperature continues to rise as heat transfers outward from the hotter core.</p>
    <p>A <b>single-node</b> equation cannot model this lag, requiring a negative $hA$ to fit. Enforcing $hA > 0$ caused the optimizer to push $hA$ to its lower bound.</p>
  </div>
</div>

---

# Step 5: Hardware & Performance Optimizations

To ensure rapid iterations on local developer hardware (NVIDIA RTX 4060 Laptop GPU), we optimized performance:

<div class="grid-2">
  <div class="card highlight-green">
    <h2>Memory & Sync Bottlenecks</h2>
    <ul>
      <li><b>Non-Blocking Transfers</b>: Configured <code>.to(device, non_blocking=True)</code> to allow GPU computations and memory copies to overlap.</li>
      <li><b>Pinned Memory</b>: Activated page-locked memory (<code>pin_memory=True</code>) in the data loader.</li>
    </ul>
  </div>
  <div class="card highlight-green">
    <h2>Tensor Core Acceleration</h2>
    <ul>
      <li><b>AMP Autocast</b>: Enabled float16 Automatic Mixed Precision (AMP) to maximize Tensor Core utilization.</li>
      <li><b>Precision Selection</b>: Used <code>torch.set_float32_matmul_precision("high")</code> to enable TF32 support.</li>
    </ul>
  </div>
</div>

<div class="card highlight-blue" style="margin-top: 15px; padding: 10px 16px;">
  <p style="margin: 0; font-size: 0.8em; text-align: center;">
    <b>Dataloader Throughput Profiling</b>: Auto-profiled configurations, selecting <b>8 workers with prefetch factor 2</b> (highest throughput of ~15,000 samples/sec).
  </p>
</div>

---

# Summary of Results: The Exposure Bias Divergence

We evaluated both models on a full **1500-second autoregressive rollout** on the unseen `WLTP2` test set:

| Evaluation Metric | Baseline MLP | Physics-Informed NN (PINN) |
| :--- | :---: | :---: |
| Single-Step Test MAE | **$0.00002$ K** | **$0.00003$ K** |
| **Rollout MAE (compounded)** | <span class="text-danger">**3.85 K**</span> | <span class="text-danger">**4.70 K**</span> |
| **Rollout Max Error** | <span class="text-danger">**8.56 K**</span> | <span class="text-success">**6.46 K**</span> |

<div class="grid-2" style="margin-top: 15px;">
  <div class="card highlight-blue" style="padding: 10px 16px;">
    <h2>Compounding Error</h2>
    <p style="font-size: 0.8em; margin: 0;">During rollout, the MLP's prediction errors accumulate, causing steady trajectory drift.</p>
  </div>
  <div class="card highlight-green" style="padding: 10px 16px;">
    <h2>Physical Bound</h2>
    <p style="font-size: 0.8em; margin: 0;">The PINN's physical loss constraints limit peak error drift, but are bound by the single-node physical model limitation.</p>
  </div>
</div>

---

# Roadmap: Next Stabilization Phases

To resolve the remaining gaps and deliver a highly accurate Digital Twin, we propose the following roadmap:

<div class="grid-2">
  <div class="card highlight-blue">
    <h2>Phase 1: Rollout-Based Loss</h2>
    <p>Train the model using <b>Truncated Backpropagation Through Time (BPTT)</b> over 50-step prediction windows instead of 1-step targets. This directly minimizes exposure bias.</p>
    <p>Add <b>input noise injection</b> during training to teach the network to recover from its own predicted errors.</p>
  </div>
  <div class="card highlight-green">
    <h2>Phase 2: Two-State PINN</h2>
    <p>Upgrade the governing equations to represent both the Core temperature ($T_c$) and Casing temperature ($T_s$):</p>
    <p>$$\frac{dT_c}{dt} = \frac{P_{loss} - R_{cs}(T_c - T_s)}{C_c}$$</p>
    <p>$$\frac{dT_s}{dt} = \frac{R_{cs}(T_c - T_s) - hA(T_s - T_{amb})}{C_s}$$</p>
    <p>This resolves the $hA$ clamping issue and accounts for core-to-surface heat lag.</p>
  </div>
</div>
