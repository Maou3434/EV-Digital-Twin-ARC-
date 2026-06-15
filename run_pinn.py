import os
import glob
import json
import random
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.dataset import ThermalDataset, FileGroupedBatchSampler, FEATURES, TARGET
from utils.preprocessing import preprocess_pipeline
from models.pinn import PhysicsInformedNN
from utils.metrics import calculate_metrics
from utils.plotting import set_style, COLOR_TRUE, COLOR_BASE, COLOR_PINN, COLOR_RESID, COLOR_TEXT

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def seed_everything(seed=42):
    """Sets seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Detects CUDA availability."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables to pass state between pipeline functions
_config = None
_device = None
_scaler_X = None
_scaler_y = None
_train_paths = None
_val_paths = None
_test_paths = None
_model = None
_history = None
_hA_history = []
_single_step_metrics = None
_rollout_metrics = None
_predictions_df = None
_rollout_df = None
_test_residuals = None

def preprocess():
    global _config, _scaler_X, _scaler_y, _train_paths, _val_paths, _test_paths
    print("\n" + "="*50)
    print(" 1. PREPROCESSING PHASE (PINN)")
    print("="*50)
    
    _config = load_config()
    paths = _config['paths']
    dataset_dir = paths['dataset_dir']
    
    # 1. Identify all train and test files
    train_files = sorted(glob.glob(os.path.join(dataset_dir, "TRAIN_*.csv")))
    _test_paths = sorted(glob.glob(os.path.join(dataset_dir, "TEST_*.csv")))
    
    if not train_files:
        raise FileNotFoundError(f"No TRAIN_*.csv files found in {dataset_dir}")
    if not _test_paths:
        raise FileNotFoundError(f"No TEST_*.csv files found in {dataset_dir}")
        
    print(f"Found {len(train_files)} training files and {len(_test_paths)} test files.")
    
    # 2. Fit/load scalers
    scaler_X_path = os.path.join(paths['scalers_dir'], "scaler_X.joblib")
    scaler_y_path = os.path.join(paths['scalers_dir'], "scaler_y.joblib")
    
    if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
        print("Loading pre-fit scalers...")
        _scaler_X = joblib.load(scaler_X_path)
        _scaler_y = joblib.load(scaler_y_path)
    else:
        print("Fitting scalers incrementally...")
        _scaler_X, _scaler_y = preprocess_pipeline()
        
    # 3. Split train files into Train/Val sets (90/10) to monitor validation loss for early stopping
    seed_everything(_config['training']['random_seed'])
    _train_paths, _val_paths = train_test_split(train_files, test_size=0.1, random_state=_config['training']['random_seed'])
    print(f"Split training files into {len(_train_paths)} train and {len(_val_paths)} validation files.")

def train():
    global _config, _device, _scaler_X, _scaler_y, _train_paths, _val_paths, _model, _history, _hA_history
    print("\n" + "="*50)
    print(" 2. TRAINING PHASE (Physics-Informed Neural Network)")
    print("="*50)
    
    _device = get_device()
    print(f"Training on device: {_device}")
    
    train_cfg = _config['training']
    pinn_cfg = _config['pinn']
    phys_cfg = _config['physics']
    
    # 1. Datasets & DataLoaders
    print("Loading training dataset...")
    train_dataset = ThermalDataset(_train_paths, _scaler_X, _scaler_y)
    train_sampler = FileGroupedBatchSampler(train_dataset, batch_size=pinn_cfg['batch_size'], shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    
    print("Loading validation dataset...")
    val_dataset = ThermalDataset(_val_paths, _scaler_X, _scaler_y)
    val_sampler = FileGroupedBatchSampler(val_dataset, batch_size=pinn_cfg['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)
    
    # 2. Instantiate Model, Optimizer, Mixed Precision Scaler
    _model = PhysicsInformedNN(
        input_dim=pinn_cfg['input_dim'],
        hidden_dim=pinn_cfg['hidden_dim'],
        mCp=phys_cfg['mCp'],
        initial_hA=phys_cfg['initial_hA']
    ).to(_device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(_model.parameters(), lr=pinn_cfg['learning_rate'])
    grad_scaler = torch.cuda.amp.GradScaler()
    
    # 3. Setup Noise std and scaler standard dev of Temperature_C to inject noise in physical space
    noise_std = train_cfg['noise_std']
    T_std = _scaler_X.scale_[0]
    scaled_noise_std = noise_std / T_std
    
    # Extract scaler tensors for fast on-device differentiable inverse-scaling of physics losses
    mean_X = torch.tensor(_scaler_X.mean_, dtype=torch.float32, device=_device)
    std_X = torch.tensor(_scaler_X.scale_, dtype=torch.float32, device=_device)
    mean_y = torch.tensor(_scaler_y.mean_, dtype=torch.float32, device=_device)
    std_y = torch.tensor(_scaler_y.scale_, dtype=torch.float32, device=_device)
    
    # 4. Training Loop
    epochs = pinn_cfg['epochs']
    lambda_physics = pinn_cfg['lambda_physics']
    patience = train_cfg['early_stopping_patience']
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None
    
    _history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_data_loss": [],
        "train_phys_loss": [],
        "val_data_loss": [],
        "val_phys_loss": [],
        "hA": []
    }
    
    _hA_history = []
    
    for epoch in range(1, epochs + 1):
        _model.train()
        train_loss = 0.0
        train_d_loss = 0.0
        train_p_loss = 0.0
        train_count = 0
        
        for batch_X, batch_y, batch_dt in train_loader:
            batch_X, batch_y, batch_dt = batch_X.to(_device), batch_y.to(_device), batch_dt.to(_device)
            
            # Inject noise on Temperature_C (index 0) in scaled equivalent of physical scale
            if noise_std > 0.0:
                batch_X_train = batch_X.clone()
                noise = torch.randn_like(batch_X_train[:, 0]) * scaled_noise_std
                batch_X_train[:, 0] += noise
            else:
                batch_X_train = batch_X
                
            optimizer.zero_grad()
            
            # Autocast mixed precision
            with torch.cuda.amp.autocast(enabled=(_device.type == "cuda")):
                pred_y_scaled = _model(batch_X_train)
                
                # A. Data Loss (scaled space)
                data_loss = criterion(pred_y_scaled, batch_y)
                
                # B. Physics Loss (physical space K/s)
                # Differentiable inverse scaling of inputs and outputs
                batch_X_phys = batch_X_train * std_X + mean_X
                pred_y_phys = pred_y_scaled * std_y + mean_y
                
                residuals = _model.compute_residual(batch_X_phys, pred_y_phys, batch_dt)
                physics_loss = torch.mean(residuals ** 2)
                
                # C. Total Loss
                total_loss = data_loss + lambda_physics * physics_loss
                
            grad_scaler.scale(total_loss).backward()
            
            # Gradient clipping
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(_model.parameters(), max_norm=train_cfg['gradient_clip'])
            
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            n = batch_X.size(0)
            train_loss += total_loss.item() * n
            train_d_loss += data_loss.item() * n
            train_p_loss += physics_loss.item() * n
            train_count += n
            
        train_loss /= train_count
        train_d_loss /= train_count
        train_p_loss /= train_count
        
        # Validation Loop
        _model.eval()
        val_loss = 0.0
        val_d_loss = 0.0
        val_p_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch_X, batch_y, batch_dt in val_loader:
                batch_X, batch_y, batch_dt = batch_X.to(_device), batch_y.to(_device), batch_dt.to(_device)
                
                pred_y_scaled = _model(batch_X)
                data_loss = criterion(pred_y_scaled, batch_y)
                
                batch_X_phys = batch_X * std_X + mean_X
                pred_y_phys = pred_y_scaled * std_y + mean_y
                
                residuals = _model.compute_residual(batch_X_phys, pred_y_phys, batch_dt)
                physics_loss = torch.mean(residuals ** 2)
                total_loss = data_loss + lambda_physics * physics_loss
                
                n = batch_X.size(0)
                val_loss += total_loss.item() * n
                val_d_loss += data_loss.item() * n
                val_p_loss += physics_loss.item() * n
                val_count += n
                
        val_loss /= val_count
        val_d_loss /= val_count
        val_p_loss /= val_count
        
        current_hA = _model.get_hA().item()
        _hA_history.append({"epoch": epoch, "hA": current_hA})
        
        # GPU Memory Logging
        gpu_mem = 0.0
        if _device.type == "cuda":
            gpu_mem = torch.cuda.memory_allocated(_device) / (1024 ** 2)
            
        # Logging
        print(f"Epoch {epoch:03d}/{epochs} | Total Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Data Loss: {train_d_loss:.6f} | Physics Loss: {train_p_loss:.6f} | hA: {current_hA:.5f} W/K | GPU Mem: {gpu_mem:.1f} MB")
        
        _history["epoch"].append(epoch)
        _history["train_loss"].append(train_loss)
        _history["val_loss"].append(val_loss)
        _history["train_data_loss"].append(train_d_loss)
        _history["train_phys_loss"].append(train_p_loss)
        _history["val_data_loss"].append(val_d_loss)
        _history["val_phys_loss"].append(val_p_loss)
        _history["hA"].append(current_hA)
        
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_weights = {k: v.cpu().clone() for k, v in _model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Restoring best weights.")
                _model.load_state_dict({k: v.to(_device) for k, v in best_weights.items()})
                break

def evaluate():
    global _model, _device, _scaler_X, _scaler_y, _test_paths, _single_step_metrics, _predictions_df, _test_residuals
    print("\n" + "="*50)
    print(" 3. SINGLE-STEP EVALUATION PHASE (PINN)")
    print("="*50)
    
    _model.eval()
    test_dataset = ThermalDataset(_test_paths, _scaler_X, _scaler_y)
    test_sampler = FileGroupedBatchSampler(test_dataset, batch_size=_config['pinn']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=0)
    
    all_preds = []
    all_targets = []
    all_temps_t = []
    all_residuals = []
    
    mean_X = torch.tensor(_scaler_X.mean_, dtype=torch.float32, device=_device)
    std_X = torch.tensor(_scaler_X.scale_, dtype=torch.float32, device=_device)
    mean_y = torch.tensor(_scaler_y.mean_, dtype=torch.float32, device=_device)
    std_y = torch.tensor(_scaler_y.scale_, dtype=torch.float32, device=_device)
    
    with torch.no_grad():
        for batch_X, batch_y, batch_dt in test_loader:
            batch_X, batch_dt = batch_X.to(_device), batch_dt.to(_device)
            pred = _model(batch_X)
            
            # Reconstruct physical features and compute residual
            batch_X_phys = batch_X * std_X + mean_X
            pred_y_phys = pred * std_y + mean_y
            
            residuals = _model.compute_residual(batch_X_phys, pred_y_phys, batch_dt)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())
            all_temps_t.append(batch_X_phys[:, 0].cpu().numpy())
            all_residuals.append(residuals.cpu().numpy())
            
    preds_scaled = np.vstack(all_preds)
    targets_scaled = np.vstack(all_targets)
    temps_t_physical = np.concatenate(all_temps_t)
    _test_residuals = np.vstack(all_residuals).flatten()
    
    # Reconstruct physical predicted and true Delta_T
    pred_delta_phys = _scaler_y.inverse_transform(preds_scaled).flatten()
    true_delta_phys = _scaler_y.inverse_transform(targets_scaled).flatten()
    
    # Reconstruct physical absolute next temperature
    pred_temp_next = temps_t_physical + pred_delta_phys
    true_temp_next = temps_t_physical + true_delta_phys
    
    _single_step_metrics = calculate_metrics(true_temp_next, pred_temp_next)
    print("Single-step Absolute Temperature Metrics on Unseen Drive Cycle (TEST):")
    for k, v in _single_step_metrics.items():
        print(f"  {k}: {v:.5f}")
    print(f"  Final trained convective coefficient hA: {_model.get_hA().item():.5f} W/K")
        
    _predictions_df = pd.DataFrame({
        "True_Temperature": true_temp_next,
        "Predicted_Temperature": pred_temp_next,
        "True_Delta_T": true_delta_phys,
        "Predicted_Delta_T": pred_delta_phys,
        "Physics_Residual": _test_residuals
    })

def rollout():
    global _model, _scaler_X, _scaler_y, _test_paths, _rollout_metrics, _rollout_df
    print("\n" + "="*50)
    print(" 4. AUTOREGRESSIVE ROLLOUT EVALUATION (PINN)")
    print("="*50)
    
    _model.eval()
    
    all_file_metrics = []
    rollout_dfs = []
    
    mean_X, std_X = _scaler_X.mean_, _scaler_X.scale_
    mean_y, std_y = _scaler_y.mean_, _scaler_y.scale_
    
    print(f"Running autoregressive rollout on all {len(_test_paths)} test files...")
    for idx, path in enumerate(_test_paths):
        df = pd.read_csv(path)
        if not df["Time_s"].is_monotonic_increasing:
            df = df.sort_values(by="Time_s").reset_index(drop=True)
            
        # Create dt
        df["Time_next"] = df["Time_s"].shift(-1)
        df["dt"] = df["Time_next"] - df["Time_s"]
        df = df.dropna().reset_index(drop=True)
        df = df[df["dt"] >= 0.001].reset_index(drop=True)
        
        N = len(df)
        if N == 0:
            continue
            
        # Ground truth absolute temperature
        T_true = df["Temperature_C"].values
        
        # Autoregressive rollout predictions
        T_pred = np.zeros(N)
        T_pred[0] = T_true[0] # Initialize with ground truth starting temperature
        
        # Extracted test features except Temperature_C which we update dynamically
        currents = df["Current_A"].values
        voltages = df["Voltage_V"].values
        socs = df["SOC_pct"].values
        plosses = df["PowerLoss_W"].values
        ambients = df["AmbientTemp_C"].values
        masses = df["MassScale"].values
        times = df["Time_s"].values
        
        # Sequentially roll out
        for t in range(N - 1):
            # 1. Construct input vector in physical scale
            x_t = np.array([[
                T_pred[t],
                currents[t],
                voltages[t],
                socs[t],
                plosses[t],
                ambients[t],
                masses[t]
            ]])
            
            # 2. Scale features
            x_scaled = (x_t - mean_X) / std_X
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=_device)
            
            # 3. Model predict Delta_T
            with torch.no_grad():
                pred_scaled = _model(x_tensor).cpu().item()
                
            # 4. Inverse scale Delta_T prediction
            delta_T_pred = pred_scaled * std_y[0] + mean_y[0]
            
            # 5. Predict next temperature
            T_pred[t + 1] = T_pred[t] + delta_T_pred
            
        mae = np.mean(np.abs(T_true - T_pred))
        rmse = np.sqrt(np.mean((T_true - T_pred) ** 2))
        max_err = np.max(np.abs(T_true - T_pred))
        
        all_file_metrics.append({
            "MAE": mae,
            "RMSE": rmse,
            "MaxError": max_err
        })
        
        # Save trajectory dataframe
        trajectory_df = pd.DataFrame({
            "Trajectory_ID": idx,
            "Time_s": times,
            "True_Temperature": T_true,
            "Predicted_Temperature": T_pred,
            "Absolute_Error": np.abs(T_true - T_pred)
        })
        rollout_dfs.append(trajectory_df)
        
    # Aggregate metrics
    metrics_summary_df = pd.DataFrame(all_file_metrics)
    _rollout_metrics = {
        "Rollout_MAE": float(metrics_summary_df["MAE"].mean()),
        "Rollout_RMSE": float(metrics_summary_df["RMSE"].mean()),
        "Maximum_Error": float(metrics_summary_df["MaxError"].max())
    }
    
    print("Rollout Evaluation Metrics across ALL test trajectories:")
    for k, v in _rollout_metrics.items():
        print(f"  {k}: {v:.5f}")
        
    _rollout_df = pd.concat(rollout_dfs, ignore_index=True)

def plots():
    global _history, _rollout_df, _test_residuals, _config
    print("\n" + "="*50)
    print(" 5. PLOTTING PHASE (PINN)")
    print("="*50)
    
    paths = _config['paths']
    out_dir = paths['results_pinn_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    set_style()
    
    # 1. Loss convergence plot (log scale)
    plt.figure(figsize=(10, 6))
    plt.plot(_history['epoch'], _history['train_loss'], label="Total Train Loss", color=COLOR_TRUE, linewidth=2.5)
    plt.plot(_history['epoch'], _history['val_loss'], label="Total Val Loss", color=COLOR_BASE, linewidth=1.8, linestyle="--")
    plt.plot(_history['epoch'], _history['train_data_loss'], label="Data Loss (Train)", color=COLOR_PINN, linewidth=1.5, linestyle="-.")
    plt.plot(_history['epoch'], _history['train_phys_loss'], label="Physics Loss (Train)", color="#10B981", linewidth=1.5, linestyle=":")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("PINN Loss Convergence Breakdown", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    loss_png = os.path.join(out_dir, "loss.png")
    plt.savefig(loss_png, dpi=300)
    plt.close()
    print(f"Saved PINN loss curve to {loss_png}")
    
    # 2. Residual Distribution Plot
    plt.figure(figsize=(10, 6))
    # Calculate statistics
    mean_res = np.mean(_test_residuals)
    std_res = np.std(_test_residuals)
    
    # Histogram
    count, bins, ignored = plt.hist(_test_residuals, bins=100, color=COLOR_RESID, edgecolor='#7FB7BE', alpha=0.8, density=True)
    # Normal curve fit
    fit_curve = (1 / (std_res * np.sqrt(2 * np.pi))) * np.exp(-((bins - mean_res) ** 2) / (2 * std_res ** 2))
    plt.plot(bins, fit_curve, linewidth=2, color=COLOR_TRUE, label=f"Normal Fit (μ={mean_res:.3f}, σ={std_res:.3f})")
    
    plt.xlabel("Physics Residual (K/s)")
    plt.ylabel("Probability Density")
    plt.title("PINN Physics Residual Distribution (TEST set)\ndT/dt - (PowerLoss - Q_loss) / (mCp * MassScale)", fontsize=13, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    residual_png = os.path.join(out_dir, "residual.png")
    plt.savefig(residual_png, dpi=300)
    plt.close()
    print(f"Saved residual distribution histogram to {residual_png}")
    
    # 3. Rollout Trajectory comparison (first test trajectory)
    traj0 = _rollout_df[_rollout_df["Trajectory_ID"] == 0]
    plt.figure(figsize=(12, 6))
    plt.plot(traj0["Time_s"], traj0["True_Temperature"], label="Ground Truth", color=COLOR_TRUE, linewidth=2.5)
    plt.plot(traj0["Time_s"], traj0["Predicted_Temperature"], label="PINN Rollout", color="#10B981", linewidth=1.8, linestyle="-")
    plt.xlabel("Time (s)")
    plt.ylabel("Battery Temperature (°C)")
    plt.title("Autoregressive Trajectory Rollout Comparison (Test Cycle 0) - PINN", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    rollout_png = os.path.join(out_dir, "rollout_trajectory.png")
    plt.savefig(rollout_png, dpi=300)
    plt.close()
    print(f"Saved PINN rollout trajectory comparison plot to {rollout_png}")

def save():
    global _model, _config, _scaler_X, _scaler_y, _single_step_metrics, _rollout_metrics, _predictions_df, _rollout_df, _hA_history
    print("\n" + "="*50)
    print(" 6. SAVING DELIVERABLES (PINN)")
    print("="*50)
    
    paths = _config['paths']
    out_dir = paths['results_pinn_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Model Weights
    model_pt = os.path.join(out_dir, "model.pt")
    torch.save({
        'model_state_dict': _model.state_dict(),
        'scaler_X_mean': _scaler_X.mean_.tolist(),
        'scaler_X_scale': _scaler_X.scale_.tolist(),
        'scaler_y_mean': _scaler_y.mean_.tolist(),
        'scaler_y_scale': _scaler_y.scale_.tolist()
    }, model_pt)
    print(f"Saved PINN weights and scalers metadata to {model_pt}")
    
    # 2. Convective Heat Transfer coefficient (hA) history CSV
    hA_df = pd.DataFrame(_hA_history)
    hA_csv = os.path.join(out_dir, "hA_history.csv")
    hA_df.to_csv(hA_csv, index=False)
    print(f"Saved convective coefficient learning history to {hA_csv}")
    
    # 3. Metrics JSON files
    metrics_json = os.path.join(out_dir, "metrics.json")
    with open(metrics_json, 'w') as f:
        json.dump(_single_step_metrics, f, indent=4)
    print(f"Saved single-step metrics to {metrics_json}")
    
    rollout_metrics_json = os.path.join(out_dir, "rollout_metrics.json")
    with open(rollout_metrics_json, 'w') as f:
        json.dump(_rollout_metrics, f, indent=4)
    print(f"Saved rollout metrics to {rollout_metrics_json}")
    
    # 4. CSV predictions and rollout files
    preds_csv = os.path.join(out_dir, "predictions.csv")
    _predictions_df.to_csv(preds_csv, index=False)
    print(f"Saved predictions CSV to {preds_csv}")
    
    rollout_csv = os.path.join(out_dir, "rollout.csv")
    _rollout_df.to_csv(rollout_csv, index=False)
    print(f"Saved rollout CSV to {rollout_csv}")
    
    print("\n" + "="*50)
    print(" PIPELINE COMPLETE")
    print("="*50)

def main():
    preprocess()
    train()
    evaluate()
    rollout()
    plots()
    save()

if __name__ == "__main__":
    print("EV Battery Thermal Digital Twin - Physics-Informed Neural Network Script prepared.")
    print("Run this script with: python run_pinn.py")
