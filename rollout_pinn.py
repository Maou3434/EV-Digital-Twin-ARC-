import os
import glob
import json
import yaml
import joblib
import time
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from utils.dataset import ThermalDataset, FEATURES
from models.pinn import PhysicsInformedNN
from utils.logging_utils import setup_logger, log_system_metrics

logger = setup_logger("rollout", "logs/rollout.log")

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_rollout(model_path, config=None, device=None):
    """
    Core function to run autoregressive rollout on GPU for PINN.
    Can be called directly from other python scripts.
    """
    logger.info(f"Starting autoregressive PINN rollout using model: {model_path}")
    start_rollout_time = time.time()
    
    if config is None:
        config = load_config()
    paths = config['paths']
    dataset_dir = paths['dataset_dir']
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    logger.info("Loading model weights and metadata...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Instantiate PINN Model
    pinn_cfg = config['pinn']
    phys_cfg = config['physics']
    model = PhysicsInformedNN(
        input_dim=pinn_cfg['input_dim'],
        hidden_dim=pinn_cfg['hidden_dim'],
        mCp=phys_cfg['mCp'],
        initial_hA=phys_cfg['initial_hA']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Reconstruct Scalers
    scaler_X_path = os.path.join(paths['scalers_dir'], "scaler_X.joblib")
    scaler_y_path = os.path.join(paths['scalers_dir'], "scaler_y.joblib")
    
    if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
    else:
        scaler_X = StandardScaler()
        scaler_X.mean_ = np.array(checkpoint['scaler_X_mean'])
        scaler_X.scale_ = np.array(checkpoint['scaler_X_scale'])
        scaler_y = StandardScaler()
        scaler_y.mean_ = np.array(checkpoint['scaler_y_mean'])
        scaler_y.scale_ = np.array(checkpoint['scaler_y_scale'])
        
    # Move scaler parameters to GPU for fast on-device scaling operations
    mean_X_gpu = torch.from_numpy(scaler_X.mean_).to(device, dtype=torch.float32)
    std_X_gpu = torch.from_numpy(scaler_X.scale_).to(device, dtype=torch.float32)
    mean_y_gpu = torch.from_numpy(scaler_y.mean_).to(device, dtype=torch.float32)
    std_y_gpu = torch.from_numpy(scaler_y.scale_).to(device, dtype=torch.float32)
    
    test_paths = sorted(glob.glob(os.path.join(dataset_dir, "TEST_*.csv")))
    if not test_paths:
        raise FileNotFoundError(f"No TEST_*.csv files found in {dataset_dir}")
        
    all_file_metrics = []
    rollout_dfs = []
    
    total_files = len(test_paths)
    logger.info(f"Running rollout on {total_files} files...")
    
    for idx, path in enumerate(test_paths):
        file_start_time = time.time()
        logger.info(f"Rollout file {idx + 1}/{total_files}: {os.path.basename(path)}")
        
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
            logger.warning(f"File {os.path.basename(path)} has 0 valid steps after preprocessing.")
            continue
            
        # Ground truth absolute temperature (physical)
        T_true_np = df["Temperature_C"].values
        T_true = torch.from_numpy(T_true_np).to(device, dtype=torch.float32)
        
        # Load features and move to GPU
        features_np = df[FEATURES].values
        features_gpu = torch.from_numpy(features_np).to(device, dtype=torch.float32)
        
        # Pre-scale all features on GPU
        features_scaled = (features_gpu - mean_X_gpu) / std_X_gpu
        
        # Pre-allocate rollout predictions tensor on GPU
        T_pred = torch.zeros(N, device=device, dtype=torch.float32)
        T_pred[0] = T_true[0]
        
        # Sequentially roll out on GPU without CPU roundtrips
        with torch.no_grad():
            for t in range(N - 1):
                T_pred_scaled = (T_pred[t] - mean_X_gpu[0]) / std_X_gpu[0]
                features_scaled[t, 0] = T_pred_scaled
                
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    pred_scaled = model(features_scaled[t:t+1])
                    
                delta_T_pred = pred_scaled * std_y_gpu[0] + mean_y_gpu[0]
                T_pred[t + 1] = T_pred[t] + delta_T_pred.squeeze()
                
                # Progress reporting for large trajectories
                if N > 20000 and (t + 1) % 50000 == 0:
                    elapsed = time.time() - file_start_time
                    steps_per_sec = (t + 1) / elapsed
                    remaining = N - 1 - t
                    eta = remaining / steps_per_sec if steps_per_sec > 0 else 0
                    logger.info(f"    Step {t + 1}/{N} | Speed: {steps_per_sec:.1f} steps/s | ETA: {eta:.1f}s")
                    
        # Copy prediction tensor back to CPU
        T_pred_np = T_pred.cpu().numpy()
        
        # Calculate metrics
        mae = np.mean(np.abs(T_true_np - T_pred_np))
        rmse = np.sqrt(np.mean((T_true_np - T_pred_np) ** 2))
        max_err = np.max(np.abs(T_true_np - T_pred_np))
        
        duration = time.time() - file_start_time
        logger.info(f"Finished rollout in {duration:.1f}s | MAE: {mae:.5f} | RMSE: {rmse:.5f} | Max Error: {max_err:.5f}")
        
        all_file_metrics.append({
            "MAE": mae,
            "RMSE": rmse,
            "MaxError": max_err
        })
        
        trajectory_df = pd.DataFrame({
            "Trajectory_ID": idx,
            "Time_s": df["Time_s"].values,
            "True_Temperature": T_true_np,
            "Predicted_Temperature": T_pred_np,
            "Absolute_Error": np.abs(T_true_np - T_pred_np)
        })
        rollout_dfs.append(trajectory_df)
        
    # Aggregate metrics
    metrics_summary_df = pd.DataFrame(all_file_metrics)
    rollout_metrics = {
        "Rollout_MAE": float(metrics_summary_df["MAE"].mean()),
        "Rollout_RMSE": float(metrics_summary_df["RMSE"].mean()),
        "Maximum_Error": float(metrics_summary_df["MaxError"].max())
    }
    
    logger.info("Rollout Evaluation Metrics across ALL test trajectories (PINN):")
    for k, v in rollout_metrics.items():
        logger.info(f"  {k}: {v:.5f}")
        
    # Save outputs
    out_dir = paths['results_pinn_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    rollout_df = pd.concat(rollout_dfs, ignore_index=True)
    rollout_csv = os.path.join(out_dir, "rollout.csv")
    rollout_df.to_csv(rollout_csv, index=False)
    logger.info(f"Saved rollout predictions to {rollout_csv}")
    
    rollout_metrics_json = os.path.join(out_dir, "rollout_metrics.json")
    with open(rollout_metrics_json, 'w') as f:
        json.dump(rollout_metrics, f, indent=4)
    logger.info(f"Saved rollout metrics to {rollout_metrics_json}")
    
    total_duration = time.time() - start_rollout_time
    logger.info(f"Rollout execution complete in {total_duration:.1f} seconds (PINN).")
    log_system_metrics(logger, prefix="Rollout Final Resource Usage (PINN)")
    return rollout_metrics

def main():
    parser = argparse.ArgumentParser(description="Autoregressive GPU Rollout for PINN")
    parser.add_argument("--model", type=str, default="results/pinn/model.pt", help="Path to trained PINN model.pt")
    args = parser.parse_args()
    
    try:
        run_rollout(model_path=args.model)
    except Exception as e:
        logger.error(f"Rollout execution failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
