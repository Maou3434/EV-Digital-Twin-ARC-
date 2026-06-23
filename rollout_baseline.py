import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
from models.baseline_mlp import BaselineMLP
from utils.logging_utils import setup_logger, log_system_metrics

logger = setup_logger("rollout", "logs/rollout.log")

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_rollout(model_path, config=None, device=None):
    """
    Core function to run autoregressive rollout on GPU.
    Can be called directly from other python scripts.
    """
    logger.info(f"Starting autoregressive rollout using model: {model_path}")
    start_rollout_time = time.time()
    
    if config is None:
        config = load_config()
    paths = config['paths']
    dataset_dir = paths['dataset_dir']
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    logger.info("Loading model weights and metadata...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Instantiate Model
    mlp_cfg = config['baseline_mlp']
    model = BaselineMLP(input_dim=mlp_cfg['input_dim'], hidden_dim=mlp_cfg['hidden_dim']).to(device)
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
        
    out_dir = paths['results_baseline_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(out_dir, "rollout_checkpoint.json")
    rollout_csv = os.path.join(out_dir, "rollout.csv")
    rollout_metrics_json = os.path.join(out_dir, "rollout_metrics.json")
    
    completed_files = []
    all_file_metrics = []
    
    if os.path.exists(checkpoint_path):
        logger.info("Found checkpoint, resuming...")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            completed_files = checkpoint_data.get("completed_files", [])
            all_file_metrics = checkpoint_data.get("metrics", [])
    else:
        # If no checkpoint, clear old rollout.csv just in case
        if os.path.exists(rollout_csv):
            os.remove(rollout_csv)
            
    total_files = len(test_paths)
    logger.info(f"Running rollout on {total_files} files...")
    
    try:
        for idx, path in enumerate(test_paths):
            file_basename = os.path.basename(path)
            if file_basename in completed_files:
                logger.info(f"Rollout file {idx + 1}/{total_files}: {file_basename} (Already completed, skipping)")
                continue
                
            file_start_time = time.time()
            logger.info(f"Rollout file {idx + 1}/{total_files}: {file_basename}")
            
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
                logger.warning(f"File {file_basename} has 0 valid steps after preprocessing.")
                completed_files.append(file_basename) # mark as completed even if empty
                # Save checkpoint
                with open(checkpoint_path, 'w') as f:
                    json.dump({"completed_files": completed_files, "metrics": all_file_metrics}, f, indent=4)
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
            T_pred[0] = T_true[0] # Initial prediction matches ground truth temperature
            
            # Sequentially roll out on GPU without CPU roundtrips
            with torch.no_grad():
                for t in range(N - 1):
                    # Update current scaled temperature in feature tensor
                    T_pred_scaled = (T_pred[t] - mean_X_gpu[0]) / std_X_gpu[0]
                    features_scaled[t, 0] = T_pred_scaled
                    
                    # Model predict scaled Delta_T
                    # Slicing keeps it as a 2D batch of size 1 (1, 7)
                    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                        pred_scaled = model(features_scaled[t:t+1])
                        
                    # Inverse scale Delta_T prediction on GPU
                    delta_T_pred = pred_scaled * std_y_gpu[0] + mean_y_gpu[0]
                    
                    # Update next temperature prediction
                    T_pred[t + 1] = T_pred[t] + delta_T_pred.squeeze()
                    
                    # Progress reporting for large trajectories
                    if N > 20000 and (t + 1) % 50000 == 0:
                        elapsed = time.time() - file_start_time
                        steps_per_sec = (t + 1) / elapsed
                        remaining = N - 1 - t
                        eta = remaining / steps_per_sec if steps_per_sec > 0 else 0
                        logger.info(f"    Step {t + 1}/{N} | Speed: {steps_per_sec:.1f} steps/s | ETA: {eta:.1f}s")
                        
            # Copy prediction tensor back to CPU at once
            T_pred_np = T_pred.cpu().numpy()
            
            # Calculate metrics
            mae = np.mean(np.abs(T_true_np - T_pred_np))
            rmse = np.sqrt(np.mean((T_true_np - T_pred_np) ** 2))
            max_err = np.max(np.abs(T_true_np - T_pred_np))
            
            duration = time.time() - file_start_time
            logger.info(f"Finished rollout in {duration:.1f}s | MAE: {mae:.5f} | RMSE: {rmse:.5f} | Max Error: {max_err:.5f}")
            
            all_file_metrics.append({
                "File": file_basename,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "MaxError": float(max_err)
            })
            
            trajectory_df = pd.DataFrame({
                "Trajectory_ID": idx,
                "Time_s": df["Time_s"].values,
                "True_Temperature": T_true_np,
                "Predicted_Temperature": T_pred_np,
                "Absolute_Error": np.abs(T_true_np - T_pred_np)
            })
            
            # Append trajectory to rollout.csv
            trajectory_df.to_csv(rollout_csv, mode='a', header=not os.path.exists(rollout_csv), index=False)
            logger.info(f"Saved trajectory outputs for {file_basename}")
            
            # Mark file as completed and save checkpoint
            completed_files.append(file_basename)
            with open(checkpoint_path, 'w') as f:
                json.dump({"completed_files": completed_files, "metrics": all_file_metrics}, f, indent=4)
                
            # Aggregate metrics incrementally
            metrics_summary_df = pd.DataFrame(all_file_metrics)
            rollout_metrics = {
                "Rollout_MAE": float(metrics_summary_df["MAE"].mean()),
                "Rollout_RMSE": float(metrics_summary_df["RMSE"].mean()),
                "Maximum_Error": float(metrics_summary_df["MaxError"].max())
            }
            with open(rollout_metrics_json, 'w') as f:
                json.dump(rollout_metrics, f, indent=4)
            logger.info(f"Saved incremental rollout metrics")
            
    except KeyboardInterrupt:
        logger.warning("Rollout interrupted by user. Saved checkpoint. You can resume later.")
        
    # Final Aggregate metrics
    if not all_file_metrics:
        logger.warning("No files were successfully processed.")
        return {}
        
    metrics_summary_df = pd.DataFrame(all_file_metrics)
    rollout_metrics = {
        "Rollout_MAE": float(metrics_summary_df["MAE"].mean()),
        "Rollout_RMSE": float(metrics_summary_df["RMSE"].mean()),
        "Maximum_Error": float(metrics_summary_df["MaxError"].max())
    }
    
    logger.info("Rollout Evaluation Metrics across ALL test trajectories:")
    for k, v in rollout_metrics.items():
        logger.info(f"  {k}: {v:.5f}")
        
    total_duration = time.time() - start_rollout_time
    logger.info(f"Rollout execution complete in {total_duration:.1f} seconds.")
    log_system_metrics(logger, prefix="Rollout Final Resource Usage")
    return rollout_metrics

def main():
    parser = argparse.ArgumentParser(description="Autoregressive GPU Rollout for Baseline MLP")
    parser.add_argument("--model", type=str, default="results/baseline/model.pt", help="Path to trained model.pt")
    args = parser.parse_args()
    
    try:
        run_rollout(model_path=args.model)
    except Exception as e:
        logger.error(f"Rollout execution failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
