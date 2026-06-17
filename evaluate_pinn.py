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
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from utils.dataset import ThermalDataset, FileGroupedBatchSampler
from models.pinn import PhysicsInformedNN
from utils.metrics import calculate_metrics
from utils.logging_utils import setup_logger, log_system_metrics

logger = setup_logger("evaluation", "logs/evaluation.log")

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_evaluation(model_path, config=None, device=None):
    """
    Core function to run single-step PINN evaluation.
    Can be called directly from other python scripts.
    """
    logger.info(f"Starting single-step PINN evaluation using model: {model_path}")
    start_eval_time = time.time()
    
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
        logger.info("Loading scalers from joblib files...")
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
    else:
        logger.info("Re-creating scalers from checkpoint metadata...")
        scaler_X = StandardScaler()
        scaler_X.mean_ = np.array(checkpoint['scaler_X_mean'])
        scaler_X.scale_ = np.array(checkpoint['scaler_X_scale'])
        scaler_y = StandardScaler()
        scaler_y.mean_ = np.array(checkpoint['scaler_y_mean'])
        scaler_y.scale_ = np.array(checkpoint['scaler_y_scale'])
        
    # GPU Scalers for residual calculation
    mean_X = torch.tensor(scaler_X.mean_, dtype=torch.float32, device=device)
    std_X = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=device)
    mean_y = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
    std_y = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)
    
    test_paths = sorted(glob.glob(os.path.join(dataset_dir, "TEST_*.csv")))
    if not test_paths:
        raise FileNotFoundError(f"No TEST_*.csv files found in {dataset_dir}")
        
    logger.info(f"Loading test dataset with {len(test_paths)} files...")
    test_dataset = ThermalDataset(test_paths, scaler_X, scaler_y)
    test_sampler = FileGroupedBatchSampler(test_dataset, batch_size=pinn_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    all_preds = []
    all_targets = []
    all_temps_t = []
    all_residuals = []
    
    logger.info("Running single-step predictions and residual calculation...")
    with torch.no_grad():
        for batch_X, batch_y, batch_dt in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_dt = batch_dt.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                pred = model(batch_X)
                
                # Reconstruct physical features and compute residual
                batch_X_phys = batch_X * std_X + mean_X
                pred_y_phys = pred * std_y + mean_y
                residuals = model.compute_residual(batch_X_phys, pred_y_phys, batch_dt)
                
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())
            all_temps_t.append(batch_X_phys[:, 0].cpu().numpy())
            all_residuals.append(residuals.cpu().numpy())
            
    preds_scaled = np.vstack(all_preds)
    targets_scaled = np.vstack(all_targets)
    temps_t_physical = np.concatenate(all_temps_t)
    test_residuals = np.vstack(all_residuals).flatten()
    
    # Reconstruct physical predicted and true Delta_T
    pred_delta_phys = scaler_y.inverse_transform(preds_scaled).flatten()
    true_delta_phys = scaler_y.inverse_transform(targets_scaled).flatten()
    
    # Reconstruct physical absolute next temperature
    pred_temp_next = temps_t_physical + pred_delta_phys
    true_temp_next = temps_t_physical + true_delta_phys
    
    # Compute metrics
    metrics = calculate_metrics(true_temp_next, pred_temp_next)
    logger.info("Single-step Absolute Temperature Metrics (PINN):")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.5f}")
    logger.info(f"  Trained convective coefficient hA: {model.get_hA().item():.5f} W/K")
    
    # Save predictions
    out_dir = paths['results_pinn_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    predictions_df = pd.DataFrame({
        "True_Temperature": true_temp_next,
        "Predicted_Temperature": pred_temp_next,
        "True_Delta_T": true_delta_phys,
        "Predicted_Delta_T": pred_delta_phys,
        "Physics_Residual": test_residuals
    })
    
    preds_csv = os.path.join(out_dir, "predictions.csv")
    predictions_df.to_csv(preds_csv, index=False)
    logger.info(f"Saved predictions to {preds_csv}")
    
    metrics_json = os.path.join(out_dir, "metrics.json")
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved single-step metrics to {metrics_json}")
    
    duration = time.time() - start_eval_time
    logger.info(f"Single-step PINN evaluation complete in {duration:.1f} seconds.")
    log_system_metrics(logger, prefix="Evaluation Final Resource Usage (PINN)")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate PINN Model")
    parser.add_argument("--model", type=str, default="results/pinn/model.pt", help="Path to trained PINN model.pt")
    args = parser.parse_args()
    
    try:
        run_evaluation(model_path=args.model)
    except Exception as e:
        logger.error(f"Single-step PINN evaluation failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
