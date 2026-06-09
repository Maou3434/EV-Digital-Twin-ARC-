import numpy as np
import torch

def calculate_metrics(y_true, y_pred):
    """
    Computes regression metrics: MAE, RMSE, and R2.
    Supports both numpy arrays and torch tensors.
    """
    # Convert PyTorch tensors to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Handle edge case where ss_tot is zero
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)
        
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
