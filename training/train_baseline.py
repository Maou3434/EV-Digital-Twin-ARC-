import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from models.baseline_mlp import BaselineMLP
from utils.metrics import calculate_metrics

def train_baseline(config_path="config.yaml"):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    paths = config['paths']
    train_cfg = config['training']
    mlp_cfg = config['baseline_mlp']
    
    # Set random seed for reproducibility
    torch.manual_seed(train_cfg['random_seed'])
    np.random.seed(train_cfg['random_seed'])
    
    print("--- Phase 2: Training Baseline Neural Network (Delta-T) ---")
    
    # 1. Load data
    data_path = paths['processed_dataset']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Please run preprocessing first.")
        
    df = pd.read_csv(data_path)
    
    # Inputs: Temperature, Current, Voltage, SOC, PowerLoss, AmbientTemp (6 inputs)
    feature_cols = ["Temperature", "Current", "Voltage", "SOC", "PowerLoss", "AmbientTemp"]
    target_col = "Delta_T" # Target is the temperature delta
    
    # 2. Generalization Split: Train on 20C, 30C, 40C (< 50C ambient) and Test on 50C (>= 50C ambient)
    train_df = df[df["AmbientTemp"] < 323.15].copy()
    test_df = df[df["AmbientTemp"] >= 323.15].copy()
    
    print(f"Dataset Split (Unseen Ambient Generalization):")
    print(f"  Training Set (20C, 30C, 40C): {len(train_df)} rows")
    print(f"  Testing Set (Unseen 50C):      {len(test_df)} rows")
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.reshape(-1, 1)
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values.reshape(-1, 1)
    
    # 3. Standard Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Save scalers for main orchestrator
    scalers = {
        "scaler_X_mean": scaler_X.mean_.tolist(),
        "scaler_X_scale": scaler_X.scale_.tolist(),
        "scaler_y_mean": scaler_y.mean_.tolist(),
        "scaler_y_scale": scaler_y.scale_.tolist()
    }
    
    # 4. Create PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=mlp_cfg['batch_size'], 
        shuffle=True
    )
    
    # 5. Device Setup (default to CUDA in WSL if available)
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    
    # 6. Instantiate Model, Loss, Optimizer
    # Under new dynamic test, input dimension is 6
    model = BaselineMLP(input_dim=mlp_cfg['input_dim'], hidden_dim=mlp_cfg['hidden_dim']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=mlp_cfg['learning_rate'])
    
    # 7. Training Loop
    epochs = mlp_cfg['epochs']
    print(f"Starting Baseline MLP training for {epochs} epochs...")
    
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss:.6f}")
            
    # 8. Evaluation on Unseen 50°C Test Set
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_preds_scaled = model(test_inputs).cpu().numpy()
        
    # Inverse-scale predicted Delta_T back to Kelvin
    y_pred_delta = scaler_y.inverse_transform(test_preds_scaled).flatten()
    
    # Reconstruct absolute temperature: T_pred_t1 = T_t + Delta_T_pred
    T_t_test = test_df["Temperature"].values
    T_true_t1_test = test_df["Temperature_next"].values
    
    y_pred_kelvin = T_t_test + y_pred_delta
    y_true_kelvin = T_true_t1_test
    
    # Compute metrics on reconstructed absolute temperature
    metrics = calculate_metrics(y_true_kelvin, y_pred_kelvin)
    print("\nBaseline MLP Generalization on Unseen 50°C Test Set:")
    for metric_name, val in metrics.items():
        print(f"  {metric_name}: {val:.4f}")
        
    # 9. Save Deliverables
    os.makedirs(os.path.dirname(mlp_cfg['model_path']), exist_ok=True)
    
    # Save model weights and scaling parameters
    torch.save({
        'model_state_dict': model.state_dict(),
        'scalers': scalers
    }, mlp_cfg['model_path'])
    print(f"Saved baseline model weights and scalers to {mlp_cfg['model_path']}")
    
    # Save metrics JSON
    with open(mlp_cfg['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved baseline metrics to {mlp_cfg['metrics_path']}")
    
    # Save predictions CSV (reconstructed predicted absolute temperature vs ground truth next temperature)
    predictions_df = pd.DataFrame({
        "True_Temperature": y_true_kelvin,
        "Predicted_Temperature": y_pred_kelvin
    })
    predictions_df.to_csv(mlp_cfg['predictions_path'], index=False)
    print(f"Saved baseline predictions to {mlp_cfg['predictions_path']}\n")
    
    return metrics

if __name__ == "__main__":
    train_baseline()
