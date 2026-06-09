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
from models.pinn import PhysicsInformedNN
from utils.metrics import calculate_metrics

def train_pinn(config_path="config.yaml"):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    paths = config['paths']
    train_cfg = config['training']
    pinn_cfg = config['pinn']
    phys_cfg = config['physics']
    
    # Set random seed for reproducibility
    torch.manual_seed(train_cfg['random_seed'])
    np.random.seed(train_cfg['random_seed'])
    
    print("--- Phase 4: Training Physics-Informed Neural Network (PINN) ---")
    
    # 1. Load data
    data_path = paths['processed_dataset']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Please run preprocessing first.")
        
    df = pd.read_csv(data_path)
    
    # PINN Inputs: [Temperature_t, Current_t, Voltage_t, SOC_t, PowerLoss_t, AmbientTemp_t] (6 inputs)
    feature_cols = ["Temperature", "Current", "Voltage", "SOC", "PowerLoss", "AmbientTemp"]
    target_col = "Delta_T" # Target is the temperature delta
    dt_col = "dt"
    time_col = "Time"
    
    # 2. Generalization Split: Train on 20C, 30C, 40C (< 50C ambient) and Test on 50C (>= 50C ambient)
    train_df = df[df["AmbientTemp"] < 323.15].copy()
    test_df = df[df["AmbientTemp"] >= 323.15].copy()
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.reshape(-1, 1)
    dt_train = train_df[dt_col].values.reshape(-1, 1)
    time_train = train_df[time_col].values.reshape(-1, 1)
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values.reshape(-1, 1)
    dt_test = test_df[dt_col].values.reshape(-1, 1)
    time_test = test_df[time_col].values.reshape(-1, 1)
    
    # 3. Standard Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Extract mean and std for differentiable inverse-scaling of physics losses
    scaler_T_mean = torch.tensor(scaler_X.mean_[0], dtype=torch.float32)
    scaler_T_scale = torch.tensor(scaler_X.scale_[0], dtype=torch.float32)
    scaler_P_mean = torch.tensor(scaler_X.mean_[4], dtype=torch.float32)
    scaler_P_scale = torch.tensor(scaler_X.scale_[4], dtype=torch.float32)
    scaler_Tamb_mean = torch.tensor(scaler_X.mean_[5], dtype=torch.float32)
    scaler_Tamb_scale = torch.tensor(scaler_X.scale_[5], dtype=torch.float32)
    
    scaler_y_mean = torch.tensor(scaler_y.mean_[0], dtype=torch.float32)
    scaler_y_scale = torch.tensor(scaler_y.scale_[0], dtype=torch.float32)
    
    # Save scalers parameters for loading later
    scalers = {
        "scaler_X_mean": scaler_X.mean_.tolist(),
        "scaler_X_scale": scaler_X.scale_.tolist(),
        "scaler_y_mean": scaler_y.mean_.tolist(),
        "scaler_y_scale": scaler_y.scale_.tolist()
    }
    
    # 4. Create PyTorch DataLoaders (including dt)
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
        torch.tensor(dt_train, dtype=torch.float32)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=pinn_cfg['batch_size'], 
        shuffle=True
    )
    
    # 5. Device Setup (CUDA in WSL if available)
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    
    # Move scaling constants to the correct device
    scaler_T_mean = scaler_T_mean.to(device)
    scaler_T_scale = scaler_T_scale.to(device)
    scaler_P_mean = scaler_P_mean.to(device)
    scaler_P_scale = scaler_P_scale.to(device)
    scaler_Tamb_mean = scaler_Tamb_mean.to(device)
    scaler_Tamb_scale = scaler_Tamb_scale.to(device)
    scaler_y_mean = scaler_y_mean.to(device)
    scaler_y_scale = scaler_y_scale.to(device)
    
    # 6. Instantiate PINN, Loss, Optimizer
    # Under new framework, both take 6 inputs and predict Delta_T
    model = PhysicsInformedNN(
        input_dim=pinn_cfg['input_dim'], 
        hidden_dim=pinn_cfg['hidden_dim'],
        m=phys_cfg['m'],
        Cp=phys_cfg['Cp'],
        initial_hA=phys_cfg['initial_hA']
    ).to(device)
    
    data_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=pinn_cfg['learning_rate'])
    
    # 7. PINN Training Loop
    epochs = pinn_cfg['epochs']
    lambda_physics = pinn_cfg['lambda_physics']
    print(f"Starting PINN training for {epochs} epochs (lambda_physics = {lambda_physics})...")
    
    loss_history = {
        "epoch": [],
        "data_loss": [],
        "physics_loss": [],
        "total_loss": [],
        "hA": []
    }
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch_X, batch_y, batch_dt in train_loader:
            batch_X, batch_y, batch_dt = batch_X.to(device), batch_y.to(device), batch_dt.to(device)
            
            optimizer.zero_grad()
            
            # Predict Delta_T in scaled space
            pred_y_scaled = model(batch_X)
            
            # A. Calculate Data Loss (MSE in scaled space)
            data_loss = data_criterion(pred_y_scaled, batch_y)
            
            # B. Calculate Physics Loss (MSE in physical units of K/s)
            # Differentiable inverse-scaling of predicted Delta_T
            Delta_T_phys = pred_y_scaled * scaler_y_scale + scaler_y_mean
            
            # Reconstruction of input features in physical units for physics computation
            T_t_physical = batch_X[:, 0] * scaler_T_scale + scaler_T_mean
            PowerLoss_physical = batch_X[:, 4] * scaler_P_scale + scaler_P_mean
            AmbientTemp_physical = batch_X[:, 5] * scaler_Tamb_scale + scaler_Tamb_mean
            
            # Reconstruct physically consistent input batch for computing residual
            batch_X_physical = torch.stack([
                T_t_physical,
                batch_X[:, 1], 
                batch_X[:, 2],
                batch_X[:, 3],
                PowerLoss_physical,
                AmbientTemp_physical
            ], dim=1)
            
            # Compute physical residuals in stable units (K/s)
            residuals = model.compute_residual(batch_X_physical, Delta_T_phys, batch_dt)
            physics_loss = torch.mean(residuals ** 2)
            
            # C. Combine Losses
            total_loss = data_loss + lambda_physics * physics_loss
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            epoch_data_loss += data_loss.item() * batch_X.size(0)
            epoch_phys_loss += physics_loss.item() * batch_X.size(0)
            epoch_total_loss += total_loss.item() * batch_X.size(0)
            
        # Compute epoch averages
        num_samples = len(train_loader.dataset)
        epoch_data_loss /= num_samples
        epoch_phys_loss /= num_samples
        epoch_total_loss /= num_samples
        current_hA = model.get_hA().item()
        
        loss_history["epoch"].append(epoch)
        loss_history["data_loss"].append(epoch_data_loss)
        loss_history["physics_loss"].append(epoch_phys_loss)
        loss_history["total_loss"].append(epoch_total_loss)
        loss_history["hA"].append(current_hA)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs} | Data Loss: {epoch_data_loss:.6f} | Physics Loss (K/s): {epoch_phys_loss:.6f} | Total Loss: {epoch_total_loss:.6f} | Trainable hA: {current_hA:.5f} W/K")
            
    # 8. Evaluation on Unseen 50°C Test Set
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_preds_scaled = model(test_inputs).cpu().numpy()
        
        # Calculate physical inputs for test residual calculation
        test_inputs_phys_T = test_inputs[:, 0] * scaler_T_scale + scaler_T_mean
        test_inputs_phys_P = test_inputs[:, 4] * scaler_P_scale + scaler_P_mean
        test_inputs_phys_Tamb = test_inputs[:, 5] * scaler_Tamb_scale + scaler_Tamb_mean
        
        # Predicted Delta_T physical
        test_pred_phys_delta = torch.tensor(test_preds_scaled, dtype=torch.float32).to(device) * scaler_y_scale + scaler_y_mean
        
        test_inputs_physical = torch.stack([
            test_inputs_phys_T,
            test_inputs[:, 1],
            test_inputs[:, 2],
            test_inputs[:, 3],
            test_inputs_phys_P,
            test_inputs_phys_Tamb
        ], dim=1)
        
        test_dt_tensor = torch.tensor(dt_test, dtype=torch.float32).to(device)
        test_residuals = model.compute_residual(test_inputs_physical, test_pred_phys_delta, test_dt_tensor).cpu().numpy()
        
    # Inverse-scale predicted Delta_T back to Kelvin
    y_pred_delta = scaler_y.inverse_transform(test_preds_scaled).flatten()
    
    # Reconstruct absolute temperature: T_pred_t1 = T_t + Delta_T_pred
    T_t_test = test_df["Temperature"].values
    T_true_t1_test = test_df["Temperature_next"].values
    
    y_pred_kelvin = T_t_test + y_pred_delta
    y_true_kelvin = T_true_t1_test
    
    # Compute metrics on reconstructed absolute temperature
    metrics = calculate_metrics(y_true_kelvin, y_pred_kelvin)
    print("\nPINN Generalization on Unseen 50°C Test Set:")
    for metric_name, val in metrics.items():
        print(f"  {metric_name}: {val:.4f}")
    print(f"  Final trained hA: {model.get_hA().item():.5f} W/K")
    
    # Save Loss history for plotting
    loss_history_df = pd.DataFrame(loss_history)
    loss_history_df.to_csv(os.path.join(paths['results_dir'], "pinn_loss_history.csv"), index=False)
    
    # 9. Save Deliverables
    os.makedirs(os.path.dirname(pinn_cfg['model_path']), exist_ok=True)
    
    # Save model weights and scaling parameters
    torch.save({
        'model_state_dict': model.state_dict(),
        'scalers': scalers
    }, pinn_cfg['model_path'])
    print(f"Saved PINN model weights and scalers to {pinn_cfg['model_path']}")
    
    # Save metrics JSON
    with open(pinn_cfg['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved PINN metrics to {pinn_cfg['metrics_path']}")
    
    # Save predictions and metadata for comparison plotting
    predictions_df = pd.DataFrame({
        "True_Temperature": y_true_kelvin,
        "Predicted_Temperature": y_pred_kelvin,
        "Time": time_test.flatten(),
        "dt": dt_test.flatten(),
        "Physics_Residual": test_residuals.flatten()
    })
    predictions_df.to_csv(os.path.join(paths['results_dir'], "pinn_predictions.csv"), index=False)
    print(f"Saved PINN predictions and residuals to {paths['results_dir']}/pinn_predictions.csv\n")
    
    return metrics

if __name__ == "__main__":
    train_pinn()
