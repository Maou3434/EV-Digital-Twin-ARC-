import os
import glob
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Temperature_C",
    "Current_A",
    "Voltage_V",
    "SOC_pct",
    "PowerLoss_W",
    "AmbientTemp_C",
    "MassScale"
]
TARGET = "Delta_T"

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_pipeline(config_path="config.yaml"):
    """
    Fits StandardScaler on features and targets incrementally over all training CSV files
    to avoid high memory consumption. Saves scalers using joblib and writes global
    summary statistics to results/summary_statistics.csv.
    """
    config = load_config(config_path)
    paths = config['paths']
    dataset_dir = paths['dataset_dir']
    scalers_dir = paths['scalers_dir']
    summary_stats_path = paths['summary_stats']
    
    os.makedirs(scalers_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary_stats_path), exist_ok=True)
    
    # Locate all TRAIN CSV files
    train_files = sorted(glob.glob(os.path.join(dataset_dir, "TRAIN_*.csv")))
    if not train_files:
        raise FileNotFoundError(f"No TRAIN_*.csv files found in {dataset_dir}")
        
    print(f"Preprocessing pipeline: found {len(train_files)} training files.")
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # We will accumulate global statistics incrementally for:
    # FEATURES + TARGET + ["Time_s", "dt"]
    stat_cols = FEATURES + [TARGET, "Time_s", "dt"]
    
    global_n = 0
    global_min = {col: np.inf for col in stat_cols}
    global_max = {col: -np.inf for col in stat_cols}
    global_sum = {col: 0.0 for col in stat_cols}
    global_sum_sq = {col: 0.0 for col in stat_cols}
    
    print("Fitting scalers and collecting statistics incrementally...")
    for idx, path in enumerate(train_files):
        df = pd.read_csv(path)
        
        # Ensure Time_s is monotonic
        if not df["Time_s"].is_monotonic_increasing:
            df = df.sort_values(by="Time_s").reset_index(drop=True)
            
        # Add shift columns
        df["Temperature_next"] = df["Temperature_C"].shift(-1)
        df["Time_next"] = df["Time_s"].shift(-1)
        
        # Calculate Delta_T and dt
        df["dt"] = df["Time_next"] - df["Time_s"]
        df["Delta_T"] = df["Temperature_next"] - df["Temperature_C"]
        
        # Drop rows with NaN or small dt
        df = df.dropna().reset_index(drop=True)
        df = df[df["dt"] >= 0.001].reset_index(drop=True)
        
        if len(df) == 0:
            continue
            
        features = df[FEATURES].values
        targets = df[[TARGET]].values
        
        # Incremental scaling fitting
        scaler_X.partial_fit(features)
        scaler_y.partial_fit(targets)
        
        # Incremental summary statistics
        for col in stat_cols:
            vals = df[col].values
            global_min[col] = min(global_min[col], np.min(vals))
            global_max[col] = max(global_max[col], np.max(vals))
            global_sum[col] += np.sum(vals)
            global_sum_sq[col] += np.sum(vals ** 2)
            
        global_n += len(df)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(train_files):
            print(f"  Processed {idx + 1}/{len(train_files)} files | Total samples: {global_n:,}")
            
    # Save scalers
    scaler_X_path = os.path.join(scalers_dir, "scaler_X.joblib")
    scaler_y_path = os.path.join(scalers_dir, "scaler_y.joblib")
    
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"Saved scalers to {scaler_X_path} and {scaler_y_path}")
    
    # Calculate global mean and standard deviation
    summary_data = []
    for col in stat_cols:
        mean_val = global_sum[col] / global_n
        # variance = E[X^2] - (E[X])^2
        var_val = (global_sum_sq[col] / global_n) - (mean_val ** 2)
        std_val = np.sqrt(max(0.0, var_val)) # clamp to 0 to prevent numerical negatives
        
        summary_data.append({
            "Variable": col,
            "min": global_min[col],
            "max": global_max[col],
            "mean": mean_val,
            "std": std_val
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_stats_path, index=False)
    print(f"Saved summary statistics to {summary_stats_path}\n")
    print(summary_df.to_string(index=False))
    
    return scaler_X, scaler_y

if __name__ == "__main__":
    preprocess_pipeline()
