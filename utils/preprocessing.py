import os
import glob
import pandas as pd
import numpy as np
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_ambient_temp(filename):
    """
    Returns the ambient temperature in Kelvin based on the file name.
    """
    if "baseline_20C" in filename:
        return 293.15
    elif "ambient_30C" in filename or "30C" in filename:
        return 303.15
    elif "ambient_40C" in filename or "40C" in filename:
        return 313.15
    elif "ambient_50C" in filename or "50C" in filename:
        return 323.15
    else:
        return None

def process_and_validate_data(config_path="config.yaml"):
    config = load_config(config_path)
    
    baseline_dir = config['paths']['baseline_dir']
    ambient_dir = config['paths']['ambient_dir']
    summary_stats_path = config['paths']['summary_stats']
    processed_path = config['paths']['processed_dataset']
    
    # Create directories if they do not exist
    os.makedirs(os.path.dirname(summary_stats_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    # Find all CSV files
    baseline_files = glob.glob(os.path.join(baseline_dir, "*.csv"))
    ambient_files = glob.glob(os.path.join(ambient_dir, "*.csv"))
    all_files = baseline_files + ambient_files
    
    # Track unique files loaded by their ambient temperature to eliminate duplicates
    loaded_datasets = {}
    
    print("--- Phase 1: Loading and Validating Datasets ---")
    for file_path in all_files:
        filename = os.path.basename(file_path)
        amb_temp = get_ambient_temp(filename)
        
        if amb_temp is None:
            print(f"Skipping unknown dataset: {filename}")
            continue
            
        if amb_temp in loaded_datasets:
            print(f"Skipping duplicate dataset for ambient temperature {amb_temp} K: {filename}")
            continue
            
        print(f"Loading dataset: {filename} (Ambient: {amb_temp} K / {amb_temp - 293.15 + 20:.2f}°C)")
        df = pd.read_csv(file_path)
        
        # Check standard columns
        required_cols = ["Time", "Current", "Voltage", "SOC", "PowerLoss", "Temperature"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column {col} in {filename}")
                
        # 1. Add AmbientTemp column
        df["AmbientTemp"] = amb_temp
        
        # 2. Data Validation
        # Check NaNs
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"WARNING: {filename} contains {nan_count} NaN values. Interpolating...")
            df = df.interpolate(method='linear').bfill().ffill()
            
        # Check Infs
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            raise ValueError(f"Dataset {filename} contains infinite (Inf) values.")
            
        # Check Time Monotonicity
        time_diffs = df["Time"].diff().dropna()
        if (time_diffs <= 0).any():
            print(f"WARNING: Time column in {filename} is not strictly monotonic! Sorting by Time...")
            df = df.sort_values(by="Time").reset_index(drop=True)
            time_diffs = df["Time"].diff().dropna()
            if (time_diffs <= 0).any():
                print("Found non-monotonic time steps even after sorting. Removing duplicate timestamps...")
                df = df.drop_duplicates(subset=["Time"]).reset_index(drop=True)
        
        # Check Temperature Trends
        temp_min = df["Temperature"].min()
        temp_max = df["Temperature"].max()
        print(f"  Validation OK: Temperature range = [{temp_min:.2f}, {temp_max:.2f}] K")
        print(f"  Time monotonicity verified. Row count = {len(df)}")
        
        loaded_datasets[amb_temp] = df

    # Combine datasets to compute global summary statistics
    combined_raw_df = pd.concat(loaded_datasets.values(), ignore_index=True)
    
    # Generate summary statistics csv
    summary_df = combined_raw_df.describe().loc[['min', 'max', 'mean', 'std']]
    summary_df.to_csv(summary_stats_path)
    print(f"Saved summary statistics to {summary_stats_path}\n")
    print(summary_df)
    
    print("\n--- Phase 3: State-Space Dataset Formulation ---")
    processed_runs = []
    
    for amb_temp, df in loaded_datasets.items():
        # Make sure dataset is sorted chronologically
        df = df.sort_values(by="Time").reset_index(drop=True)
        
        # Prepare inputs at time t
        df_t = df.copy()
        
        # Target: Temperature at t+1
        df_t["Temperature_next"] = df["Temperature"].shift(-1)
        df_t["Time_next"] = df["Time"].shift(-1)
        
        # Calculate time step dt
        df_t["dt"] = df_t["Time_next"] - df_t["Time"]
        
        # Drop the last row of the run as it does not have a t+1 target
        df_t = df_t.dropna().reset_index(drop=True)
        
        # Calculate Delta_T target to eliminate identity leakage
        df_t["Delta_T"] = df_t["Temperature_next"] - df_t["Temperature"]
        
        # Ensure dt is valid and not extremely small to prevent numerical derivative explosion
        if (df_t["dt"] < 0.005).any():
            print(f"WARNING: Extremely small or non-positive dt found in run {amb_temp} K. Filtering out those rows...")
            df_t = df_t[df_t["dt"] >= 0.005].reset_index(drop=True)
            
        print(f"Processed state-space for run {amb_temp} K: {len(df_t)} rows generated.")
        processed_runs.append(df_t)
        
    # Combine state-space formulations of all runs
    state_space_df = pd.concat(processed_runs, ignore_index=True)
    
    # Save processed state space
    state_space_df.to_csv(processed_path, index=False)
    print(f"Saved completed state-space dataset to {processed_path} (Total rows: {len(state_space_df)})\n")
    
    return state_space_df

if __name__ == "__main__":
    process_and_validate_data()
