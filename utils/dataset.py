import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

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

def preprocess_df(df):
    """Applies the standard filtering and preprocessing steps to the dataframe."""
    # Monotonicity check and fix
    if not df["Time_s"].is_monotonic_increasing:
        df = df.sort_values(by="Time_s").reset_index(drop=True)
        
    # Add shift target columns
    df["Temperature_next"] = df["Temperature_C"].shift(-1)
    df["Time_next"] = df["Time_s"].shift(-1)
    
    # Calculate Delta_T and dt
    df["dt"] = df["Time_next"] - df["Time_s"]
    df["Delta_T"] = df["Temperature_next"] - df["Temperature_C"]
    
    # Drop rows with NaN or small/invalid dt to prevent numerical derivative issues
    df = df.dropna().reset_index(drop=True)
    df = df[df["dt"] >= 0.001].reset_index(drop=True)
    return df


class ThermalDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for large EV battery electrothermal datasets.
    Avoids loading all 78M rows into RAM by loading and preprocessing one CSV file at a time.
    """
    def __init__(self, file_paths, scaler_X=None, scaler_y=None, cache_lengths=True):
        self.file_paths = sorted(file_paths)
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        # Lengths caching
        cache_dir = "datasets"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, "file_lengths_v2.json")
        
        file_lengths_dict = {}
        if cache_lengths and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    file_lengths_dict = json.load(f)
            except Exception:
                pass
                
        self.file_lengths = []
        updated_cache = False
        
        print(f"Initializing ThermalDataset: scanning lengths of {len(self.file_paths)} files...")
        for path in self.file_paths:
            # Normalize path for key matching in cache
            norm_path = os.path.normpath(path).replace("\\", "/")
            if norm_path in file_lengths_dict:
                self.file_lengths.append(file_lengths_dict[norm_path])
            else:
                # Read CSV and preprocess to count exact length
                df = pd.read_csv(path)
                df = preprocess_df(df)
                length = len(df)
                self.file_lengths.append(length)
                file_lengths_dict[norm_path] = length
                updated_cache = True
                
        if updated_cache:
            with open(self.cache_path, "w") as f:
                json.dump(file_lengths_dict, f, indent=4)
                
        self.cumulative_lengths = np.array(
            [0] + list(np.cumsum(self.file_lengths)),
            dtype=np.int64
        )
        self.total_length = self.cumulative_lengths[-1]
        print(f"Dataset initialized with total samples: {self.total_length:,}")
        
        # Precompute index-to-file mapping for O(1) file lookup
        self.idx_to_file = np.empty(self.total_length, dtype=np.int16)
        for file_idx in range(len(self.file_paths)):
            start = self.cumulative_lengths[file_idx]
            end = self.cumulative_lengths[file_idx + 1]
            self.idx_to_file[start:end] = file_idx
            
        # Cache for currently loaded file to avoid I/O thrashing
        self.current_file_idx = -1
        self.current_data = None
        
    def __len__(self):
        return self.total_length
        
    def _load_file(self, file_idx):
        """Loads and preprocesses a CSV file, caching it in memory."""
        if file_idx == self.current_file_idx:
            return
            
        path = self.file_paths[file_idx]
        df = pd.read_csv(path)
        
        df = preprocess_df(df)
        
        # Features and target matrices
        features = df[FEATURES].values
        targets = df[[TARGET]].values
        dts = df[["dt"]].values
        
        # Apply scaling if standard scalers are provided
        if self.scaler_X is not None:
            features = self.scaler_X.transform(features)
        if self.scaler_y is not None:
            targets = self.scaler_y.transform(targets)
            
        self.current_file_idx = file_idx
        self.current_data = {
            "features": features.astype(np.float32),
            "targets": targets.astype(np.float32),
            "dts": dts.astype(np.float32)
        }
        
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of bounds")
            
        # Find which file index idx belongs to
        file_idx = int(self.idx_to_file[idx])
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        self._load_file(file_idx)
        
        x = self.current_data["features"][local_idx]
        y = self.current_data["targets"][local_idx]
        dt = self.current_data["dts"][local_idx]
        
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(dt)
        )


class FileGroupedBatchSampler(Sampler):
    """
    Custom Batch Sampler that groups indices by CSV file.
    This guarantees that batches only request indices from the same file, 
    allowing us to load a CSV file, process all its batches, and move on.
    Prevents cache misses and random-access file thrashing.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        file_order = list(range(len(self.dataset.file_paths)))
        if self.shuffle:
            np.random.shuffle(file_order)
            
        for file_idx in file_order:
            start = self.dataset.cumulative_lengths[file_idx]
            end = self.dataset.cumulative_lengths[file_idx + 1]
            indices = np.arange(start, end)
            if self.shuffle:
                np.random.shuffle(indices)
                
            for i in range(0, len(indices), self.batch_size):
                yield indices[i : i + self.batch_size].tolist()
        
    def __len__(self):
        total_batches = 0
        for file_idx in range(len(self.dataset.file_paths)):
            start = self.dataset.cumulative_lengths[file_idx]
            end = self.dataset.cumulative_lengths[file_idx + 1]
            n_samples = end - start
            total_batches += (n_samples + self.batch_size - 1) // self.batch_size
        return total_batches
