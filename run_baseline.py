import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import glob
import json
import random
import yaml
import joblib
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.dataset import ThermalDataset, FileGroupedBatchSampler, FEATURES, TARGET
from utils.preprocessing import preprocess_pipeline
from models.baseline_mlp import BaselineMLP
from utils.logging_utils import setup_logger, log_system_metrics, get_system_metrics

# Initialize logger
logger = setup_logger("training", "logs/training.log")

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

def benchmark_dataloader(train_dataset, train_sampler, batch_size):
    """Benchmarks different DataLoader configurations to select the best one."""
    logger.info("Benchmarking DataLoader configuration for optimal performance...")
    best_throughput = 0.0
    best_config = (4, 2) # Default fallback
    
    # Enable test configs
    configs = []
    for num_workers in [4, 6, 8]:
        for prefetch_factor in [2, 4]:
            configs.append((num_workers, prefetch_factor))
            
    for num_workers, prefetch_factor in configs:
        try:
            temp_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
                persistent_workers=True
            )
            
            start_time = time.time()
            count = 0
            iterator = iter(temp_loader)
            # Fetch 5 batches
            for _ in range(5):
                _ = next(iterator)
                count += batch_size
            duration = time.time() - start_time
            throughput = count / duration if duration > 0 else 0
            logger.info(f"DataLoader Config: num_workers={num_workers}, prefetch_factor={prefetch_factor} | Throughput: {throughput:.2f} samples/sec")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = (num_workers, prefetch_factor)
                
            # Clean up
            del iterator
            del temp_loader
        except Exception as e:
            logger.warning(f"DataLoader Config num_workers={num_workers}, prefetch_factor={prefetch_factor} failed: {e}")
            
    logger.info(f"Selected DataLoader Config: num_workers={best_config[0]}, prefetch_factor={best_config[1]} (Throughput: {best_throughput:.2f} samples/sec)")
    return best_config

def main():
    parser = argparse.ArgumentParser(description="Train Baseline MLP Model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    
    try:
        logger.info("=== STARTING BASELINE MLP TRAINING PIPELINE ===")
        
        # 1. Preprocessing
        logger.info("--- PHASE 1: PREPROCESSING ---")
        config = load_config()
        paths = config['paths']
        dataset_dir = paths['dataset_dir']
        
        train_files = sorted(glob.glob(os.path.join(dataset_dir, "TRAIN_*.csv")))
        test_paths = sorted(glob.glob(os.path.join(dataset_dir, "TEST_*.csv")))
        
        if not train_files:
            raise FileNotFoundError(f"No TRAIN_*.csv files found in {dataset_dir}")
        if not test_paths:
            raise FileNotFoundError(f"No TEST_*.csv files found in {dataset_dir}")
            
        logger.info(f"Found {len(train_files)} training files and {len(test_paths)} test files.")
        
        scaler_X_path = os.path.join(paths['scalers_dir'], "scaler_X.joblib")
        scaler_y_path = os.path.join(paths['scalers_dir'], "scaler_y.joblib")
        
        if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
            logger.info("Loading pre-fit scalers...")
            scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)
        else:
            logger.info("Fitting scalers incrementally...")
            scaler_X, scaler_y = preprocess_pipeline()
            
        seed_everything(config['training']['random_seed'])
        train_paths, val_paths = train_test_split(
            train_files, 
            test_size=0.1, 
            random_state=config['training']['random_seed']
        )
        logger.info(f"Split training files into {len(train_paths)} train and {len(val_paths)} validation files.")
        
        # 2. Training Setup
        logger.info("--- PHASE 2: TRAINING SETUP ---")
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Enable CUDA performance flags
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("CUDA optimizations enabled: high float32 matmul precision, TF32 allowed")
            
        train_cfg = config['training']
        mlp_cfg = config['baseline_mlp']
        
        logger.info("Initializing datasets...")
        train_dataset = ThermalDataset(train_paths, scaler_X, scaler_y)
        train_sampler = FileGroupedBatchSampler(train_dataset, batch_size=mlp_cfg['batch_size'], shuffle=True)
        
        val_dataset = ThermalDataset(val_paths, scaler_X, scaler_y)
        val_sampler = FileGroupedBatchSampler(val_dataset, batch_size=mlp_cfg['batch_size'], shuffle=False)
        
        # Auto-tune DataLoader
        num_workers, prefetch_factor = benchmark_dataloader(train_dataset, train_sampler, mlp_cfg['batch_size'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=True
        )
        
        model = BaselineMLP(input_dim=mlp_cfg['input_dim'], hidden_dim=mlp_cfg['hidden_dim']).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=mlp_cfg['learning_rate'])
        grad_scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
        
        # Noise settings
        noise_std = train_cfg['noise_std']
        T_std = scaler_X.scale_[0]
        scaled_noise_std = noise_std / T_std
        
        # Recovery state
        start_epoch = 1
        best_val_loss = float('inf')
        epochs_no_improve = 0
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": []
        }
        
        # Resume training if requested
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            if 'history' in checkpoint:
                history = checkpoint['history']
            logger.info(f"Loaded checkpoint successfully. Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.6f}")
            
        # 3. Training Loop
        logger.info("--- PHASE 3: TRAINING LOOP ---")
        epochs = mlp_cfg['epochs']
        patience = train_cfg['early_stopping_patience']
        
        checkpoint_dir = os.path.join(paths['results_baseline_dir'], "../checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Pre-measure first batch loading
        start_time = time.time()
        _ = next(iter(train_loader))
        logger.info(f"First training batch load time: {time.time() - start_time:.2f}s")
        
        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time.time()
            logger.info(f"Epoch {epoch}/{epochs} started")
            
            model.train()
            train_loss = 0.0
            train_ae = 0.0
            train_count = 0
            
            epoch_train_start = time.time()
            last_log_time = epoch_train_start
            last_log_count = 0
            
            for i, (batch_X, batch_y, _) in enumerate(train_loader):
                # Use non-blocking to match pinned memory
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                # Noise injection
                if noise_std > 0.0:
                    batch_X_train = batch_X.clone()
                    noise = torch.randn_like(batch_X_train[:, 0]) * scaled_noise_std
                    batch_X_train[:, 0] += noise
                else:
                    batch_X_train = batch_X
                    
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    pred = model(batch_X_train)
                    loss = criterion(pred, batch_y)
                    
                grad_scaler.scale(loss).backward()
                
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg['gradient_clip'])
                
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                n = batch_X.size(0)
                train_loss += loss.item() * n
                train_ae += torch.sum(torch.abs(pred - batch_y)).item()
                train_count += n
                
                # Log step progress every 30 seconds
                current_time = time.time()
                if current_time - last_log_time >= 30.0:
                    elapsed = current_time - epoch_train_start
                    interval_elapsed = current_time - last_log_time
                    interval_samples = train_count - last_log_count
                    step_throughput = interval_samples / interval_elapsed if interval_elapsed > 0 else 0
                    percent_complete = (i + 1) / len(train_loader) * 100
                    logger.info(f"  Epoch {epoch} | Batch {i+1}/{len(train_loader)} ({percent_complete:.1f}%) | Loss: {loss.item():.6f} | Speed: {step_throughput:.1f} samples/s | Elapsed: {elapsed:.1f}s")
                    last_log_time = current_time
                    last_log_count = train_count
                
            train_loss /= train_count
            train_mae = train_ae / train_count
            train_duration = time.time() - epoch_train_start
            
            # Validation
            epoch_val_start = time.time()
            model.eval()
            val_loss = 0.0
            val_ae = 0.0
            val_count = 0
            
            with torch.no_grad():
                for batch_X, batch_y, _ in val_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                        pred = model(batch_X)
                        loss = criterion(pred, batch_y)
                        
                    n = batch_X.size(0)
                    val_loss += loss.item() * n
                    val_ae += torch.sum(torch.abs(pred - batch_y)).item()
                    val_count += n
                    
            val_loss /= val_count
            val_mae = val_ae / val_count
            val_duration = time.time() - epoch_val_start
            
            epoch_duration = time.time() - epoch_start_time
            throughput = train_count / train_duration if train_duration > 0 else 0
            
            # System Metrics Logging
            metrics = get_system_metrics(device)
            logger.info(f"Epoch {epoch:03d}/{epochs} Complete | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Train MAE: {train_mae:.5f} | Val MAE: {val_mae:.5f} | Duration: {epoch_duration:.1f}s (Train: {train_duration:.1f}s, Val: {val_duration:.1f}s) | Throughput: {throughput:.1f} samples/sec")
            log_system_metrics(logger, prefix=f"Epoch {epoch:03d} System Resource Usage")
            
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_mae"].append(train_mae)
            history["val_mae"].append(val_mae)
            
            # Checkpoint management - save immediately on validation improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_baseline.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'scaler_state': {
                        'mean_X': scaler_X.mean_.tolist(),
                        'scale_X': scaler_X.scale_.tolist(),
                        'mean_y': scaler_y.mean_.tolist(),
                        'scale_y': scaler_y.scale_.tolist()
                    },
                    'history': history,
                    'config': config,
                    'seed': config['training']['random_seed']
                }, best_checkpoint_path)
                logger.info(f"Validation loss improved. Saved best model checkpoint to {best_checkpoint_path}")
            else:
                epochs_no_improve += 1
                
            # Periodic checkpoints (every 10 epochs)
            if epoch % 10 == 0:
                periodic_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'scaler_state': {
                        'mean_X': scaler_X.mean_.tolist(),
                        'scale_X': scaler_X.scale_.tolist(),
                        'mean_y': scaler_y.mean_.tolist(),
                        'scale_y': scaler_y.scale_.tolist()
                    },
                    'history': history
                }, periodic_path)
                logger.info(f"Saved periodic checkpoint to {periodic_path}")
                
            # Early Stopping Check
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best model weights.")
                best_checkpoint = torch.load(os.path.join(checkpoint_dir, "best_baseline.pt"), map_location=device)
                model.load_state_dict(best_checkpoint['model_state_dict'])
                break
                
        # 4. Save Final Outputs
        logger.info("--- PHASE 4: SAVING DELIVERABLES ---")
        out_dir = paths['results_baseline_dir']
        os.makedirs(out_dir, exist_ok=True)
        
        # Save model.pt for use in evaluate/rollout/plot phase
        model_pt_path = os.path.join(out_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X_mean': scaler_X.mean_.tolist(),
            'scaler_X_scale': scaler_X.scale_.tolist(),
            'scaler_y_mean': scaler_y.mean_.tolist(),
            'scaler_y_scale': scaler_y.scale_.tolist(),
            'history': history
        }, model_pt_path)
        logger.info(f"Saved final baseline model and scalers metadata to {model_pt_path}")
        
        # Run sub-pipelines automatically
        logger.info("--- TRIGGERING EVALUATION AND ROLLOUT SUB-PIPELINES ---")
        
        # Import evaluation and rollout scripts dynamically to run
        try:
            import evaluate_baseline
            logger.info("Running evaluate_baseline.py...")
            evaluate_baseline.run_evaluation(model_path=model_pt_path, config=config, device=device)
        except Exception as eval_err:
            logger.error(f"Error running evaluate_baseline sub-pipeline: {eval_err}", exc_info=True)
            
        try:
            import rollout_baseline
            logger.info("Running rollout_baseline.py...")
            rollout_baseline.run_rollout(model_path=model_pt_path, config=config, device=device)
        except Exception as roll_err:
            logger.error(f"Error running rollout_baseline sub-pipeline: {roll_err}", exc_info=True)
            
        try:
            import plot_baseline
            logger.info("Running plot_baseline.py...")
            plot_baseline.run_plotting(out_dir=out_dir)
        except Exception as plot_err:
            logger.error(f"Error running plot_baseline sub-pipeline: {plot_err}", exc_info=True)
            
        logger.info("=== BASELINE MLP TRAINING PIPELINE FINISHED SUCCESSFULY ===")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user! Saving emergency checkpoint...")
        try:
            checkpoint_dir = os.path.join(paths['results_baseline_dir'], "../checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            emergency_path = os.path.join(checkpoint_dir, f"interrupted_baseline_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'scaler_state': {
                    'mean_X': scaler_X.mean_.tolist(),
                    'scale_X': scaler_X.scale_.tolist(),
                    'mean_y': scaler_y.mean_.tolist(),
                    'scale_y': scaler_y.scale_.tolist()
                },
                'history': history
            }, emergency_path)
            logger.warning(f"Saved emergency checkpoint to {emergency_path}")
        except Exception as save_err:
            logger.error(f"Failed to save emergency checkpoint: {save_err}")
        logger.warning("Exiting training pipeline.")
        
    except Exception as e:
        logger.error(f"Pipeline crashed due to an unhandled exception: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
