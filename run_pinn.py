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
from models.pinn import PhysicsInformedNN
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
    logger.info("Benchmarking DataLoader configuration for PINN...")
    best_throughput = 0.0
    best_config = (4, 2) # Default fallback
    
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
            for _ in range(5):
                _ = next(iterator)
                count += batch_size
            duration = time.time() - start_time
            throughput = count / duration if duration > 0 else 0
            logger.info(f"DataLoader Config: num_workers={num_workers}, prefetch_factor={prefetch_factor} | Throughput: {throughput:.2f} samples/sec")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = (num_workers, prefetch_factor)
                
            del iterator
            del temp_loader
        except Exception as e:
            logger.warning(f"DataLoader Config num_workers={num_workers}, prefetch_factor={prefetch_factor} failed: {e}")
            
    logger.info(f"Selected DataLoader Config: num_workers={best_config[0]}, prefetch_factor={best_config[1]} (Throughput: {best_throughput:.2f} samples/sec)")
    return best_config

def main():
    parser = argparse.ArgumentParser(description="Train Physics-Informed NN (PINN) Model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    
    try:
        logger.info("=== STARTING PINN TRAINING PIPELINE ===")
        
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
        
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("CUDA optimizations enabled: high float32 matmul precision, TF32 allowed")
            
        train_cfg = config['training']
        pinn_cfg = config['pinn']
        phys_cfg = config['physics']
        
        logger.info("Initializing datasets...")
        train_dataset = ThermalDataset(train_paths, scaler_X, scaler_y)
        train_sampler = FileGroupedBatchSampler(train_dataset, batch_size=pinn_cfg['batch_size'], shuffle=True)
        
        val_dataset = ThermalDataset(val_paths, scaler_X, scaler_y)
        val_sampler = FileGroupedBatchSampler(val_dataset, batch_size=pinn_cfg['batch_size'], shuffle=False)
        
        num_workers, prefetch_factor = benchmark_dataloader(train_dataset, train_sampler, pinn_cfg['batch_size'])
        
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
        
        model = PhysicsInformedNN(
            input_dim=pinn_cfg['input_dim'],
            hidden_dim=pinn_cfg['hidden_dim'],
            mCp=phys_cfg['mCp'],
            initial_hA=phys_cfg['initial_hA']
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=pinn_cfg['learning_rate'])
        grad_scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
        
        # Noise settings
        noise_std = train_cfg['noise_std']
        T_std = scaler_X.scale_[0]
        scaled_noise_std = noise_std / T_std
        
        # GPU Caching of Scaler Tensors
        mean_X_gpu = torch.tensor(scaler_X.mean_, dtype=torch.float32, device=device)
        std_X_gpu = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=device)
        mean_y_gpu = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
        std_y_gpu = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)
        
        # Recovery state
        start_epoch = 1
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_data_loss": [],
            "train_phys_loss": [],
            "val_data_loss": [],
            "val_phys_loss": [],
            "hA": []
        }
        hA_history = []
        
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
            if 'hA_history' in checkpoint:
                hA_history = checkpoint['hA_history']
            logger.info(f"Loaded checkpoint successfully. Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.6f}")
            
        # 3. Training Loop
        logger.info("--- PHASE 3: TRAINING LOOP ---")
        epochs = pinn_cfg['epochs']
        lambda_physics = pinn_cfg['lambda_physics']
        patience = train_cfg['early_stopping_patience']
        
        checkpoint_dir = os.path.join(paths['results_pinn_dir'], "../checkpoints")
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
            train_d_loss = 0.0
            train_p_loss = 0.0
            train_count = 0
            
            epoch_train_start = time.time()
            for i, (batch_X, batch_y, batch_dt) in enumerate(train_loader):
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_dt = batch_dt.to(device, non_blocking=True)
                
                # Noise injection
                if noise_std > 0.0:
                    batch_X_train = batch_X.clone()
                    noise = torch.randn_like(batch_X_train[:, 0]) * scaled_noise_std
                    batch_X_train[:, 0] += noise
                else:
                    batch_X_train = batch_X
                    
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    pred_y_scaled = model(batch_X_train)
                    
                    # Data loss
                    data_loss = criterion(pred_y_scaled, batch_y)
                    
                    # Physics loss
                    # Differentiable inverse-scaling of inputs & outputs on GPU
                    batch_X_phys = batch_X_train * std_X_gpu + mean_X_gpu
                    pred_y_phys = pred_y_scaled * std_y_gpu + mean_y_gpu
                    
                    residuals = model.compute_residual(batch_X_phys, pred_y_phys, batch_dt)
                    physics_loss = torch.mean(residuals ** 2)
                    
                    # Total Loss
                    total_loss = data_loss + lambda_physics * physics_loss
                    
                grad_scaler.scale(total_loss).backward()
                
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg['gradient_clip'])
                
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                n = batch_X.size(0)
                train_loss += total_loss.item() * n
                train_d_loss += data_loss.item() * n
                train_p_loss += physics_loss.item() * n
                train_count += n
                
            train_loss /= train_count
            train_d_loss /= train_count
            train_p_loss /= train_count
            train_duration = time.time() - epoch_train_start
            
            # Validation
            epoch_val_start = time.time()
            model.eval()
            val_loss = 0.0
            val_d_loss = 0.0
            val_p_loss = 0.0
            val_count = 0
            
            with torch.no_grad():
                for batch_X, batch_y, batch_dt in val_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    batch_dt = batch_dt.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                        pred_y_scaled = model(batch_X)
                        data_loss = criterion(pred_y_scaled, batch_y)
                        
                        batch_X_phys = batch_X * std_X_gpu + mean_X_gpu
                        pred_y_phys = pred_y_scaled * std_y_gpu + mean_y_gpu
                        
                        residuals = model.compute_residual(batch_X_phys, pred_y_phys, batch_dt)
                        physics_loss = torch.mean(residuals ** 2)
                        
                        total_loss = data_loss + lambda_physics * physics_loss
                        
                    n = batch_X.size(0)
                    val_loss += total_loss.item() * n
                    val_d_loss += data_loss.item() * n
                    val_p_loss += physics_loss.item() * n
                    val_count += n
                    
            val_loss /= val_count
            val_d_loss /= val_count
            val_p_loss /= val_count
            val_duration = time.time() - epoch_val_start
            
            current_hA = model.get_hA().item()
            hA_history.append({"epoch": epoch, "hA": current_hA})
            
            epoch_duration = time.time() - epoch_start_time
            throughput = train_count / train_duration if train_duration > 0 else 0
            
            # System Metrics Logging
            metrics = get_system_metrics(device)
            logger.info(f"Epoch {epoch:03d}/{epochs} Complete | Total Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Data Loss: {train_d_loss:.6f} | Physics Loss: {train_p_loss:.6f} | hA: {current_hA:.5f} W/K | Duration: {epoch_duration:.1f}s | Throughput: {throughput:.1f} samples/s")
            log_system_metrics(logger, prefix=f"Epoch {epoch:03d} System Resource Usage")
            
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_data_loss"].append(train_d_loss)
            history["train_phys_loss"].append(train_p_loss)
            history["val_data_loss"].append(val_d_loss)
            history["val_phys_loss"].append(val_p_loss)
            history["hA"].append(current_hA)
            
            # Checkpoint management - save immediately on validation improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_pinn.pt")
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
                    'hA_history': hA_history,
                    'config': config,
                    'seed': config['training']['random_seed']
                }, best_checkpoint_path)
                logger.info(f"Validation loss improved. Saved best PINN checkpoint to {best_checkpoint_path}")
            else:
                epochs_no_improve += 1
                
            # Periodic checkpoints (every 10 epochs)
            if epoch % 10 == 0:
                periodic_path = os.path.join(checkpoint_dir, f"checkpoint_pinn_epoch_{epoch}.pt")
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
                    'hA_history': hA_history
                }, periodic_path)
                logger.info(f"Saved periodic PINN checkpoint to {periodic_path}")
                
            # Early Stopping Check
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best model weights.")
                best_checkpoint = torch.load(os.path.join(checkpoint_dir, "best_pinn.pt"), map_location=device)
                model.load_state_dict(best_checkpoint['model_state_dict'])
                break
                
        # 4. Save Final Outputs
        logger.info("--- PHASE 4: SAVING DELIVERABLES ---")
        out_dir = paths['results_pinn_dir']
        os.makedirs(out_dir, exist_ok=True)
        
        # Save model.pt for use in evaluate/rollout/plot phase
        model_pt_path = os.path.join(out_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X_mean': scaler_X.mean_.tolist(),
            'scaler_X_scale': scaler_X.scale_.tolist(),
            'scaler_y_mean': scaler_y.mean_.tolist(),
            'scaler_y_scale': scaler_y.scale_.tolist(),
            'history': history,
            'hA_history': hA_history
        }, model_pt_path)
        logger.info(f"Saved final PINN model, scalers, and training history metadata to {model_pt_path}")
        
        # Save convective coefficient history CSV
        hA_df = pd.DataFrame(hA_history)
        hA_csv = os.path.join(out_dir, "hA_history.csv")
        hA_df.to_csv(hA_csv, index=False)
        logger.info(f"Saved learning history of convective coefficient (hA) to {hA_csv}")
        
        # Run sub-pipelines automatically
        logger.info("--- TRIGGERING EVALUATION AND ROLLOUT SUB-PIPELINES (PINN) ---")
        
        try:
            import evaluate_pinn
            logger.info("Running evaluate_pinn.py...")
            evaluate_pinn.run_evaluation(model_path=model_pt_path, config=config, device=device)
        except Exception as eval_err:
            logger.error(f"Error running evaluate_pinn sub-pipeline: {eval_err}", exc_info=True)
            
        try:
            import rollout_pinn
            logger.info("Running rollout_pinn.py...")
            rollout_pinn.run_rollout(model_path=model_pt_path, config=config, device=device)
        except Exception as roll_err:
            logger.error(f"Error running rollout_pinn sub-pipeline: {roll_err}", exc_info=True)
            
        try:
            import plot_pinn
            logger.info("Running plot_pinn.py...")
            plot_pinn.run_plotting(out_dir=out_dir)
        except Exception as plot_err:
            logger.error(f"Error running plot_pinn sub-pipeline: {plot_err}", exc_info=True)
            
        logger.info("=== PINN TRAINING PIPELINE FINISHED SUCCESSFULY ===")
        
    except KeyboardInterrupt:
        logger.warning("PINN Training interrupted by user! Saving emergency checkpoint...")
        try:
            checkpoint_dir = os.path.join(paths['results_pinn_dir'], "../checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            emergency_path = os.path.join(checkpoint_dir, f"interrupted_pinn_epoch_{epoch}.pt")
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
                'hA_history': hA_history
            }, emergency_path)
            logger.warning(f"Saved emergency checkpoint to {emergency_path}")
        except Exception as save_err:
            logger.error(f"Failed to save emergency checkpoint: {save_err}")
        logger.warning("Exiting training pipeline.")
        
    except Exception as e:
        logger.error(f"PINN Pipeline crashed due to an unhandled exception: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
