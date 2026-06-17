import os
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import set_style, COLOR_TRUE, COLOR_BASE, COLOR_PINN, COLOR_RESID, COLOR_TEXT
from utils.logging_utils import setup_logger

logger = setup_logger("plotting", "logs/evaluation.log")

def run_plotting(out_dir="results/pinn"):
    """
    Generates loss convergence plots, residual distributions, and rollout trajectory plots for PINN.
    """
    logger.info(f"Generating PINN plots and saving to: {out_dir}")
    
    # 1. Load history from model.pt
    model_pt = os.path.join(out_dir, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"Model file not found at {model_pt}")
        
    checkpoint = torch.load(model_pt, map_location="cpu")
    if "history" not in checkpoint:
        raise KeyError(f"No history key found in model checkpoint {model_pt}")
        
    history = checkpoint["history"]
    
    set_style()
    
    # Plot 1: Loss convergence breakdown
    loss_png = os.path.join(out_dir, "loss.png")
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], label="Total Train Loss", color=COLOR_TRUE, linewidth=2.5)
    plt.plot(history['epoch'], history['val_loss'], label="Total Val Loss", color=COLOR_BASE, linewidth=1.8, linestyle="--")
    plt.plot(history['epoch'], history['train_data_loss'], label="Data Loss (Train)", color=COLOR_PINN, linewidth=1.5, linestyle="-.")
    plt.plot(history['epoch'], history['train_phys_loss'], label="Physics Loss (Train)", color="#10B981", linewidth=1.5, linestyle=":")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("PINN Loss Convergence Breakdown", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    plt.savefig(loss_png, dpi=300)
    plt.close()
    logger.info(f"Saved PINN loss curve to {loss_png}")
    
    # Plot 2: Residual Distribution
    preds_csv = os.path.join(out_dir, "predictions.csv")
    if os.path.exists(preds_csv):
        preds_df = pd.read_csv(preds_csv)
        if "Physics_Residual" in preds_df.columns:
            residuals = preds_df["Physics_Residual"].values
            residual_png = os.path.join(out_dir, "residual.png")
            
            plt.figure(figsize=(10, 6))
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            
            # Histogram
            count, bins, ignored = plt.hist(residuals, bins=100, color=COLOR_RESID, edgecolor='#7FB7BE', alpha=0.8, density=True)
            # Normal distribution fit
            fit_curve = (1 / (std_res * np.sqrt(2 * np.pi))) * np.exp(-((bins - mean_res) ** 2) / (2 * std_res ** 2))
            plt.plot(bins, fit_curve, linewidth=2, color=COLOR_TRUE, label=f"Normal Fit (μ={mean_res:.3f}, σ={std_res:.3f})")
            
            plt.xlabel("Physics Residual (K/s)")
            plt.ylabel("Probability Density")
            plt.title("PINN Physics Residual Distribution (TEST set)\ndT/dt - (PowerLoss - Q_loss) / (mCp * MassScale)", fontsize=13, fontweight="bold", pad=15, color=COLOR_TEXT)
            plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
            plt.tight_layout()
            plt.savefig(residual_png, dpi=300)
            plt.close()
            logger.info(f"Saved residual distribution to {residual_png}")
        else:
            logger.warning("Physics_Residual column not found in predictions.csv. Skipping residual plot.")
    else:
        logger.warning(f"predictions.csv not found at {preds_csv}. Skipping residual plot.")
        
    # Plot 3: Rollout Trajectory comparison for Trajectory 0
    rollout_csv = os.path.join(out_dir, "rollout.csv")
    if os.path.exists(rollout_csv):
        rollout_df = pd.read_csv(rollout_csv)
        traj0 = rollout_df[rollout_df["Trajectory_ID"] == 0]
        
        if len(traj0) > 0:
            rollout_png = os.path.join(out_dir, "rollout_trajectory.png")
            plt.figure(figsize=(12, 6))
            plt.plot(traj0["Time_s"], traj0["True_Temperature"], label="Ground Truth", color=COLOR_TRUE, linewidth=2.5)
            plt.plot(traj0["Time_s"], traj0["Predicted_Temperature"], label="PINN Rollout", color="#10B981", linewidth=1.8, linestyle="-")
            plt.xlabel("Time (s)")
            plt.ylabel("Battery Temperature (°C)")
            plt.title("Autoregressive Trajectory Rollout Comparison (Test Cycle 0) - PINN", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
            plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
            plt.tight_layout()
            plt.savefig(rollout_png, dpi=300)
            plt.close()
            logger.info(f"Saved PINN rollout trajectory comparison to {rollout_png}")
        else:
            logger.warning("No data found for Trajectory_ID = 0 in rollout.csv")
    else:
        logger.warning(f"rollout.csv not found at {rollout_csv}. Skipping rollout plot.")

def main():
    parser = argparse.ArgumentParser(description="Generate PINN Plots")
    parser.add_argument("--results-dir", type=str, default="results/pinn", help="Directory containing PINN model.pt and csv files")
    args = parser.parse_args()
    
    try:
        run_plotting(out_dir=args.results_dir)
    except Exception as e:
        logger.error(f"PINN Plotting generation failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
