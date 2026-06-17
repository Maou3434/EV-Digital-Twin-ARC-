import os
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt

from utils.plotting import set_style, COLOR_TRUE, COLOR_BASE, COLOR_TEXT
from utils.logging_utils import setup_logger

logger = setup_logger("plotting", "logs/evaluation.log")

def run_plotting(out_dir="results/baseline"):
    """
    Generates training loss convergence plot and rollout trajectory plot.
    """
    logger.info(f"Generating plots and saving to: {out_dir}")
    
    # 1. Load history from model.pt
    model_pt = os.path.join(out_dir, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"Model file not found at {model_pt}")
        
    checkpoint = torch.load(model_pt, map_location="cpu")
    if "history" not in checkpoint:
        raise KeyError(f"No history key found in model checkpoint {model_pt}")
        
    history = checkpoint["history"]
    
    # Set styling
    set_style()
    
    # Plot 1: Loss Curve
    loss_png = os.path.join(out_dir, "loss.png")
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], label="Train Loss", color=COLOR_TRUE, linewidth=2.5)
    plt.plot(history['epoch'], history['val_loss'], label="Val Loss", color=COLOR_BASE, linewidth=1.8, linestyle="--")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.title("Baseline MLP Loss Convergence", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    plt.savefig(loss_png, dpi=300)
    plt.close()
    logger.info(f"Saved loss curve to {loss_png}")
    
    # Plot 2: Rollout trajectory comparison for Trajectory 0
    rollout_csv = os.path.join(out_dir, "rollout.csv")
    if os.path.exists(rollout_csv):
        rollout_df = pd.read_csv(rollout_csv)
        traj0 = rollout_df[rollout_df["Trajectory_ID"] == 0]
        
        if len(traj0) > 0:
            rollout_png = os.path.join(out_dir, "rollout_trajectory.png")
            plt.figure(figsize=(12, 6))
            plt.plot(traj0["Time_s"], traj0["True_Temperature"], label="Ground Truth", color=COLOR_TRUE, linewidth=2.5)
            plt.plot(traj0["Time_s"], traj0["Predicted_Temperature"], label="Baseline MLP Rollout", color=COLOR_BASE, linewidth=1.8, linestyle="--")
            plt.xlabel("Time (s)")
            plt.ylabel("Battery Temperature (°C)")
            plt.title("Autoregressive Trajectory Rollout Comparison (Test Cycle 0)", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
            plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
            plt.tight_layout()
            plt.savefig(rollout_png, dpi=300)
            plt.close()
            logger.info(f"Saved rollout trajectory comparison to {rollout_png}")
        else:
            logger.warning("No data found for Trajectory_ID = 0 in rollout.csv")
    else:
        logger.warning(f"rollout.csv not found at {rollout_csv}. Skipping rollout plot.")

def main():
    parser = argparse.ArgumentParser(description="Generate Baseline MLP Plots")
    parser.add_argument("--results-dir", type=str, default="results/baseline", help="Directory containing model.pt and rollout.csv")
    args = parser.parse_args()
    
    try:
        run_plotting(out_dir=args.results_dir)
    except Exception as e:
        logger.error(f"Plotting generation failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
