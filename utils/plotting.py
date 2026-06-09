import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define aesthetic color scheme
COLOR_TRUE = "#1D3557"    # Sleek dark navy for ground truth
COLOR_BASE = "#E63946"    # Crimson red for baseline model
COLOR_PINN = "#457B9D"    # Steel blue for PINN model
COLOR_RESID = "#A8DADC"   # Pale cyan for residuals
COLOR_TEXT = "#2B2D42"    # Dark grey for text
COLOR_GRID = "#F1FAEE"    # Light minty white for grid lines

def set_style():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#FAFAFA'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#EAEAEA'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelcolor'] = COLOR_TEXT
    plt.rcParams['xtick.color'] = COLOR_TEXT
    plt.rcParams['ytick.color'] = COLOR_TEXT

def plot_loss_curve(epochs, data_losses, physics_losses, total_losses, save_path):
    """
    Plots the training loss curve showing the breakdown of Total Loss,
    Data Loss, and Physics Loss over epochs.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, total_losses, label="Total Loss", color=COLOR_TRUE, linewidth=2.5, linestyle="-")
    plt.plot(epochs, data_losses, label="Data Loss", color=COLOR_BASE, linewidth=1.8, linestyle="--")
    plt.plot(epochs, physics_losses, label="Physics Loss (λ-weighted)", color=COLOR_PINN, linewidth=1.8, linestyle="-.")
    
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("PINN Training Loss Convergence", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss curve plot to {save_path}")

def plot_prediction_vs_truth(y_true, y_pred_base, y_pred_pinn, save_path):
    """
    Plots predicted vs true temperatures as a scatter plot with a 45-degree reference line.
    """
    set_style()
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred_base, label="Baseline MLP", color=COLOR_BASE, alpha=0.4, s=6, marker="o")
    plt.scatter(y_true, y_pred_pinn, label="PINN (Ours)", color=COLOR_PINN, alpha=0.4, s=6, marker="s")
    
    # Perfect correlation line
    min_val = min(y_true.min(), y_pred_base.min(), y_pred_pinn.min()) - 1.0
    max_val = max(y_true.max(), y_pred_base.max(), y_pred_pinn.max()) + 1.0
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Ideal (y=x)", color=COLOR_TRUE, alpha=0.8, linewidth=1.5)
    
    plt.xlabel("Ground Truth Temperature (K)")
    plt.ylabel("Predicted Temperature (K)")
    plt.title("Model Correlation: Prediction vs. Ground Truth", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved prediction vs truth scatter plot to {save_path}")

def plot_residual_distribution(residuals, save_path):
    """
    Plots the distribution (histogram) of the physics residuals to check conservation consistency.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    # Calculate statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    
    # Histogram
    count, bins, ignored = plt.hist(residuals, bins=100, color=COLOR_RESID, edgecolor='#7FB7BE', alpha=0.8, density=True)
    
    # Normal distribution approximation curve
    fit_curve = (1 / (std_res * np.sqrt(2 * np.pi))) * np.exp(-((bins - mean_res) ** 2) / (2 * std_res ** 2))
    plt.plot(bins, fit_curve, linewidth=2, color=COLOR_TRUE, label=f"Normal Fit (μ={mean_res:.3f}, σ={std_res:.3f})")
    
    plt.xlabel("Physics Equation Residual (W)")
    plt.ylabel("Probability Density")
    plt.title("PINN Physics Residual Distribution\nmCp * dT/dt - (PowerLoss - Q_loss)", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved residual distribution plot to {save_path}")

def plot_trajectory_comparison(times, y_true, y_pred_base, y_pred_pinn, save_path):
    """
    Plots temperature evolution trajectories over time for a subset of the test sequence.
    """
    set_style()
    plt.figure(figsize=(12, 6))
    
    # Sort by time for clean line plotting
    sort_idx = np.argsort(times)
    times_sorted = times[sort_idx]
    y_true_sorted = y_true[sort_idx]
    y_base_sorted = y_pred_base[sort_idx]
    y_pinn_sorted = y_pred_pinn[sort_idx]
    
    # Let's plot only the first 5000 points to make the line chart readable
    subset_len = min(5000, len(times_sorted))
    plt.plot(times_sorted[:subset_len], y_true_sorted[:subset_len], label="Ground Truth", color=COLOR_TRUE, linewidth=2.5)
    plt.plot(times_sorted[:subset_len], y_base_sorted[:subset_len], label="Baseline MLP", color=COLOR_BASE, linewidth=1.8, linestyle="--")
    plt.plot(times_sorted[:subset_len], y_pinn_sorted[:subset_len], label="PINN", color="#10B981", linewidth=2.0, linestyle="-")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Battery Pack Temperature (K)")
    plt.title("Battery Temperature Evolution Trajectory Comparison", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved trajectory comparison plot to {save_path}")

def plot_error_distribution(y_true, y_pred_base, y_pred_pinn, save_path):
    """
    Plots the error distribution (absolute errors) of the two models.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    error_base = np.abs(y_true - y_pred_base)
    error_pinn = np.abs(y_true - y_pred_pinn)
    
    plt.hist(error_base, bins=80, alpha=0.6, label=f"Baseline MLP (Mean Err: {np.mean(error_base):.3f} K)", color=COLOR_BASE, edgecolor='#9D0208')
    plt.hist(error_pinn, bins=80, alpha=0.6, label=f"PINN (Mean Err: {np.mean(error_pinn):.3f} K)", color="#10B981", edgecolor='#065F46')
    
    plt.xlabel("Absolute Temperature Error (Kelvin)")
    plt.ylabel("Frequency")
    plt.title("Absolute Prediction Error Distribution Comparison", fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved error distribution plot to {save_path}")
