import os
import json
import yaml
import pandas as pd
import numpy as np
from utils.preprocessing import process_and_validate_data
from training.train_baseline import train_baseline
from training.train_pinn import train_pinn
from utils.plotting import (
    plot_loss_curve,
    plot_prediction_vs_truth,
    plot_residual_distribution,
    plot_trajectory_comparison,
    plot_error_distribution
)

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_comparison_report(config):
    paths = config['paths']
    baseline_metrics_path = config['baseline_mlp']['metrics_path']
    pinn_metrics_path = config['pinn']['metrics_path']
    
    # Load metrics
    with open(baseline_metrics_path, 'r') as f:
        baseline_metrics = json.load(f)
    with open(pinn_metrics_path, 'r') as f:
        pinn_metrics = json.load(f)
        
    # Build comparison dataframe
    comparison_data = {
        "Metric": ["MAE (Kelvin)", "RMSE (Kelvin)", "R² Score"],
        "Baseline MLP (Static)": [
            f"{baseline_metrics['MAE']:.5f}",
            f"{baseline_metrics['RMSE']:.5f}",
            f"{baseline_metrics['R2']:.5f}"
        ],
        "Physics-Informed NN (PINN)": [
            f"{pinn_metrics['MAE']:.5f}",
            f"{pinn_metrics['RMSE']:.5f}",
            f"{pinn_metrics['R2']:.5f}"
        ]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Display in console
    print("\n" + "="*60)
    print("         MODEL PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    print(comp_df.to_string(index=False))
    print("="*60 + "\n")
    
    # Save as Markdown table
    report_path = os.path.join(paths['results_dir'], "comparison_table.md")
    with open(report_path, 'w') as f:
        f.write("# Model Performance Comparison\n\n")
        f.write("A quantitative comparison between the conventional baseline Multi-Layer Perceptron (MLP) and the Physics-Informed Neural Network (PINN) battery digital twin.\n\n")
        f.write("| Metric | Baseline MLP (Static) | Physics-Informed NN (PINN) |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **MAE (K)** | {baseline_metrics['MAE']:.5f} | {pinn_metrics['MAE']:.5f} |\n")
        f.write(f"| **RMSE (K)** | {baseline_metrics['RMSE']:.5f} | {pinn_metrics['RMSE']:.5f} |\n")
        f.write(f"| **R² Score** | {baseline_metrics['R2']:.5f} | {pinn_metrics['R2']:.5f} |\n\n")
        
        # Load final trained parameter from PINN predictions or log
        loss_history_path = os.path.join(paths['results_dir'], "pinn_loss_history.csv")
        if os.path.exists(loss_history_path):
            lh_df = pd.read_csv(loss_history_path)
            final_hA = lh_df["hA"].iloc[-1]
            f.write(f"### Trainable Physical Properties Learned by PINN\n")
            f.write(f"* **Convective Heat Transfer Coefficient ($hA$):** {final_hA:.5f} W/K\n")
            
    print(f"Saved comparison report table to {report_path}")

def run_visualizations(config):
    print("Generating and saving diagnostic visualizations...")
    paths = config['paths']
    pinn_cfg = config['pinn']
    
    # 1. Load predictions
    base_pred_path = config['baseline_mlp']['predictions_path']
    pinn_pred_path = os.path.join(paths['results_dir'], "pinn_predictions.csv")
    loss_history_path = os.path.join(paths['results_dir'], "pinn_loss_history.csv")
    
    base_df = pd.read_csv(base_pred_path)
    pinn_df = pd.read_csv(pinn_pred_path)
    loss_df = pd.read_csv(loss_history_path)
    
    # 2. Extract values
    y_true_base = base_df["True_Temperature"].values
    y_pred_base = base_df["Predicted_Temperature"].values
    
    y_true_pinn = pinn_df["True_Temperature"].values
    y_pred_pinn = pinn_df["Predicted_Temperature"].values
    times = pinn_df["Time"].values
    residuals = pinn_df["Physics_Residual"].values
    
    # 3. Generate individual plots
    
    # Loss convergence
    plot_loss_curve(
        epochs=loss_df["epoch"].values,
        data_losses=loss_df["data_loss"].values,
        physics_losses=loss_df["physics_loss"].values,
        total_losses=loss_df["total_loss"].values,
        save_path=pinn_cfg['loss_curve_path']
    )
    
    # Prediction Correlation
    plot_prediction_vs_truth(
        y_true=y_true_pinn,
        y_pred_base=y_pred_base, # We align them since test sets are identically split
        y_pred_pinn=y_pred_pinn,
        save_path=pinn_cfg['prediction_path']
    )
    
    # Residual Histogram
    plot_residual_distribution(
        residuals=residuals,
        save_path=pinn_cfg['residual_path']
    )
    
    # Trajectory plot
    plot_trajectory_comparison(
        times=times,
        y_true=y_true_pinn,
        y_pred_base=y_pred_base,
        y_pred_pinn=y_pred_pinn,
        save_path=pinn_cfg['trajectory_path']
    )
    
    # Error distribution
    plot_error_distribution(
        y_true=y_true_pinn,
        y_pred_base=y_pred_base,
        y_pred_pinn=y_pred_pinn,
        save_path=pinn_cfg['error_path']
    )
    
    print("All plots saved successfully!\n")

def main():
    print("="*70)
    print("       THERMAL BATTERY DIGITAL TWIN PINN - MASTER PIPELINE")
    print("="*70 + "\n")
    
    # 1. Load configurations
    config = load_config()
    
    # 2. Preprocess, validate, and formulate state space datasets
    process_and_validate_data()
    
    # 3. Train Baseline MLP
    train_baseline()
    
    # 4. Train Physics-Informed NN
    train_pinn()
    
    # 5. Generate Quantitative Comparison Report
    generate_comparison_report(config)
    
    # 6. Generate and Save Visualizations
    run_visualizations(config)
    
    print("="*70)
    print("            PIPELINE EXECUTED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()
