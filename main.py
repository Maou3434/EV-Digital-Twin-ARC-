import os
import json
import yaml
import pandas as pd

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_comparison_report():
    print("="*70)
    print("   EV BATTERY THERMAL DIGITAL TWIN - REPORT GENERATION")
    print("="*70 + "\n")
    
    config = load_config()
    paths = config['paths']
    baseline_dir = paths['results_baseline_dir']
    pinn_dir = paths['results_pinn_dir']
    
    base_metrics_path = os.path.join(baseline_dir, "metrics.json")
    base_rollout_metrics_path = os.path.join(baseline_dir, "rollout_metrics.json")
    pinn_metrics_path = os.path.join(pinn_dir, "metrics.json")
    pinn_rollout_metrics_path = os.path.join(pinn_dir, "rollout_metrics.json")
    hA_history_path = os.path.join(pinn_dir, "hA_history.csv")
    
    # Check if files exist
    missing = []
    if not os.path.exists(base_metrics_path):
        missing.append("Baseline metrics.json")
    if not os.path.exists(base_rollout_metrics_path):
        missing.append("Baseline rollout_metrics.json")
    if not os.path.exists(pinn_metrics_path):
        missing.append("PINN metrics.json")
    if not os.path.exists(pinn_rollout_metrics_path):
        missing.append("PINN rollout_metrics.json")
        
    if missing:
        print("Error: Missing training run outputs for comparison.")
        print("Please run the training scripts first:")
        print("  1. python run_baseline.py")
        print("  2. python run_pinn.py")
        print("\nMissing items:")
        for item in missing:
            print(f"  - {item}")
        return
        
    # Load all metrics
    with open(base_metrics_path, 'r') as f:
        base_single = json.load(f)
    with open(base_rollout_metrics_path, 'r') as f:
        base_rollout = json.load(f)
        
    with open(pinn_metrics_path, 'r') as f:
        pinn_single = json.load(f)
    with open(pinn_rollout_metrics_path, 'r') as f:
        pinn_rollout = json.load(f)
        
    # Extract hA parameter
    final_hA = None
    if os.path.exists(hA_history_path):
        hA_df = pd.read_csv(hA_history_path)
        if len(hA_df) > 0:
            final_hA = hA_df["hA"].iloc[-1]
            
    # Build comparison data
    comparison_data = {
        "Evaluation Dimension": [
            "Single-step MAE (°C)", 
            "Single-step RMSE (°C)", 
            "Single-step R² Score",
            "Autoregressive Rollout MAE (°C)",
            "Autoregressive Rollout RMSE (°C)",
            "Maximum Error over Rollout (°C)"
        ],
        "Baseline MLP": [
            f"{base_single.get('MAE', 0.0):.5f}",
            f"{base_single.get('RMSE', 0.0):.5f}",
            f"{base_single.get('R2', 0.0):.5f}",
            f"{base_rollout.get('Rollout_MAE', 0.0):.5f}",
            f"{base_rollout.get('Rollout_RMSE', 0.0):.5f}",
            f"{base_rollout.get('Maximum_Error', 0.0):.5f}"
        ],
        "Physics-Informed NN (PINN)": [
            f"{pinn_single.get('MAE', 0.0):.5f}",
            f"{pinn_single.get('RMSE', 0.0):.5f}",
            f"{pinn_single.get('R2', 0.0):.5f}",
            f"{pinn_rollout.get('Rollout_MAE', 0.0):.5f}",
            f"{pinn_rollout.get('Rollout_RMSE', 0.0):.5f}",
            f"{pinn_rollout.get('Maximum_Error', 0.0):.5f}"
        ]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Display in console
    print("="*60)
    print("         MODEL PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    print(comp_df.to_string(index=False))
    print("="*60)
    if final_hA is not None:
        print(f"Physics Parameter Learned by PINN (hA): {final_hA:.5f} W/K (Initial: 0.10000 W/K)")
    print("="*60 + "\n")
    
    # Save as Markdown table
    report_path = os.path.join(paths['summary_stats'].split("/")[0], "comparison_table.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Model Performance Comparison\n\n")
        f.write("A quantitative comparison between the conventional baseline Multi-Layer Perceptron (MLP) and the Physics-Informed Neural Network (PINN) battery digital twin.\n\n")
        f.write("| Evaluation Dimension | Baseline MLP | Physics-Informed NN (PINN) |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Single-step MAE (°C)** | {base_single.get('MAE', 0.0):.5f} | {pinn_single.get('MAE', 0.0):.5f} |\n")
        f.write(f"| **Single-step RMSE (°C)** | {base_single.get('RMSE', 0.0):.5f} | {pinn_single.get('RMSE', 0.0):.5f} |\n")
        f.write(f"| **Single-step R² Score** | {base_single.get('R2', 0.0):.5f} | {pinn_single.get('R2', 0.0):.5f} |\n")
        f.write(f"| **Autoregressive Rollout MAE (°C)** | {base_rollout.get('Rollout_MAE', 0.0):.5f} | {pinn_rollout.get('Rollout_MAE', 0.0):.5f} |\n")
        f.write(f"| **Autoregressive Rollout RMSE (°C)** | {base_rollout.get('Rollout_RMSE', 0.0):.5f} | {pinn_rollout.get('Rollout_RMSE', 0.0):.5f} |\n")
        f.write(f"| **Maximum Error over Rollout (°C)** | {base_rollout.get('Maximum_Error', 0.0):.5f} | {pinn_rollout.get('Maximum_Error', 0.0):.5f} |\n\n")
        
        if final_hA is not None:
            f.write(f"### Trainable Physical Properties Learned by PINN\n")
            f.write(f"* **Convective Heat Transfer Coefficient ($hA$):** {final_hA:.5f} W/K\n")
            
    print(f"Saved comparison report table to {report_path}")

if __name__ == "__main__":
    generate_comparison_report()
