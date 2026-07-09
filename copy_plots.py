import os
import shutil
import urllib.request

def copy_if_exists(src, dest):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)
        print(f"Copied {src} to {dest}")
    else:
        print(f"Source not found: {src}")

def main():
    print("Setting up paper assets...")
    
    # 1. Copy PINN plots
    copy_if_exists("results/pinn/loss.png", "paper/figures/loss.png")
    copy_if_exists("results/pinn/residual.png", "paper/figures/residual.png")
    copy_if_exists("results/pinn/rollout_trajectory.png", "paper/figures/rollout_trajectory.png")
    
    # 2. Copy Baseline plots
    copy_if_exists("results/baseline/loss.png", "paper/figures/baseline_loss.png")
    copy_if_exists("results/baseline/rollout_trajectory.png", "paper/figures/baseline_rollout_trajectory.png")
    
    # 3. Try to download IEEEtran.cls
    dest_cls = "paper/IEEEtran.cls"
    if not os.path.exists(dest_cls):
        url = "https://raw.githubusercontent.com/ieeeorg/IEEEtran/master/IEEEtran.cls"
        print(f"Attempting to download IEEEtran.cls from {url}...")
        try:
            urllib.request.urlretrieve(url, dest_cls)
            print(f"Successfully downloaded IEEEtran.cls to {dest_cls}")
        except Exception as e:
            print(f"Could not download IEEEtran.cls: {e}")
            print("Note: If you compile this paper on Overleaf or a local system with LaTeX pre-installed, IEEEtran.cls is usually included globally by the compiler.")

if __name__ == "__main__":
    main()
