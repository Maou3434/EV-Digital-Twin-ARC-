import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for dynamic battery temperature prediction.
    
    Architecture:
    - Input: 7 features (Temperature_C, Current_A, Voltage_V, SOC_pct, PowerLoss_W, AmbientTemp_C, MassScale)
    - Hidden Layer 1: 64 neurons + ReLU
    - Hidden Layer 2: 64 neurons + ReLU
    - Hidden Layer 3: 32 neurons + ReLU
    - Output Layer: 1 neuron (predicted Temperature Delta)
    
    Trainable Physical Parameters:
    - hA: Convective heat transfer coefficient, initialized at 0.1 W/K, constrained positive via softplus.
    """
    def __init__(self, input_dim=7, hidden_dim=64, mCp=900.0, initial_hA=0.1):
        super(PhysicsInformedNN, self).__init__()
        
        self.mCp = mCp # Base thermal capacity (J/K)
        
        # Dense network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Constrain hA positive via softplus. 
        # To start hA at exactly initial_hA, we initialize self.hA_raw to log(exp(initial_hA) - 1)
        raw_initial = np.log(np.exp(initial_hA) - 1.0)
        self.hA_raw = nn.Parameter(torch.tensor(raw_initial, dtype=torch.float32, requires_grad=True))
        
    def forward(self, x):
        """
        Predicts the battery temperature delta over the time step.
        x: tensor of shape (batch_size, 7) containing the scaled features.
        """
        return self.network(x)
        
    def get_hA(self):
        """
        Helper method to return positive convective heat transfer coefficient.
        Uses F.softplus to enforce positivity instead of clipping.
        """
        return F.softplus(self.hA_raw)
        
    def compute_residual(self, x_phys, Delta_T_pred_phys, dt):
        """
        Computes the physics residual of the governing lumped thermal equation in K/s:
        dT/dt = (PowerLoss - Q_loss) / (mCp * MassScale)
        Residual = dTdt_pred - (PowerLoss - Q_loss) / (mCp * MassScale)
        
        Inputs:
        - x_phys: inputs at time t in PHYSICAL units (batch_size, 7)
          - x_phys[:, 0:1]: Temperature_C
          - x_phys[:, 4:5]: PowerLoss_W
          - x_phys[:, 5:6]: AmbientTemp_C
          - x_phys[:, 6:7]: MassScale
        - Delta_T_pred_phys: predicted temperature delta in physical units (batch_size, 1)
        - dt: time step size in seconds (batch_size, 1)
        """
        T_t = x_phys[:, 0:1]
        PowerLoss = x_phys[:, 4:5]
        AmbientTemp = x_phys[:, 5:6]
        MassScale = x_phys[:, 6:7]
        
        # Rate of change: dTdt_pred = Delta_T_pred_phys / dt
        dTdt_pred = Delta_T_pred_phys / dt
        
        # Newton cooling loss Q_loss = hA * (T_t - T_amb)
        hA = self.get_hA()
        Q_loss = hA * (T_t - AmbientTemp)
        
        # Governing differential equation residual
        residual = dTdt_pred - (PowerLoss - Q_loss) / (self.mCp * MassScale)
        
        return residual
