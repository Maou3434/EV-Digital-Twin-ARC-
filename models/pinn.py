import torch
import torch.nn as nn

class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for dynamic battery temperature prediction.
    
    Architecture:
    - Input: 6 features (Temperature_t, Current_t, Voltage_t, SOC_t, PowerLoss_t, AmbientTemp_t)
    - Hidden Layer 1: 64 neurons + ReLU
    - Hidden Layer 2: 64 neurons + ReLU
    - Hidden Layer 3: 32 neurons + ReLU
    - Output Layer: 1 neuron (predicted Temperature at t+1)
    
    Trainable Physical Parameters:
    - hA: Convective heat transfer coefficient (Newton cooling), initialized at 0.1 W/K.
    """
    def __init__(self, input_dim=6, hidden_dim=64, m=4.0, Cp=1000.0, initial_hA=0.1):
        super(PhysicsInformedNN, self).__init__()
        
        self.mCp = m * Cp # Thermal capacity in J/K (4000 J/K by default)
        
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
        
        # Trainable physical parameter hA. We define it as a parameter so that the
        # optimizer learns it alongside network weights.
        self.hA_param = nn.Parameter(torch.tensor(initial_hA, dtype=torch.float32, requires_grad=True))
        
    def forward(self, x):
        """
        Predicts the battery temperature at the next time step t+1.
        x: tensor of shape (batch_size, 6) containing:
           [Temperature_t, Current_t, Voltage_t, SOC_t, PowerLoss_t, AmbientTemp_t]
        """
        return self.network(x)
        
    def get_hA(self):
        """
        Helper method to return positive hA by constraining it (e.g. using ReLU or clamp
        to prevent negative heat transfer coefficients, which is physically impossible).
        """
        return torch.clamp(self.hA_param, min=1e-5)
        
    def compute_residual(self, x, y_pred, dt):
        """
        Computes the physics residual of the governing lumped thermal equation in K/s:
        Residual = dTdt_pred - (PowerLoss - Q_loss) / mCp = 0
        
        Inputs:
        - x: network inputs at time t (batch_size, 6) in physical units
          - x[:, 0]: Temperature_t
          - x[:, 4]: PowerLoss_t
          - x[:, 5]: AmbientTemp_t
        - y_pred: network prediction for Delta_T_t (batch_size, 1) in physical units
        - dt: time step size (batch_size, 1)
        """
        T_t = x[:, 0]
        PowerLoss = x[:, 4]
        AmbientTemp = x[:, 5]
        
        Delta_T_pred = y_pred.squeeze()
        dt = dt.squeeze()
        
        # Reconstruct next temperature: T_pred = T_t + Delta_T_pred
        T_pred_t1 = T_t + Delta_T_pred
        
        # Rate of change: dTdt_pred = Delta_T_pred / dt
        dTdt_pred = Delta_T_pred / dt
        
        # Newton cooling loss Q_loss = hA * (T_pred_t1 - T_amb)
        hA = self.get_hA()
        Q_loss = hA * (T_pred_t1 - AmbientTemp)
        
        # Scale the equation by dividing by mCp to avoid massive numerical gradients.
        # Residual = dTdt_pred - (PowerLoss - Q_loss) / mCp (both sides in Kelvin/second)
        residual = dTdt_pred - (PowerLoss - Q_loss) / self.mCp
        
        return residual
