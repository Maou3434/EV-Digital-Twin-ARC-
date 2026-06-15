import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    Conventional Neural Network (Multi-Layer Perceptron) for battery
    temperature delta prediction based on standard electrothermal features.
    
    Architecture:
    - Input: 7 features (Temperature_C, Current_A, Voltage_V, SOC_pct, PowerLoss_W, AmbientTemp_C, MassScale)
    - Hidden Layer 1: 64 neurons + ReLU
    - Hidden Layer 2: 64 neurons + ReLU
    - Hidden Layer 3: 32 neurons + ReLU
    - Output Layer: 1 neuron (predicted Temperature Delta)
    """
    def __init__(self, input_dim=7, hidden_dim=64):
        super(BaselineMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)
