# Model Performance Comparison

A quantitative comparison between the conventional baseline Multi-Layer Perceptron (MLP) and the Physics-Informed Neural Network (PINN) battery digital twin.

| Metric | Baseline MLP (Static) | Physics-Informed NN (PINN) |
| :--- | :---: | :---: |
| **MAE (K)** | 0.00005 | 0.00010 |
| **RMSE (K)** | 0.00006 | 0.00012 |
| **R² Score** | 1.00000 | 1.00000 |

### Trainable Physical Properties Learned by PINN
* **Convective Heat Transfer Coefficient ($hA$):** 0.00001 W/K
