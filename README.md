# Physics-Based Thermal Model using Physics-Informed Neural Networks

## Overview
This project presents a physics-informed neural network (PINN) framework for control-oriented building thermal modeling. The approach combines data-driven learning with physical laws governing heat transfer to predict room temperature, power usage, and hidden thermal states with improved accuracy, robustness, and data efficiency.

The model integrates neural networks with explicit physics constraints, enabling physically consistent long-horizon predictions suitable for building energy management and predictive control applications.

---

## Key Objectives
- Develop control-oriented thermal models for buildings
- Reduce training data requirements through physics-informed learning
- Improve long-horizon temperature and energy predictions
- Maintain physical interpretability while leveraging neural networks

---

## Model Architecture

### PhysRegMLP (Physics-Regularized MLP)

The core model is implemented using **PyTorch Lightning** and consists of:
- A data-driven MLP for state prediction
- Trainable physical parameters representing thermal dynamics
- A physics-based constraint block embedded in the loss function

### Network Inputs
The MLP receives the following inputs at time step `k`:
- Time index
- Current room temperature `T_r(k)`
- Previous room temperatures `T_r(k-1), T_r(k-2)`
- Control action `u(k)`
- Outside temperature `T_a(k)`

### Network Outputs
The model predicts:
- Next-step room temperature `T_r(k+1)`
- Physical control input `u_phys(k)`
- Hidden thermal mass temperature `T_m(k)`

---

## Physics-Informed Components

### Trainable Physical Parameters
The model includes learnable parameters representing thermal coefficients:
- Heat transfer coefficients (`c11`, `c12`, `c21`, `c22`)
- Control influence coefficient (`b1`)
- Additional thermal dynamics coefficients (`d11`–`d23`)

These parameters are optimized jointly with the neural network while being constrained to remain physically valid.

---

## Physics Block
A dedicated physics block enforces heat transfer dynamics by:
- Estimating temperature derivatives using finite differences
- Applying a physics-based equation to estimate thermal mass temperature
- Smoothing estimates using convolutional filtering
- Scaling outputs to match learned representations

This block ensures the latent thermal state adheres to realistic physical behavior.

---

## Loss Function Design

The training objective is a **physics-constrained composite loss**:

### 1. Prediction Loss
Mean squared error (MSE) for:
- Room temperature prediction
- Control action prediction

### 2. Model Loss
MSE between:
- Neural network latent thermal state
- Physics-based thermal mass estimate

### 3. Constraint Loss
Hard physical constraints enforced via penalty terms:
- Positivity of thermal coefficients
- Stability constraints on parameter relationships

The total loss is a weighted sum: 

Total Loss = λ₁ · Prediction Loss + λ₂ · Model Loss + λ₂ · Constraint Loss

---

## Optimization Strategy

- Optimizer: Adam
- Separate learning rates:
  - Neural network parameters
  - Physical parameters (smaller learning rate)
- Learning rate scheduler: ReduceLROnPlateau
- Framework: PyTorch Lightning

This setup stabilizes training while preserving physical interpretability.

---

## Training Pipeline

1. Load aggregated state and label data
2. Construct time-shifted input-output pairs
3. Forward pass through MLP
4. Physics-based latent state estimation
5. Compute constrained loss
6. Update neural and physical parameters jointly

---

## Data Handling

### Training Data
The model expects:
- `x_agg_k`: Aggregated state at time `k`
- `label_k`: Ground truth at time `k`
- `x_agg_k1`: Aggregated state at time `k+1`
- `label_k1`: Ground truth at time `k+1`

Data is loaded using PyTorch `TensorDataset` and `DataLoader`.

---

## Inference Capabilities

- **Temperature Prediction:** Predict next-step room temperature
- **Latent State Estimation:** Infer hidden thermal mass temperature
- **Model-Based Control:** Enable downstream MPC or RL integration

---

## Design Principles

- Physics-Consistency: Explicit enforcement of thermal laws
- Data Efficiency: Reduced reliance on large datasets
- Interpretability: Learnable parameters map to physical quantities
- Modularity: Clear separation of network, physics, and loss components
- Control-Oriented: Designed for predictive control applications

---

## Scalability and Extensions

This framework can be extended to:
- Multi-zone building models
- HVAC system integration
- Reinforcement learning for energy optimization
- Model Predictive Control (MPC)
- Real-time building energy management systems

---

## Constraints and Limitations

- Assumes lumped thermal dynamics
- Requires reasonably accurate sensor data
- Physics model structure must be predefined
- Designed primarily for control-oriented applications

---

## Dependencies

- Python 3.8+
- PyTorch
- PyTorch Lightning
- NumPy

---

## Key Takeaways
- Physics-informed neural networks improve thermal modeling accuracy
- Physical constraints enhance robustness and interpretability
- Latent thermal state estimation is critical for long-horizon prediction
- PINNs provide a practical balance between physics-based and data-driven models

---

## License
This project is intended for educational and research use.  
Refer to applicable licenses for PyTorch, PyTorch Lightning, and associated datasets.
