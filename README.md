# PINN for 4th-Order Boussinesq Equation

This repository contains a self-developed implementation of a **Physics-Informed Neural Network (PINN)** designed to solve the nonlinear 4th-order Boussinesq equation. 

The project is rooted in the first principles of PINNs, as introduced by Raissi et al., and specifically adapts the methodology presented in the provided supplementary research paper to evaluate high-order PDE convergence.

## 🌊 Problem Statement
The Boussinesq equation describes the evolution of long waves in shallow water. We solve the following non-linear 4th-order PDE:

$$u_{tt} - u_{xx} - 3(u^2)_{xx} - u_{xxxx} = 0$$

## 🚀 Key Features
*   **Self-Developed Implementation:** While inspired by existing literature, the codebase is built from scratch to explore gradient stability in high-order derivatives.
*   **Physics-Constraint Integration:** The loss function incorporates the PDE residual directly via Automatic Differentiation (AD).
*   **Comparative Analysis:** Results are validated against the benchmarks provided in the reference paper to ensure accuracy in capturing wave propagation and peak amplitudes.

## 🏗 Project Architecture
The implementation is modularized for clarity and scalability:

1.  **`import.py`**: Centralized management of dependencies (PyTorch/TensorFlow, NumPy, Matplotlib).
2.  **`network.py`**: Definition of the Deep Neural Network architecture, including custom initialization and activation functions (e.g., Tanh/Swish) suitable for higher-order derivatives.
3.  **`main.py`**: The training, including the sampling, loss function formulation (Data + Physics + Boundary), and optimizer scheduling (Adam + L-BFGS).
4.  **`testing+sample_plot.py`**: Evaluation scripts to calculate L2 relative error and generate contour plots, 3D surface visualizations, and temporal snapshots.

## 📊 Results
Our implementation successfully captures the nonlinear dynamics of the Boussinesq system. Detailed comparisons regarding convergence rates and residual minimization are documented in the report, which also provides a brief description of the work.
