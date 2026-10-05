# Hyperparameter Scaling Analysis of Deep Neural Networks

This repository contains a physics-inspired, systematic analysis of Neural Network training dynamics, capacity scaling, and systematic errors. Instead of treating deep learning as a black-box optimization problem, this project explores the scaling laws and convergence behaviors of a Multi-Layer Perceptron (MLP) trained on the EMNIST dataset.

## Key Concepts Explored

* **Capacity Scaling & Saturation:** Analyzing how network degrees of freedom (width $2^N$ and depth) relate to the generalization error, identifying the transition from under- to over-parameterization.
* **Stochastic Dynamics & Power Laws:** Fitting power-law scaling models ($T \propto B^{-\alpha}$) to investigate the trade-off between mini-batch stochasticity and GPU computational efficiency.
* **Loss Landscapes & Non-linearities:** Evaluating the asymptotic error convergence rates of different activation functions (e.g., mitigating the vanishing gradient problem).
* **Intrinsic Dataset Limits:** Differentiating between statistical noise and systematic topological confusion using normalized confusion matrices.

## Repository Structure

```text
.
├── configs/                  # Experiment configurations (YAML/JSON)
├── figures/                  # Publication-ready vector graphics (SVG/PDF)
├── notebooks/
│   └── hyperparameter_scaling_analysis.ipynb  # Main analytical showcase 
├── RawNetworkData/           # Serialized training histories and metrics (.pkl)
└── src/                      # Modular Python source code
    ├── data.py               # EMNIST data loading and normalization
    ├── models.py             # Flexible Keras model architectures
    ├── training.py           # Training loops and custom callbacks
    └── visualization.py      # Standardized plotting routines
    