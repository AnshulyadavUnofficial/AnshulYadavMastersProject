# Overcoming Magnitude Hallucinations: A Variance-Normalized Deep Learning Framework for Intraday Trading

This repository contains the official codebase for the Master's dissertation on variance-normalized spatial-temporal deep learning in high-frequency quantitative finance. It includes the complete data pipeline, Walk-Forward Optimization (WFO) framework, network architectures, and downstream Markowitz portfolio simulations.

## Abstract
Deploying deep spatial-temporal neural networks in high-frequency algorithmic trading is fundamentally bottlenecked by the inherently low financial Signal-to-Noise Ratio. Standard deep learning paradigms frequently succumb to "magnitude hallucinations," where unnormalized variance causes networks to over-predict expected returns, leading to catastrophic downstream portfolio allocation. This dissertation addresses this challenge by adapting and structurally refining the Return-Volatility (ReVol) normalization framework for intraday execution. Through rigorous Walk-Forward Optimization across the DJ30 universe, this research empirically validates that aggressively constraining the network's input space to an information bottleneck of four normalized error residuals successfully neutralizes volatility spikes and stabilizes out-of-sample prediction. 

A major contribution of this work is the structural optimization of the architecture itself. Rigorous statistical diagnostics reveal that parameterized Return Volatility Estimators (RVEs) are computationally redundant; replacing them with a deterministic statistical bottleneck drastically reduces computational overhead while preserving predictive integrity. Furthermore, portfolio simulations identify the "Smoothed Target Trap," proving that targeting artificially smoothed risk-adjusted proxy scores paralyzes Mean-Variance optimizers. Conversely, dynamic log-return reconstruction yields safely constrained signal magnitudes.

Finally, extending this philosophy to portfolio risk, the research proposes a novel Spectral Covariance Score ($S_A$). This decomposition proves that while absolute market energy is highly volatile, the underlying geometric risk hierarchy remains exceptionally persistent. Ultimately, integrating these variance-normalized predictions into a deterministic, shallow-wide spatial-temporal architecture ($L=1$, $d_{model}=256$) generated a stable, compounding out-of-sample equity curve, definitively demonstrating the viability of variance-normalized deep learning in high-frequency quantitative finance.

---

## Repository Structure

All primary scripts and notebooks required to reproduce this research are located in the `Code/` directory:

* **`Code/utility.py`**: The core backbone of the project. Contains hyperparameters, the intraday data processing pipeline, custom PyTorch neural network architectures (including `MuBackBone` and `ReVolSuperNetwork`), and the CVXPY Markowitz optimizer wrappers.
* **`Code/data_collection.ipynb`**: Handles the fetching, cleaning, and interval-alignment of the raw intraday tick data for the DJ30 universe.
* **`Code/main_nb_1.ipynb`**: The primary experiment pipeline. This notebook executes the Walk-Forward Optimization (WFO), trains the deep learning models across the ablation grid, generates predictions, and runs the downstream portfolio backtests.
* **`Code/covariance_tests.ipynb`**: Contains the empirical diagnostics and visualizations for the Spectral Covariance Score ($S_A$), proving the stability of the eigenvector subspace and the volatility of the absolute market energy scale.

---

## Installation & Setup

###  Create a Virtual Environment
It is highly recommended to run this project inside an isolated Python virtual environment.

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Install Dependencies**

Once your virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

**MOSEK License Configuration (Crucial)**
The downstream Mean-Variance portfolio optimization heavily relies on the MOSEK solver via CVXPY to handle the quadratic risk penalties efficiently. MOSEK requires a valid license to run.

Acquire a License: If you are a student or faculty member, you can obtain a free Personal Academic License from the MOSEK website here: MOSEK Academic Licenses.

Install the License: Once you receive the mosek.lic file via email, you must place it in the default MOSEK directory on your machine