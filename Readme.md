# PLSQR--ML-SKCW Framework

Reference Python implementation for the manuscript:

> **Matrix-Free Krylov Regularization with Multi-Level Second-Kind Chebyshev Wavelets for Large-Scale Inverse Retarded Stochastic Itô–Volterra Systems**  
> M. Salehi, M. Ahmadinia (University of Qom, Iran)

## Overview

This repository provides a **matrix-free $\mathcal{O}(RN)$** solver for inverse trajectory and source reconstruction in retarded stochastic Itô–Volterra integral equations (R-SIVIEs) without truncating the wavelet approximation space.

Key components implemented in this codebase:
- **$L^2$-Orthonormal Multi-Level Second-Kind Chebyshev Wavelets (ML-SKCW):** Exact $L^2$-unitary two-scale fast wavelet transforms (`fwt_synthesis`, `fwt_analysis`) on local interior Gauss–Chebyshev collocation nodes.
- **Matrix-Free Volterra & Itô Operators:** Separable kernel evaluations and running block-cumulative prefix sums (`cumsum`) for forward and adjoint actions in $\mathcal{O}(RN)$ operations, avoiding $\mathcal{O}(N^2)$ dense storage.
- **Two-Sided Hierarchical Diagonal Preconditioning (HDP):** Combines exact $\mathcal{O}(R^2 N)$ left row-norm equilibration ($\mathbf{D}_L^{-1}$) for general asymmetric kernels with right dyadic Sobolev scaling ($\mathbf{D}_S^{-1} = \operatorname{diag}(2^{-js})$), including support for the $s > 1/2$ stochastic operator stabilization regime.
- **Preconditioned LSQR with Discrepancy-Based Stopping Criterion (DSC):** Paige–Saunders bidiagonalization with $\mathcal{O}(1)$ recursive residual monitoring, an adapted Morozov discrepancy floor $\delta_{\mathrm{prior}}$, and a stagnation safeguard.

---

## Repository Structure

```text
.
├── ml_skcw_core.py            # Core ML-SKCW transforms, matrix-free operators, O(R^2 N) HDP, LSQR-DSC, & unit tests
├── run_upgraded_benchmarks.py # Reproduces Examples 1–6, Section 7.7–7.8 benchmarks, PDF figures, & CSV tables
├── baseline_comparison.csv    # Generated baseline comparison results (Section 7.8)
├── sensitivity_analysis.csv   # Generated 5-seed hyperparameter sensitivity results (Section 7.8)
└── README.md
```

---

## Requirements

- Python $\ge 3.10$ (tested on Python 3.13)
- `numpy`
- `scipy`
- `matplotlib`

Install dependencies via:

```bash
pip install numpy scipy matplotlib
```

---

## Quick Start & Reproducibility

### 1. Run Mathematical Verification Tests
Verifies multi-level wavelet unitarity, exact matrix-free adjointness $\langle \mathbf{A}\mathbf{v}, \mathbf{u}\rangle = \langle \mathbf{v}, \mathbf{A}^\top\mathbf{u}\rangle$, and exactness of the $\mathcal{O}(R^2 N)$ row-norm evaluator on asymmetric kernels:

```bash
python ml_skcw_core.py
```

Expected output:
```text
Running mathematical verification tests...
  [PASS] Multi-level SKCW FWT unitary check: err = 1.91e-15
  [PASS] Matrix-free Adjoint test <Av,u>==<v,A^Tu>: rel_err = 8.85e-16
  [PASS] O(R^2 N) Row norm vs Explicit A: rel_err = 3.03e-15
```

### 2. Reproduce All Benchmarks, Figures, and CSV Tables
Runs all numerical experiments from the manuscript on asymmetric kernels:
- **Example 1:** Ensemble pathwise consistency ($L = 100$, $N = 1024$)
- **Example 2:** Three-way spectral conditioning analysis ($N = 32 \dots 512$, strong $K_2 = 0.25$)
- **Example 3:** Semi-convergence diagnostics and DSC stopping ($N = 1024$)
- **Example 4:** Large-scale scalability up to $N = 262,144$ ($1.83\text{ s}$, $100.8\text{ MB}$ peak RAM)
- **Example 5:** Multi-frequency spectral energy retention ($95.76\%$ on $50$--$150\text{ Hz}$ at $N = 131,072$)
- **Example 6:** Verification of the $s > 1/2$ stochastic stabilization threshold ($K_2 = 0.25$, non-smooth jump/pulse target vs. 64-point averaging rival)
- **Section 7.7–7.8:** Tripartite error decomposition, ablation study, matched/full-budget Tikhonov-LSQR & CGLS baselines, and 5-seed hyperparameter sensitivity

```bash
python run_upgraded_benchmarks.py
```

Generated figures:
- `Example1_Aligned_Consistency.pdf`
- `Example2_Conditioning_Stable.pdf`
- `Example3_SemiConvergence.pdf`
- `Example5_Error_Field.pdf`
- `Example6_Stochastic_Threshold.pdf`

Generated tables:
- `baseline_comparison.csv`
- `sensitivity_analysis.csv`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{SalehiAhmadinia2026MLSKCW,
  title   = {Matrix-Free Krylov Regularization with Multi-Level Second-Kind Chebyshev Wavelets for Large-Scale Inverse Retarded Stochastic It\^o--Volterra Systems},
  author  = {Salehi, M. and Ahmadinia, M.},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Contact

For questions or bug reports, please contact: **M. Salehi** (`m.salehi@qom.ac.ir`).
