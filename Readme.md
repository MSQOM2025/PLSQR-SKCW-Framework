# PLSQR--ML-SKCW Framework

Reference Python implementation for the manuscript:

> **Matrix-Free Krylov Regularization with Multi-Level Second-Kind Chebyshev Wavelets for Large-Scale Inverse Retarded Stochastic It\^o--Volterra Systems**  
> M. Salehi, M. Ahmadinia (University of Qom, Iran)

## Overview

This repository provides a **matrix-free $\mathcal{O}(N)$** solver for inverse trajectory and source reconstruction in retarded stochastic Itô–Volterra integral equations (R-SIVIEs) without truncating the multi-level wavelet approximation space.

Key components implemented in this codebase:
- **Multi-Level Second-Kind Chebyshev Wavelets (ML-SKCW):** Exact $L^2$-orthonormal two-scale fast wavelet transforms (`fwt_synthesis`, `fwt_analysis`) on local interior Gauss–Chebyshev collocation nodes.
- **Matrix-Free $\mathcal{O}(N)$ Volterra & Itô Operators:** Separable kernel action and running block-cumulative prefix sums (`cumsum`) for forward and adjoint evaluations, avoiding $\mathcal{O}(N^2)$ dense matrix storage.
- **Two-Sided Hierarchical Diagonal Preconditioning (HDP):** Combines left temporal row equilibration ($\mathbf{D}_L^{-1}$) with right dyadic Sobolev scaling ($\mathbf{D}_S^{-1} = \operatorname{diag}(2^{-js})$).
- **Preconditioned LSQR with Spectral Stop Sign (SSS):** Paige–Saunders bidiagonalization with $\mathcal{O}(1)$ recursive residual monitoring and an estimated discretization-and-noise floor $\delta_{\mathrm{prior}}$.

---

## Repository Structure

```text
.
├── ml_skcw_core.py          # Core ML-SKCW transforms, matrix-free operators, HDP, LSQR-SSS, & unit tests
├── run_benchmarks_v2.py     # Reproduces Examples 1–5, Section 7.6 error decomposition, & PDF figures
├── run_extra_benchmarks.py  # Reproduces Section 7.7 (Tikhonov/CGLS baselines & 5-seed sensitivity + CSVs)
├── baseline_comparison.csv  # Generated baseline comparison table (Section 7.7)
├── sensitivity_analysis.csv # Generated multi-seed sensitivity table (Section 7.7)
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

### 1. Run Mathematical Self-Tests
Verifies multi-level wavelet unitarity, exact matrix-free adjointness $\langle \mathbf{A}\mathbf{v}, \mathbf{u}\rangle = \langle \mathbf{v}, \mathbf{A}^\top\mathbf{u}\rangle$, and $\mathcal{O}(N \log N)$ column-norm equivalence:

```bash
python ml_skcw_core.py
```

Expected output:
```text
Running mathematical verification tests...
  [PASS] Multi-level SKCW FWT unitary check: err = 1.91e-15
  [PASS] Matrix-free Adjoint test <Av,u>==<v,A^Tu>: rel_err = 1.94e-16
  [PASS] Vectorized O(N log N) Column norm vs Explicit A: rel_err = 5.05e-17
```

### 2. Reproduce Main Benchmarks and Figures (Examples 1–5 & Section 7.6)
Runs ensemble consistency ($L=100$), conditioning comparison ($N=32\dots 512$), semi-convergence diagnostics, large-scale scalability up to $N = 262,144$, multi-frequency spectral retention ($N = 131,072$), and tripartite error decomposition:

```bash
python run_benchmarks_v2.py
```

This generates the four publication figures:
- `Example1_Aligned_Consistency.pdf`
- `Example2_Conditioning_Stable.pdf`
- `Example3_SemiConvergence.pdf`
- `Proposed_Error_Field_Forensic_Refined.pdf`

### 3. Reproduce Baseline Comparisons & Multi-Seed Sensitivity (Section 7.7)
Compares against matrix-free CGLS+SSS and Tikhonov-LSQR (full and matched iteration budgets) and runs 5-seed hyperparameter sensitivity for $(s, \tau_{\mathrm{safe}}, \varepsilon_s)$:

```bash
python run_extra_benchmarks.py
```

Outputs are exported to `baseline_comparison.csv` and `sensitivity_analysis.csv`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{SalehiAhmadinia2026PLSQR,
  title   = {{Matrix-Free Krylov Regularization with Multi-Level Second-Kind Chebyshev Wavelets for Large-Scale Inverse Retarded Stochastic It\^o--Volterra Systems},
  author  = {Salehi, M. and Ahmadinia, M.},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Contact

For questions or bug reports, please contact: **M. Salehi** (`m.salehi@qom.ac.ir`).
