# PLSQR--ML-SKCW Framework

Matrix-free Krylov regularization for large-scale **inverse retarded stochastic Itô--Volterra integral equations (R-SIVIEs)** using an $L^2$-orthonormalized Chebyshev-seeded multiwavelet basis (Alpert-type ML-SKCW), $\mathcal{O}(R N)$ prefix-sum operator evaluation, exact $\mathcal{O}(R^2 N)$ two-sided hierarchical diagonal preconditioning (HDP), and a Discrepancy-Based Stopping Criterion (DSC).

## Repository Contents

- `ml_skcw_core.py`: Core library implementing:
  - Exact $\mathcal{O}(pN)$ two-scale $L^2$-orthonormal multiwavelet transforms (`fwt_synthesis`, `fwt_analysis`),
  - Local interior Gauss--Chebyshev (type-I) collocation and $\mathcal{O}(RN)$ forward/adjoint Volterra and Itô operators (`RetardedSIVIEOperator`),
  - Chebyshev separable expansion for smooth non-separable kernels (`chebyshev_separable_kernel`),
  - Exact $\mathcal{O}(R^2N)$ matrix-free row-norm equilibration $\mathbf{D}_L^{-1}$ and right dyadic Sobolev/Besov scaling $\mathbf{D}_S^{-1}$ (`build_two_sided_hdp`),
  - $\mathcal{O}(N)$ banded-Cholesky tridiagonal $H^1$ ($s=1$) baseline preconditioner (`build_h1_preconditioner`),
  - Paige--Saunders LSQR with DSC (`solve_plsqr_dsc`) and Hybrid Golub--Kahan with DP/GCV (`solve_hybrid_gk`),
  - Machine-precision self-verification tests (`run_self_tests`).
- `run_upgraded_benchmarks.py`: Full non-inverse-crime benchmark suite generating observations on fine reference meshes ($k_{\mathrm{ref}} \ge k + 2$, up to $N_{\mathrm{ref}} = 1,048,576$):
  - **Example 1**: Ensemble pathwise consistency ($L = 100$, $N = 1,024$),
  - **Example 2**: Three-way spectral conditioning ($k = 4 \dots 8$, $\mathrm{amp}_2 = 0.25$),
  - **Example 3**: Semi-convergence trajectory and DSC stopping ($N = 1,024$),
  - **Example 4**: Large-scale scalability up to $N = 262,144$ ($1.938\text{ s}$, $100.80\text{ MB}$),
  - **Example 5**: Multi-frequency spectral retention at $N = 131,072$ ($96.85\%$ high-frequency retention),
  - **Example 6**: Verification of the $s > 1/2$ spectral phase transition under strong Itô forcing ($\mathrm{amp}_2 = 0.25$) on a jump-and-spike target vs. $s = 0$, left-only HDP, $s = 0.20$, and tridiagonal $H^1$,
  - **Example 7**: Non-separable smooth Gaussian kernel ($R_0 = 16$, $R_{\mathrm{tot}} = 18$) up to $N = 131,072$,
  - **Tripartite Error, Baselines, & Sensitivity**: Projection-based error decomposition, baseline comparisons (CGLS, Tikhonov-LSQR, Left-Only HDP, Tridiagonal $H^1$, Hybrid GK DP/GCV), and 5-seed hyperparameter sensitivity.

## Requirements

- Python $\ge 3.10$ (tested on Python 3.13)
- `numpy`, `scipy`, `matplotlib`

```bash
pip install numpy scipy matplotlib
