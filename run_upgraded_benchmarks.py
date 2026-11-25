"""
run_upgraded_benchmarks.py
Complete, non-inverse-crime benchmark suite aligned with the updated theory:
- Reference forward data generated on a fine mesh k_ref >= k + 2 using continuous X(t),
  exact delay t - tau, true history phi, and Brownian-bridge sub-increments.
- Exact timing separated from tracemalloc overhead.
- Examples 1-5: Ensemble consistency, 3-way conditioning, semi-convergence, scalability
  up to N = 262,144, and spectral retention.
- Example 6: Strong Ito (amp2=0.25) on multi-scale Jump + 220 Hz localized burst,
  showing BOTH why coarse N<=2048 fails (E_proj is large) AND why s > 1/2 is mandatory
  at large N (comparing against s=0, s=0.20, Left-Only, and tridiagonal H^1 s=1).
- Example 7: Non-separable Gaussian kernel (R_0=12 via Chebyshev expansion) up to N=131,072.
- Section 7.7/7.8: Genuine tripartite error decomposition (E_proj, E_R, E_S), full
  baseline comparison (including Left-Only HDP, H^1, and Hybrid Golub-Kahan DP/GCV),
  and 5-seed hyperparameter sensitivity + CSV exports.
"""

from __future__ import annotations

import csv
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator

from ml_skcw_core import (
    eval_chebyshev_U,
    build_local_basis,
    fwt_analysis,
    RetardedSIVIEOperator,
    SeparableKernel,
    chebyshev_separable_kernel,
    build_sobolev_weights,
    build_two_sided_hdp,
    make_two_sided_operator,
    build_h1_preconditioner,
    make_h1_operator,
    solve_plsqr_dsc,
    solve_hybrid_gk,
    run_self_tests,
)

OUT_DIR = Path(__file__).resolve().parent

S_BASE = 0.20
TAU_SAFE = 1.10
EPS_S = 1e-4


def project_function_to_ml_coeffs(func, k: int, p: int, T_end: float, basis) -> np.ndarray:
    M_int = 1 << (k - 1)
    h = T_end / M_int
    xg, wg = np.polynomial.legendre.leggauss(2 * p)
    Phi_g = eval_chebyshev_U(xg, p) @ basis.R_gram
    centers = (np.arange(M_int) + 0.5) * h
    t_quad = centers[:, None] + 0.5 * h * xg[None, :]
    f_val = func(t_quad)
    f_fine = np.sqrt(0.5 * h) * ((f_val * wg[None, :]) @ Phi_g)
    return fwt_analysis(f_fine, k, basis)


def _eval_ref_cumulative_term(
    a_solver: np.ndarray,
    b_ref: np.ndarray,
    x_ref: np.ndarray,
    w_ref: np.ndarray,
    h_ref: float,
    v_idx: np.ndarray,
    theta: np.ndarray,
    stochastic: bool = False,
    dW_ref: Optional[np.ndarray] = None,
    dW_local: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate a(t) * int_0^t b(s) x(s) d(s or W) at solver collocation nodes via k_ref mesh."""
    M_ref = x_ref.shape[0]
    g_ref = b_ref * x_ref
    if not stochastic:
        S_ref = 0.5 * h_ref * (g_ref @ w_ref)
        F_left = np.empty(M_ref, dtype=float)
        F_left[0] = 0.0
        F_left[1:] = np.cumsum(S_ref[:-1])
        F_at_t = F_left[v_idx] + theta * S_ref[v_idx]
    else:
        assert dW_ref is not None and dW_local is not None
        avg_ref = 0.5 * (g_ref @ w_ref)
        S_ref_s = dW_ref * avg_ref
        F_left_s = np.empty(M_ref, dtype=float)
        F_left_s[0] = 0.0
        F_left_s[1:] = np.cumsum(S_ref_s[:-1])
        F_at_t = F_left_s[v_idx] + dW_local * avg_ref[v_idx]
    return a_solver * F_at_t


def build_reference_problem(
    k: int,
    p: int,
    tau: float,
    x_func: Callable[[np.ndarray], np.ndarray],
    phi_func: Callable[[np.ndarray], np.ndarray],
    dphi_func: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    amp2: float = 0.01,
    rel_noise: float = 5e-4,
    basis=None,
    k_ref: Optional[int] = None,
    nonsep_k0: bool = False,
    R0_solver: int = 12,
):
    """
    Generate non-inverse-crime synthetic observations on a fine reference grid k_ref >= k + 2:
      - Uses continuous X(t) and exact t - tau (with history phi on [-tau, 0])
      - Aggregates exact pathwise Brownian increments dW_solver = sum(dW_ref)
      - Solver subtracts history integral on its own grid k (producing real delta_Ito,h)
    """
    T_end = 1.0
    if basis is None:
        basis = build_local_basis(p)
    if k_ref is None:
        k_ref = max(k + 2, 14)

    M_int = 1 << (k - 1)
    M_ref = 1 << (k_ref - 1)
    m_sub = 1 << (k_ref - k)
    h = T_end / M_int
    h_ref = T_end / M_ref

    dW_ref = rng.normal(0.0, np.sqrt(h_ref), size=M_ref)
    dW_solver = dW_ref.reshape(M_int, m_sub).sum(axis=1)

    t_grid = RetardedSIVIEOperator.collocation_grid(k, p, T_end=T_end, basis=basis)
    t_ref = RetardedSIVIEOperator.collocation_grid(k_ref, p, T_end=T_end, basis=basis)

    # Fine-mesh node indices corresponding to solver nodes t_grid
    loc_idx = np.clip(np.floor(0.5 * (basis.x_nodes + 1.0) * m_sub).astype(int), 0, m_sub - 1)
    v_idx = (np.arange(M_int)[:, None] * m_sub) + loc_idx[None, :]
    theta = (t_grid - v_idx * h_ref) / h_ref
    z_bb = rng.standard_normal((M_int, p))
    dW_local = theta * dW_ref[v_idx] + np.sqrt(np.maximum(theta * (1.0 - theta), 0.0) * h_ref) * z_bb

    x_ref = x_func(t_ref)
    t_del = t_ref - tau
    x_del_ref = np.where(t_del < 0.0, phi_func(t_del), x_func(t_del))

    # Build K0 (either asymmetric exponential or non-separable Gaussian kernel)
    if not nonsep_k0:
        a0_sol = 4.0 * np.exp(3.0 * t_grid) * np.exp(-2.0 * (t_grid - 0.5))
        b0_sol = np.exp(2.0 * (t_grid - 0.5))
        K0_solver = SeparableKernel([a0_sol], [b0_sol])

        a0_ref = a0_sol
        b0_ref = np.exp(2.0 * (t_ref - 0.5))
        g_k0 = _eval_ref_cumulative_term(a0_ref, b0_ref, x_ref, basis.w_ref, h_ref, v_idx, theta)
        k0_bar = 4.0 * np.exp(3.0)
    else:
        k0_fn = lambda t, s: (2.0 + np.sin(2.0 * np.pi * t)) * np.exp(-2.0 * (t - s) ** 2)
        K0_solver = chebyshev_separable_kernel(k0_fn, t_grid, R=R0_solver, T_end=T_end)
        # Reference uses R_ref = max(R0_solver + 6, 18) for zero inverse crime on kernel rank
        R_ref = max(R0_solver + 6, 18)
        K0_ref_at_ref = chebyshev_separable_kernel(k0_fn, t_ref, R=R_ref, T_end=T_end)
        K0_ref_at_sol = chebyshev_separable_kernel(k0_fn, t_grid, R=R_ref, T_end=T_end)
        g_k0 = np.zeros((M_int, p), dtype=float)
        for a_s, b_r in zip(K0_ref_at_sol.a_factors, K0_ref_at_ref.b_factors):
            g_k0 += _eval_ref_cumulative_term(a_s, b_r, x_ref, basis.w_ref, h_ref, v_idx, theta)
        k0_bar = 3.0

    # K1 and K2 (asymmetric)
    a1_sol = 1.2 * np.exp(-1.5 * t_grid) * np.exp(-1.0 * (t_grid - 0.5))
    b1_sol = np.exp(1.0 * (t_grid - 0.5))
    K1_solver = SeparableKernel([a1_sol], [b1_sol])
    b1_ref = np.exp(1.0 * (t_ref - 0.5))
    g_k1 = _eval_ref_cumulative_term(a1_sol, b1_ref, x_del_ref, basis.w_ref, h_ref, v_idx, theta)

    a2_sol = amp2 * (1.0 + 0.5 * np.sin(2.0 * np.pi * t_grid))
    K2_solver = SeparableKernel([a2_sol], [np.ones_like(t_grid)])
    b2_ref = np.ones_like(t_ref)
    g_k2 = _eval_ref_cumulative_term(
        a2_sol, b2_ref, x_del_ref, basis.w_ref, h_ref, v_idx, theta,
        stochastic=True, dW_ref=dW_ref, dW_local=dW_local
    )

    op = RetardedSIVIEOperator(
        k, p, tau=tau, K0=K0_solver, K1=K1_solver, K2=K2_solver,
        dW=dW_solver, T_end=T_end, basis=basis
    )

    g_full_w = (g_k0 + g_k1 + g_k2).ravel() / np.sqrt(op.N)
    phi_at_grid = phi_func(t_grid - tau)
    g_hist_w_solver = op.eval_history_response(phi_at_grid)
    b_w_exact = g_full_w - g_hist_w_solver

    b_norm = float(np.linalg.norm(b_w_exact))
    noise_std = rel_noise * b_norm
    b_w_noisy = b_w_exact + (noise_std / np.sqrt(op.N)) * rng.standard_normal(op.N)

    c_true = project_function_to_ml_coeffs(x_func, k, p, T_end, basis)
    x_true = x_func(t_grid).ravel()

    # Discretization-and-noise floor (Definition 4.7)
    t_hist = np.linspace(-tau, 0.0, 200)
    phi_l2 = float(np.sqrt(np.mean(phi_func(t_hist) ** 2) * tau))
    dphi_l2 = float(np.sqrt(np.mean(dphi_func(t_hist) ** 2) * tau))
    k2_bar = 1.5 * amp2
    L_K_hist = 1.2 * np.exp(0.5) + np.pi * amp2
    delta_ito = (k2_bar * dphi_l2 + L_K_hist * phi_l2) * np.sqrt(tau) * h
    tau_h = op.d_shift * op.h
    rms_x_est = float(np.sqrt(np.mean(x_true ** 2)))
    delta_tau = k2_bar * np.sqrt(abs(tau - tau_h)) * rms_x_est
    delta_prior = float(np.sqrt(noise_std ** 2 + delta_ito ** 2 + delta_tau ** 2))

    return op, t_grid.ravel(), c_true, x_true, b_w_exact, b_w_noisy, delta_prior, noise_std


# ============================================================================
# Example 1: Ensemble Consistency (L = 100 paths, N = 1024)
# ============================================================================

def run_example_1():
    print("\n" + "=" * 78)
    print("EXAMPLE 1: Ensemble Consistency & Pathwise Stability (L=100, N=1024)")
    print("=" * 78)
    k, p, tau = 9, 4, 0.1
    basis = build_local_basis(p)
    x_func = lambda t: np.exp(t) * np.sin(2.0 * np.pi * t) + 0.5 * np.cos(np.pi * t)
    phi_func = lambda t: 0.5 * np.cos(np.pi * t)
    dphi_func = lambda t: -0.5 * np.pi * np.sin(np.pi * t)

    rng = np.random.default_rng(42)
    paths_rec, stops, rel_errs = [], [], []

    for _ in range(100):
        op, t_vec, _, x_true, _, b_w_noisy, delta_prior, _ = build_reference_problem(
            k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=1e-3, basis=basis, k_ref=13
        )
        hdp = build_two_sided_hdp(op, s=S_BASE)
        A_tilde = make_two_sided_operator(op, hdp)
        res = solve_plsqr_dsc(
            A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=100
        )
        x_rec = op.eval_basis_at_nodes(res.c_sol).ravel()
        paths_rec.append(x_rec)
        stops.append(res.m_stop)
        rel_errs.append(np.linalg.norm(x_rec - x_true) / np.linalg.norm(x_true))

    paths_arr = np.asarray(paths_rec)
    mean_path = np.mean(paths_arr, axis=0)
    ci_low = np.percentile(paths_arr, 2.5, axis=0)
    ci_high = np.percentile(paths_arr, 97.5, axis=0)

    print(f"Stopping index m*: mean={np.mean(stops):.2f}, std={np.std(stops):.2f}, range=[{min(stops)}, {max(stops)}]")
    print(f"Mean relative L2 error: {np.mean(rel_errs):.4e} (+/- {np.std(rel_errs):.4e})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.fill_between(t_vec, ci_low, ci_high, color="salmon", alpha=0.35, label="95% Ensemble Band")
    ax1.plot(t_vec, x_true, "k-", lw=1.6, label="Exact Signal")
    ax1.plot(t_vec, mean_path, "r--", lw=1.3, label="Mean PLSQR Reconstruction")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$X(t)$")
    ax1.set_title("(a) Ensemble Path Reconstruction ($N=1024$)")
    ax1.legend()
    ax1.grid(alpha=0.2)

    bins = np.arange(min(stops) - 0.5, max(stops) + 1.5, 1.0)
    ax2.hist(stops, bins=bins, edgecolor="black", alpha=0.8)
    ax2.set_xlabel(r"Stopping Index $m^*$")
    ax2.set_ylabel("Frequency (out of 100 paths)")
    ax2.set_title("(b) Empirical Distribution of DSC Index $m^*$")
    ax2.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Example1_Aligned_Consistency.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example 2: Full 3-Way Conditioning Report (k=4..8, p=4, Strong K2=0.25)
# ============================================================================

def run_example_2():
    print("\n" + "=" * 78)
    print("EXAMPLE 2: 3-Way Conditioning Analysis (Asymmetric Kernels, amp2=0.25)")
    print("=" * 78)
    p, tau = 4, 0.1
    basis = build_local_basis(p)
    rng = np.random.default_rng(7)
    x_func = lambda t: np.sin(2.0 * np.pi * t)
    z_func = lambda t: np.zeros_like(t)

    k_vals = [4, 5, 6, 7, 8]
    k_orig_list, k_L_list, k_tilde_list = [], [], []
    print(f"{'k':>3} | {'N':>5} | {'kappa(A_w)':>12} | {'kappa(D_L^-1 A_w)':>18} | {'Gain_L':>7} | {'kappa(A_tilde, s=0.60)':>22}")
    print("-" * 80)
    for k in k_vals:
        op, *_ = build_reference_problem(k, p, tau, x_func, z_func, z_func, rng, amp2=0.25, basis=basis, k_ref=12)
        hdp = build_two_sided_hdp(op, s=0.60)
        A_w = op.assemble_explicit_for_diagnostics()
        A_L = hdp.w_left[:, None] * A_w
        A_tilde = A_L * hdp.d_s_inv[None, :]
        k0 = np.linalg.cond(A_w)
        kL = np.linalg.cond(A_L)
        kt = np.linalg.cond(A_tilde)
        k_orig_list.append(k0)
        k_L_list.append(kL)
        k_tilde_list.append(kt)
        print(f"{k:>3} | {op.N:>5} | {k0:>12.2e} | {kL:>18.2e} | {k0/kL:>6.2f}x | {kt:>22.2e}")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.semilogy(k_vals, k_orig_list, "r-o", lw=2, label=r"Unpreconditioned $\kappa(\mathbf{A}_w)$")
    ax.semilogy(k_vals, k_L_list, "b-s", lw=2, label=r"Left-Equilibrated $\kappa(\mathbf{D}_L^{-1}\mathbf{A}_w)$")
    ax.semilogy(k_vals, k_tilde_list, "g--^", lw=1.8, label=r"Two-Sided $\kappa(\widetilde{\mathbf{A}})$ ($s=0.60$)")
    ax.set_xlabel(r"Dyadic Resolution Level $k$ ($N = 2^{k-1} p$)")
    ax.set_ylabel("Spectral Condition Number")
    ax.set_title(r"Three-Way Conditioning on Asymmetric Kernels ($K_2 = 0.25$, $p=4$)")
    ax.set_xticks(k_vals)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Example2_Conditioning_Stable.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example 3: Semi-Convergence V-Curve & DSC Verification (N = 1024)
# ============================================================================

def run_example_3():
    print("\n" + "=" * 78)
    print("EXAMPLE 3: Semi-Convergence & DSC Identification (N = 1024)")
    print("=" * 78)
    k, p, tau = 9, 4, 0.1
    rng = np.random.default_rng(101)
    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.3 * np.cos(6.0 * np.pi * t)
    phi_func = lambda t: 0.3 * np.cos(6.0 * np.pi * t)
    dphi_func = lambda t: -1.8 * np.pi * np.sin(6.0 * np.pi * t)

    op, _, _, x_true, _, b_w_noisy, delta_prior, _ = build_reference_problem(
        k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=1e-3, k_ref=14
    )
    hdp = build_two_sided_hdp(op, s=S_BASE)
    A_tilde = make_two_sided_operator(op, hdp)

    m_max = 100
    res_full = solve_plsqr_dsc(
        A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior, use_dsc=False, m_max=m_max, record_history=True
    )
    res_dsc = solve_plsqr_dsc(
        A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
        tau_safe=TAU_SAFE, eps_s=EPS_S, use_dsc=True, m_max=m_max
    )

    rel_errors = [
        np.linalg.norm(op.eval_basis_at_nodes(cm).ravel() - x_true) / np.linalg.norm(x_true)
        for cm in res_full.c_history
    ]
    m_oracle = int(np.argmin(rel_errors)) + 1
    print(f"Estimated floor delta_prior: {delta_prior:.4e}")
    print(f"DSC stopping index m*:       {res_dsc.m_stop} (rel_err = {rel_errors[res_dsc.m_stop - 1]:.4e})")
    print(f"Oracle optimal m_opt:        {m_oracle} (rel_err = {rel_errors[m_oracle - 1]:.4e})")

    iters = np.arange(1, m_max + 1)
    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    ax1.semilogy(iters, rel_errors, "r-", lw=2, label="Relative Reconstruction Error")
    ax1.scatter(res_dsc.m_stop, rel_errors[res_dsc.m_stop - 1], c="black", s=70, zorder=5,
                label=f"DSC Stop ($m^*={res_dsc.m_stop}$)")
    ax1.set_xlabel("Krylov Iteration Index $m$")
    ax1.set_ylabel("Relative $L^2$ Error", color="r")
    ax1.tick_params(axis="y", labelcolor="r")

    ax2 = ax1.twinx()
    ax2.semilogy(iters, res_full.omega_history[1:], "b--", lw=1.6, label=r"Normalized Residual $\omega_m$")
    ax2.axhline(TAU_SAFE * delta_prior, color="b", ls=":", alpha=0.7, label=r"$\tau_{\mathrm{safe}}\delta_{\mathrm{prior}}$")
    ax2.set_ylabel(r"Normalized Residual $\omega_m$", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right")
    ax1.set_title("Semi-Convergence Trajectory and DSC Identification ($N=1024$)")
    ax1.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Example3_SemiConvergence.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example 4: Large-Scale Scalability up to N = 262,144 (Clean Timing!)
# ============================================================================

def run_example_4():
    print("\n" + "=" * 78)
    print("EXAMPLE 4: Large-Scale Matrix-Free Scalability (N up to 262,144)")
    print("=" * 78)
    p, tau = 4, 0.1
    basis = build_local_basis(p)
    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.35 * np.sin(20.0 * np.pi * t) + 0.15 * np.sin(80.0 * np.pi * t)
    phi_func = lambda t: np.sin(2.0 * np.pi * t)
    dphi_func = lambda t: 2.0 * np.pi * np.cos(2.0 * np.pi * t)

    k_list = [12, 14, 15, 16, 17]
    print(f"{'k':>3} | {'N':>8} | {'m*':>4} | {'Err PLSQR':>11} | {'Err RawDSC':>11} | {'Err Raw300':>11} | {'Time(s)':>8} | {'MB':>6}")
    print("-" * 82)

    for k in k_list:
        rng = np.random.default_rng(42)
        op, _, _, x_true, _, b_w_noisy, delta_prior, noise_std = build_reference_problem(
            k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=5e-4, basis=basis, k_ref=19
        )

        # Pass 1: Pure wall-clock timing WITHOUT tracemalloc overhead
        t0 = time.perf_counter()
        hdp = build_two_sided_hdp(op, s=S_BASE)
        A_tilde = make_two_sided_operator(op, hdp)
        res = solve_plsqr_dsc(
            A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
        )
        elapsed = time.perf_counter() - t0

        # Pass 2: Peak RAM measurement
        tracemalloc.start()
        hdp_m = build_two_sided_hdp(op, s=S_BASE)
        A_tilde_m = make_two_sided_operator(op, hdp_m)
        _ = solve_plsqr_dsc(
            A_tilde_m, hdp_m.w_left * b_w_noisy, hdp_m.d_s_inv, delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        err_plsqr = np.linalg.norm(op.eval_basis_at_nodes(res.c_sol).ravel() - x_true) / np.linalg.norm(x_true)

        res_raw_dsc = solve_plsqr_dsc(
            op, b_w_noisy, np.ones(op.N), noise_std, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
        )
        err_raw_dsc = np.linalg.norm(op.eval_basis_at_nodes(res_raw_dsc.c_sol).ravel() - x_true) / np.linalg.norm(x_true)

        res_raw300 = solve_plsqr_dsc(
            op, b_w_noisy, np.ones(op.N), noise_std, use_dsc=False, m_max=300
        )
        err_raw300 = np.linalg.norm(op.eval_basis_at_nodes(res_raw300.c_sol).ravel() - x_true) / np.linalg.norm(x_true)

        print(
            f"{k:>3} | {op.N:>8} | {res.m_stop:>4} | {err_plsqr:>11.4e} | {err_raw_dsc:>11.4e} | "
            f"{err_raw300:>11.4e} | {elapsed:>8.4f} | {peak_bytes / (1024**2):>6.2f}"
        )


# ============================================================================
# Example 5: High-Frequency Spectral Retention (N = 131,072)
# ============================================================================

def run_example_5():
    print("\n" + "=" * 78)
    print("EXAMPLE 5: Multi-Frequency Spectral Retention (N = 131,072)")
    print("=" * 78)
    k, p, tau = 16, 4, 0.1
    rng = np.random.default_rng(42)

    x_func = lambda t: (
        1.0 * np.sin(2.0 * np.pi * 2.0 * t)
        + 0.45 * np.sin(2.0 * np.pi * 25.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 80.0 * t)
    )
    phi_func = lambda t: 1.0 * np.sin(2.0 * np.pi * 2.0 * t)
    dphi_func = lambda t: 4.0 * np.pi * np.cos(2.0 * np.pi * 2.0 * t)

    op, t_vec, _, x_true, _, b_w_noisy, delta_prior, noise_std = build_reference_problem(
        k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=2e-4, k_ref=18
    )
    hdp = build_two_sided_hdp(op, s=S_BASE)
    A_tilde = make_two_sided_operator(op, hdp)

    res = solve_plsqr_dsc(
        A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
        tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200
    )
    x_rec = op.eval_basis_at_nodes(res.c_sol).ravel()

    res_raw_dsc = solve_plsqr_dsc(
        op, b_w_noisy, np.ones(op.N), noise_std, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200
    )
    x_raw_dsc = op.eval_basis_at_nodes(res_raw_dsc.c_sol).ravel()

    res_raw300 = solve_plsqr_dsc(
        op, b_w_noisy, np.ones(op.N), noise_std, use_dsc=False, m_max=300
    )
    x_raw300 = op.eval_basis_at_nodes(res_raw300.c_sol).ravel()

    rms_plsqr = float(np.sqrt(np.mean((x_rec - x_true) ** 2)))
    rms_raw_dsc = float(np.sqrt(np.mean((x_raw_dsc - x_true) ** 2)))
    rms_raw300 = float(np.sqrt(np.mean((x_raw300 - x_true) ** 2)))
    print(f"PLSQR (Two-Sided HDP+DSC) m*={res.m_stop}: RMS = {rms_plsqr:.4e}")
    print(f"Raw LSQR (DSC)            m*={res_raw_dsc.m_stop}: RMS = {rms_raw_dsc:.4e}")
    print(f"Raw LSQR(300):                      RMS = {rms_raw300:.4e} (Gain vs Raw300: {rms_raw300 / rms_plsqr:.2f}x)")

    t_uni = np.linspace(0.0, 1.0, op.N, endpoint=False)
    x_true_uni = x_func(t_uni)
    x_rec_uni = np.interp(t_uni, t_vec, x_rec)

    freqs = np.fft.rfftfreq(op.N, d=1.0 / op.N)
    P_true = (np.abs(np.fft.rfft(x_true_uni)) / op.N) ** 2
    P_rec = (np.abs(np.fft.rfft(x_rec_uni)) / op.N) ** 2
    total_true = float(np.sum(P_true))

    bands = [("Low (0-10 Hz)", 0.0, 10.0), ("Mid (10-50 Hz)", 10.0, 50.0), ("High (50-150 Hz)", 50.0, 150.0)]
    print(f"\n{'Frequency Band':<18} | {'Nominal (%)':>11} | {'Recovered (%)':>13} | {'Retention (%)':>13}")
    print("-" * 64)
    for name, f1, f2 in bands:
        mask = (freqs >= f1) & (freqs < f2)
        e_true = float(np.sum(P_true[mask]))
        e_rec = float(np.sum(P_rec[mask]))
        nom_pct = 100.0 * e_true / total_true
        rec_pct = 100.0 * e_rec / total_true
        ret_pct = 100.0 * e_rec / max(e_true, 1e-30)
        print(f"{name:<18} | {nom_pct:>10.2f}% | {rec_pct:>12.2f}% | {ret_pct:>12.2f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.2))
    zoom = (t_vec >= 0.4) & (t_vec <= 0.6)
    ax1.plot(t_vec[zoom], x_true[zoom], "k--", lw=1.6, label="True Multi-Frequency Signal")
    ax1.plot(t_vec[zoom], x_rec[zoom], "r-", lw=1.1, label=f"PLSQR--ML-SKCW ($m^*={res.m_stop}$)")
    ax1.set_title(r"(a) Trajectory Reconstruction Zoom ($t \in [0.4, 0.6]$, $N=131,072$)")
    ax1.set_ylabel(r"$X(t)$")
    ax1.legend()
    ax1.grid(alpha=0.2)

    ax2.plot(t_vec, x_raw300 - x_true, color="silver", lw=0.5, label=f"Unpreconditioned LSQR(300) (RMS={rms_raw300:.3f})")
    ax2.plot(t_vec, x_rec - x_true, color="red", lw=0.9, label=f"PLSQR--ML-SKCW (RMS={rms_plsqr:.4f})")
    ax2.set_title("(b) Pointwise Reconstruction Error Field")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$e(t) = X_{\mathrm{num}}(t) - X_{\mathrm{true}}(t)$")
    ax2.legend()
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Example5_Error_Field.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example 6: Strong Ito Threshold (s > 1/2) on Jump + 220 Hz Localized Burst
#            + Left-Only + Tridiagonal H^1 (s=1) Comparison
# ============================================================================

def multiscale_nonsmooth_target(t: np.ndarray) -> np.ndarray:
    """
    Multi-scale non-smooth signal combining:
      - Smooth background wave
      - Step discontinuity on [0.25, 0.65]
      - Sharp localized Gaussian spike at t = 0.80 (sigma = 0.003, under-resolved on coarse N <= 512)
    """
    bg = 0.8 * np.sin(2.0 * np.pi * t)
    jump = 0.6 * ((t >= 0.25) & (t <= 0.65)).astype(float)
    pulse = 3.0 * np.exp(-0.5 * ((t - 0.80) / 0.003) ** 2)
    return bg + jump + pulse

def run_example_6():
    print("\n" + "=" * 78)
    print("EXAMPLE 6: Strong Ito (amp2=0.25) on Jump + Sharp Spike (k=8..16)")
    print("=" * 78)
    p, tau, amp2 = 4, 0.1, 0.25
    basis = build_local_basis(p)
    z_func = lambda t: np.zeros_like(t)

    k_list = [7, 8, 10, 12, 14, 16]
    s_test = [0.00, 0.20, 0.55]
    results = {s: [] for s in s_test}
    left_only_errs, h1_errs = [], []
    t_plot, x_true_plot, x_rec20_plot, x_rec65_plot, x_h1_plot = None, None, None, None, None

    print(
        f"{'k':>2} | {'N':>7} | {'s=0.00 (Raw)':>15} | {'Left-Only (s=0)':>15} | "
        f"{'s=0.20 (<1/2)':>15} | {'H^1 (s=1)':>15} | {'s=0.55 (>1/2)':>15}"
    )
    print("-" * 108)

    for k in k_list:
        rng = np.random.default_rng(42)
        op, t_vec, _, x_true, _, b_w_noisy, delta_prior, noise_std = build_reference_problem(
            k, p, tau, multiscale_nonsmooth_target, z_func, z_func, rng,
            amp2=amp2, rel_noise=5e-4, basis=basis, k_ref=18
        )
        norm_x = float(np.linalg.norm(x_true))

        # 1. Raw (s=0, no D_L)
        r_raw = solve_plsqr_dsc(
            op, b_w_noisy, np.ones(op.N), delta_prior, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200
        )
        e_raw = float(np.linalg.norm(op.eval_basis_at_nodes(r_raw.c_sol).ravel() - x_true)) / norm_x
        results[0.00].append(e_raw)

        # 2. Left-Only (D_L^{-1}, s=0)
        h_left = build_two_sided_hdp(op, s=0.0)
        A_left = make_two_sided_operator(op, h_left)
        r_left = solve_plsqr_dsc(
            A_left, h_left.w_left * b_w_noisy, h_left.d_s_inv, delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200
        )
        e_left = float(np.linalg.norm(op.eval_basis_at_nodes(r_left.c_sol).ravel() - x_true)) / norm_x
        left_only_errs.append(e_left)

        # 3. Two-Sided s=0.20 (< 1/2)
        h_20 = build_two_sided_hdp(op, s=0.20)
        A_20 = make_two_sided_operator(op, h_20)
        r_20 = solve_plsqr_dsc(
            A_20, h_20.w_left * b_w_noisy, h_20.d_s_inv, delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200
        )
        x_20 = op.eval_basis_at_nodes(r_20.c_sol).ravel()
        e_20 = float(np.linalg.norm(x_20 - x_true)) / norm_x
        results[0.20].append(e_20)

        # 4. Continuous Tridiagonal H^1 (s=1) competitor
        h1_prec = build_h1_preconditioner(op, t_vec.reshape(op.M_int, p), h_left.w_left, alpha=0.1)
        A_h1 = make_h1_operator(h1_prec)
        r_h1 = solve_plsqr_dsc(
            A_h1, h_left.w_left * b_w_noisy, np.ones(op.N), delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200, y_to_c_fn=h1_prec.y_to_c
        )
        x_h1 = op.eval_basis_at_nodes(r_h1.c_sol).ravel()
        e_h1 = float(np.linalg.norm(x_h1 - x_true)) / norm_x
        h1_errs.append(e_h1)

        # 5. Two-Sided s=0.65 (> 1/2)
        h_65 = build_two_sided_hdp(op, s=0.55)
        A_65 = make_two_sided_operator(op, h_65)
        r_65 = solve_plsqr_dsc(
            A_65, h_65.w_left * b_w_noisy, h_65.d_s_inv, delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200
        )
        x_65 = op.eval_basis_at_nodes(r_65.c_sol).ravel()
        e_65 = float(np.linalg.norm(x_65 - x_true)) / norm_x
        results[0.55].append(e_65)

        print(
            f"{k:>2} | {op.N:>7} | {e_raw:>9.3e}({r_raw.m_stop:3d}) | "
            f"{e_left:>9.3e}({r_left.m_stop:3d}) | {e_20:>9.3e}({r_20.m_stop:3d}) | "
            f"{e_h1:>9.3e}({r_h1.m_stop:3d}) | {e_65:>9.3e}({r_65.m_stop:3d})"
        )

        if k == 16:
            t_plot, x_true_plot = t_vec, x_true
            x_rec20_plot, x_rec65_plot, x_h1_plot = x_20, x_65, x_h1

    N_vals = [(1 << (k - 1)) * p for k in k_list]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax1.loglog(N_vals, results[0.00], "k--^", lw=1.8, label=r"$s = 0.00$ (Raw+DSC)")
    ax1.loglog(N_vals, left_only_errs, "m:d", lw=1.8, label=r"Left-Only HDP ($s = 0$)")
    ax1.loglog(N_vals, results[0.20], "r-o", lw=1.8, label=r"$s = 0.20$ ($s < 1/2$, blow-up)")
    ax1.loglog(N_vals, h1_errs, "g-.v", lw=1.8, label=r"Tridiagonal $H^1$ ($s=1$)")
    ax1.loglog(N_vals, results[0.55], "b-s", lw=2.3, label=r"Two-Sided HDP ($s = 0.55 > 1/2$)")
    ax1.set_xlabel("System Dimension $N$")
    ax1.set_ylabel(r"Relative $L^2$ Error")
    ax1.set_title(r"(a) Multi-Scale Error vs. $N$ ($K_2 = 0.25$)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.25)

    zoom = (t_plot >= 0.22) & (t_plot <= 0.83)
    ax2.plot(t_plot[zoom], x_true_plot[zoom], "k-", lw=1.6, label="Exact Jump + Spike")
    ax2.plot(t_plot[zoom], x_h1_plot[zoom], "g-.", lw=1.2, label=r"Tridiagonal $H^1$ ($s=1$)")
    ax2.plot(t_plot[zoom], x_rec20_plot[zoom], "r-", alpha=0.45, lw=0.7, label=r"$s = 0.20$ (stochastic noise)")
    ax2.plot(t_plot[zoom], x_rec65_plot[zoom], "b--", lw=1.4, label=r"$s = 0.55$ (stabilized)")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$X(t)$")
    ax2.set_title(r"(b) Jump + Spike at $N=131,072$ ($t \in [0.22, 0.83]$)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Example6_Stochastic_Threshold.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example 7: Non-Separable Smooth Gaussian Kernel (R_0 = 16, R_tot = 18)
# ============================================================================

def run_example_7_nonseparable():
    print("\n" + "=" * 78)
    print("EXAMPLE 7: Non-Separable Gaussian Kernel K0 (R0=16, R_tot=18)")
    print("=" * 78)
    p, tau = 4, 0.1
    basis = build_local_basis(p)
    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.4 * np.cos(14.0 * np.pi * t)
    phi_func = lambda t: 0.4 * np.cos(14.0 * np.pi * t)
    dphi_func = lambda t: -5.6 * np.pi * np.sin(14.0 * np.pi * t)

    print(f"{'k':>3} | {'N':>7} | {'m*':>4} | {'Err PLSQR':>11} | {'Err Raw+DSC':>11} | {'Time(s)':>8}")
    print("-" * 62)
    for k in [10, 12, 14, 16]:
        rng = np.random.default_rng(77)
        op, _, _, x_true, _, b_w_noisy, delta_prior, _ = build_reference_problem(
            k, p, tau, x_func, phi_func, dphi_func, rng,
            amp2=0.10, rel_noise=5e-4, basis=basis, k_ref=max(k + 2, 16),
            nonsep_k0=True, R0_solver=12
        )
        nx = float(np.linalg.norm(x_true))
        t0 = time.perf_counter()
        hdp = build_two_sided_hdp(op, s=0.60)
        A_tilde = make_two_sided_operator(op, hdp)
        res = solve_plsqr_dsc(
            A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
        )
        elapsed = time.perf_counter() - t0
        err_p = float(np.linalg.norm(op.eval_basis_at_nodes(res.c_sol).ravel() - x_true)) / nx

        res_raw = solve_plsqr_dsc(
            op, b_w_noisy, np.ones(op.N), delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
        )
        err_r = float(np.linalg.norm(op.eval_basis_at_nodes(res_raw.c_sol).ravel() - x_true)) / nx
        print(f"{k:>3} | {op.N:>7} | {res.m_stop:>4} | {err_p:>11.4e} | {err_r:>11.4e} | {elapsed:>8.4f}")


# ============================================================================
# Section 5 & 7.8: Genuine Tripartite Decomposition, Baselines & Sensitivity
# ============================================================================

def run_tripartite_baselines_sensitivity():
    print("\n" + "=" * 78)
    print("SECTION 5 & 7.8: Genuine Tripartite Decomposition, Baselines & 5-Seed Sensitivity")
    print("=" * 78)
    k, p, tau = 14, 4, 0.1
    basis = build_local_basis(p)
    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.35 * np.sin(20.0 * np.pi * t) + 0.15 * np.sin(80.0 * np.pi * t)
    phi_func = lambda t: np.sin(2.0 * np.pi * t)
    dphi_func = lambda t: 2.0 * np.pi * np.cos(2.0 * np.pi * t)

    rng = np.random.default_rng(42)
    op, t_vec, c_proj, x_true, b_w_exact, b_w_noisy, delta_prior, noise_std = build_reference_problem(
        k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=5e-4, basis=basis, k_ref=18
    )
    norm_x = float(np.linalg.norm(x_true))
    hdp = build_two_sided_hdp(op, s=S_BASE)
    A_tilde = make_two_sided_operator(op, hdp)

    # --- Tripartite Error Decomposition around Pi_N X (Theorem 5.3) ---
    x_proj = op.eval_basis_at_nodes(c_proj).ravel()
    b_proj_L = hdp.w_left * op.matvec(c_proj)
    res_dsc = solve_plsqr_dsc(
        A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
        tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
    )
    m_star = res_dsc.m_stop
    x_reg_noisy = op.eval_basis_at_nodes(res_dsc.c_sol).ravel()

    res_clean = solve_plsqr_dsc(
        A_tilde, b_proj_L, hdp.d_s_inv, delta_prior, use_dsc=False, m_max=m_star
    )
    x_reg_clean = op.eval_basis_at_nodes(res_clean.c_sol).ravel()

    E_proj = float(np.linalg.norm(x_true - x_proj)) / norm_x
    E_R = float(np.linalg.norm(x_proj - x_reg_clean)) / norm_x
    E_S = float(np.linalg.norm(x_reg_clean - x_reg_noisy)) / norm_x
    E_total = float(np.linalg.norm(x_true - x_reg_noisy)) / norm_x
    disc_norm = float(np.linalg.norm(hdp.w_left * b_w_exact - b_proj_L))
    print(f"Discrepancy ||delta_disc||_2={disc_norm:.4e}, noise_std={noise_std:.4e}, delta_prior={delta_prior:.4e}")
    print(f"Tripartite (m*={m_star}): E_proj={E_proj:.4e}, E_R={E_R:.4e}, E_S={E_S:.4e}, Bound={E_proj+E_R+E_S:.4e}, Total={E_total:.4e}")

    # --- Panel A Baselines (amp2=0.01, N=32,768) ---
    rows_base = []

    def flag_str(istop: int) -> str:
        return {1: "tol(Ax=b)", 2: "tol(A^Tr)", 3: "conlim", 7: "max_iter"}.get(istop, f"flag({istop})")

    t0 = time.perf_counter()
    res_prop = solve_plsqr_dsc(
        A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
        tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
    )
    t_prop = time.perf_counter() - t0
    rows_base.append(["Panel A: PLSQR--ML-SKCW (s=0.20, DSC)", res_prop.m_stop, "DSC", f"{E_total:.6e}", f"{t_prop:.4f}"])

    t0 = time.perf_counter()
    res_cgls = solve_plsqr_dsc(op, b_w_noisy, np.ones(op.N), delta_prior, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150)
    t_cgls = time.perf_counter() - t0
    e_cgls = float(np.linalg.norm(op.eval_basis_at_nodes(res_cgls.c_sol).ravel() - x_true)) / norm_x
    rows_base.append(["Panel A: Unprecond. LSQR/CGLS + DSC", res_cgls.m_stop, "DSC", f"{e_cgls:.6e}", f"{t_cgls:.4f}"])

    h_left_m = build_two_sided_hdp(op, s=0.0)
    A_left_m = make_two_sided_operator(op, h_left_m)
    t0 = time.perf_counter()
    res_ld = solve_plsqr_dsc(
        A_left_m, h_left_m.w_left * b_w_noisy, h_left_m.d_s_inv, delta_prior,
        tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
    )
    t_ld = time.perf_counter() - t0
    e_ld = float(np.linalg.norm(op.eval_basis_at_nodes(res_ld.c_sol).ravel() - x_true)) / norm_x
    rows_base.append(["Panel A: Left-Only HDP + DSC (s=0)", res_ld.m_stop, "DSC", f"{e_ld:.6e}", f"{t_ld:.4f}"])

    t0 = time.perf_counter()
    c_lt, istop, itn, *_ = spla.lsqr(A_left_m, h_left_m.w_left * b_w_noisy, damp=1e-2, iter_lim=300, atol=1e-10, btol=1e-10)
    t_lt = time.perf_counter() - t0
    e_lt = float(np.linalg.norm(op.eval_basis_at_nodes(c_lt).ravel() - x_true)) / norm_x
    rows_base.append(["Panel A: Left-Only HDP + Tikhonov (lam=1e-2)", itn, flag_str(istop), f"{e_lt:.6e}", f"{t_lt:.4f}"])

    for mode_gk in ["dp", "gcv"]:
        t0 = time.perf_counter()
        r_gk = solve_hybrid_gk(A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, mode=mode_gk, delta_prior=delta_prior, tau_safe=TAU_SAFE, m_max=60)
        t_gk = time.perf_counter() - t0
        e_gk = float(np.linalg.norm(op.eval_basis_at_nodes(r_gk.c_sol).ravel() - x_true)) / norm_x
        rows_base.append([f"Panel A: HDP + Hybrid GK ({mode_gk.upper()})", r_gk.m_stop, mode_gk.upper(), f"{e_gk:.6e}", f"{t_gk:.4f}"])

    for lam, budget in [(1e-2, 300), (1e-2, m_star)]:
        t0 = time.perf_counter()
        c_tik, istop, itn, *_ = spla.lsqr(op, b_w_noisy, damp=lam, iter_lim=budget, atol=1e-10, btol=1e-10)
        el = time.perf_counter() - t0
        err = float(np.linalg.norm(op.eval_basis_at_nodes(c_tik).ravel() - x_true)) / norm_x
        rows_base.append([f"Panel A: Tikhonov-LSQR (lam={lam:.0e}, lim={budget})", itn, flag_str(istop), f"{err:.6e}", f"{el:.4f}"])

    for lam, budget in [(5e-3, 300), (5e-3, m_star), (1e-2, 300)]:
        t0 = time.perf_counter()
        y_tik, istop, itn, *_ = spla.lsqr(A_tilde, hdp.w_left * b_w_noisy, damp=lam, iter_lim=budget, atol=1e-10, btol=1e-10)
        el = time.perf_counter() - t0
        err = float(np.linalg.norm(op.eval_basis_at_nodes(hdp.d_s_inv * y_tik).ravel() - x_true)) / norm_x
        rows_base.append([f"Panel A: HDP-Tikhonov-LSQR (lam={lam:.0e}, lim={budget})", itn, flag_str(istop), f"{err:.6e}", f"{el:.4f}"])

    # --- Panel B Baselines (amp2=0.25, s=0.65, N=32,768, jump+220Hz burst) ---
    rng_st = np.random.default_rng(42)
    z_f = lambda t: np.zeros_like(t)
    op_st, tv_st, _, x_st, _, bn_st, dp_st, _ = build_reference_problem(
        k, p, tau, multiscale_nonsmooth_target, z_f, z_f, rng_st, amp2=0.25, rel_noise=5e-4, basis=basis, k_ref=18
    )
    nx_st = float(np.linalg.norm(x_st))
    hdp_st = build_two_sided_hdp(op_st, s=0.55)
    A_st = make_two_sided_operator(op_st, hdp_st)

    t0 = time.perf_counter()
    res_st = solve_plsqr_dsc(
        A_st, hdp_st.w_left * bn_st, hdp_st.d_s_inv, dp_st, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200
    )
    t_st = time.perf_counter() - t0
    e_st = float(np.linalg.norm(op_st.eval_basis_at_nodes(res_st.c_sol).ravel() - x_st)) / nx_st
    rows_base.append(["Panel B: PLSQR--ML-SKCW (s=0.55, DSC)", res_st.m_stop, "DSC", f"{e_st:.6e}", f"{t_st:.4f}"])

    h_left_st = build_two_sided_hdp(op_st, s=0.0)
    A_left_st = make_two_sided_operator(op_st, h_left_st)
    t0 = time.perf_counter()
    r_ls_dsc = solve_plsqr_dsc(A_left_st, h_left_st.w_left * bn_st, h_left_st.d_s_inv, dp_st, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200)
    t_ls_dsc = time.perf_counter() - t0
    e_ls_dsc = float(np.linalg.norm(op_st.eval_basis_at_nodes(r_ls_dsc.c_sol).ravel() - x_st)) / nx_st
    rows_base.append(["Panel B: Left-Only HDP + DSC (s=0)", r_ls_dsc.m_stop, "DSC", f"{e_ls_dsc:.6e}", f"{t_ls_dsc:.4f}"])

    t0 = time.perf_counter()
    c_ls_tik, istop, itn, *_ = spla.lsqr(A_left_st, h_left_st.w_left * bn_st, damp=5e-3, iter_lim=300, atol=1e-10, btol=1e-10)
    t_ls_tik = time.perf_counter() - t0
    e_ls_tik = float(np.linalg.norm(op_st.eval_basis_at_nodes(c_ls_tik).ravel() - x_st)) / nx_st
    rows_base.append(["Panel B: Left-Only HDP + Tikhonov (lam=5e-3)", itn, flag_str(istop), f"{e_ls_tik:.6e}", f"{t_ls_tik:.4f}"])

    h1_st = build_h1_preconditioner(op_st, tv_st.reshape(op_st.M_int, p), h_left_st.w_left, alpha=0.10)
    A_h1_st = make_h1_operator(h1_st)
    t0 = time.perf_counter()
    r_h1_st = solve_plsqr_dsc(
        A_h1_st, h_left_st.w_left * bn_st, np.ones(op_st.N), dp_st,
        tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200, y_to_c_fn=h1_st.y_to_c
    )
    t_h1_st = time.perf_counter() - t0
    e_h1_st = float(np.linalg.norm(op_st.eval_basis_at_nodes(r_h1_st.c_sol).ravel() - x_st)) / nx_st
    rows_base.append(["Panel B: Tridiagonal H^1 (s=1, alpha=0.10) + DSC", r_h1_st.m_stop, "DSC", f"{e_h1_st:.6e}", f"{t_h1_st:.4f}"])

    for mode_gk in ["dp", "gcv"]:
        t0 = time.perf_counter()
        r_gk_st = solve_hybrid_gk(A_st, hdp_st.w_left * bn_st, hdp_st.d_s_inv, mode=mode_gk, delta_prior=dp_st, tau_safe=TAU_SAFE, m_max=80)
        t_gk_st = time.perf_counter() - t0
        e_gk_st = float(np.linalg.norm(op_st.eval_basis_at_nodes(r_gk_st.c_sol).ravel() - x_st)) / nx_st
        rows_base.append([f"Panel B: HDP + Hybrid GK ({mode_gk.upper()})", r_gk_st.m_stop, mode_gk.upper(), f"{e_gk_st:.6e}", f"{t_gk_st:.4f}"])

    # Unpreconditioned Hybrid GK (to prove HDP s>1/2 is also essential for HGK!)
    t0 = time.perf_counter()
    r_gk_raw = solve_hybrid_gk(op_st, bn_st, np.ones(op_st.N), mode="gcv", m_max=80)
    t_gk_raw = time.perf_counter() - t0
    e_gk_raw = float(np.linalg.norm(op_st.eval_basis_at_nodes(r_gk_raw.c_sol).ravel() - x_st)) / nx_st
    rows_base.append(["Panel B: Unprecond. Hybrid GK (GCV)", r_gk_raw.m_stop, "GCV", f"{e_gk_raw:.6e}", f"{t_gk_raw:.4f}"])

    for lam, budget in [(1e-2, 300)]:
        t0 = time.perf_counter()
        c_t, istop, itn, *_ = spla.lsqr(op_st, bn_st, damp=lam, iter_lim=budget, atol=1e-10, btol=1e-10)
        el = time.perf_counter() - t0
        err = float(np.linalg.norm(op_st.eval_basis_at_nodes(c_t).ravel() - x_st)) / nx_st
        rows_base.append([f"Panel B: Tikhonov-LSQR (lam={lam:.0e}, lim={budget})", itn, flag_str(istop), f"{err:.6e}", f"{el:.4f}"])

    for lam, budget in [(5e-3, 300), (5e-3, res_st.m_stop)]:
        t0 = time.perf_counter()
        y_t, istop, itn, *_ = spla.lsqr(A_st, hdp_st.w_left * bn_st, damp=lam, iter_lim=budget, atol=1e-10, btol=1e-10)
        el = time.perf_counter() - t0
        err = float(np.linalg.norm(op_st.eval_basis_at_nodes(hdp_st.d_s_inv * y_t).ravel() - x_st)) / nx_st
        rows_base.append([f"Panel B: HDP-Tikhonov-LSQR (lam={lam:.0e}, lim={budget})", itn, flag_str(istop), f"{err:.6e}", f"{el:.4f}"])

    print("\n--- Full Baseline Comparison (Clean Timings, No Inverse Crime) ---")
    for r in rows_base:
        print(f"{r[0]:<50} | itn={r[1]:>3} | {r[2]:<9} | Err={float(r[3]):.4e} | {float(r[4]):.4f}s")

    with open(OUT_DIR / "baseline_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Iterations", "StopReason", "RelError", "Time_s"])
        w.writerows(rows_base)

    # --- 5-Seed Sensitivity ---
    seeds = [42, 101, 202, 303, 404]
    instances = []
    for sd in seeds:
        rng_s = np.random.default_rng(sd)
        op_s, _, _, xt_s, _, bn_s, dp_s, _ = build_reference_problem(
            k, p, tau, x_func, phi_func, dphi_func, rng_s, amp2=0.01, rel_noise=5e-4, basis=basis, k_ref=18
        )
        hdp_base = build_two_sided_hdp(op_s, s=S_BASE)
        instances.append((op_s, xt_s, float(np.linalg.norm(xt_s)), bn_s, hdp_base.w_left, dp_s))

    rows_sens = []
    print("\n--- 5-Seed Hyperparameter Sensitivity ---")

    def run_sens(grp: str, lbl: str, s_val: float, ts_val: float, es_val: float):
        d_inv = build_sobolev_weights(k, p, s=s_val)
        ms, es = [], []
        for op_s, xt_s, nx_s, bn_s, wl_s, dp_s in instances:
            A_t = LinearOperator(
                dtype=float, shape=op_s.shape,
                matvec=lambda y, o=op_s, w=wl_s, d=d_inv: w * o.matvec(d * y),
                rmatvec=lambda u, o=op_s, w=wl_s, d=d_inv: d * o.rmatvec(w * u),
            )
            res_s = solve_plsqr_dsc(A_t, wl_s * bn_s, d_inv, dp_s, tau_safe=ts_val, eps_s=es_val, m_max=150)
            ms.append(res_s.m_stop)
            es.append(float(np.linalg.norm(op_s.eval_basis_at_nodes(res_s.c_sol).ravel() - xt_s)) / nx_s)
        mm, msd = float(np.mean(ms)), float(np.std(ms))
        em, esd = float(np.mean(es)), float(np.std(es))
        print(f"{grp:<8} {lbl:<8} | m*={mm:5.2f}+/-{msd:4.2f} | Err={em:.4e}+/-{esd:.2e}")
        rows_sens.append([grp, lbl, f"{mm:.2f}", f"{msd:.2f}", f"{em:.6e}", f"{esd:.6e}"])

    for sv in [0.00, 0.10, 0.15, 0.20, 0.25, 0.35]:
        run_sens("s", f"{sv:.2f}", sv, TAU_SAFE, EPS_S)
    for tv in [1.02, 1.05, 1.10, 1.15, 1.25, 1.40]:
        run_sens("tau_safe", f"{tv:.2f}", S_BASE, tv, EPS_S)
    for ev in [1e-4, 5e-4, 1e-3, 5e-3]:
        run_sens("eps_s", f"{ev:.1e}", S_BASE, TAU_SAFE, ev)

    with open(OUT_DIR / "sensitivity_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Value", "m_mean", "m_std", "RelErr_mean", "RelErr_std"])
        w.writerows(rows_sens)


if __name__ == "__main__":
    run_self_tests()
    run_example_1()
    run_example_2()
    run_example_3()
    run_example_4()
    run_example_5()
    run_example_6()
    run_example_7_nonseparable()
    run_tripartite_baselines_sensitivity()