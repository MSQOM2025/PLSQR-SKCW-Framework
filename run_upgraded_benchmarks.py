"""
run_upgraded_benchmarks.py
Complete, self-consistent benchmark suite using asymmetric non-factored kernels
and consistent hyperparameters:
- Example 1: Ensemble consistency & DSC stopping stability (L=100, N=1024)
- Example 2: Full 3-way spectral conditioning report (k=4..8, p=4, strong K2=0.25)
- Example 3: Semi-convergence V-curve & DSC verification (N=1024)
- Example 4: Large-scale scalability up to N = 262,144
- Example 5: High-frequency spectral retention at N = 131,072
- Example 6: Strong Ito regime (amp2=0.25) + non-smooth jump/pulse target,
             verifying the s > 1/2 stabilization threshold vs. s < 1/2 & 64-pt rival
- Section 7.7: Tripartite error decomposition, baseline comparison (Tikhonov/CGLS,
               matched & full budget), and 5-seed hyperparameter sensitivity + CSVs.
"""

from __future__ import annotations
from scipy.sparse.linalg import LinearOperator
import csv
import time
import tracemalloc
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from ml_skcw_core import (
    eval_chebyshev_U,
    build_local_basis,
    fwt_analysis,
    RetardedSIVIEOperator,
    SeparableKernel,
    build_sobolev_weights,
    build_two_sided_hdp,
    make_two_sided_operator,
    solve_plsqr_dsc,
    run_self_tests,
)

OUT_DIR = Path(__file__).resolve().parent


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


def build_asymmetric_problem(
    k: int,
    p: int,
    tau: float,
    x_func,
    phi_func,
    dphi_func,
    rng: np.random.Generator,
    amp2: float = 0.01,
    rel_noise: float = 5e-4,
    basis=None,
):
    """
    Asymmetric kernels with NO common temporal factor:
      K0(t,s) = 4.0 * exp(3.0*t) * exp(-2.0*(t-s))
      K1(t,s) = 1.2 * exp(-1.5*t) * exp(-(t-s))
      K2(t,s) = amp2 * (1 + 0.5*sin(2*pi*t))
    """
    T_end = 1.0
    if basis is None:
        basis = build_local_basis(p)
    M_int = 1 << (k - 1)
    h = T_end / M_int
    t_grid = RetardedSIVIEOperator.collocation_grid(k, p, T_end=T_end, basis=basis)

    a0 = 4.0 * np.exp(3.0 * t_grid) * np.exp(-2.0 * (t_grid - 0.5))
    b0 = np.exp(2.0 * (t_grid - 0.5))
    K0 = SeparableKernel([a0], [b0])

    a1 = 1.2 * np.exp(-1.5 * t_grid) * np.exp(-1.0 * (t_grid - 0.5))
    b1 = np.exp(1.0 * (t_grid - 0.5))
    K1 = SeparableKernel([a1], [b1])

    a2 = amp2 * (1.0 + 0.5 * np.sin(2.0 * np.pi * t_grid))
    K2 = SeparableKernel([a2], [np.ones_like(t_grid)])
    dW = rng.normal(0.0, np.sqrt(h), size=M_int)

    op = RetardedSIVIEOperator(k, p, tau=tau, K0=K0, K1=K1, K2=K2, dW=dW, T_end=T_end, basis=basis)
    c_true = project_function_to_ml_coeffs(x_func, k, p, T_end, basis)
    x_true = x_func(t_grid).ravel()

    b_w_exact = op.matvec(c_true)
    b_norm = float(np.linalg.norm(b_w_exact))
    noise_std = rel_noise * b_norm
    b_w_noisy = b_w_exact + (noise_std / np.sqrt(op.N)) * rng.standard_normal(op.N)

    # Discretization-and-noise floor: (K2_bar * ||phi'|| + L_K * ||phi||) * sqrt(tau) * h
    t_hist = np.linspace(-tau, 0.0, 200)
    phi_l2 = float(np.sqrt(np.mean(phi_func(t_hist) ** 2) * tau))
    dphi_l2 = float(np.sqrt(np.mean(dphi_func(t_hist) ** 2) * tau))
    k2_bar = 1.5 * amp2
    delta_ito = k2_bar * dphi_l2 * np.sqrt(tau) * h
    delta_prior = float(np.sqrt(noise_std ** 2 + delta_ito ** 2))

    return op, t_grid.ravel(), c_true, x_true, b_w_exact, b_w_noisy, delta_prior, noise_std


# Consistent baseline hyperparameters across Examples 1, 3, 4, 5, 7.6, 7.7
S_BASE = 0.20
TAU_SAFE = 1.10
EPS_S = 1e-4


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
        op, t_vec, _, x_true, _, b_w_noisy, delta_prior, _ = build_asymmetric_problem(
            k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=1e-3, basis=basis
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
        op, *_ = build_asymmetric_problem(k, p, tau, x_func, z_func, z_func, rng, amp2=0.25, basis=basis)
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

    op, _, _, x_true, _, b_w_noisy, delta_prior, _ = build_asymmetric_problem(
        k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=1e-3
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
# Example 4: Large-Scale Scalability up to N = 262,144
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
        op, _, _, x_true, _, b_w_noisy, delta_prior, noise_std = build_asymmetric_problem(
            k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=5e-4, basis=basis
        )

        tracemalloc.start()
        t0 = time.perf_counter()
        hdp = build_two_sided_hdp(op, s=S_BASE)
        A_tilde = make_two_sided_operator(op, hdp)
        res = solve_plsqr_dsc(
            A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
            tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
        )
        elapsed = time.perf_counter() - t0
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

    op, t_vec, _, x_true, _, b_w_noisy, delta_prior, noise_std = build_asymmetric_problem(
        k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=2e-4
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
# Example 6: Strong Ito Kernel (amp2=0.25), Non-Smooth Target, & s > 1/2 Threshold
# ============================================================================

def nonsmooth_target(t: np.ndarray) -> np.ndarray:
    """Step jump + sharp localized pulses + smooth background."""
    bg = 0.8 * np.sin(2.0 * np.pi * t)
    jump = 0.5 * ((t >= 0.25) & (t <= 0.65)).astype(float)
    pulse1 = 0.9 * np.exp(-0.5 * ((t - 0.38) / 0.0025) ** 2) * np.cos(2.0 * np.pi * 120.0 * t)
    pulse2 = 0.8 * np.exp(-0.5 * ((t - 0.78) / 0.0018) ** 2) * np.sin(2.0 * np.pi * 180.0 * t)
    return bg + jump + pulse1 + pulse2


def run_example_6():
    print("\n" + "=" * 78)
    print("EXAMPLE 6: Strong Ito Threshold (s > 1/2) on Non-Smooth Jump/Pulse Target")
    print("=" * 78)
    p, tau, amp2 = 4, 0.1, 0.25
    basis = build_local_basis(p)
    z_func = lambda t: np.zeros_like(t)

    k_list = [10, 12, 14, 16]
    s_test = [0.00, 0.20, 0.65]
    results = {s: [] for s in s_test}

    print(f"{'k':>3} | {'N':>7} | {'s=0.00 (Raw+DSC)':>16} | {'s=0.20 (s<1/2)':>16} | {'s=0.65 (s>1/2)':>16}")
    print("-" * 74)
    for k in k_list:
        rng = np.random.default_rng(42)
        op, _, _, x_true, _, b_w_noisy, delta_prior, noise_std = build_asymmetric_problem(
            k, p, tau, nonsmooth_target, z_func, z_func, rng, amp2=amp2, rel_noise=5e-4, basis=basis
        )
        norm_x = float(np.linalg.norm(x_true))
        line_parts = []
        for s_val in s_test:
            if s_val == 0.0:
                res = solve_plsqr_dsc(
                    op, b_w_noisy, np.ones(op.N), noise_std, tau_safe=TAU_SAFE, eps_s=1e-3, m_max=200
                )
            else:
                hdp = build_two_sided_hdp(op, s=s_val)
                A_tilde = make_two_sided_operator(op, hdp)
                res = solve_plsqr_dsc(
                    A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
                    tau_safe=TAU_SAFE, eps_s=1e-3, m_max=200
                )
            err = float(np.linalg.norm(op.eval_basis_at_nodes(res.c_sol).ravel() - x_true)) / norm_x
            results[s_val].append(err)
            line_parts.append(f"{err:>10.3e}({res.m_stop:3d})")
        print(f"{k:>3} | {op.N:>7} | {line_parts[0]:>16} | {line_parts[1]:>16} | {line_parts[2]:>16}")

    # Fair same-path 64-pt averaging rival (131,072 -> 2,048)
    rng = np.random.default_rng(42)
    op_f, t_f, _, x_true_f, _, b_w_noisy_f, _, noise_f = build_asymmetric_problem(
        16, p, tau, nonsmooth_target, z_func, z_func, rng, amp2=amp2, rel_noise=5e-4, basis=basis
    )
    dW_c = op_f.dW.reshape(1 << 9, 64).sum(axis=1)
    t_c_grid = RetardedSIVIEOperator.collocation_grid(10, p, T_end=1.0, basis=basis)
    a0 = 4.0 * np.exp(3.0 * t_c_grid) * np.exp(-2.0 * (t_c_grid - 0.5))
    b0 = np.exp(2.0 * (t_c_grid - 0.5))
    a1 = 1.2 * np.exp(-1.5 * t_c_grid) * np.exp(-1.0 * (t_c_grid - 0.5))
    b1 = np.exp(1.0 * (t_c_grid - 0.5))
    a2 = amp2 * (1.0 + 0.5 * np.sin(2.0 * np.pi * t_c_grid))
    op_c = RetardedSIVIEOperator(
        10, p, tau=tau,
        K0=SeparableKernel([a0], [b0]), K1=SeparableKernel([a1], [b1]),
        K2=SeparableKernel([a2], [np.ones_like(t_c_grid)]), dW=dW_c, T_end=1.0, basis=basis
    )
    hdp_c = build_two_sided_hdp(op_c, s=0.65)
    A_tilde_c = make_two_sided_operator(op_c, hdp_c)
    b_w_coarse = np.mean((np.sqrt(op_f.N) * b_w_noisy_f).reshape(2048, 64), axis=1) / np.sqrt(op_c.N)
    res_c = solve_plsqr_dsc(A_tilde_c, hdp_c.w_left * b_w_coarse, hdp_c.d_s_inv, noise_f / 8.0, tau_safe=TAU_SAFE, eps_s=1e-3, m_max=200)
    x_rec_c_fine = np.interp(t_f, t_c_grid.ravel(), op_c.eval_basis_at_nodes(res_c.c_sol).ravel())
    err_rival = float(np.linalg.norm(x_rec_c_fine - x_true_f)) / float(np.linalg.norm(x_true_f))
    print(f"Fair Same-Path 64-pt Averaging Rival (2,048 pts, s=0.65): m*={res_c.m_stop}, RelErr={err_rival:.4e}")

    N_vals = [(1 << (k - 1)) * p for k in k_list]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.loglog(N_vals, results[0.00], "k--^", lw=2, label=r"$s = 0.00$ (Unpreconditioned + DSC)")
    ax.loglog(N_vals, results[0.20], "r-o", lw=2, label=r"$s = 0.20$ ($s < 1/2$, stochastic blow-up)")
    ax.loglog(N_vals, results[0.65], "b-s", lw=2.2, label=r"$s = 0.65$ ($s > 1/2$, stabilized)")
    ax.axhline(err_rival, color="gray", ls=":", lw=1.6, label=f"64-pt Averaging Rival (RelErr={err_rival:.3f})")
    ax.set_xlabel("System Dimension $N$")
    ax.set_ylabel(r"Relative $L^2$ Error")
    ax.set_title(r"Stochastic Stabilization Threshold $s > 1/2$ ($K_2 = 0.25$)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Example6_Stochastic_Threshold.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Section 7.6 & 7.7: Tripartite Error, Baselines (Matched & Full), Sensitivity
# ============================================================================

def run_tripartite_baselines_sensitivity():
    print("\n" + "=" * 78)
    print("SECTION 7.6 & 7.7: Tripartite Error, Baselines & 5-Seed Sensitivity (N=32,768)")
    print("=" * 78)
    k, p, tau = 14, 4, 0.1
    basis = build_local_basis(p)
    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.35 * np.sin(20.0 * np.pi * t) + 0.15 * np.sin(80.0 * np.pi * t)
    phi_func = lambda t: np.sin(2.0 * np.pi * t)
    dphi_func = lambda t: 2.0 * np.pi * np.cos(2.0 * np.pi * t)

    rng = np.random.default_rng(42)
    op, _, c_true, x_true, b_w_exact, b_w_noisy, delta_prior, noise_std = build_asymmetric_problem(
        k, p, tau, x_func, phi_func, dphi_func, rng, amp2=0.01, rel_noise=5e-4, basis=basis
    )
    norm_x = float(np.linalg.norm(x_true))
    hdp = build_two_sided_hdp(op, s=S_BASE)
    A_tilde = make_two_sided_operator(op, hdp)

    # --- Tripartite Error Decomposition & Ablation ---
    x_disc = op.eval_basis_at_nodes(c_true).ravel()
    res_dsc = solve_plsqr_dsc(
        A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
        tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
    )
    m_star = res_dsc.m_stop
    x_reg_noisy = op.eval_basis_at_nodes(res_dsc.c_sol).ravel()

    res_clean = solve_plsqr_dsc(
        A_tilde, hdp.w_left * b_w_exact, hdp.d_s_inv, delta_prior, use_dsc=False, m_max=m_star
    )
    x_reg_clean = op.eval_basis_at_nodes(res_clean.c_sol).ravel()

    E_A = float(np.linalg.norm(x_true - x_disc)) / norm_x
    E_R = float(np.linalg.norm(x_disc - x_reg_clean)) / norm_x
    E_S = float(np.linalg.norm(x_reg_clean - x_reg_noisy)) / norm_x
    E_total = float(np.linalg.norm(x_true - x_reg_noisy)) / norm_x
    print(f"Tripartite (m*={m_star}): E_A={E_A:.4e}, E_R={E_R:.4e}, E_S={E_S:.4e}, Bound={E_A+E_R+E_S:.4e}, Total={E_total:.4e}")

    res_raw300 = solve_plsqr_dsc(op, b_w_noisy, np.ones(op.N), noise_std, use_dsc=False, m_max=300)
    res_hdp300 = solve_plsqr_dsc(A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior, use_dsc=False, m_max=300)
    E_raw300 = float(np.linalg.norm(op.eval_basis_at_nodes(res_raw300.c_sol).ravel() - x_true)) / norm_x
    E_hdp300 = float(np.linalg.norm(op.eval_basis_at_nodes(res_hdp300.c_sol).ravel() - x_true)) / norm_x
    print(f"Ablation: Raw(300)={E_raw300:.4e}, HDP-LSQR(300)={E_hdp300:.4e}, PLSQR(m*={m_star})={E_total:.4e}")

    # --- Baseline Comparisons (Full & Matched Budget) ---
    rows_base = []
    t0 = time.perf_counter()
    res_prop = solve_plsqr_dsc(
        A_tilde, hdp.w_left * b_w_noisy, hdp.d_s_inv, delta_prior,
        tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150
    )
    t_prop = time.perf_counter() - t0
    rows_base.append(["PLSQR--ML-SKCW (s=0.20, DSC)", m_star, "DSC", f"{E_total:.6e}", f"{t_prop:.4f}"])

    t0 = time.perf_counter()
    res_cgls = solve_plsqr_dsc(op, b_w_noisy, np.ones(op.N), noise_std, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=150)
    t_cgls = time.perf_counter() - t0
    e_cgls = float(np.linalg.norm(op.eval_basis_at_nodes(res_cgls.c_sol).ravel() - x_true)) / norm_x
    rows_base.append(["Unprecond. LSQR/CGLS + DSC", res_cgls.m_stop, "DSC", f"{e_cgls:.6e}", f"{t_cgls:.4f}"])

    def flag_str(istop: int) -> str:
        return {1: "tol(Ax=b)", 2: "tol(A^Tr)", 3: "conlim", 7: "max_iter"}.get(istop, f"flag({istop})")

    for lam, budget in [(1e-3, 300), (1e-2, 300), (1e-2, m_star), (5e-2, 300)]:
        t0 = time.perf_counter()
        c_tik, istop, itn, *_ = spla.lsqr(op, b_w_noisy, damp=lam, iter_lim=budget, atol=1e-10, btol=1e-10)
        el = time.perf_counter() - t0
        err = float(np.linalg.norm(op.eval_basis_at_nodes(c_tik).ravel() - x_true)) / norm_x
        rows_base.append([f"Tikhonov-LSQR (lam={lam:.0e}, lim={budget})", itn, flag_str(istop), f"{err:.6e}", f"{el:.4f}"])

    for lam, budget in [(5e-3, 300), (5e-3, m_star), (1e-2, 300), (1e-2, m_star), (2e-2, 300)]:
        t0 = time.perf_counter()
        y_tik, istop, itn, *_ = spla.lsqr(A_tilde, hdp.w_left * b_w_noisy, damp=lam, iter_lim=budget, atol=1e-10, btol=1e-10)
        el = time.perf_counter() - t0
        err = float(np.linalg.norm(op.eval_basis_at_nodes(hdp.d_s_inv * y_tik).ravel() - x_true)) / norm_x
        rows_base.append([f"HDP-Tikhonov-LSQR (lam={lam:.0e}, lim={budget})", itn, flag_str(istop), f"{err:.6e}", f"{el:.4f}"])

    print("\n--- Baseline Comparison ---")
    for r in rows_base:
        print(f"{r[0]:<44} | itn={r[1]:>3} | {r[2]:<9} | Err={float(r[3]):.4e} | {float(r[4]):.4f}s")

    with open(OUT_DIR / "baseline_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Iterations", "StopReason", "RelError", "Time_s"])
        w.writerows(rows_base)

    # --- 5-Seed Sensitivity ---
    seeds = [42, 101, 202, 303, 404]
    instances = []
    for sd in seeds:
        rng_s = np.random.default_rng(sd)
        op_s, _, _, xt_s, _, bn_s, dp_s, _ = build_asymmetric_problem(
            k, p, tau, x_func, phi_func, dphi_func, rng_s, amp2=0.01, rel_noise=5e-4, basis=basis
        )
        # Precompute row weights once per seed
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
    run_tripartite_baselines_sensitivity()