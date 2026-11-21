"""
run_benchmarks_v2.py
Complete benchmark suite using Two-Sided Hierarchical Preconditioning:
  - Left Row Equilibration D_L^{-1} (eliminates temporal kernel scaling imbalance)
  - Right Hierarchical Sobolev Scaling D_S^{-1} = diag(2^{-j s}) (scale-invariant noise shielding)
  - Matrix-Free O(N) ML-SKCW operators and Scale-Consistent SSS
"""

from __future__ import annotations

import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator

from ml_skcw_core import (
    eval_chebyshev_U,
    build_local_basis,
    fwt_analysis,
    make_constant_kernel,
    make_exponential_kernel,
    RetardedSIVIEOperator,
    SeparableKernel,
    solve_plsqr_sss,
    run_self_tests,
)


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


def build_sobolev_weights(k: int, p: int, s: float) -> np.ndarray:
    """Right hierarchical Sobolev scaling D_S^{-1}: 2^{-j s} at dyadic level j."""
    N = (1 << (k - 1)) * p
    d_inv = np.ones(N, dtype=float)
    offset = p
    for j in range(1, k):
        cnt = (1 << (j - 1)) * p
        d_inv[offset: offset + cnt] = 2.0 ** (-j * s)
        offset += cnt
    return d_inv


def build_problem(
    k: int,
    p: int,
    tau: float,
    x_func,
    phi_func,
    rng: np.random.Generator,
    amp0: float = 5.0,
    lam0: float = 2.0,
    growth_gamma: float = 0.0,
    amp1: float = 1.0,
    lam1: float = 1.0,
    amp2: float = 0.005,
    rel_noise: float = 1e-3,
    basis=None,
):
    T_end = 1.0
    if basis is None:
        basis = build_local_basis(p)
    M_int = 1 << (k - 1)
    h = T_end / M_int
    t_grid = RetardedSIVIEOperator.collocation_grid(k, p, T_end=T_end, basis=basis)

    a0 = amp0 * np.exp(growth_gamma * t_grid) * np.exp(-lam0 * (t_grid - 0.5))
    b0 = np.exp(lam0 * (t_grid - 0.5))
    K0 = SeparableKernel([a0], [b0])

    a1 = amp1 * np.exp(growth_gamma * t_grid) * np.exp(-lam1 * (t_grid - 0.5))
    b1 = np.exp(lam1 * (t_grid - 0.5))
    K1 = SeparableKernel([a1], [b1])

    a2 = amp2 * np.exp(growth_gamma * t_grid)
    K2 = SeparableKernel([a2], [np.ones_like(t_grid)])
    dW = rng.normal(0.0, np.sqrt(h), size=M_int)

    op = RetardedSIVIEOperator(k, p, tau=tau, K0=K0, K1=K1, K2=K2, dW=dW, T_end=T_end, basis=basis)
    c_true = project_function_to_ml_coeffs(x_func, k, p, T_end, basis)
    x_true = x_func(t_grid).ravel()

    b_w_exact = op.matvec(c_true)
    b_norm = float(np.linalg.norm(b_w_exact))
    noise_std = rel_noise * b_norm
    eta_w = (noise_std / np.sqrt(op.N)) * rng.standard_normal(op.N)
    b_w_noisy = b_w_exact + eta_w

    # Left temporal kernel envelope equilibration (O(N))
    w_left = np.exp(-growth_gamma * t_grid.ravel())
    # Normalize w_left so mean squared weight is 1 (keeps residual scale O(1))
    w_left = w_left / np.sqrt(np.mean(w_left ** 2))

    # Prior noise floor in left-preconditioned coordinates
    phi_grid = phi_func(np.linspace(-tau, 0.0, 200))
    phi_l2 = float(np.sqrt(np.mean(phi_grid ** 2) * tau))
    delta_ito = amp2 * h * phi_l2
    delta_obs_L = float(noise_std * np.sqrt(np.mean(w_left ** 2)))
    delta_prior_L = float(np.sqrt(delta_obs_L ** 2 + delta_ito ** 2))

    return op, t_grid.ravel(), c_true, x_true, b_w_exact, b_w_noisy, w_left, delta_prior_L, noise_std


def make_two_sided_operator(op: RetardedSIVIEOperator, w_left: np.ndarray, d_s_inv: np.ndarray) -> LinearOperator:
    return LinearOperator(
        dtype=float,
        shape=op.shape,
        matvec=lambda y: w_left * op.matvec(d_s_inv * y),
        rmatvec=lambda u: d_s_inv * op.rmatvec(w_left * u),
    )


# ============================================================================
# Example 1: Ensemble Consistency (L = 100 paths)
# ============================================================================

def run_example_1():
    print("\n" + "=" * 72)
    print("EXAMPLE 1: Ensemble Consistency & Pathwise Stability (L=100, N=1024)")
    print("=" * 72)
    k, p, tau = 9, 4, 0.1
    s_sob = 0.15
    basis = build_local_basis(p)
    x_func = lambda t: np.exp(t) * np.sin(2.0 * np.pi * t) + 0.5 * np.cos(np.pi * t)
    phi_func = lambda t: 0.5 * np.cos(np.pi * t)

    rng = np.random.default_rng(42)
    d_s_inv = build_sobolev_weights(k, p, s=s_sob)
    paths_rec, stops, rel_errs = [], [], []

    for _ in range(100):
        op, t_vec, _, x_true, _, b_w_noisy, w_left, delta_prior_L, _ = build_problem(
            k, p, tau, x_func, phi_func, rng, growth_gamma=2.0, rel_noise=1e-3, basis=basis
        )
        A_tilde = make_two_sided_operator(op, w_left, d_s_inv)
        res = solve_plsqr_sss(
            A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L, tau_safe=1.15, eps_s=1e-3, m_max=100
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
    print(f"Terminal MSE at t~1:    {np.mean((paths_arr[:, -1] - x_true[-1])**2):.4e}")

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
    ax2.set_title("(b) Empirical Distribution of SSS Index $m^*$")
    ax2.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig("Example1_Aligned_Consistency.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example 2: Temporal Conditioning Reduction via Left Equilibration (k=4..8)
# ============================================================================

def run_example_2():
    print("\n" + "=" * 72)
    print("EXAMPLE 2: Conditioning Reduction via Left Kernel Equilibration (p=4)")
    print("=" * 72)
    p, tau, growth_gamma = 4, 0.1, 4.0
    basis = build_local_basis(p)
    rng = np.random.default_rng(7)
    x_func = lambda t: np.sin(2.0 * np.pi * t)
    phi_func = lambda t: np.zeros_like(t)

    k_vals = [4, 5, 6, 7, 8]
    cond_raw, cond_eq = [], []
    print(f"{'k':>3} | {'p':>3} | {'N':>5} | {'kappa(A_w)':>13} | {'kappa(D_L^-1 A_w)':>18} | {'Gain':>7}")
    print("-" * 64)
    for k in k_vals:
        op, _, _, _, _, _, w_left, _, _ = build_problem(
            k, p, tau, x_func, phi_func, rng, growth_gamma=growth_gamma, basis=basis
        )
        A_w = op.assemble_explicit_for_diagnostics()
        A_L = w_left[:, None] * A_w
        k_orig = np.linalg.cond(A_w)
        k_eq = np.linalg.cond(A_L)
        cond_raw.append(k_orig)
        cond_eq.append(k_eq)
        print(f"{k:>3} | {p:>3} | {op.N:>5} | {k_orig:>13.2e} | {k_eq:>18.2e} | {k_orig / k_eq:>6.2f}x")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogy(k_vals, cond_raw, "r-o", lw=2, label=r"Unpreconditioned $\kappa(\mathbf{A}_w)$")
    ax.semilogy(k_vals, cond_eq, "b-s", lw=2, label=r"Left-Equilibrated $\kappa(\mathbf{D}_L^{-1}\mathbf{A}_w)$")
    ax.set_xlabel(r"Dyadic Resolution Level $k$ ($N = 2^{k-1} p$)")
    ax.set_ylabel("Spectral Condition Number")
    ax.set_title(r"Conditioning Stabilization under Exponential Kernel Growth ($\gamma=4, p=4$)")
    ax.set_xticks(k_vals)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig("Example2_Conditioning_Stable.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example 3: Semi-Convergence V-Curve & SSS Verification
# ============================================================================

def run_example_3():
    print("\n" + "=" * 72)
    print("EXAMPLE 3: Semi-Convergence & Spectral Stop Sign (N = 1024)")
    print("=" * 72)
    k, p, tau, s_sob = 9, 4, 0.1, 0.15
    rng = np.random.default_rng(101)
    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.3 * np.cos(6.0 * np.pi * t)
    phi_func = lambda t: 0.3 * np.cos(6.0 * np.pi * t)

    op, _, _, x_true, _, b_w_noisy, w_left, delta_prior_L, _ = build_problem(
        k, p, tau, x_func, phi_func, rng, growth_gamma=2.0, rel_noise=1e-3
    )
    d_s_inv = build_sobolev_weights(k, p, s=s_sob)
    A_tilde = make_two_sided_operator(op, w_left, d_s_inv)

    m_max = 100
    res_full = solve_plsqr_sss(
        A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L, use_sss=False, m_max=m_max, record_history=True
    )
    res_sss = solve_plsqr_sss(
        A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L, tau_safe=1.15, eps_s=1e-3, use_sss=True, m_max=m_max
    )

    rel_errors = [
        np.linalg.norm(op.eval_basis_at_nodes(cm).ravel() - x_true) / np.linalg.norm(x_true)
        for cm in res_full.c_history
    ]
    m_oracle = int(np.argmin(rel_errors)) + 1
    print(f"Prior noise floor delta_prior_L: {delta_prior_L:.4e}")
    print(f"SSS stopping index m*:           {res_sss.m_stop} (rel_err = {rel_errors[res_sss.m_stop - 1]:.4e})")
    print(f"Oracle optimal index m_opt:      {m_oracle} (rel_err = {rel_errors[m_oracle - 1]:.4e})")

    iters = np.arange(1, m_max + 1)
    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    ax1.semilogy(iters, rel_errors, "r-", lw=2, label="Relative Reconstruction Error")
    ax1.scatter(res_sss.m_stop, rel_errors[res_sss.m_stop - 1], c="black", s=70, zorder=5,
                label=f"SSS Stop ($m^*={res_sss.m_stop}$)")
    ax1.set_xlabel("Krylov Iteration Index $m$")
    ax1.set_ylabel("Relative $L^2$ Error", color="r")
    ax1.tick_params(axis="y", labelcolor="r")

    ax2 = ax1.twinx()
    ax2.semilogy(iters, res_full.omega_history[1:], "b--", lw=1.6, label=r"Normalized Residual $\omega_m$")
    ax2.axhline(1.15 * delta_prior_L, color="b", ls=":", alpha=0.7, label=r"$\tau_{\mathrm{safe}}\delta_{\mathrm{prior}}$")
    ax2.set_ylabel(r"Normalized Residual $\omega_m$", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right")
    ax1.set_title("Semi-Convergence Trajectory and SSS Identification ($N=1024$)")
    ax1.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig("Example3_SemiConvergence.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example 4: Large-Scale Scalability up to N = 262,144
# ============================================================================

def run_example_4():
    print("\n" + "=" * 72)
    print("EXAMPLE 4: Large-Scale Matrix-Free Scalability (N up to 262,144)")
    print("=" * 72)
    p, tau, s_sob = 4, 0.1, 0.20
    basis = build_local_basis(p)
    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.35 * np.sin(20.0 * np.pi * t) + 0.15 * np.sin(80.0 * np.pi * t)
    phi_func = lambda t: np.sin(2.0 * np.pi * t)

    k_list = [12, 14, 15, 16, 17]  # N = 8192 .. 262144
    print(f"{'k':>3} | {'N':>8} | {'m*':>4} | {'Err PLSQR':>11} | {'Err RawSSS':>11} | {'Err Raw300':>11} | {'Time(s)':>8} | {'MB':>6}")
    print("-" * 82)

    for k in k_list:
        rng = np.random.default_rng(42)
        op, _, _, x_true, _, b_w_noisy, w_left, delta_prior_L, noise_std = build_problem(
            k, p, tau, x_func, phi_func, rng, growth_gamma=2.0, rel_noise=5e-4, basis=basis
        )

        tracemalloc.start()
        t0 = time.perf_counter()
        d_s_inv = build_sobolev_weights(k, p, s=s_sob)
        A_tilde = make_two_sided_operator(op, w_left, d_s_inv)
        res = solve_plsqr_sss(
            A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L, tau_safe=1.1, eps_s=1e-3, m_max=150
        )
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        err_plsqr = np.linalg.norm(op.eval_basis_at_nodes(res.c_sol).ravel() - x_true) / np.linalg.norm(x_true)

        res_raw_sss = solve_plsqr_sss(
            op, b_w_noisy, np.ones(op.N), noise_std, tau_safe=1.1, eps_s=1e-3, m_max=150
        )
        err_raw_sss = np.linalg.norm(op.eval_basis_at_nodes(res_raw_sss.c_sol).ravel() - x_true) / np.linalg.norm(x_true)

        res_raw300 = solve_plsqr_sss(
            op, b_w_noisy, np.ones(op.N), noise_std, use_sss=False, m_max=300
        )
        err_raw300 = np.linalg.norm(op.eval_basis_at_nodes(res_raw300.c_sol).ravel() - x_true) / np.linalg.norm(x_true)

        print(
            f"{k:>3} | {op.N:>8} | {res.m_stop:>4} | {err_plsqr:>11.4e} | {err_raw_sss:>11.4e} | "
            f"{err_raw300:>11.4e} | {elapsed:>8.4f} | {peak_bytes / (1024**2):>6.2f}"
        )


# ============================================================================
# Example 5: Signal-Noise Decoupling & Spectral Retention (N = 131,072)
# ============================================================================

def run_example_5():
    print("\n" + "=" * 72)
    print("EXAMPLE 5: Multi-Frequency Signal Reconstruction (N = 131,072)")
    print("=" * 72)
    k, p, tau, s_sob = 16, 4, 0.1, 0.15
    rng = np.random.default_rng(42)

    # Low (2 Hz), Mid (25 Hz), High (80 Hz)
    x_func = lambda t: (
        1.0 * np.sin(2.0 * np.pi * 2.0 * t)
        + 0.45 * np.sin(2.0 * np.pi * 25.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 80.0 * t)
    )
    phi_func = lambda t: 1.0 * np.sin(2.0 * np.pi * 2.0 * t)

    op, t_vec, _, x_true, _, b_w_noisy, w_left, delta_prior_L, noise_std = build_problem(
        k, p, tau, x_func, phi_func, rng, growth_gamma=2.0, rel_noise=2e-4
    )
    d_s_inv = build_sobolev_weights(k, p, s=s_sob)
    A_tilde = make_two_sided_operator(op, w_left, d_s_inv)

    res = solve_plsqr_sss(
        A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L, tau_safe=1.08, eps_s=5e-4, m_max=150
    )
    x_rec = op.eval_basis_at_nodes(res.c_sol).ravel()

    res_raw_sss = solve_plsqr_sss(
        op, b_w_noisy, np.ones(op.N), noise_std, tau_safe=1.08, eps_s=5e-4, m_max=150
    )
    x_raw_sss = op.eval_basis_at_nodes(res_raw_sss.c_sol).ravel()

    res_raw300 = solve_plsqr_sss(
        op, b_w_noisy, np.ones(op.N), noise_std, use_sss=False, m_max=300
    )
    x_raw300 = op.eval_basis_at_nodes(res_raw300.c_sol).ravel()

    rms_plsqr = float(np.sqrt(np.mean((x_rec - x_true) ** 2)))
    rms_raw_sss = float(np.sqrt(np.mean((x_raw_sss - x_true) ** 2)))
    rms_raw300 = float(np.sqrt(np.mean((x_raw300 - x_true) ** 2)))
    print(f"PLSQR (Two-Sided HDP+SSS) m*={res.m_stop}: RMS = {rms_plsqr:.4e}")
    print(f"Raw LSQR (SSS)            m*={res_raw_sss.m_stop}: RMS = {rms_raw_sss:.4e}")
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
    fig.savefig("Proposed_Error_Field_Forensic_Refined.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Section 6.6: Empirical Validation of Tripartite Error Decomposition
# ============================================================================

def run_section_6_6():
    print("\n" + "=" * 72)
    print("SECTION 6.6: Empirical Tripartite Error Decomposition (N = 32,768)")
    print("=" * 72)
    k, p, tau, s_sob = 14, 4, 0.1, 0.20
    rng = np.random.default_rng(42)

    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.35 * np.sin(20.0 * np.pi * t) + 0.15 * np.sin(80.0 * np.pi * t)
    phi_func = lambda t: np.sin(2.0 * np.pi * t)

    op, _, c_true, x_true, b_w_exact, b_w_noisy, w_left, delta_prior_L, noise_std = build_problem(
        k, p, tau, x_func, phi_func, rng, growth_gamma=2.0, rel_noise=5e-4
    )
    d_s_inv = build_sobolev_weights(k, p, s=s_sob)
    A_tilde = make_two_sided_operator(op, w_left, d_s_inv)

    x_disc = op.eval_basis_at_nodes(c_true).ravel()
    res_sss = solve_plsqr_sss(
        A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L, tau_safe=1.1, eps_s=1e-3, m_max=150
    )
    m_star = res_sss.m_stop
    x_reg_noisy = op.eval_basis_at_nodes(res_sss.c_sol).ravel()

    res_reg_clean = solve_plsqr_sss(
        A_tilde, w_left * b_w_exact, d_s_inv, delta_prior_L, use_sss=False, m_max=m_star
    )
    x_reg_clean = op.eval_basis_at_nodes(res_reg_clean.c_sol).ravel()

    norm_x = float(np.linalg.norm(x_true))
    E_A = float(np.linalg.norm(x_true - x_disc)) / norm_x
    E_R = float(np.linalg.norm(x_disc - x_reg_clean)) / norm_x
    E_S = float(np.linalg.norm(x_reg_clean - x_reg_noisy)) / norm_x
    E_total = float(np.linalg.norm(x_true - x_reg_noisy)) / norm_x
    upper_bound = E_A + E_R + E_S

    print(f"Stopping index m*:                    {m_star}")
    print(f"E_A (Discretization error):           {E_A:.6e}")
    print(f"E_R (Regularization bias):            {E_R:.6e}")
    print(f"E_S (Propagated noise sensitivity):   {E_S:.6e}")
    print(f"Upper bound E_A + E_R + E_S:          {upper_bound:.6e}")
    print(f"Measured E_total:                     {E_total:.6e}")
    print(f"Strict triangle inequality satisfied: {E_total <= upper_bound + 1e-12}")

    res_raw = solve_plsqr_sss(op, b_w_noisy, np.ones(op.N), noise_std, use_sss=False, m_max=300)
    res_raw_sss = solve_plsqr_sss(op, b_w_noisy, np.ones(op.N), noise_std, tau_safe=1.1, eps_s=1e-3, m_max=150)
    res_hdp_over = solve_plsqr_sss(A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L, use_sss=False, m_max=300)

    E_raw = float(np.linalg.norm(op.eval_basis_at_nodes(res_raw.c_sol).ravel() - x_true)) / norm_x
    E_raw_sss = float(np.linalg.norm(op.eval_basis_at_nodes(res_raw_sss.c_sol).ravel() - x_true)) / norm_x
    E_hdp_over = float(np.linalg.norm(op.eval_basis_at_nodes(res_hdp_over.c_sol).ravel() - x_true)) / norm_x

    res_hist = solve_plsqr_sss(
        A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L, use_sss=False, m_max=80, record_history=True
    )
    errs_m = [float(np.linalg.norm(op.eval_basis_at_nodes(cm).ravel() - x_true)) / norm_x for cm in res_hist.c_history]
    m_oracle = int(np.argmin(errs_m)) + 1

    print("\n--- Component Ablation ---")
    print(f"Raw LSQR (unprecond, m=300):            {E_raw:.6e}")
    print(f"Raw LSQR + SSS (m*={res_raw_sss.m_stop:2d}):                 {E_raw_sss:.6e}")
    print(f"Two-Sided HDP-LSQR (no SSS, m=300):     {E_hdp_over:.6e}")
    print(f"PLSQR--ML-SKCW (Two-Sided + SSS, m*={m_star}): {E_total:.6e}")
    print(f"Oracle stopping index m_oracle:         {m_oracle} (error = {errs_m[m_oracle - 1]:.6e})")


if __name__ == "__main__":
    run_self_tests()
    run_example_1()
    run_example_2()
    run_example_3()
    run_example_4()
    run_example_5()
    run_section_6_6()