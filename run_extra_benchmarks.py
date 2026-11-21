"""
run_extra_benchmarks.py
Reviewer-requested supplementary benchmarks at N = 32,768 (k = 14, p = 4):
1. Baseline comparison against CGLS/LSQR+SSS and matrix-free Tikhonov-LSQR
   under BOTH matched iteration budget (itn = m*) and full budget (iter_lim = 300),
   reporting SciPy stopping flag (istop).
2. Multi-seed (5 independent Brownian & noise seeds) sensitivity analysis for
   Sobolev exponent s, safety factor tau_safe, and stagnation tolerance eps_s.
3. Automatic CSV export for GitHub reproducibility.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla

from ml_skcw_core import build_local_basis, solve_plsqr_sss
from run_benchmarks_v2 import build_problem, build_sobolev_weights, make_two_sided_operator


OUT_DIR = Path(__file__).resolve().parent


def istop_label(istop: int) -> str:
    return {
        1: "tol(Ax=b)",
        2: "tol(A^Tr)",
        3: "conlim",
        7: "max_iter",
    }.get(istop, f"flag({istop})")


def run_baseline_comparison() -> None:
    print("=" * 88)
    print("BASELINE COMPARISON: PLSQR--ML-SKCW vs. Tikhonov-LSQR & CGLS (N = 32,768)")
    print("=" * 88)
    k, p, tau = 14, 4, 0.1
    basis = build_local_basis(p)
    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.35 * np.sin(20.0 * np.pi * t) + 0.15 * np.sin(80.0 * np.pi * t)
    phi_func = lambda t: np.sin(2.0 * np.pi * t)

    rng = np.random.default_rng(42)
    op, _, _, x_true, _, b_w_noisy, w_left, delta_prior_L, noise_std = build_problem(
        k, p, tau, x_func, phi_func, rng, growth_gamma=2.0, rel_noise=5e-4, basis=basis
    )
    norm_x = float(np.linalg.norm(x_true))
    rows_csv = []

    # 1. Proposed PLSQR--ML-SKCW (s = 0.20)
    t0 = time.perf_counter()
    d_s_inv = build_sobolev_weights(k, p, s=0.20)
    A_tilde = make_two_sided_operator(op, w_left, d_s_inv)
    res_prop = solve_plsqr_sss(
        A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L, tau_safe=1.10, eps_s=1e-3, m_max=150
    )
    t_prop = time.perf_counter() - t0
    err_prop = float(np.linalg.norm(op.eval_basis_at_nodes(res_prop.c_sol).ravel() - x_true)) / norm_x
    m_star = res_prop.m_stop
    print(f"{'Method':<44} | {'itn':>4} | {'Stop Reason':<10} | {'RelErr':>11} | {'Time(s)':>8}")
    print("-" * 88)
    print(f"{'PLSQR--ML-SKCW (s=0.20, SSS)':<44} | {m_star:>4d} | {'SSS':<10} | {err_prop:>11.4e} | {t_prop:>8.4f}")
    rows_csv.append(["PLSQR--ML-SKCW (s=0.20, SSS)", m_star, "SSS", f"{err_prop:.6e}", f"{t_prop:.4f}"])

    # 2. Unpreconditioned CGLS/LSQR + SSS
    t0 = time.perf_counter()
    res_cgls = solve_plsqr_sss(
        op, b_w_noisy, np.ones(op.N), noise_std, tau_safe=1.10, eps_s=1e-3, m_max=150
    )
    t_cgls = time.perf_counter() - t0
    err_cgls = float(np.linalg.norm(op.eval_basis_at_nodes(res_cgls.c_sol).ravel() - x_true)) / norm_x
    print(f"{'Unprecond. LSQR/CGLS + SSS':<44} | {res_cgls.m_stop:>4d} | {'SSS':<10} | {err_cgls:>11.4e} | {t_cgls:>8.4f}")
    rows_csv.append(["Unprecond. LSQR/CGLS + SSS", res_cgls.m_stop, "SSS", f"{err_cgls:.6e}", f"{t_cgls:.4f}"])

    # 3. Unpreconditioned Tikhonov-LSQR (full budget iter_lim=300 & matched budget iter_lim=m_star)
    for lam, budget in [(1e-3, 300), (1e-2, 300), (1e-2, m_star), (5e-2, 300)]:
        t0 = time.perf_counter()
        c_tik, istop, itn, *_ = spla.lsqr(op, b_w_noisy, damp=lam, iter_lim=budget, atol=1e-10, btol=1e-10)
        t_tik = time.perf_counter() - t0
        err_tik = float(np.linalg.norm(op.eval_basis_at_nodes(c_tik).ravel() - x_true)) / norm_x
        name = f"Tikhonov-LSQR (lam={lam:.0e}, lim={budget})"
        flag = istop_label(istop)
        print(f"{name:<44} | {itn:>4d} | {flag:<10} | {err_tik:>11.4e} | {t_tik:>8.4f}")
        rows_csv.append([name, itn, flag, f"{err_tik:.6e}", f"{t_tik:.4f}"])

    # 4. Two-Sided HDP-Tikhonov-LSQR (full budget & matched budget iter_lim=m_star)
    for lam, budget in [(5e-3, 300), (5e-3, m_star), (1e-2, 300), (1e-2, m_star), (2e-2, 300)]:
        t0 = time.perf_counter()
        y_tik, istop, itn, *_ = spla.lsqr(
            A_tilde, w_left * b_w_noisy, damp=lam, iter_lim=budget, atol=1e-10, btol=1e-10
        )
        c_tik = d_s_inv * y_tik
        t_tik = time.perf_counter() - t0
        err_tik = float(np.linalg.norm(op.eval_basis_at_nodes(c_tik).ravel() - x_true)) / norm_x
        name = f"HDP-Tikhonov-LSQR (lam={lam:.0e}, lim={budget})"
        flag = istop_label(istop)
        print(f"{name:<44} | {itn:>4d} | {flag:<10} | {err_tik:>11.4e} | {t_tik:>8.4f}")
        rows_csv.append([name, itn, flag, f"{err_tik:.6e}", f"{t_tik:.4f}"])

    csv_path = OUT_DIR / "baseline_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Iterations", "StopReason", "RelError", "Time_s"])
        writer.writerows(rows_csv)
    print(f"\n[Saved] {csv_path}")


def run_multiseed_sensitivity() -> None:
    print("\n" + "=" * 88)
    print("MULTI-SEED HYPERPARAMETER SENSITIVITY (N = 32,768, 5 independent seeds)")
    print("=" * 88)
    k, p, tau = 14, 4, 0.1
    seeds = [42, 101, 202, 303, 404]
    basis = build_local_basis(p)
    x_func = lambda t: np.sin(2.0 * np.pi * t) + 0.35 * np.sin(20.0 * np.pi * t) + 0.15 * np.sin(80.0 * np.pi * t)
    phi_func = lambda t: np.sin(2.0 * np.pi * t)

    # Pre-build problem instances once per seed
    instances = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        op, _, _, x_true, _, b_w_noisy, w_left, delta_prior_L, _ = build_problem(
            k, p, tau, x_func, phi_func, rng, growth_gamma=2.0, rel_noise=5e-4, basis=basis
        )
        instances.append((op, x_true, float(np.linalg.norm(x_true)), b_w_noisy, w_left, delta_prior_L))

    rows_csv = []

    def eval_config(param_group: str, param_label: str, s: float, tau_safe: float, eps_s: float):
        d_s_inv = build_sobolev_weights(k, p, s=s)
        m_list, e_list = [], []
        for op, x_true, norm_x, b_w_noisy, w_left, delta_prior_L in instances:
            A_tilde = make_two_sided_operator(op, w_left, d_s_inv)
            res = solve_plsqr_sss(
                A_tilde, w_left * b_w_noisy, d_s_inv, delta_prior_L,
                tau_safe=tau_safe, eps_s=eps_s, m_max=150
            )
            err = float(np.linalg.norm(op.eval_basis_at_nodes(res.c_sol).ravel() - x_true)) / norm_x
            m_list.append(res.m_stop)
            e_list.append(err)
        m_mean, m_std = float(np.mean(m_list)), float(np.std(m_list))
        e_mean, e_std = float(np.mean(e_list)), float(np.std(e_list))
        print(f"{param_label:<10} | {m_mean:5.2f} +/- {m_std:4.2f} | {e_mean:.4e} +/- {e_std:.2e}")
        rows_csv.append([param_group, param_label, f"{m_mean:.2f}", f"{m_std:.2f}", f"{e_mean:.6e}", f"{e_std:.6e}"])

    print("\n1) Sensitivity to Sobolev exponent s (tau_safe=1.10, eps_s=1e-3):")
    print(f"{'s':<10} | {'m* (mean+/-std)':<15} | {'RelErr (mean+/-std)'}")
    print("-" * 55)
    for s in [0.00, 0.10, 0.15, 0.20, 0.25, 0.35]:
        eval_config("s", f"{s:.2f}", s=s, tau_safe=1.10, eps_s=1e-3)

    print("\n2) Sensitivity to discrepancy factor tau_safe (s=0.20, eps_s=1e-3):")
    print(f"{'tau_safe':<10} | {'m* (mean+/-std)':<15} | {'RelErr (mean+/-std)'}")
    print("-" * 55)
    for ts in [1.02, 1.05, 1.10, 1.15, 1.25, 1.40]:
        eval_config("tau_safe", f"{ts:.2f}", s=0.20, tau_safe=ts, eps_s=1e-3)

    print("\n3) Sensitivity to stagnation tolerance eps_s (s=0.20, tau_safe=1.10):")
    print(f"{'eps_s':<10} | {'m* (mean+/-std)':<15} | {'RelErr (mean+/-std)'}")
    print("-" * 55)
    for es in [1e-4, 5e-4, 1e-3, 5e-3]:
        eval_config("eps_s", f"{es:.1e}", s=0.20, tau_safe=1.10, eps_s=es)

    csv_path = OUT_DIR / "sensitivity_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Group", "Value", "m_mean", "m_std", "RelErr_mean", "RelErr_std"])
        writer.writerows(rows_csv)
    print(f"\n[Saved] {csv_path}")


if __name__ == "__main__":
    run_baseline_comparison()
    run_multiseed_sensitivity()