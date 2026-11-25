"""
test_left_only_rival.py
Checks the exact ablation requested by the critic:
Left-Only row equilibration D_L^{-1} (NO D_S^{-1}, i.e., s=0) + DSC and + Tikhonov.
"""

import numpy as np
import scipy.sparse.linalg as spla
from ml_skcw_core import build_local_basis, build_two_sided_hdp, make_two_sided_operator, solve_plsqr_dsc
from run_upgraded_benchmarks import build_asymmetric_problem, nonsmooth_target, TAU_SAFE, EPS_S


def run_left_only_check():
    p, tau = 4, 0.1
    basis = build_local_basis(p)
    z_f = lambda t: np.zeros_like(t)

    print("=== 1. Strong Ito Regime (Example 6, amp2=0.25): Left-Only (s=0) vs Two-Sided (s=0.65) ===")
    for k in [10, 12, 14, 16]:
        rng = np.random.default_rng(42)
        op, _, _, x_true, _, bn, dp, _ = build_asymmetric_problem(
            k, p, tau, nonsmooth_target, z_f, z_f, rng, amp2=0.25, rel_noise=5e-4, basis=basis
        )
        nx = float(np.linalg.norm(x_true))

        # Left-Only HDP (s = 0.0 -> D_S = I, only D_L^{-1} active)
        h_left = build_two_sided_hdp(op, s=0.0)
        A_left = make_two_sided_operator(op, h_left)
        r_left_dsc = solve_plsqr_dsc(A_left, h_left.w_left * bn, h_left.d_s_inv, dp, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200)
        e_left_dsc = float(np.linalg.norm(op.eval_basis_at_nodes(r_left_dsc.c_sol).ravel() - x_true)) / nx

        # Full Two-Sided (s = 0.65)
        h_full = build_two_sided_hdp(op, s=0.65)
        A_full = make_two_sided_operator(op, h_full)
        r_full = solve_plsqr_dsc(A_full, h_full.w_left * bn, h_full.d_s_inv, dp, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200)
        e_full = float(np.linalg.norm(op.eval_basis_at_nodes(r_full.c_sol).ravel() - x_true)) / nx

        print(f"k={k:2d} N={op.N:6d} | Left-Only+DSC: {e_left_dsc:.3e}({r_left_dsc.m_stop:2d}) | Two-Sided(s=0.65)+DSC: {e_full:.3e}({r_full.m_stop:2d})")

    # Also test Left-Only + Tikhonov at N=32,768 (k=14)
    rng = np.random.default_rng(42)
    op, _, _, x_true, _, bn, dp, _ = build_asymmetric_problem(
        14, p, tau, nonsmooth_target, z_f, z_f, rng, amp2=0.25, rel_noise=5e-4, basis=basis
    )
    nx = float(np.linalg.norm(x_true))
    h_left = build_two_sided_hdp(op, s=0.0)
    A_left = make_two_sided_operator(op, h_left)
    print("\n--- Left-Only D_L^-1 + Tikhonov (amp2=0.25, N=32,768) ---")
    for lam in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        c_t, istop, itn, *_ = spla.lsqr(A_left, h_left.w_left * bn, damp=lam, iter_lim=300, atol=1e-10, btol=1e-10)
        err = float(np.linalg.norm(op.eval_basis_at_nodes(c_t).ravel() - x_true)) / nx
        print(f"  Left-Only + Tikhonov (lam={lam:.0e}, itn={itn:3d}): RelErr = {err:.4e}")

    print("\n=== 2. Moderate Regime (Example 4/7.8, amp2=0.01, N=32,768): Left-Only (s=0) ===")
    x_mf = lambda t: np.sin(2.0 * np.pi * t) + 0.35 * np.sin(20.0 * np.pi * t) + 0.15 * np.sin(80.0 * np.pi * t)
    phi_mf = lambda t: np.sin(2.0 * np.pi * t)
    dphi_mf = lambda t: 2.0 * np.pi * np.cos(2.0 * np.pi * t)
    rng = np.random.default_rng(42)
    op_m, _, _, xm, _, bnm, dpm, _ = build_asymmetric_problem(
        14, p, tau, x_mf, phi_mf, dphi_mf, rng, amp2=0.01, rel_noise=5e-4, basis=basis
    )
    nxm = float(np.linalg.norm(xm))
    hm_left = build_two_sided_hdp(op_m, s=0.0)
    Am_left = make_two_sided_operator(op_m, hm_left)
    rm_dsc = solve_plsqr_dsc(Am_left, hm_left.w_left * bnm, hm_left.d_s_inv, dpm, tau_safe=TAU_SAFE, eps_s=EPS_S, m_max=200)
    em_dsc = float(np.linalg.norm(op_m.eval_basis_at_nodes(rm_dsc.c_sol).ravel() - xm)) / nxm
    print(f"Left-Only + DSC (m*={rm_dsc.m_stop}): RelErr = {em_dsc:.4e}")
    for lam in [5e-3, 1e-2, 5e-2]:
        c_t, istop, itn, *_ = spla.lsqr(Am_left, hm_left.w_left * bnm, damp=lam, iter_lim=300, atol=1e-10, btol=1e-10)
        err = float(np.linalg.norm(op_m.eval_basis_at_nodes(c_t).ravel() - xm)) / nxm
        print(f"  Left-Only + Tikhonov (lam={lam:.0e}, itn={itn:3d}): RelErr = {err:.4e}")


if __name__ == "__main__":
    run_left_only_check()