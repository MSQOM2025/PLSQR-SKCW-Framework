"""
ml_skcw_core.py
Matrix-Free Multi-Level Second-Kind Chebyshev Wavelet (ML-SKCW) Solver
for Inverse Retarded Stochastic Ito-Volterra Integral Equations.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator
from dataclasses import dataclass
from typing import List, Optional


# ============================================================================
# 1. Local SKCW Polynomials & Two-Scale Multi-Level Orthogonal Transform
# ============================================================================

def eval_chebyshev_U(x: np.ndarray, p: int) -> np.ndarray:
    """Evaluate Second-Kind Chebyshev polynomials U_0,...,U_{p-1} at x in [-1, 1]."""
    x = np.asarray(x, dtype=float)
    U = np.zeros((x.size, p), dtype=float)
    if p >= 1:
        U[:, 0] = 1.0
    if p >= 2:
        U[:, 1] = 2.0 * x
    for r in range(2, p):
        U[:, r] = 2.0 * x * U[:, r - 1] - U[:, r - 2]
    return U


@dataclass
class LocalBasisData:
    p: int
    R_gram: np.ndarray       # Orthonormalization factor from U_r to L2([-1,1]) basis phi_r
    Q: np.ndarray            # (2p, 2p) orthogonal two-scale matrix [Q_low; Q_high]
    x_nodes: np.ndarray      # (p,) interior Gauss-Chebyshev nodes in (-1, 1)
    E: np.ndarray            # (p, p) evaluation matrix E[q, r] = phi_r(x_q)
    w_ref: np.ndarray        # (p,) exact polynomial quadrature weights on [-1, 1]
    I_ref: np.ndarray        # (p, p) exact local antiderivative matrix at x_nodes


def build_local_basis(p: int) -> LocalBasisData:
    """
    Construct L2([-1, 1])-orthonormal basis spanning {U_0, ..., U_{p-1}},
    its exact orthogonal two-scale multiwavelet refinement matrix Q,
    and local Gauss-Chebyshev collocation/integration operators.
    """
    xg, wg = np.polynomial.legendre.leggauss(2 * p)
    U_g = eval_chebyshev_U(xg, p)

    W_sqrt = np.sqrt(wg)[:, None]
    Q_qr, R_qr = np.linalg.qr(W_sqrt * U_g)
    if (Q_qr[0, 0] / W_sqrt[0, 0]) < 0:
        Q_qr[:, 0] *= -1.0
        R_qr[0, :] *= -1.0
    R_inv = np.linalg.inv(R_qr)

    def eval_phi(x: np.ndarray) -> np.ndarray:
        return eval_chebyshev_U(x, p) @ R_inv

    Phi_u = eval_phi(xg)
    Phi_left = eval_phi(0.5 * (xg - 1.0))
    Phi_right = eval_phi(0.5 * (xg + 1.0))

    L_blk = (1.0 / np.sqrt(2.0)) * (Phi_left.T @ (wg[:, None] * Phi_u))
    R_blk = (1.0 / np.sqrt(2.0)) * (Phi_right.T @ (wg[:, None] * Phi_u))
    Q_low = np.hstack([L_blk, R_blk])

    _, _, Vt = np.linalg.svd(Q_low, full_matrices=True)
    Q_high = Vt[p:, :]
    Q = np.vstack([Q_low, Q_high])

    q_idx = np.arange(p)
    x_nodes = -np.cos((2.0 * q_idx + 1.0) * np.pi / (2.0 * p))
    E = eval_phi(x_nodes)
    E_inv = np.linalg.inv(E)

    int_phi_full = wg @ Phi_u
    w_ref = int_phi_full @ E_inv

    Phi_int = np.zeros((p, p), dtype=float)
    for i, xi in enumerate(x_nodes):
        half_len = 0.5 * (xi + 1.0)
        mid = 0.5 * (xi - 1.0)
        x_sub = mid + half_len * xg
        w_sub = half_len * wg
        Phi_int[i, :] = w_sub @ eval_phi(x_sub)

    I_ref = Phi_int @ E_inv

    return LocalBasisData(
        p=p, R_gram=R_inv, Q=Q, x_nodes=x_nodes, E=E, w_ref=w_ref, I_ref=I_ref
    )


def fwt_synthesis(c: np.ndarray, k: int, basis: LocalBasisData) -> np.ndarray:
    """Map multi-level wavelet coefficients c (N,) -> fine coefficients (2^(k-1), p) in O(p N)."""
    p = basis.p
    Q = basis.Q
    curr = c[:p].reshape(1, p)
    offset = p
    for j in range(1, k):
        m_blk = 1 << (j - 1)
        detail = c[offset: offset + m_blk * p].reshape(m_blk, p)
        offset += m_blk * p
        combined = np.hstack([curr, detail])
        children = combined @ Q
        curr = children.reshape(2 * m_blk, p)
    return curr


def fwt_analysis(f_fine: np.ndarray, k: int, basis: LocalBasisData) -> np.ndarray:
    """Exact transpose/inverse of fwt_synthesis: fine (2^(k-1), p) -> multi-level c (N,)."""
    p = basis.p
    QT = basis.Q.T
    curr = f_fine
    details: List[np.ndarray] = []
    for j in range(k - 1, 0, -1):
        m_blk = 1 << (j - 1)
        paired = curr.reshape(m_blk, 2 * p)
        parent = paired @ QT
        curr = parent[:, :p]
        details.append(parent[:, p:].ravel())
    parts = [curr.ravel()] + details[::-1]
    return np.concatenate(parts)


# ============================================================================
# 2. Separable Kernel Representation & Matrix-Free Volterra/Ito Operator
# ============================================================================

@dataclass
class SeparableKernel:
    """Represents K(t, s) = sum_{q=1}^R a_q(t) * b_q(s) on collocation grid (M_int, p)."""
    a_factors: List[np.ndarray]
    b_factors: List[np.ndarray]


def make_exponential_kernel(t_grid: np.ndarray, amp: float, lam: float) -> SeparableKernel:
    """Exact rank-1 decomposition of K(t, s) = amp * exp(-lam * (t - s))."""
    a = amp * np.exp(-lam * (t_grid - 0.5))
    b = np.exp(lam * (t_grid - 0.5))
    return SeparableKernel(a_factors=[a], b_factors=[b])


def make_constant_kernel(t_grid: np.ndarray, amp: float) -> SeparableKernel:
    return SeparableKernel(
        a_factors=[np.full_like(t_grid, amp)],
        b_factors=[np.ones_like(t_grid)],
    )


class RetardedSIVIEOperator(LinearOperator):
    """
    Matrix-free O(N) LinearOperator for:
        (K_0 + K_{d,tau} + K_{s,tau}) X = g_phi^delta
    on ML-SKCW coefficients c in R^N, scaled by 1/sqrt(N).
    """

    def __init__(
        self,
        k: int,
        p: int,
        tau: float,
        K0: SeparableKernel,
        K1: SeparableKernel,
        K2: SeparableKernel,
        dW: np.ndarray,
        T_end: float = 1.0,
        basis: Optional[LocalBasisData] = None,
    ):
        self.k = k
        self.p = p
        self.M_int = 1 << (k - 1)
        self.N = self.M_int * p
        self.T_end = T_end
        self.h = T_end / self.M_int
        self.tau = tau
        self.d_shift = int(np.round(tau / self.h))
        self.basis = basis if basis is not None else build_local_basis(p)
        self.K0 = K0
        self.K1 = K1
        self.K2 = K2
        self.dW = dW
        self.scale_node = np.sqrt(2.0 / self.h)
        self.inv_sqrtN = 1.0 / np.sqrt(self.N)

        super().__init__(dtype=float, shape=(self.N, self.N))

    @staticmethod
    def collocation_grid(
        k: int, p: int, T_end: float = 1.0, basis: Optional[LocalBasisData] = None
    ) -> np.ndarray:
        """Return local interior Gauss-Chebyshev nodes of shape (M_int, p) on [0, T_end]."""
        if basis is None:
            basis = build_local_basis(p)
        M_int = 1 << (k - 1)
        h = T_end / M_int
        centers = (np.arange(M_int) + 0.5) * h
        return centers[:, None] + 0.5 * h * basis.x_nodes[None, :]

    def _volterra_forward(self, g: np.ndarray) -> np.ndarray:
        """Compute int_0^{t_{n,q}} g(s) ds in O(N) via local integration + prefix sum."""
        half_h = 0.5 * self.h
        S = half_h * (g @ self.basis.w_ref)
        C = np.empty(self.M_int, dtype=float)
        C[0] = 0.0
        if self.M_int > 1:
            C[1:] = np.cumsum(S[:-1])
        return C[:, None] + half_h * (g @ self.basis.I_ref.T)

    def _volterra_adjoint(self, z: np.ndarray) -> np.ndarray:
        """Exact adjoint of _volterra_forward in O(N) via reverse prefix sum."""
        half_h = 0.5 * self.h
        Z = np.sum(z, axis=1)
        T_rev = np.empty(self.M_int, dtype=float)
        T_rev[-1] = 0.0
        if self.M_int > 1:
            T_rev[:-1] = np.cumsum(Z[:0:-1])[::-1]
        return half_h * (T_rev[:, None] * self.basis.w_ref[None, :]) + half_h * (z @ self.basis.I_ref)

    def _apply_kernel_forward(
        self, x_node: np.ndarray, kernel: SeparableKernel, delayed: bool, stochastic: bool
    ) -> np.ndarray:
        out = np.zeros_like(x_node)
        if delayed and self.d_shift >= self.M_int:
            return out

        if delayed and self.d_shift > 0:
            x_eff = np.zeros_like(x_node)
            x_eff[self.d_shift:, :] = x_node[:-self.d_shift, :]
        else:
            x_eff = x_node

        for a, b in zip(kernel.a_factors, kernel.b_factors):
            g = b * x_eff
            if stochastic:
                g = (self.dW / self.h)[:, None] * g
            out += a * self._volterra_forward(g)
        return out

    def _apply_kernel_adjoint(
        self, z_node: np.ndarray, kernel: SeparableKernel, delayed: bool, stochastic: bool
    ) -> np.ndarray:
        out = np.zeros_like(z_node)
        if delayed and self.d_shift >= self.M_int:
            return out

        for a, b in zip(kernel.a_factors, kernel.b_factors):
            g_adj = self._volterra_adjoint(a * z_node)
            if stochastic:
                g_adj = (self.dW / self.h)[:, None] * g_adj
            contrib = b * g_adj
            if delayed and self.d_shift > 0:
                shifted_back = np.zeros_like(contrib)
                shifted_back[:-self.d_shift, :] = contrib[self.d_shift:, :]
                out += shifted_back
            else:
                out += contrib
        return out

    def eval_basis_at_nodes(self, c: np.ndarray) -> np.ndarray:
        """Evaluate X_N(t) = c^T Psi(t) at local collocation nodes (M_int, p)."""
        f_fine = fwt_synthesis(c, self.k, self.basis)
        return self.scale_node * (f_fine @ self.basis.E.T)

    def _matvec(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=float).ravel()
        x_node = self.eval_basis_at_nodes(c)
        y_node = (
            self._apply_kernel_forward(x_node, self.K0, delayed=False, stochastic=False)
            + self._apply_kernel_forward(x_node, self.K1, delayed=True, stochastic=False)
            + self._apply_kernel_forward(x_node, self.K2, delayed=True, stochastic=True)
        )
        return self.inv_sqrtN * y_node.ravel()

    def _rmatvec(self, u: np.ndarray) -> np.ndarray:
        z_node = self.inv_sqrtN * np.asarray(u, dtype=float).reshape(self.M_int, self.p)
        x_adj = (
            self._apply_kernel_adjoint(z_node, self.K0, delayed=False, stochastic=False)
            + self._apply_kernel_adjoint(z_node, self.K1, delayed=True, stochastic=False)
            + self._apply_kernel_adjoint(z_node, self.K2, delayed=True, stochastic=True)
        )
        f_adj = self.scale_node * (x_adj @ self.basis.E)
        return fwt_analysis(f_adj, self.k, self.basis)

    def assemble_explicit_for_diagnostics(self) -> np.ndarray:
        """Assemble explicit dense matrix ONLY for small-N condition diagnostics (N <= 2048)."""
        if self.N > 2048:
            raise MemoryError("Explicit assembly disabled for N > 2048; use matrix-free mode.")
        A = np.zeros((self.N, self.N), dtype=float)
        e = np.zeros(self.N, dtype=float)
        for ell in range(self.N):
            e[ell] = 1.0
            A[:, ell] = self._matvec(e)
            e[ell] = 0.0
        return A


# ============================================================================
# 3. Fast Vectorized O(N log N) Column Norms & Damped HDP
# ============================================================================

def compute_fast_column_norms(op: RetardedSIVIEOperator) -> np.ndarray:
    """
    Compute exact column 2-norms ||A_w[:, ell]||_2 in vectorized O(N log N)
    without assembling A_w.
    """
    k, p, M_int = op.k, op.p, op.M_int
    norms_sq = np.zeros(op.N, dtype=float)

    a_all: List[np.ndarray] = []
    tail_coeffs_all: List[np.ndarray] = []

    def add_kernel_tails(kernel: SeparableKernel, delayed: bool, stochastic: bool):
        for a, b in zip(kernel.a_factors, kernel.b_factors):
            a_all.append(a.reshape(M_int, p))
            b_eff = b.copy()
            if stochastic:
                b_eff = (op.dW / op.h)[:, None] * b_eff
            weighted_b = (0.5 * op.h) * (b_eff * op.basis.w_ref[None, :])
            if delayed and op.d_shift > 0:
                shifted_back = np.zeros_like(weighted_b)
                if op.d_shift < M_int:
                    shifted_back[:-op.d_shift, :] = weighted_b[op.d_shift:, :]
                weighted_b = shifted_back
            f_adj = op.scale_node * (weighted_b @ op.basis.E)
            tail_coeffs_all.append(fwt_analysis(f_adj, k, op.basis))

    add_kernel_tails(op.K0, delayed=False, stochastic=False)
    add_kernel_tails(op.K1, delayed=True, stochastic=False)
    add_kernel_tails(op.K2, delayed=True, stochastic=True)

    n_k0 = len(op.K0.a_factors)
    R_tot = len(a_all)
    tail_gram = np.zeros((R_tot, R_tot, M_int + 1), dtype=float)
    for q1 in range(R_tot):
        for q2 in range(q1, R_tot):
            blk_sum = np.sum(a_all[q1] * a_all[q2], axis=1)
            rev_cs = np.cumsum(blk_sum[::-1])[::-1]
            tail_gram[q1, q2, :M_int] = rev_cs
            tail_gram[q2, q1, :M_int] = rev_cs

    e = np.zeros(op.N, dtype=float)
    for r in range(p):
        e[r] = 1.0
        norms_sq[r] = np.sum(op._matvec(e) ** 2)
        e[r] = 0.0

    invN = 1.0 / op.N
    offset = p
    for j in range(1, k):
        m_blk = 1 << (j - 1)
        span_fine = M_int // m_blk
        if op.d_shift == 0 or span_fine >= op.d_shift:
            for idx in range(m_blk * p):
                ell = offset + idx
                e[ell] = 1.0
                norms_sq[ell] = np.sum(op._matvec(e) ** 2)
                e[ell] = 0.0
            offset += m_blk * p
            continue

        blk_ids = np.arange(m_blk)
        n_end_arr = (blk_ids + 1) * span_fine
        del_start_arr = np.minimum(blk_ids * span_fine + op.d_shift, M_int)
        del_end_arr = np.minimum(n_end_arr + op.d_shift, M_int)
        n_full = (M_int - op.d_shift) // span_fine
        full_end = op.d_shift + n_full * span_fine

        for r in range(p):
            curr_sub = op.basis.Q[p + r: p + r + 1, :].reshape(2, p)
            for _ in range(j + 1, k):
                n_cur = curr_sub.shape[0]
                comb = np.hstack([curr_sub, np.zeros((n_cur, p))])
                curr_sub = (comb @ op.basis.Q).reshape(2 * n_cur, p)

            f_fine = np.tile(curr_sub, (m_blk, 1))
            x_node = op.scale_node * (f_fine @ op.basis.E.T)

            # 1) Non-delayed K0 local energy on I_{j,n}
            y0_loc = np.zeros_like(x_node)
            for a0, b0 in zip(op.K0.a_factors, op.K0.b_factors):
                g0 = (b0 * x_node).reshape(m_blk, span_fine, p)
                S0 = 0.5 * op.h * (g0 @ op.basis.w_ref)
                C0 = np.zeros_like(S0)
                if span_fine > 1:
                    C0[:, 1:] = np.cumsum(S0[:, :-1], axis=1)
                loc0 = C0[:, :, None] + 0.5 * op.h * (g0 @ op.basis.I_ref.T)
                y0_loc += a0 * loc0.reshape(M_int, p)

            loc0_sq = np.sum((y0_loc * op.inv_sqrtN).reshape(m_blk, span_fine * p) ** 2, axis=1)

            # 2) Delayed K1 + K2 local energy on shifted intervals [n*span_fine + d_shift, ...)
            yd_loc = np.zeros_like(x_node)
            x_del = np.zeros_like(x_node)
            x_del[op.d_shift:, :] = x_node[:-op.d_shift, :]

            for kernel, stoch in [(op.K1, False), (op.K2, True)]:
                for ad, bd in zip(kernel.a_factors, kernel.b_factors):
                    gd = bd * x_del
                    if stoch:
                        gd = (op.dW / op.h)[:, None] * gd

                    if n_full > 0:
                        gd_blk = gd[op.d_shift: full_end, :].reshape(n_full, span_fine, p)
                        Sd = 0.5 * op.h * (gd_blk @ op.basis.w_ref)
                        Cd = np.zeros_like(Sd)
                        if span_fine > 1:
                            Cd[:, 1:] = np.cumsum(Sd[:, :-1], axis=1)
                        loc_d = Cd[:, :, None] + 0.5 * op.h * (gd_blk @ op.basis.I_ref.T)
                        yd_loc[op.d_shift: full_end, :] += (
                            ad[op.d_shift: full_end, :] * loc_d.reshape(n_full * span_fine, p)
                        )

                    if full_end < M_int:
                        len_p = M_int - full_end
                        g_p = gd[full_end: M_int, :]
                        S_p = 0.5 * op.h * (g_p @ op.basis.w_ref)
                        C_p = np.zeros(len_p, dtype=float)
                        if len_p > 1:
                            C_p[1:] = np.cumsum(S_p[:-1])
                        loc_p = C_p[:, None] + 0.5 * op.h * (g_p @ op.basis.I_ref.T)
                        yd_loc[full_end: M_int, :] += ad[full_end: M_int, :] * loc_p

            ell_indices = offset + blk_ids * p + r
            for q0 in range(n_k0):
                tc_q0 = tail_coeffs_all[q0][ell_indices]
                tc_rep = np.repeat(tc_q0, span_fine)[: M_int - op.d_shift]
                yd_loc[op.d_shift:, :] += tc_rep[:, None] * a_all[q0][op.d_shift:, :]

            pure_yd_sq = np.zeros(m_blk, dtype=float)
            if n_full > 0:
                blk_vals = (yd_loc[op.d_shift: full_end, :] * op.inv_sqrtN).reshape(n_full, span_fine * p)
                pure_yd_sq[:n_full] = np.sum(blk_vals ** 2, axis=1)
            if full_end < M_int and n_full < m_blk:
                pure_yd_sq[n_full] = np.sum((yd_loc[full_end: M_int, :] * op.inv_sqrtN) ** 2)

            # 3) Remaining tail energies (vectorized einsum)
            C0_mat = np.column_stack([tail_coeffs_all[q0][ell_indices] for q0 in range(n_k0)])
            G_A = tail_gram[:n_k0, :n_k0, n_end_arr] - tail_gram[:n_k0, :n_k0, del_start_arr]
            segA_sq = invN * np.einsum("ni,ijn,nj->n", C0_mat, G_A, C0_mat)
            segA_sq *= (del_start_arr > n_end_arr)

            Call_mat = np.column_stack([tail_coeffs_all[q][ell_indices] for q in range(R_tot)])
            G_B = tail_gram[:, :, del_end_arr]
            segB_sq = invN * np.einsum("ni,ijn,nj->n", Call_mat, G_B, Call_mat)
            segB_sq *= (del_end_arr < M_int)

            norms_sq[ell_indices] = loc0_sq + pure_yd_sq + segA_sq + segB_sq

        offset += m_blk * p

    return np.sqrt(np.maximum(norms_sq, 0.0))


@dataclass
class DampedHDP:
    d_raw: np.ndarray
    d_damp: np.ndarray
    d_inv: np.ndarray
    lambda0: float


def build_damped_hdp(op: RetardedSIVIEOperator, lambda0: float) -> DampedHDP:
    d_raw = compute_fast_column_norms(op)
    d_damp = np.sqrt(d_raw ** 2 + lambda0 ** 2)
    d_inv = 1.0 / d_damp
    return DampedHDP(d_raw=d_raw, d_damp=d_damp, d_inv=d_inv, lambda0=lambda0)


def make_preconditioned_operator(op: RetardedSIVIEOperator, hdp: DampedHDP) -> LinearOperator:
    """Return A_tilde = A_w @ D_H^{-1} as a matrix-free LinearOperator."""
    d_inv = hdp.d_inv
    return LinearOperator(
        dtype=float,
        shape=op.shape,
        matvec=lambda y: op.matvec(d_inv * y),
        rmatvec=lambda u: d_inv * op.rmatvec(u),
    )


# ============================================================================
# 4. Paige-Saunders LSQR with Spectral Stop Sign (SSS)
# ============================================================================

@dataclass
class LSQRResult:
    c_sol: np.ndarray
    m_stop: int
    omega_history: np.ndarray
    c_history: Optional[List[np.ndarray]] = None


def solve_plsqr_sss(
    A_tilde: LinearOperator,
    b_w: np.ndarray,
    d_inv: np.ndarray,
    delta_prior: float,
    tau_safe: float = 1.15,
    eps_s: float = 1e-3,
    m_min: int = 2,
    m_max: int = 80,
    use_sss: bool = True,
    record_history: bool = False,
) -> LSQRResult:
    """Execute Algorithm 1: Paige-Saunders LSQR with O(1) residual monitoring and SSS."""
    N = len(b_w)
    beta = float(np.linalg.norm(b_w))
    u = b_w / beta if beta > 0.0 else np.zeros(N, dtype=float)
    q0 = A_tilde.rmatvec(u)
    alpha = float(np.linalg.norm(q0))
    v = q0 / alpha if alpha > 0.0 else np.zeros(N, dtype=float)

    w = v.copy()
    y = np.zeros(N, dtype=float)
    phi_bar = beta
    rho_bar = alpha

    omega_hist: List[float] = [phi_bar]
    c_hist: Optional[List[np.ndarray]] = [] if record_history else None
    m_stop = m_max

    for m in range(1, m_max + 1):
        p_vec = A_tilde.matvec(v) - alpha * u
        beta = float(np.linalg.norm(p_vec))
        u = p_vec / beta if beta > 1e-15 else np.zeros(N, dtype=float)

        q_vec = A_tilde.rmatvec(u) - beta * v
        alpha = float(np.linalg.norm(q_vec))
        v = q_vec / alpha if alpha > 1e-15 else np.zeros(N, dtype=float)

        rho = float(np.hypot(rho_bar, beta))
        if rho < 1e-15:
            m_stop = m
            break
        c_giv = rho_bar / rho
        s_giv = beta / rho
        theta = s_giv * alpha
        rho_bar = -c_giv * alpha
        phi = c_giv * phi_bar
        phi_bar = s_giv * phi_bar

        y = y + (phi / rho) * w
        w = v - (theta / rho) * w

        omega = phi_bar
        omega_hist.append(omega)

        if record_history and c_hist is not None:
            c_hist.append(d_inv * y)

        if use_sss and m >= m_min:
            omega_prev = omega_hist[-2]
            rel_drop = abs(omega - omega_prev) / max(omega_prev, 1e-15)
            cond_disc = omega <= tau_safe * delta_prior
            cond_stag = (omega <= 2.0 * tau_safe * delta_prior) and (rel_drop <= eps_s)
            if cond_disc or cond_stag:
                m_stop = m
                break

    return LSQRResult(
        c_sol=d_inv * y,
        m_stop=m_stop,
        omega_history=np.asarray(omega_hist),
        c_history=c_hist,
    )


# ============================================================================
# 5. Self-Tests (Unitarity, Adjoint, and Fast Column Norm Verification)
# ============================================================================

def run_self_tests() -> None:
    print("Running mathematical verification tests...")
    rng = np.random.default_rng(123)

    k, p = 6, 4  # N = 128
    basis = build_local_basis(p)
    N = (1 << (k - 1)) * p
    c_rand = rng.standard_normal(N)
    f_syn = fwt_synthesis(c_rand, k, basis)
    c_rec = fwt_analysis(f_syn, k, basis)
    err_fwt = np.linalg.norm(c_rand - c_rec) / np.linalg.norm(c_rand)
    err_norm = abs(np.linalg.norm(f_syn) - np.linalg.norm(c_rand)) / np.linalg.norm(c_rand)
    assert err_fwt < 1e-13 and err_norm < 1e-13, f"FWT test failed: {err_fwt}, {err_norm}"
    print(f"  [PASS] Multi-level SKCW FWT unitary check: err = {err_fwt:.2e}")

    t_grid = RetardedSIVIEOperator.collocation_grid(k, p, basis=basis)
    K0 = make_exponential_kernel(t_grid, amp=1.0, lam=2.0)
    K1 = make_exponential_kernel(t_grid, amp=0.5, lam=1.0)
    K2 = make_constant_kernel(t_grid, amp=0.1)
    dW = rng.normal(0.0, np.sqrt(1.0 / (1 << (k - 1))), size=(1 << (k - 1)))
    op = RetardedSIVIEOperator(k, p, tau=0.125, K0=K0, K1=K1, K2=K2, dW=dW, basis=basis)

    v = rng.standard_normal(N)
    u = rng.standard_normal(N)
    lhs = float(np.dot(op.matvec(v), u))
    rhs = float(np.dot(v, op.rmatvec(u)))
    err_adj = abs(lhs - rhs) / max(abs(lhs), 1.0)
    assert err_adj < 1e-13, f"Adjoint test failed: {err_adj}"
    print(f"  [PASS] Matrix-free Adjoint test <Av,u>==<v,A^Tu>: rel_err = {err_adj:.2e}")

    A_exp = op.assemble_explicit_for_diagnostics()
    d_exact = np.linalg.norm(A_exp, axis=0)
    d_fast = compute_fast_column_norms(op)
    err_cols = np.max(np.abs(d_exact - d_fast)) / np.max(d_exact)
    assert err_cols < 1e-12, f"Fast column norm mismatch: {err_cols}"
    print(f"  [PASS] Vectorized O(N log N) Column norm vs Explicit A: rel_err = {err_cols:.2e}")


if __name__ == "__main__":
    run_self_tests()