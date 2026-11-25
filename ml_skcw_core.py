"""
ml_skcw_core.py
Matrix-Free Multi-Level L2-Orthonormalized Chebyshev-Seeded Multiwavelet (ML-SKCW) Solver
for Inverse Retarded Stochastic Ito-Volterra Integral Equations.

Implements:
1. Exact L2-orthonormal Alpert-type Chebyshev-seeded multiwavelet FWT in O(p N).
2. Local interior Gauss-Chebyshev (type-I) collocation + O(R N) prefix-sum Volterra & Ito action.
3. Chebyshev separable expansion for non-separable smooth kernels (R > 1).
4. Exact O(R^2 N) row-norm equilibration D_L^{-1} and dyadic Sobolev scaling D_S^{-1}.
5. O(N) tridiagonal H^1 (s=1) preconditioner baseline via banded Cholesky.
6. Paige-Saunders LSQR with Discrepancy-Based Stopping Criterion (DSC) and Hybrid Golub-Kahan (DP/GCV).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky_banded, solve_banded
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse.linalg import LinearOperator
from dataclasses import dataclass
from typing import Callable, List, Optional


# ============================================================================
# 1. Local SKCW Polynomials & Two-Scale Multi-Level Orthogonal Transform
# ============================================================================

def eval_chebyshev_U(x: np.ndarray, p: int) -> np.ndarray:
    """Evaluate Second-Kind Chebyshev polynomials U_0,...,U_{p-1} at x in [-1, 1]."""
    x_flat = np.asarray(x, dtype=float).ravel()
    U = np.zeros((x_flat.size, p), dtype=float)
    if p >= 1:
        U[:, 0] = 1.0
    if p >= 2:
        U[:, 1] = 2.0 * x_flat
    for r in range(2, p):
        U[:, r] = 2.0 * x_flat * U[:, r - 1] - U[:, r - 2]
    return U


def eval_chebyshev_T(x: np.ndarray, R: int) -> np.ndarray:
    """Evaluate First-Kind Chebyshev polynomials T_0,...,T_{R-1} at x in [-1, 1]."""
    x_flat = np.asarray(x, dtype=float).ravel()
    T = np.zeros((x_flat.size, R), dtype=float)
    if R >= 1:
        T[:, 0] = 1.0
    if R >= 2:
        T[:, 1] = x_flat
    for q in range(2, R):
        T[:, q] = 2.0 * x_flat * T[:, q - 1] - T[:, q - 2]
    return T


@dataclass
class LocalBasisData:
    p: int
    R_gram: np.ndarray
    Q: np.ndarray
    x_nodes: np.ndarray
    E: np.ndarray
    E_inv: np.ndarray
    E_inv_T: np.ndarray
    w_ref: np.ndarray
    I_ref: np.ndarray


def build_local_basis(p: int) -> LocalBasisData:
    """
    Construct L2([-1, 1])-orthonormal basis spanning {U_0, ..., U_{p-1}},
    its exact orthogonal two-scale multiwavelet refinement matrix Q,
    and local Gauss-Chebyshev (type-I) collocation/integration operators.
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
    E_inv_T = E_inv.T.copy()

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
        p=p,
        R_gram=R_inv,
        Q=Q,
        x_nodes=x_nodes,
        E=E,
        E_inv=E_inv,
        E_inv_T=E_inv_T,
        w_ref=w_ref,
        I_ref=I_ref,
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
# 2. Separable Kernels & Matrix-Free Volterra/Ito Operator
# ============================================================================

@dataclass
class SeparableKernel:
    """Represents K(t, s) = sum_{q=1}^R a_q(t) * b_q(s) on collocation grid (M_int, p)."""
    a_factors: List[np.ndarray]
    b_factors: List[np.ndarray]


def chebyshev_separable_kernel(
    kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t_grid: np.ndarray,
    R: int,
    T_end: float = 1.0,
) -> SeparableKernel:
    """
    Construct an R-term Chebyshev interpolation expansion in s:
      K^{(R)}(t, s) = sum_{q=0}^{R-1} a_q(t) T_q(2s/T_end - 1)
    on collocation grid t_grid of shape (M_int, p).
    """
    m = np.arange(R)
    x_m = -np.cos((2.0 * m + 1.0) * np.pi / (2.0 * R))
    s_m = 0.5 * T_end * (x_m + 1.0)

    T_nodes = eval_chebyshev_T(x_m, R)  # (R, R)
    K_tm = kernel_fn(t_grid[..., None], s_m[None, None, :])  # (M_int, p, R)

    x_grid = (2.0 / T_end) * t_grid - 1.0
    T_grid = eval_chebyshev_T(x_grid, R).reshape(t_grid.shape[0], t_grid.shape[1], R)

    a_factors: List[np.ndarray] = []
    b_factors: List[np.ndarray] = []
    for q in range(R):
        scale_q = (1.0 / R) if q == 0 else (2.0 / R)
        a_q = scale_q * np.sum(K_tm * T_nodes[None, None, :, q], axis=2)
        b_q = T_grid[:, :, q].copy()
        a_factors.append(a_q)
        b_factors.append(b_q)
    return SeparableKernel(a_factors=a_factors, b_factors=b_factors)


class RetardedSIVIEOperator(LinearOperator):
    """
    Matrix-free O(R N) LinearOperator for:
        A_w c = (1/sqrt(N)) (K_0 + K_{d,tau} + K_{s,tau}) X
    on ML-SKCW coefficients c in R^N.
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
        """Return local interior Gauss-Chebyshev (type-I) nodes of shape (M_int, p) on [0, T_end]."""
        if basis is None:
            basis = build_local_basis(p)
        M_int = 1 << (k - 1)
        h = T_end / M_int
        centers = (np.arange(M_int) + 0.5) * h
        return centers[:, None] + 0.5 * h * basis.x_nodes[None, :]

    def _volterra_forward(self, g: np.ndarray) -> np.ndarray:
        half_h = 0.5 * self.h
        S = half_h * (g @ self.basis.w_ref)
        C = np.empty(self.M_int, dtype=float)
        C[0] = 0.0
        if self.M_int > 1:
            C[1:] = np.cumsum(S[:-1])
        return C[:, None] + half_h * (g @ self.basis.I_ref.T)

    def _volterra_adjoint(self, z: np.ndarray) -> np.ndarray:
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

    def eval_history_response(self, phi_vals_at_grid: np.ndarray) -> np.ndarray:
        """
        Evaluate the solver-grid history contribution (over 0 <= s < tau_h) for K1 and K2:
            sum_{u < d_shift} (K1 + K2 dW) phi(t_u - tau_h), scaled by 1/sqrt(N).
        phi_vals_at_grid must hold phi(t_u - tau) (or phi(t_u - tau_h)) of shape (M_int, p)
        (only rows < d_shift are used).
        """
        out = np.zeros((self.M_int, self.p), dtype=float)
        if self.d_shift <= 0:
            return out.ravel()
        d_lim = min(self.d_shift, self.M_int)
        x_hist = np.zeros((self.M_int, self.p), dtype=float)
        x_hist[:d_lim, :] = phi_vals_at_grid[:d_lim, :]
        for kernel, stoch in [(self.K1, False), (self.K2, True)]:
            for a, b in zip(kernel.a_factors, kernel.b_factors):
                g = b * x_hist
                if stoch:
                    g = (self.dW / self.h)[:, None] * g
                out += a * self._volterra_forward(g)
        return self.inv_sqrtN * out.ravel()

    def eval_basis_at_nodes(self, c: np.ndarray) -> np.ndarray:
        """Evaluate X_N(t) = c^T Psi(t) at local collocation nodes (M_int, p)."""
        f_fine = fwt_synthesis(c, self.k, self.basis)
        return self.scale_node * (f_fine @ self.basis.E.T)

    def nodes_to_coeffs(self, x_node: np.ndarray) -> np.ndarray:
        """Inverse of eval_basis_at_nodes: nodal values (N,) -> wavelet coefficients (N,)."""
        x_blk = np.asarray(x_node, dtype=float).reshape(self.M_int, self.p)
        f_fine = (x_blk / self.scale_node) @ self.basis.E_inv_T
        return fwt_analysis(f_fine, self.k, self.basis)

    def nodes_to_coeffs_adjoint(self, r_c: np.ndarray) -> np.ndarray:
        """Exact transpose of nodes_to_coeffs: wavelet vector (N,) -> nodal vector (N,)."""
        f_syn = fwt_synthesis(np.asarray(r_c, dtype=float), self.k, self.basis)
        x_adj = (f_syn @ self.basis.E_inv) / self.scale_node
        return x_adj.ravel()

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
# 3. Exact O(R^2 N) Row Equilibration & Two-Sided HDP
# ============================================================================

def compute_fast_row_norms(op: RetardedSIVIEOperator) -> np.ndarray:
    """
    Compute exact Euclidean row norms ||A_w[i, :]||_2 for all i=0..N-1
    in O(R^2 N) without assembling A_w (using unitarity of multi-level FWT).
    """
    M_int, p, h = op.M_int, op.p, op.h
    d_shift = op.d_shift
    scale = 0.5 * h * op.scale_node

    a_list = []
    u_list = []
    L_loc_list = []

    n_k0 = len(op.K0.a_factors)
    for a0, b0 in zip(op.K0.a_factors, op.K0.b_factors):
        a_list.append(a0.reshape(M_int, p))
        u0 = scale * ((b0 * op.basis.w_ref[None, :]) @ op.basis.E)
        u_list.append(u0)
        L0 = scale * np.einsum("nj,qj,jl->nql", b0, op.basis.I_ref, op.basis.E)
        L_loc_list.append(a0[:, :, None] * L0)

    for kernel, stoch in [(op.K1, False), (op.K2, True)]:
        for ad, bd in zip(kernel.a_factors, kernel.b_factors):
            a_list.append(ad.reshape(M_int, p))
            bd_eff = (op.dW / h)[:, None] * bd if stoch else bd
            ud = np.zeros((M_int, p), dtype=float)
            Ld = np.zeros((M_int, p, p), dtype=float)
            if d_shift == 0:
                ud = scale * ((bd_eff * op.basis.w_ref[None, :]) @ op.basis.E)
                L_tmp = scale * np.einsum("nj,qj,jl->nql", bd_eff, op.basis.I_ref, op.basis.E)
                Ld = ad[:, :, None] * L_tmp
            elif d_shift < M_int:
                ud[:-d_shift, :] = scale * ((bd_eff[d_shift:, :] * op.basis.w_ref[None, :]) @ op.basis.E)
                L_tmp = scale * np.einsum("nj,qj,jl->nql", bd_eff[d_shift:, :], op.basis.I_ref, op.basis.E)
                Ld[d_shift:, :, :] = ad[d_shift:, :, None] * L_tmp
            u_list.append(ud)
            L_loc_list.append(Ld)

    R_tot = len(a_list)
    G_cum_all = np.zeros((R_tot, R_tot, M_int + 1), dtype=float)
    for r1 in range(R_tot):
        for r2 in range(r1, R_tot):
            dots = np.sum(u_list[r1] * u_list[r2], axis=1)
            cs = np.zeros(M_int + 1, dtype=float)
            cs[1:] = np.cumsum(dots)
            G_cum_all[r1, r2, :] = cs
            G_cum_all[r2, r1, :] = cs

    a_arr = np.stack(a_list, axis=0)
    row_sq = np.zeros((M_int, p), dtype=float)

    if d_shift == 0:
        n_idx = np.arange(M_int)
        tail_sq = np.einsum("rnq,rsn,snq->nq", a_arr, G_cum_all[:, :, n_idx], a_arr)
        loc_vec = sum(L_loc_list)
        row_sq = np.maximum(tail_sq, 0.0) + np.sum(loc_vec ** 2, axis=2)
    else:
        n_idx = np.arange(M_int)
        L0_sum = sum(L_loc_list[:n_k0])
        row_sq += np.sum(L0_sum ** 2, axis=2)

        if d_shift < M_int:
            Ld_sum = sum(L_loc_list[n_k0:])[d_shift:, :, :]
            u0_stack = np.stack(u_list[:n_k0], axis=0)[:, :-d_shift, :]
            a0_slice = a_arr[:n_k0, d_shift:, :]
            k0_tail_vec = np.einsum("rnq,rnl->nql", a0_slice, u0_stack)
            row_sq[d_shift:, :] += np.sum((Ld_sum + k0_tail_vec) ** 2, axis=2)

        v_low = np.maximum(0, n_idx - d_shift + 1)
        G_mid = G_cum_all[:n_k0, :n_k0, n_idx] - G_cum_all[:n_k0, :n_k0, v_low]
        mid_sq = np.einsum("rnq,rsn,snq->nq", a_arr[:n_k0], G_mid, a_arr[:n_k0])
        row_sq += np.maximum(mid_sq, 0.0)

        if d_shift + 1 < M_int:
            n_hi = n_idx[d_shift + 1:] - d_shift
            G_hi = G_cum_all[:, :, n_hi]
            hi_sq = np.einsum("rnq,rsn,snq->nq", a_arr[:, d_shift + 1:, :], G_hi, a_arr[:, d_shift + 1:, :])
            row_sq[d_shift + 1:, :] += np.maximum(hi_sq, 0.0)

    return np.sqrt(np.maximum(row_sq.ravel() / op.N, 0.0))


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


@dataclass
class TwoSidedHDP:
    w_left: np.ndarray
    d_s_inv: np.ndarray


def build_two_sided_hdp(
    op: RetardedSIVIEOperator, s: float, floor_ratio: float = 1e-2
) -> TwoSidedHDP:
    """
    Construct O(R^2 N) Two-Sided HDP:
      - Left row equilibration w_left from exact matrix-free row norms (mean(w^2)=1)
      - Right dyadic Sobolev weights d_s_inv = 2^{-j s}
    """
    r_norms = compute_fast_row_norms(op)
    floor = floor_ratio * float(np.max(r_norms))
    w_raw = 1.0 / np.sqrt(r_norms ** 2 + floor ** 2)
    w_left = w_raw / np.sqrt(np.mean(w_raw ** 2))
    d_s_inv = build_sobolev_weights(op.k, op.p, s=s)
    return TwoSidedHDP(w_left=w_left, d_s_inv=d_s_inv)


def make_two_sided_operator(op: RetardedSIVIEOperator, hdp: TwoSidedHDP) -> LinearOperator:
    w_left = hdp.w_left
    d_s_inv = hdp.d_s_inv
    return LinearOperator(
        dtype=float,
        shape=op.shape,
        matvec=lambda y: w_left * op.matvec(d_s_inv * y),
        rmatvec=lambda u: d_s_inv * op.rmatvec(w_left * u),
    )


# ============================================================================
# 4. O(N) Tridiagonal H^1 (s = 1) Right Preconditioner Baseline
# ============================================================================

@dataclass
class H1Preconditioner:
    U_band: np.ndarray
    L_band: np.ndarray
    op: RetardedSIVIEOperator
    w_left: np.ndarray

    def y_to_c(self, y: np.ndarray) -> np.ndarray:
        x_node = solve_banded((0, 1), self.U_band, y)
        return self.op.nodes_to_coeffs(x_node)

    def c_adj_to_y(self, r_c: np.ndarray) -> np.ndarray:
        x_adj = self.op.nodes_to_coeffs_adjoint(r_c)
        return solve_banded((1, 0), self.L_band, x_adj)


def build_h1_preconditioner(
    op: RetardedSIVIEOperator,
    t_grid: np.ndarray,
    w_left: np.ndarray,
    alpha: float = 0.05,
) -> H1Preconditioner:
    """
    Build O(N) SPD tridiagonal H^1 metric G = W_q + alpha^2 D_t^T W_{dt} D_t = U^T U
    on collocation nodes t_grid, returning banded Cholesky factors for right preconditioning.
    """
    t_flat = t_grid.ravel()
    N = t_flat.size
    dt = np.diff(t_flat)
    w_q = np.empty(N, dtype=float)
    w_q[0] = 0.5 * dt[0] + t_flat[0]
    w_q[-1] = 0.5 * dt[-1] + (op.T_end - t_flat[-1])
    w_q[1:-1] = 0.5 * (dt[:-1] + dt[1:])

    inv_dt = (alpha ** 2) / dt
    diag = w_q.copy()
    diag[:-1] += inv_dt
    diag[1:] += inv_dt
    off = -inv_dt

    ab = np.zeros((2, N), dtype=float)
    ab[0, 1:] = off
    ab[1, :] = diag
    U_band = cholesky_banded(ab, lower=False)

    L_band = np.zeros((2, N), dtype=float)
    L_band[0, :] = U_band[1, :]
    L_band[1, :-1] = U_band[0, 1:]
    return H1Preconditioner(U_band=U_band, L_band=L_band, op=op, w_left=w_left)


def make_h1_operator(h1_prec: H1Preconditioner) -> LinearOperator:
    op = h1_prec.op
    w_left = h1_prec.w_left
    return LinearOperator(
        dtype=float,
        shape=op.shape,
        matvec=lambda y: w_left * op.matvec(h1_prec.y_to_c(y)),
        rmatvec=lambda u: h1_prec.c_adj_to_y(op.rmatvec(w_left * u)),
    )


# ============================================================================
# 5. Paige-Saunders LSQR with DSC & Hybrid Golub-Kahan (DP / GCV)
# ============================================================================

@dataclass
class LSQRResult:
    c_sol: np.ndarray
    m_stop: int
    omega_history: np.ndarray
    c_history: Optional[List[np.ndarray]] = None
    lam_opt: float = 0.0


def solve_plsqr_dsc(
    A_tilde: LinearOperator,
    b_L: np.ndarray,
    d_s_inv: np.ndarray,
    delta_prior: float,
    tau_safe: float = 1.10,
    eps_s: float = 1e-4,
    m_min: int = 2,
    m_max: int = 200,
    use_dsc: bool = True,
    record_history: bool = False,
    y_to_c_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> LSQRResult:
    """Paige-Saunders LSQR with O(1) residual monitoring and Discrepancy-Based Stopping (DSC)."""
    N = len(b_L)
    beta = float(np.linalg.norm(b_L))
    u = b_L / beta if beta > 0.0 else np.zeros(N, dtype=float)
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
    map_c = y_to_c_fn if y_to_c_fn is not None else (lambda vec: d_s_inv * vec)

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
            c_hist.append(map_c(y))

        if use_dsc and m >= m_min:
            omega_prev = omega_hist[-2]
            rel_drop = abs(omega - omega_prev) / max(omega_prev, 1e-15)
            cond_disc = omega <= tau_safe * delta_prior
            cond_stag = (omega <= 2.0 * tau_safe * delta_prior) and (rel_drop <= eps_s)
            if cond_disc or cond_stag:
                m_stop = m
                break

    return LSQRResult(
        c_sol=map_c(y),
        m_stop=m_stop,
        omega_history=np.asarray(omega_hist),
        c_history=c_hist,
    )


def solve_hybrid_gk(
    A_tilde: LinearOperator,
    b_L: np.ndarray,
    d_s_inv: np.ndarray,
    mode: str = "gcv",
    delta_prior: float = 0.0,
    tau_safe: float = 1.10,
    m_min: int = 4,
    m_max: int = 80,
) -> LSQRResult:
    """
    Hybrid Golub-Kahan (HGK) solver: applies Golub-Kahan bidiagonalization to A_tilde
    and solves projected Tikhonov subproblems with automatic parameter selection:
      - mode="dp": Morozov discrepancy principle on projected residual
      - mode="gcv": Generalized Cross-Validation on projected bidiagonal system
    """
    N = len(b_L)
    beta1 = float(np.linalg.norm(b_L))
    if beta1 == 0.0:
        return LSQRResult(c_sol=np.zeros(N), m_stop=0, omega_history=np.array([0.0]))

    u = b_L / beta1
    q0 = A_tilde.rmatvec(u)
    alpha = float(np.linalg.norm(q0))
    v = q0 / alpha if alpha > 0.0 else np.zeros(N, dtype=float)

    V_basis: List[np.ndarray] = [v]
    alphas: List[float] = [alpha]
    betas: List[float] = []
    omega_hist: List[float] = [beta1]

    lam_prev = None
    lam_opt = 0.0
    z_best = np.zeros(1)
    m_stop = m_max

    for m in range(1, m_max + 1):
        p_vec = A_tilde.matvec(V_basis[-1]) - alphas[-1] * u
        beta_next = float(np.linalg.norm(p_vec))
        betas.append(beta_next)
        u = p_vec / beta_next if beta_next > 1e-15 else np.zeros(N, dtype=float)

        q_vec = A_tilde.rmatvec(u) - beta_next * V_basis[-1]
        alpha_next = float(np.linalg.norm(q_vec))

        B_m = np.zeros((m + 1, m), dtype=float)
        for j in range(m):
            B_m[j, j] = alphas[j]
            B_m[j + 1, j] = betas[j]

        U_b, s_b, Vt_b = np.linalg.svd(B_m, full_matrices=True)
        g_proj = beta1 * U_b[0, :]  # length m + 1
        tail_sq = float(g_proj[m] ** 2)
        omega_hist.append(float(np.sqrt(tail_sq)))

        def res_sq_fn(lam: float) -> float:
            f_rem = (lam ** 2) / (s_b ** 2 + lam ** 2)
            return float(np.sum((f_rem * g_proj[:m]) ** 2) + tail_sq)

        if mode == "dp":
            target = tau_safe * delta_prior
            res0 = np.sqrt(res_sq_fn(0.0))
            if res0 <= target:
                lam_opt = float(
                    brentq(
                        lambda l: np.sqrt(res_sq_fn(l)) - target,
                        0.0,
                        max(1e3 * s_b[0], 1.0),
                    )
                )
            else:
                lam_opt = 0.0
        elif mode == "gcv":
            lo = np.log10(max(s_b[-1] * 1e-3, 1e-12 * s_b[0]))
            hi = np.log10(max(10.0 * s_b[0], 1e-6))

            def gcv_obj(log_lam: float) -> float:
                lam = 10.0 ** log_lam
                df = float(np.sum((s_b ** 2) / (s_b ** 2 + lam ** 2)))
                denom = max((m + 1) - df, 1e-12)
                return ((m + 1) * res_sq_fn(lam)) / (denom ** 2)

            opt = minimize_scalar(gcv_obj, bounds=(lo, hi), method="bounded")
            lam_opt = float(10.0 ** opt.x)
        else:
            raise ValueError(f"Unknown Hybrid GK mode: {mode}")

        filt = s_b / (s_b ** 2 + lam_opt ** 2)
        z_best = Vt_b.T @ (filt * g_proj[:m])

        stop_now = False
        if m >= m_min:
            if mode == "dp" and np.sqrt(res_sq_fn(0.0)) <= tau_safe * delta_prior:
                if lam_prev is not None and lam_opt > 0.0:
                    if abs(lam_opt - lam_prev) / max(lam_opt, 1e-15) < 1e-2:
                        stop_now = True
            elif mode == "gcv":
                if lam_prev is not None and abs(lam_opt - lam_prev) / max(lam_opt, 1e-15) < 5e-3:
                    stop_now = True

        lam_prev = lam_opt
        if stop_now or alpha_next < 1e-15:
            m_stop = m
            break

        if m < m_max:
            alphas.append(alpha_next)
            V_basis.append(q_vec / alpha_next)

    V_mat = np.column_stack(V_basis[: z_best.size])
    y_sol = V_mat @ z_best
    return LSQRResult(
        c_sol=d_s_inv * y_sol,
        m_stop=m_stop,
        omega_history=np.asarray(omega_hist),
        lam_opt=lam_opt,
    )


# ============================================================================
# 6. Self-Tests (Unitarity, Adjoint, Row Norms, Chebyshev Kernel, H^1 Adjoint)
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
    assert err_fwt < 1e-13, f"FWT test failed: {err_fwt}"
    print(f"  [PASS] Multi-level FWT unitary check: err = {err_fwt:.2e}")

    t_grid = RetardedSIVIEOperator.collocation_grid(k, p, basis=basis)
    k0_fn = lambda t, s: (2.0 + np.sin(2.0 * np.pi * t)) * np.exp(-2.0 * (t - s) ** 2)
    K0 = chebyshev_separable_kernel(k0_fn, t_grid, R=18, T_end=1.0)
    K0_exact = k0_fn(t_grid.ravel()[:, None], t_grid.ravel()[None, :])
    K0_approx = sum(
        np.outer(a.ravel(), b.ravel()) for a, b in zip(K0.a_factors, K0.b_factors)
    )
    err_sep = np.max(np.abs(K0_exact - K0_approx))
    assert err_sep < 1e-10, f"Chebyshev separable expansion failed: {err_sep}"
    print(f"  [PASS] Non-separable Gaussian kernel (R=18): max err = {err_sep:.2e}")

    a1 = 1.2 * np.exp(-1.5 * t_grid) * np.exp(-1.0 * (t_grid - 0.5))
    b1 = np.exp(1.0 * (t_grid - 0.5))
    a2 = 0.25 * (1.0 + 0.5 * np.sin(2.0 * np.pi * t_grid))
    K1 = SeparableKernel([a1], [b1])
    K2 = SeparableKernel([a2], [np.ones_like(t_grid)])
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
    r_exact = np.linalg.norm(A_exp, axis=1)
    r_fast = compute_fast_row_norms(op)
    err_rows = np.max(np.abs(r_exact - r_fast)) / np.max(r_exact)
    assert err_rows < 1e-13, f"Fast row norm mismatch: {err_rows}"
    print(f"  [PASS] O(R^2 N) Row norm (R_tot=20) vs Explicit A: rel_err = {err_rows:.2e}")

    hdp = build_two_sided_hdp(op, s=0.6)
    h1_prec = build_h1_preconditioner(op, t_grid, hdp.w_left, alpha=0.05)
    A_h1 = make_h1_operator(h1_prec)
    lhs_h1 = float(np.dot(A_h1.matvec(v), u))
    rhs_h1 = float(np.dot(v, A_h1.rmatvec(u)))
    err_h1 = abs(lhs_h1 - rhs_h1) / max(abs(lhs_h1), 1.0)
    assert err_h1 < 1e-13, f"H^1 adjoint test failed: {err_h1}"
    print(f"  [PASS] Tridiagonal H^1 preconditioner adjoint check: rel_err = {err_h1:.2e}")


if __name__ == "__main__":
    run_self_tests()