"""
ml_skcw_core.py
Matrix-Free Multi-Level Second-Kind Chebyshev Wavelet (ML-SKCW) Solver
for Inverse Retarded Stochastic Ito-Volterra Integral Equations.

Implements:
1. Exact L2-orthonormal multi-level SKCW two-scale transforms in O(p N).
2. Local interior Gauss-Chebyshev collocation + O(N) prefix-sum Volterra & Ito action.
3. Exact adjoint operator matching <A v, u> == <v, A^T u> to machine precision.
4. Exact O(R^2 N) row-norm equilibration D_L^{-1} and dyadic Sobolev scaling D_S^{-1}.
5. Paige-Saunders LSQR with Discrepancy-Based Stopping Criterion (DSC).
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
    R_gram: np.ndarray
    Q: np.ndarray
    x_nodes: np.ndarray
    E: np.ndarray
    w_ref: np.ndarray
    I_ref: np.ndarray


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
# 2. Separable Kernels & Matrix-Free Volterra/Ito Operator
# ============================================================================

@dataclass
class SeparableKernel:
    """Represents K(t, s) = sum_{q=1}^R a_q(t) * b_q(s) on collocation grid (M_int, p)."""
    a_factors: List[np.ndarray]
    b_factors: List[np.ndarray]


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
      - Left row equilibration w_left from exact matrix-free row norms (normalized so mean(w^2)=1)
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
# 4. Paige-Saunders LSQR with Discrepancy-Based Stopping Criterion (DSC)
# ============================================================================

@dataclass
class LSQRResult:
    c_sol: np.ndarray
    m_stop: int
    omega_history: np.ndarray
    c_history: Optional[List[np.ndarray]] = None


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
            c_hist.append(d_s_inv * y)

        if use_dsc and m >= m_min:
            omega_prev = omega_hist[-2]
            rel_drop = abs(omega - omega_prev) / max(omega_prev, 1e-15)
            cond_disc = omega <= tau_safe * delta_prior
            cond_stag = (omega <= 2.0 * tau_safe * delta_prior) and (rel_drop <= eps_s)
            if cond_disc or cond_stag:
                m_stop = m
                break

    return LSQRResult(
        c_sol=d_s_inv * y,
        m_stop=m_stop,
        omega_history=np.asarray(omega_hist),
        c_history=c_hist,
    )


# ============================================================================
# 5. Self-Tests (Unitarity, Adjoint, and O(R^2 N) Row Norm Exactness)
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
    print(f"  [PASS] Multi-level SKCW FWT unitary check: err = {err_fwt:.2e}")

    t_grid = RetardedSIVIEOperator.collocation_grid(k, p, basis=basis)
    a0 = 4.0 * np.exp(3.0 * t_grid) * np.exp(-2.0 * (t_grid - 0.5))
    b0 = np.exp(2.0 * (t_grid - 0.5))
    a1 = 1.2 * np.exp(-1.5 * t_grid) * np.exp(-1.0 * (t_grid - 0.5))
    b1 = np.exp(1.0 * (t_grid - 0.5))
    a2 = 0.25 * (1.0 + 0.5 * np.sin(2.0 * np.pi * t_grid))
    K0 = SeparableKernel([a0], [b0])
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
    print(f"  [PASS] O(R^2 N) Row norm vs Explicit A: rel_err = {err_rows:.2e}")


if __name__ == "__main__":
    run_self_tests()