"""
ManNet-based fully connected hybrid beamforming (FC-HBF) for wideband THz massive MIMO.

Reference: N. T. Nguyen et al., "Deep Unfolding Hybrid Beamforming Designs for THz
Massive MIMO Systems," IEEE Trans. Signal Process., vol. 71, 2023.

Modules:
    1. Wideband Saleh-Valenzuela channel model with UPA and beam squint (Eqs. 2-3).
    2. Unconstrained optimal digital precoder via SVD (Eq. 8).
    3. ManNet: unfolded projected gradient descent network (Eqs. 20-22).
    4. Unsupervised training procedure (Algorithm 1).
    5. FC-HBF inference and optimal digital precoder (Algorithm 2, Eq. 29).
    6. Baselines: fully digital beamforming and wideband OMP.
    7. Spectral efficiency evaluation (Eq. 6).
    8. Dynamic sub-connected HBF: RF chain-antenna mapping (Algorithm 3), heuristic
       ManNet-based design (Algorithm 4), and subManNet-based design (Algorithm 5).

Usage:
    python mannet_hbf.py            # full-scale experiment
    python mannet_hbf.py --quick    # reduced-size functional test
    python mannet_hbf.py --skip_sc  # fully connected designs only
"""
# %%
import os
import json
import sys
import math
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

CDTYPE = torch.complex64


@dataclass
class Cfg:
    """Simulation, network, and evaluation parameters."""
    # System parameters
    Nth: int = 16                 # BS UPA size, horizontal (Nt = Nth * Ntv)
    Ntv: int = 8                  # BS UPA size, vertical
    Nrh: int = 2                  # MS array size, horizontal
    Nrv: int = 1                  # MS array size, vertical
    NRF: int = 2                  # number of RF chains
    Ns: int = 2                   # number of data streams
    K: int = 128                  # number of subcarriers
    P: int = 4                    # number of propagation paths
    fc: float = 300e9             # center frequency (Hz)
    BW: float = 30e9              # bandwidth (Hz)
    # ManNet parameters
    L: int = 3                    # number of layers
    I_train: int = 3              # unfolding iterations during training
    I_net: int = 10               # unfolding iterations during inference
    epochs: int = 30
    batch: int = 32
    n_train: int = 320
    lr: float = 1e-2              # Adam learning rate
    u_scale: float = 1.0          # fixed scaling applied to the gradient input u
    normalize_loop: bool = True   # project F_RF onto the unit-modulus set after each outer iteration
    # Alternating minimization baseline parameters
    mo_outer: int = 10            # MO-AltMin: outer (alternating) iterations
    mo_inner: int = 50            # MO-AltMin: inner Riemannian conjugate-gradient iterations
    ao_max_iter: int = 100        # AO: maximum number of alternating sweeps
    tol: float = 1e-3             # relative convergence tolerance of the baselines
    # Evaluation parameters
    n_test: int = 100
    snr_db: tuple = (-10, -5, 0, 5, 10)
    seed: int = 0
    out: str = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"

    @property
    def Nt(self):
        return self.Nth * self.Ntv

    @property
    def Nr(self):
        return self.Nrh * self.Nrv


# %% Channel model
def upa_response(Nh, Nv, phi, theta, fscale):
    """Frequency-dependent UPA array response (Eq. 3).

    phi, theta: (N, 1, P) azimuth and elevation angles; fscale = f_k / f_c: (1, K, 1).
    Returns a tensor of shape (N, K, P, Nh * Nv).
    """
    ih = torch.arange(Nh, dtype=torch.float64).repeat_interleave(Nv)
    iv = torch.arange(Nv, dtype=torch.float64).repeat(Nh)
    sx = (torch.sin(phi) * torch.sin(theta))[..., None]
    sy = torch.cos(theta)[..., None]
    phase = math.pi * fscale[..., None] * (ih * sx + iv * sy)
    return torch.exp(1j * phase) / math.sqrt(Nh * Nv)


def generate_channels(cfg: Cfg, N, gen, chunk=20):
    """Generate N wideband channels (Eq. 2); returns H of shape (N, K, Nr, Nt), complex64.

    Computation is performed in double precision to preserve phase accuracy.
    """
    K, P = cfg.K, cfg.P
    k = torch.arange(1, K + 1, dtype=torch.float64)
    fk = cfg.fc + cfg.BW * (2 * k - 1 - K) / (2 * K)
    fs = (fk / cfg.fc)[None, :, None]
    tau_max = (K / 4) / cfg.BW                     # cyclic prefix length K/4, sampling period 1/BW
    xi = math.sqrt(cfg.Nr * cfg.Nt / P)
    out = []
    for s in range(0, N, chunk):
        n = min(chunk, N - s)

        def U(lo, hi):
            return lo + (hi - lo) * torch.rand(n, 1, P, generator=gen, dtype=torch.float64)

        phi_t, phi_r = U(0, 2 * math.pi), U(0, 2 * math.pi)
        th_t, th_r = U(-math.pi / 2, math.pi / 2), U(-math.pi / 2, math.pi / 2)
        tau = tau_max * torch.rand(n, 1, P, generator=gen, dtype=torch.float64)
        alpha = torch.complex(
            torch.randn(n, 1, P, generator=gen, dtype=torch.float64),
            torch.randn(n, 1, P, generator=gen, dtype=torch.float64),
        ) / math.sqrt(2)
        g = alpha * torch.exp(-2j * math.pi * tau * fk[None, :, None])      # (n, K, P)
        at = upa_response(cfg.Nth, cfg.Ntv, phi_t, th_t, fs)                # (n, K, P, Nt)
        ar = upa_response(cfg.Nrh, cfg.Nrv, phi_r, th_r, fs)                # (n, K, P, Nr)
        H = xi * torch.einsum("nkp,nkpr,nkpt->nkrt", g, ar, at.conj())
        out.append(H.to(CDTYPE))
    return torch.cat(out, 0)


# %% Linear algebra
def hermitian_eigh(A):
    """Eigendecomposition of a batch of small Hermitian matrices (ascending eigenvalues).

    The decomposition is executed on the CPU because the batched cuSOLVER routine can fail
    for very large batches of small matrices. Results are returned on the original device.
    """
    lam, U = torch.linalg.eigh(A.cpu())
    return lam.to(A.device), U.to(A.device)


def hermitian_eigvalsh(A):
    """Eigenvalues of a batch of small Hermitian matrices (ascending), computed on the CPU."""
    return torch.linalg.eigvalsh(A.cpu()).to(A.device)


def top_right_sv(H, Ns):
    """Return the Ns principal right singular vectors V (..., Nt, Ns) of H and the
    corresponding squared singular values (..., Ns), computed via eigendecomposition of H H^H."""
    lam, U = hermitian_eigh(H @ H.mH)
    lam, U = lam.flip(-1)[..., :Ns], U.flip(-1)[..., :Ns]
    V = (H.mH @ U) / torch.sqrt(lam.clamp_min(1e-20))[..., None, :]
    return V, lam


def waterfill(gain, total):
    """Water-filling power allocation.

    gain: (..., M), sorted in descending order. Returns powers p with sum(p) = total that
    maximize sum(log(1 + gain * p)).
    """
    M = gain.shape[-1]
    inv = 1.0 / gain.clamp_min(1e-12)
    m = torch.arange(1, M + 1, device=gain.device)
    mu = (total + inv.cumsum(-1)) / m
    idx = torch.where(mu > inv, m, torch.zeros_like(m)).max(-1).values.clamp_min(1)
    mu_star = mu.gather(-1, (idx - 1)[..., None])
    return (mu_star - inv).clamp_min(0)


def unit_modulus(F):
    """Project each entry onto the unit circle."""
    return F / F.abs().clamp_min(1e-8)


def random_unit_modulus(S, Nt, NRF, device):
    """Random analog precoder with i.i.d. uniform phases."""
    return torch.exp(2j * math.pi * torch.rand(S, Nt, NRF, device=device)).to(CDTYPE)


def ls_bb(F_rf, Fopt):
    """Least-squares digital precoder F_BB[k] = pinv(F_RF) Fopt[k] (Eqs. 24, 27).

    F_rf: (N, Nt, NRF); Fopt: (N, K, Nt, Ns).
    """
    A = F_rf.mH
    pinv = torch.linalg.solve(A @ F_rf, A)
    return pinv[:, None] @ Fopt


def optimal_digital(H, F_rf, snr, Ns):
    """Spectral-efficiency-optimal digital precoder for a given analog precoder (Eq. 29).

    The transmit power constraint ||F_RF F_BB[k]||_F^2 = Ns is satisfied by construction.
    """
    Q = F_rf.mH @ F_rf
    lam, U = hermitian_eigh(Q)
    Qis = (U * lam.clamp_min(1e-12).rsqrt()[:, None, :]) @ U.mH            # Q^{-1/2}
    Ht = H @ (F_rf @ Qis)[:, None]                                         # (N, K, Nr, NRF)
    V, lam_h = top_right_sv(Ht, Ns)                                        # (N, K, NRF, Ns)
    p = waterfill(snr / Ns * lam_h, float(Ns))
    return Qis[:, None] @ (V * p.sqrt()[..., None, :].to(CDTYPE))


def spectral_efficiency(H, F, snr, Ns):
    """Average per-subcarrier spectral efficiency (Eq. 6) with the optimal combiner.

    H: (N, K, Nr, Nt); F: (N, K, Nt, Ns). Returns (N,) in bits/s/Hz.
    """
    M = H @ F
    lam = hermitian_eigvalsh(M.mH @ M).clamp_min(0)
    return torch.log2(1 + snr / Ns * lam).sum(-1).mean(-1)


def se_digital(H, snr, Ns):
    """Spectral efficiency of optimal fully digital beamforming with water-filling."""
    _, lam = top_right_sv(H, Ns)
    p = waterfill(snr / Ns * lam, float(Ns))
    return torch.log2(1 + snr / Ns * lam * p).sum(-1).mean(-1)


# %% ManNet
class ManNet(nn.Module):
    """Sparsely connected network obtained by unfolding projected gradient descent.

    Layer l computes  x_l = tanh(w1_l * x_{l-1} + w2_l * u_{l-1}),  with x_0 = 0, where
    u = -z_bar + sum_k B_bar[k] x is the gradient of the least-squares objective (Eqs. 20-21).
    Equivalently, u is the real-valued stacking of the matrix gradient
        G = sum_k (F_RF F_BB[k] - Fopt[k]) F_BB[k]^H,
    which avoids explicit construction of B[k] (Eq. 16).

    For the sub-connected architecture (subManNet), a binary mask c replaces tanh(.) by
    c * tanh(.) and u by c * u (Eqs. 40-41), so that only connected coefficients are non-zero.

    The number of trainable parameters is 4 * L * Nt * NRF.
    """

    def __init__(self, Nt, NRF, L=3, u_scale=1.0):
        super().__init__()
        self.Nt, self.NRF, self.L, self.u_scale = Nt, NRF, L, u_scale
        n = 2 * Nt * NRF
        self.w1 = nn.Parameter(0.1 * torch.randn(L, n))    # initialization N(0, 0.01)
        self.w2 = nn.Parameter(0.1 * torch.randn(L, n))

    def to_c(self, x):
        """Real vector (S, 2 Nt NRF) to complex matrix (S, Nt, NRF); inverse of Eq. 18."""
        re, im = x.chunk(2, -1)
        return torch.complex(re, im).view(-1, self.NRF, self.Nt).transpose(1, 2)

    def to_r(self, F):
        """Complex matrix (S, Nt, NRF) to real vector (S, 2 Nt NRF); mapping of Eq. 18."""
        F = F.transpose(1, 2).reshape(F.shape[0], -1)
        return torch.cat([F.real, F.imag], -1)

    def forward(self, Fopt, Fbb, c=None):
        """Return the list of layer outputs [x_1, ..., x_L], each of shape (S, 2 Nt NRF).

        c: optional real mask of shape (S, 2 Nt NRF) selecting the connected coefficients.
        """
        x = torch.zeros(Fopt.shape[0], 2 * self.Nt * self.NRF, device=Fopt.device)
        outs = []
        for l in range(self.L):
            G = ((self.to_c(x)[:, None] @ Fbb - Fopt) @ Fbb.mH).sum(1)
            u = self.u_scale * self.to_r(G)
            if c is not None:
                u = c * u
            x = torch.tanh(self.w1[l] * x + self.w2[l] * u)
            if c is not None:
                x = c * x
            outs.append(x)
        return outs


def mask_to_real(C):
    """Map a binary connection matrix (S, Nt, NRF) to the real mask (S, 2 Nt NRF) of Eq. 37."""
    v = C.transpose(1, 2).reshape(C.shape[0], -1)
    return torch.cat([v, v], -1)


def train_mannet(cfg: Cfg, Fopt_train, device, masks=None, log=print):
    """Unsupervised training of ManNet (Algorithm 1).

    Fopt serves only as network input; no labels are used. The loss is the layer-weighted
    least-squares factorization error of Eq. 26. If masks (n, Nt, NRF) are provided, the
    network is trained as subManNet with the corresponding connection matrix per sample.
    """
    model = ManNet(cfg.Nt, cfg.NRF, cfg.L, cfg.u_scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    D = Fopt_train.to(device)
    Mk = masks.to(device) if masks is not None else None
    n = D.shape[0]
    hist = []
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        tot, cnt = 0.0, 0
        for b in range(0, n, cfg.batch):
            idx = perm[b:b + cfg.batch]
            Fo = D[idx]
            F_rf = random_unit_modulus(Fo.shape[0], cfg.Nt, cfg.NRF, device)
            cb = None
            if Mk is not None:
                F_rf = F_rf * Mk[idx]
                cb = mask_to_real(Mk[idx])
            Fbb = ls_bb(F_rf, Fo)
            for _ in range(cfg.I_train):
                outs = model(Fo, Fbb, cb)
                loss = 0.0
                for l, x in enumerate(outs, start=1):
                    err = Fo - model.to_c(x)[:, None] @ Fbb
                    loss = loss + math.log(l) * (err.real ** 2 + err.imag ** 2).sum((-1, -2)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    F_rf = model.to_c(outs[-1])
                    if cfg.normalize_loop:
                        F_rf = unit_modulus(F_rf)
                    Fbb = ls_bb(F_rf, Fo)
                tot += loss.item()
                cnt += 1
        hist.append(tot / cnt)
        log(f"epoch {ep + 1:3d}/{cfg.epochs}  loss {hist[-1]:.4f}")
    return model, hist


@torch.no_grad()
def mannet_fc(model, Fopt, cfg: Cfg, I_net=None, C=None):
    """Analog precoder design with ManNet (Algorithm 2, analog part).

    Returns a unit-modulus F_RF of shape (N, Nt, NRF); the result does not depend on the SNR.
    If a connection matrix C (N, Nt, NRF) is given, the network is operated as subManNet and
    the returned precoder is zero outside the connected coefficients.
    """
    I_net = I_net or cfg.I_net
    F_rf = random_unit_modulus(Fopt.shape[0], cfg.Nt, cfg.NRF, Fopt.device)
    cmask = None
    if C is not None:
        F_rf = F_rf * C
        cmask = mask_to_real(C)
    Fbb = ls_bb(F_rf, Fopt)
    for _ in range(I_net):
        F_rf = model.to_c(model(Fopt, Fbb, cmask)[-1])
        if cfg.normalize_loop:
            F_rf = unit_modulus(F_rf)
        Fbb = ls_bb(F_rf, Fopt)
    return unit_modulus(F_rf)


# %% OMP baseline
def omp_dictionary(cfg: Cfg, device, oversample=2):
    """Unit-modulus UPA steering vectors at the center frequency on a uniform
    spatial-frequency grid. Returns a tensor of shape (Nt, G)."""
    Gx, Gy = oversample * cfg.Nth, oversample * cfg.Ntv
    wx = -1 + 2 * torch.arange(Gx, dtype=torch.float64) / Gx
    wy = -1 + 2 * torch.arange(Gy, dtype=torch.float64) / Gy
    ih = torch.arange(cfg.Nth, dtype=torch.float64).repeat_interleave(cfg.Ntv)
    iv = torch.arange(cfg.Ntv, dtype=torch.float64).repeat(cfg.Nth)
    W = torch.stack(torch.meshgrid(wx, wy, indexing="ij"), -1).reshape(-1, 2)
    phase = math.pi * (ih[:, None] * W[None, :, 0] + iv[:, None] * W[None, :, 1])
    return torch.exp(1j * phase).to(CDTYPE).to(device)


@torch.no_grad()
def omp_fc(Fopt, cfg: Cfg, A):
    """Wideband orthogonal matching pursuit with a frequency-flat analog precoder
    selected jointly over all subcarriers."""
    Fres, cols, chosen = Fopt.clone(), [], []
    for _ in range(cfg.NRF):
        score = (A.mH @ Fres).abs().pow(2).sum((1, 3))
        for c in chosen:
            score[torch.arange(score.shape[0]), c] = -1.0
        g = score.argmax(-1)
        chosen.append(g)
        cols.append(A[:, g].T)
        F_rf = torch.stack(cols, -1)
        Fbb = ls_bb(F_rf, Fopt)
        Fres = Fopt - F_rf[:, None] @ Fbb
        Fres = Fres / Fres.flatten(-2).norm(dim=-1).clamp_min(1e-12)[..., None, None]
    return F_rf


# %% Alternating minimization baselines
def ls_statistics(Fbb, Fopt):
    """Sufficient statistics of the objective  sum_k ||Fopt[k] - F_RF F_BB[k]||_F^2.

    With A = sum_k F_BB[k] F_BB[k]^H and C = sum_k Fopt[k] F_BB[k]^H, the objective equals
        const - 2 Re sum(conj(C) * F_RF) + Re tr(F_RF A F_RF^H),
    which allows evaluation and differentiation in O(Nt NRF^2) operations, independent of K.
    Returns A: (N, NRF, NRF) and C: (N, Nt, NRF).
    """
    A = torch.einsum("nkrs,nkts->nrt", Fbb, Fbb.conj())
    C = torch.einsum("nkas,nkrs->nar", Fopt, Fbb.conj())
    return A, C


def factorization_error(F_rf, Fopt, Fbb):
    """Objective sum_k ||Fopt[k] - F_RF F_BB[k]||_F^2 for each channel; returns shape (N,)."""
    err = Fopt - F_rf[:, None] @ Fbb
    return (err.real ** 2 + err.imag ** 2).sum((1, 2, 3))


@torch.no_grad()
def ao_fc(Fopt, cfg: Cfg, mask=None):
    """Alternating optimization (AO) of Sohrabi and Yu.

    If a connection matrix mask (N, Nt, NRF) is given, only the connected coefficients are
    optimized and all others remain zero (sub-connected architecture).

    Each analog phase is updated in closed form by coordinate descent with all other entries
    fixed, followed by a least-squares update of the digital precoder. Since the objective
    separates over the rows of F_RF, all antennas of one RF chain are updated simultaneously,
    which is equivalent to a sequential sweep over the Nt * NRF coefficients.
    """
    F = random_unit_modulus(Fopt.shape[0], cfg.Nt, cfg.NRF, Fopt.device)
    if mask is not None:
        F = F * mask
    Fbb = ls_bb(F, Fopt)
    prev = factorization_error(F, Fopt, Fbb)
    for _ in range(cfg.ao_max_iter):
        A, C = ls_statistics(Fbb, Fopt)
        for j in range(cfg.NRF):
            s = (F @ A[:, :, j:j + 1])[..., 0] - F[:, :, j] * A[:, j, j][:, None]
            F[:, :, j] = unit_modulus(C[:, :, j] - s)
            if mask is not None:
                F[:, :, j] = F[:, :, j] * mask[:, :, j]
        Fbb = ls_bb(F, Fopt)
        cur = factorization_error(F, Fopt, Fbb)
        if ((prev - cur).abs() / prev.clamp_min(1e-12)).max() < cfg.tol:
            break
        prev = cur
    return F


@torch.no_grad()
def riemannian_cg(F, A, C, iters, c1=1e-4, n_cand=8):
    """Riemannian conjugate gradient on the product of complex circles.

    Minimizes  -2 Re sum(conj(C) * F) + Re tr(F A F^H)  subject to |F_ij| = 1, using the
    tangent-space projection, the elementwise retraction (x + v) / |x + v|, projection-based
    vector transport, the Polak-Ribiere+ update, and Armijo backtracking over the step sizes
    alpha_0 * 2^p, where alpha_0 is the inverse of a Lipschitz bound of the gradient.
    """
    def cost(X):
        return -2 * (X * C.conj()).real.sum((-2, -1)) + ((X @ A) * X.conj()).real.sum((-2, -1))

    def project(X, Z):
        return Z - (Z * X.conj()).real * X

    def rgrad(X):
        return project(X, 2 * (X @ A - C))

    def inner(U, V):
        return (U.conj() * V).real.sum((-2, -1))

    alpha0 = 1.0 / (2 * A.diagonal(dim1=-2, dim2=-1).real.sum(-1)).clamp_min(1e-12)
    powers = 2.0 ** torch.arange(2, 2 - n_cand, -1, device=F.device, dtype=torch.float32)
    alphas = powers[:, None] * alpha0[None, :]                                  # (M, N)
    g = rgrad(F)
    d = -g
    for _ in range(iters):
        slope = inner(g, d)
        reset = slope >= 0
        d = torch.where(reset[:, None, None], -g, d)
        slope = torch.where(reset, -inner(g, g), slope)
        f0 = cost(F)
        cand = F[None] + alphas[..., None, None] * d[None]
        cand = unit_modulus(cand)                                               # (M, N, Nt, NRF)
        fc = cost(cand)
        ok = fc <= f0[None] + c1 * alphas * slope[None]
        idx = torch.where(ok.any(0), ok.int().argmax(0), torch.full_like(ok[0], n_cand - 1, dtype=torch.long))
        Fn = cand[idx, torch.arange(F.shape[0], device=F.device)]
        accept = fc.gather(0, idx[None])[0] <= f0
        Fn = torch.where(accept[:, None, None], Fn, F)
        gn = rgrad(Fn)
        beta = (inner(gn, gn - project(Fn, g)) / inner(g, g).clamp_min(1e-20)).clamp_min(0)
        d = -gn + beta[:, None, None] * project(Fn, d)
        F, g = Fn, gn
    return F


@torch.no_grad()
def mo_altmin_fc(Fopt, cfg: Cfg):
    """Manifold optimization based alternating minimization (MO-AltMin) of Yu et al.

    The outer loop alternates between the least-squares digital precoder and the analog
    precoder obtained by Riemannian conjugate gradient; the objective is accumulated over
    all subcarriers.
    """
    F = random_unit_modulus(Fopt.shape[0], cfg.Nt, cfg.NRF, Fopt.device)
    Fbb = ls_bb(F, Fopt)
    prev = factorization_error(F, Fopt, Fbb)
    for _ in range(cfg.mo_outer):
        A, C = ls_statistics(Fbb, Fopt)
        F = riemannian_cg(F, A, C, cfg.mo_inner)
        Fbb = ls_bb(F, Fopt)
        cur = factorization_error(F, Fopt, Fbb)
        if ((prev - cur).abs() / prev.clamp_min(1e-12)).max() < cfg.tol:
            break
        prev = cur
    return F


# %% Dynamic sub-connected HBF
@torch.no_grad()
def rf_antenna_mapping(H, NRF):
    """RF chain-antenna mapping (Algorithm 3).

    H: (B, Nr, Nt) channel matrices. The NRF rows of H with the largest norms are selected and
    the magnitudes of their entries define the gain between each antenna and each RF chain.
    The RF chains then select, in a round-robin order, the not yet assigned antenna with the
    largest gain until each chain is connected to M = Nt / NRF antennas.
    Returns the binary connection matrices C of shape (B, Nt, NRF) satisfying Eqs. 32-34.
    """
    B, Nr, Nt = H.shape
    M = Nt // NRF
    k = min(NRF, Nr)
    rows = H.abs().pow(2).sum(-1).topk(k, dim=-1).indices
    rows = rows[:, [n % k for n in range(NRF)]]
    Ht = torch.gather(H, 1, rows[..., None].expand(-1, -1, Nt))
    gain = Ht.abs().transpose(1, 2).contiguous()
    C = torch.zeros_like(gain)
    taken = torch.zeros(B, Nt, dtype=torch.bool, device=H.device)
    ar = torch.arange(B, device=H.device)
    for _ in range(M):
        for n in range(NRF):
            m0 = gain[:, :, n].masked_fill(taken, -1.0).argmax(-1)
            C[ar, m0, n] = 1.0
            taken[ar, m0] = True
    return C


@torch.no_grad()
def strongest_subcarrier_mapping(H, NRF):
    """Connection matrices designed on the subcarrier with the largest Frobenius norm (Algorithm 5)."""
    k_star = H.abs().pow(2).sum((-2, -1)).argmax(1)
    return rf_antenna_mapping(H[torch.arange(H.shape[0], device=H.device), k_star], NRF)


def fixed_mask(cfg: Cfg, device, N):
    """Fixed sub-connected architecture: RF chain n is connected to antennas nM, ..., (n+1)M - 1."""
    M = cfg.Nt // cfg.NRF
    C = torch.zeros(cfg.Nt, cfg.NRF, device=device)
    C[torch.arange(cfg.Nt), torch.arange(cfg.Nt) // M] = 1.0
    return C[None].expand(N, -1, -1).contiguous()


@torch.no_grad()
def dynamic_sc_masks(H, NRF, k_set):
    """Connection matrices for every subcarrier in k_set; returns shape (N, |k_set|, Nt, NRF)."""
    N, _, Nr, Nt = H.shape
    C = rf_antenna_mapping(H[:, k_set].reshape(-1, Nr, Nt), NRF)
    return C.view(N, len(k_set), Nt, NRF)


@torch.no_grad()
def manet_dyn_sc(H, F_tilde, masks, snr, Ns):
    """Heuristic ManNet-based dynamic SC-HBF (Algorithm 4).

    The fully connected analog precoder F_tilde is masked by each candidate connection matrix,
    and for every channel the candidate yielding the largest spectral efficiency is returned.
    """
    best_se = torch.full((H.shape[0],), -1.0, device=H.device)
    best_F = torch.zeros_like(F_tilde)
    for q in range(masks.shape[1]):
        F = F_tilde * masks[:, q]
        se = spectral_efficiency(H, F[:, None] @ optimal_digital(H, F, snr, Ns), snr, Ns)
        better = se > best_se
        best_se = torch.where(better, se, best_se)
        best_F = torch.where(better[:, None, None], F, best_F)
    return best_F


# %% Experiment
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size functional test")
    ap.add_argument("--skip_sc", action="store_true", help="evaluate fully connected designs only")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--n_test", type=int)
    ap.add_argument("--Nth", type=int)
    ap.add_argument("--Ntv", type=int)
    ap.add_argument("--I_net", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--u_scale", type=float)
    ap.add_argument("--out", type=str)
    ap.add_argument("--no_normalize_loop", action="store_true")
    a, _ = ap.parse_known_args(argv)

    cfg = Cfg()
    if a.quick:
        # Nt=16, K=16, 2 epochs, 64 training / 5 test channels (smoke test only)
        cfg.Nth, cfg.Ntv, cfg.K, cfg.epochs, cfg.n_train, cfg.n_test = 4, 4, 16, 2, 64, 5
    for k in ("epochs", "n_test", "Nth", "Ntv", "I_net", "out", "lr", "u_scale"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.no_normalize_loop:
        cfg.normalize_loop = False
    run_sc = not a.skip_sc
    os.makedirs(cfg.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    N = cfg.n_test
    print(f"device={device}  Nt={cfg.Nt} Nr={cfg.Nr} NRF={cfg.NRF} Ns={cfg.Ns} K={cfg.K}")

    # Data generation. Fopt uses equal power allocation so that training is SNR-independent.
    H_tr = generate_channels(cfg, cfg.n_train, gen).to(device)
    Fopt_tr = top_right_sv(H_tr, cfg.Ns)[0]
    masks_tr = strongest_subcarrier_mapping(H_tr, cfg.NRF) if run_sc else None
    del H_tr
    H_te = generate_channels(cfg, N, gen).to(device)
    Fopt_te = top_right_sv(H_te, cfg.Ns)[0]

    # Training of ManNet and, for the sub-connected designs, subManNet.
    t0 = time.time()
    model, hist = train_mannet(cfg, Fopt_tr, device)
    print(f"ManNet training time {time.time() - t0:.1f} s, trainable parameters = "
          f"{sum(p.numel() for p in model.parameters())}")
    torch.save({"state": model.state_dict(), "cfg": cfg.__dict__, "loss": hist},
               f"{cfg.out}/mannet_fc.pt")
    hist_sub = None
    if run_sc:
        t0 = time.time()
        sub, hist_sub = train_mannet(cfg, Fopt_tr, device, masks=masks_tr)
        print(f"subManNet training time {time.time() - t0:.1f} s")
        torch.save({"state": sub.state_dict(), "cfg": cfg.__dict__, "loss": hist_sub},
                   f"{cfg.out}/submannet.pt")

    def timed(fn):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        r = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return r, (time.time() - t) / N

    # Fully connected analog precoder design (SNR-independent, computed once).
    F_man, t_man = timed(lambda: mannet_fc(model, Fopt_te, cfg))
    A = omp_dictionary(cfg, device)
    F_omp, t_omp = timed(lambda: omp_fc(Fopt_te, cfg, A))
    F_man1 = mannet_fc(model, Fopt_te, cfg, I_net=1)
    F_mo, t_mo = timed(lambda: mo_altmin_fc(Fopt_te, cfg))
    F_ao, t_ao = timed(lambda: ao_fc(Fopt_te, cfg))
    print(f"mean analog design time per channel: ManNet (I_net={cfg.I_net}) {t_man * 1e3:.2f} ms, "
          f"MO-AltMin {t_mo * 1e3:.2f} ms, AO {t_ao * 1e3:.2f} ms, OMP {t_omp * 1e3:.2f} ms")

    fc_designs = [("ManNet-FC", F_man), ("ManNet-FC (I=1)", F_man1),
                  ("MO-AltMin-FC", F_mo), ("AO-FC", F_ao), ("OMP-FC", F_omp)]

    # Sub-connected designs.
    sc_designs, sc_names = [], []
    if run_sc:
        k_set = list(range(0, cfg.K, 2))
        masks_dyn, t_map = timed(lambda: dynamic_sc_masks(H_te, cfg.NRF, k_set))
        (F_sub, C_star), t_sub = timed(lambda: (lambda C: (mannet_fc(sub, Fopt_te, cfg, C=C), C))(
            strongest_subcarrier_mapping(H_te, cfg.NRF)))
        C_fix = fixed_mask(cfg, device, N)
        F_fix = F_man * C_fix
        F_aod, t_aod = timed(lambda: ao_fc(Fopt_te, cfg, C_star))
        F_aof = ao_fc(Fopt_te, cfg, C_fix)
        _, t_sel = timed(lambda: manet_dyn_sc(H_te, F_man, masks_dyn, 10.0, cfg.Ns))
        print(f"mean SC design time per channel: ManNet-Dyn-SC {(t_man + t_map + t_sel) * 1e3:.2f} ms "
              f"(mapping {t_map * 1e3:.2f} ms, selection at 10 dB {t_sel * 1e3:.2f} ms), "
              f"subManNet-Dyn-SC {t_sub * 1e3:.2f} ms, AO-Dyn-SC {t_aod * 1e3:.2f} ms")
        sc_names = ["ManNet-Dyn-SC", "subManNet-Dyn-SC", "ManNet-Fix-SC", "AO-Dyn-SC", "AO-Fix-SC"]

    # Spectral efficiency versus SNR.
    fc_names = ["ManNet-FC", "ManNet-FC (I=1)", "MO-AltMin-FC", "AO-FC", "OMP-FC"]
    res = {"snr_db": list(cfg.snr_db), "DBF": []}
    for n_ in fc_names + sc_names:
        res[n_] = []

    def record(name, F_rf, snr):
        F = F_rf[:, None] @ optimal_digital(H_te, F_rf, snr, cfg.Ns)
        pw = (F.mH @ F).diagonal(dim1=-2, dim2=-1).real.sum(-1).mean().item()
        assert abs(pw - cfg.Ns) < 1e-2 * cfg.Ns, f"transmit power constraint violated: {pw}"
        res[name].append(spectral_efficiency(H_te, F, snr, cfg.Ns).mean().item())

    for sdb in cfg.snr_db:
        snr = 10 ** (sdb / 10)
        res["DBF"].append(se_digital(H_te, snr, cfg.Ns).mean().item())
        for name, F_rf in fc_designs:
            record(name, F_rf, snr)
        if run_sc:
            for name, F_rf in (("ManNet-Dyn-SC", manet_dyn_sc(H_te, F_man, masks_dyn, snr, cfg.Ns)),
                               ("subManNet-Dyn-SC", F_sub), ("ManNet-Fix-SC", F_fix),
                               ("AO-Dyn-SC", F_aod), ("AO-Fix-SC", F_aof)):
                record(name, F_rf, snr)

    def show(title, names):
        print(f"\n{title}")
        print("SNR(dB) " + "".join(f"{n:>19s}" for n in names))
        for i, s_ in enumerate(cfg.snr_db):
            print(f"{s_:7d} " + "".join(
                f"{res[n][i]:8.3f} ({100 * res[n][i] / res['DBF'][i]:5.1f}%)" for n in names))

    show("Spectral efficiency [bits/s/Hz] (percentage of fully digital beamforming): fully connected",
         ["DBF"] + fc_names)
    if run_sc:
        show("Spectral efficiency [bits/s/Hz] (percentage of fully digital beamforming): sub-connected",
             sc_names)
        i10 = len(cfg.snr_db) - 1
        print(f"\nSub-connected designs relative to MO-AltMin-FC at {cfg.snr_db[i10]} dB: " +
              ", ".join(f"{n} {100 * res[n][i10] / res['MO-AltMin-FC'][i10]:.1f}%" for n in sc_names))

    out = {k: np.array(v) for k, v in res.items()}
    out["loss"] = np.array(hist)
    if hist_sub is not None:
        out["loss_sub"] = np.array(hist_sub)
    np.savez(f"{cfg.out}/results.npz", **out)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3 if run_sc else 2, figsize=(16 if run_sc else 11, 4))
        for n_, m in zip(["DBF"] + fc_names, ("k-", "r-o", "r--s", "g-d", "m-v", "b-^")):
            ax[0].plot(cfg.snr_db, res[n_], m, label=n_)
        ax[0].set_title("Fully connected")
        panels = [ax[0]]
        if run_sc:
            for n_, m in zip(sc_names, ("r-o", "c-s", "r--^", "m-d", "m--v")):
                ax[1].plot(cfg.snr_db, res[n_], m, label=n_)
            ax[1].plot(cfg.snr_db, res["MO-AltMin-FC"], "g:", label="MO-AltMin-FC")
            ax[1].set_title("Sub-connected")
            panels.append(ax[1])
        for p in panels:
            p.set_xlabel("SNR (dB)")
            p.set_ylabel("Spectral efficiency (bits/s/Hz)")
            p.grid(alpha=0.3)
            p.legend(fontsize=7)
        axl = ax[-1]
        axl.plot(range(1, len(hist) + 1), hist, label="ManNet")
        if hist_sub is not None:
            axl.plot(range(1, len(hist_sub) + 1), hist_sub, label="subManNet")
        axl.set_xlabel("Epoch")
        axl.set_ylabel("Training loss")
        axl.set_yscale("log")
        axl.grid(alpha=0.3)
        axl.legend()
        plt.tight_layout()
        plt.savefig(f"{cfg.out}/se_vs_snr.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    return model, res


# %% Fig. 6: SE vs Nt
NT_GRID = {16: (4, 4), 32: (4, 8), 64: (8, 8), 128: (8, 16)}   # Nt -> (Nth, Ntv)


def run_fig6(argv):
    """Retrain ManNet/subManNet for each Nt and record SE at 10 dB (paper Fig. 6)."""
    base = Cfg().out
    quick = "--quick" in argv
    root = os.path.join(base, "fig6")
    os.makedirs(root, exist_ok=True)
    rows = {}
    for nt, (nh, nv) in NT_GRID.items():
        d = os.path.join(root, f"Nt{nt}")
        main(list(argv) + ["--Nth", str(nh), "--Ntv", str(nv), "--out", d])
        r = np.load(f"{d}/results.npz")
        i10 = list(r["snr_db"]).index(10)
        rows[nt] = {k: float(r[k][i10]) for k in r.files
                    if k not in ("snr_db", "loss", "loss_sub") and r[k].shape == r["snr_db"].shape}
    names = list(rows[min(rows)].keys())
    np.savez(f"{root}/fig6.npz", Nt=np.array(list(rows)), **{n: np.array([rows[nt].get(n, np.nan) for nt in rows]) for n in names})
    c = Cfg()
    with open(f"{root}/fig6.json", "w") as f:
        json.dump({"snr_db": 10, "quick": quick, "argv": list(argv), "seed": c.seed, "lr": c.lr,
                   "epochs": c.epochs, "n_test": c.n_test, "K": c.K, "se_at_10dB": rows}, f, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nts = list(rows)
        plt.figure(figsize=(7, 4.5))
        for n in names:
            plt.plot(nts, [rows[nt].get(n, np.nan) for nt in nts], "-o", label=n, ms=4)
        plt.xscale("log", base=2)
        plt.xticks(nts, nts)
        plt.xlabel("Nt")
        plt.ylabel("Spectral efficiency (bits/s/Hz) @ 10 dB")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{root}/fig6.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    print("\nFig. 6 SE @10 dB:")
    for nt in rows:
        print(f"Nt={nt}: " + ", ".join(f"{n} {rows[nt][n]:.3f}" for n in names if n in rows[nt]))
    return rows


# %% Fig. 4: convergence
@torch.no_grad()
def mannet_traj(model, Fopt, cfg: Cfg, I_max, C=None):
    """F_RF after each unfolding iteration 1..I_max (each iteration = L layers). Mirrors mannet_fc."""
    F_rf = random_unit_modulus(Fopt.shape[0], cfg.Nt, cfg.NRF, Fopt.device)
    cmask = None
    if C is not None:
        F_rf = F_rf * C
        cmask = mask_to_real(C)
    Fbb = ls_bb(F_rf, Fopt)
    traj = []
    for _ in range(I_max):
        F_rf = model.to_c(model(Fopt, Fbb, cmask)[-1])
        if cfg.normalize_loop:
            F_rf = unit_modulus(F_rf)
        Fbb = ls_bb(F_rf, Fopt)
        traj.append(unit_modulus(F_rf))
    return traj


@torch.no_grad()
def ao_traj(Fopt, cfg: Cfg, n_iter):
    """AO-FC after each alternating sweep 1..n_iter, without early stopping. Mirrors ao_fc."""
    F = random_unit_modulus(Fopt.shape[0], cfg.Nt, cfg.NRF, Fopt.device)
    Fbb = ls_bb(F, Fopt)
    traj = []
    for _ in range(n_iter):
        A, C = ls_statistics(Fbb, Fopt)
        for j in range(cfg.NRF):
            s = (F @ A[:, :, j:j + 1])[..., 0] - F[:, :, j] * A[:, j, j][:, None]
            F[:, :, j] = unit_modulus(C[:, :, j] - s)
        Fbb = ls_bb(F, Fopt)
        traj.append(F.clone())
    return traj


@torch.no_grad()
def mo_traj(Fopt, cfg: Cfg, n_iter, per_outer=5):
    """MO-AltMin-FC versus the total number of Riemannian-CG (inner) iterations.

    The digital precoder is updated every per_outer inner iterations (our reading of the paper's
    "total number of inner iterations"); no early stopping. Returns F_RF at multiples of per_outer.
    """
    F = random_unit_modulus(Fopt.shape[0], cfg.Nt, cfg.NRF, Fopt.device)
    Fbb = ls_bb(F, Fopt)
    traj = []
    for _ in range(n_iter // per_outer):
        A, C = ls_statistics(Fbb, Fopt)
        F = riemannian_cg(F, A, C, per_outer)
        Fbb = ls_bb(F, Fopt)
        traj.append(F.clone())
    return traj


def run_fig4(argv):
    """SE versus iteration count for Nt=128 at 10 and 20 dB (paper Fig. 4); FC and subManNet-SC."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--n_test", type=int)
    ap.add_argument("--out", type=str)
    a, _ = ap.parse_known_args(argv)
    cfg = Cfg()
    if a.quick:
        cfg.Nth, cfg.Ntv, cfg.K, cfg.epochs, cfg.n_train, cfg.n_test = 4, 4, 16, 2, 64, 5
    for k in ("epochs", "n_test", "out"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    out_dir = os.path.join(cfg.out, "fig4") if a.out is None else a.out
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    snrs = (10, 20)
    I_max = 20                       # ManNet iterations evaluated (trained with I_train, default inference I_net)
    n_ao = 100 if not a.quick else 20
    n_mo = 100 if not a.quick else 20
    print(f"device={device} Nt={cfg.Nt} K={cfg.K}")

    H_tr = generate_channels(cfg, cfg.n_train, gen).to(device)
    Fopt_tr = top_right_sv(H_tr, cfg.Ns)[0]
    masks_tr = strongest_subcarrier_mapping(H_tr, cfg.NRF)
    del H_tr
    H = generate_channels(cfg, cfg.n_test, gen).to(device)
    Fopt = top_right_sv(H, cfg.Ns)[0]
    model, _ = train_mannet(cfg, Fopt_tr, device)
    sub, _ = train_mannet(cfg, Fopt_tr, device, masks=masks_tr)
    C_star = strongest_subcarrier_mapping(H, cfg.NRF)

    def se(F_rf, sdb):
        snr = 10 ** (sdb / 10)
        F = F_rf[:, None] @ optimal_digital(H, F_rf, snr, cfg.Ns)
        return spectral_efficiency(H, F, snr, cfg.Ns).mean().item()

    ones = np.arange(1, I_max + 1)
    curves = {
        "ManNet-FC": (ones * cfg.L, mannet_traj(model, Fopt, cfg, I_max)),
        "subManNet-SC": (ones * cfg.L, mannet_traj(sub, Fopt, cfg, I_max, C=C_star)),
        "AO-FC": (np.arange(1, n_ao + 1), ao_traj(Fopt, cfg, n_ao)),
        "MO-AltMin-FC": (np.arange(5, n_mo + 1, 5), mo_traj(Fopt, cfg, n_mo)),
    }
    omp = omp_fc(Fopt, cfg, omp_dictionary(cfg, device))
    res = {"snr_db": list(snrs), "DBF": [se_digital(H, 10 ** (s / 10), cfg.Ns).mean().item() for s in snrs],
           "OMP-FC": [se(omp, s) for s in snrs]}
    for name, (x, traj) in curves.items():
        res[name + "_iters"] = [int(v) for v in x]
        res[name] = [[se(F, s) for F in traj] for s in snrs]
    np.savez(f"{out_dir}/fig4.npz", **{k: np.array(v) for k, v in res.items()})
    with open(f"{out_dir}/fig4.json", "w") as f:
        json.dump({"argv": list(argv), "seed": cfg.seed, "lr": cfg.lr, "epochs": cfg.epochs, "n_test": cfg.n_test,
                   "K": cfg.K, "Nt": cfg.Nt, "L": cfg.L, "I_train": cfg.I_train, "mo_inner_per_outer": 5,
                   "results": res}, f, indent=1)
    for i, s in enumerate(snrs):
        print(f"\nSNR {s} dB: DBF {res['DBF'][i]:.3f}, OMP-FC {res['OMP-FC'][i]:.3f}")
        for name in curves:
            x, y = res[name + "_iters"], res[name][i]
            pick = [j for j, v in enumerate(x) if v in (3, 6, 9, 15, 30, 60, 100, 10, 20, 50)]
            print(f"  {name:14s} " + ", ".join(f"{x[j]}:{y[j]:.3f}" for j in pick) + f", final {x[-1]}:{y[-1]:.3f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(snrs), figsize=(11, 4))
        for i, s in enumerate(snrs):
            ax[i].axhline(res["DBF"][i], color="k", label="DBF-FC")
            ax[i].axhline(res["OMP-FC"][i], color="m", ls="--", label="OMP-FC")
            for name, m in zip(curves, ("r-o", "c-s", "g-d", "b-^")):
                ax[i].plot(res[name + "_iters"], res[name][i], m, ms=3, label=name)
            ax[i].set_xscale("log")
            ax[i].set_title(f"SNR = {s} dB")
            ax[i].set_xlabel("Iterations (ManNet: I_net * L)")
            ax[i].set_ylabel("Spectral efficiency (bits/s/Hz)")
            ax[i].grid(alpha=0.3)
            ax[i].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/fig4.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--exp", default="main", choices=["main", "fig4", "fig6"])
    # Kaggle scripts take no CLI args: set the experiment to run there by editing this list.
    KAGGLE_ARGV = ["--exp", "fig4"]
    argv = sys.argv[1:] if (len(sys.argv) > 1 or not os.path.isdir("/kaggle/working")) else KAGGLE_ARGV
    a, rest = ap.parse_known_args(argv)
    if a.exp == "fig6":
        run_fig6(rest)
    elif a.exp == "fig4":
        run_fig4(rest)
    else:
        main(rest)
