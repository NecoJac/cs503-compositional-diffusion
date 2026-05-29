"""
K-particle SMC bridge worker — correct FKC-PoE implementation per Skreta+ 2025
Prop. 3.3, adapted from jieke (a-green-hand-jack/physgen)'s `FKCPoEPipeline`.

Replaces `ProposalFinalBridgeWorker`'s three deviations:
  1. K=1 single-sample  →  K particles (default 4) with separate latents
  2. Explicit `-s_y_implicit` subtract per step  →  weight absorbs marginal implicitly
  3. Explicit `+Δs` Tweedie correction per step  →  SMC weight + resample accumulates

The corrected algorithm per ODE step (per particle k):
  ① Pairwise predictions for all K particles at once (batched).
  ② FK weight increment: `dlog_w = -β · ||x0_overlap_i - x0_overlap_{i+1}||² / D · dt`
     where x0 estimates come from Tweedie (x_t - sqrt(β)*eps) / sqrt(α).
     Negative x0 disagreement in the overlap region directly proxies seam MSE.
  ③ Systematic resampling BEFORE the ODE step (FKC-correct order).
  ④ Naive PoE merge: `s_xy ⊕ s_yz` + endpoint guidance, NO `-s_y` subtract.
  ⑤ Heun/Euler step on the K-particle ensemble using merged_eps.
  ⑥ SNIS final selection: per-batch keep the max-log-weight particle.

References:
  - jieke FKC-PoE pipeline: code_jk/physgen-fkc-poe-physics-refinement/
    condition_diffusion/fkc_poe_pipeline.py:73 FKCPoEPipeline
  - Score-space derivation: memory/reference_infdiff_fkc_vs_diffcollage.md
  - Implementation gap analysis: memory/reference_fkc_correct_impl_jieke.md
"""

from __future__ import annotations

from typing import Optional

import torch as th
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from proposal_methods.workers import BridgeWorkerBase


class ProposalSMCBridgeWorker(BridgeWorkerBase):
    """K-particle SMC bridge worker (correct FKC-PoE).

    Returns a NAIVE merged eps (no implicit-y subtract, no Δs) from
    `get_eps_t_fn`. The SMC weight + resample + SNIS happen in the
    `smc_sample()` driver, NOT inside the eps function (which has to
    remain stateless to fit the BaseWorker contract).
    """

    def __init__(
        self,
        shape,
        eps_scalar_t_fn,
        num_img,
        left_fixed_images,
        right_fixed_images,
        overlap_size=32,
        guidance_scale=1.0,
        sigma_max=80.0,
        sigma_min=1e-3,
        num_particles: int = 4,
        beta: float = 1.0,
        t_resample_start: float = 0.0,
        t_resample_end: float = 0.8,
    ):
        # Stash the raw eps fn so smc_sample can call _pairwise_factor_predictions
        # directly (bypassing the BaseWorker eps_fn that returns merged eps).
        self._raw_eps_fn = eps_scalar_t_fn
        self.num_particles = num_particles
        self.beta = beta
        # Time window for resampling, expressed as fractions of sigma_max.
        self.t_resample_start = float(t_resample_start)
        self.t_resample_end = float(t_resample_end)
        super().__init__(
            shape=shape,
            eps_scalar_t_fn=eps_scalar_t_fn,
            num_img=num_img,
            left_fixed_images=left_fixed_images,
            right_fixed_images=right_fixed_images,
            overlap_size=overlap_size,
            guidance_scale=guidance_scale,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    # ------------------------------------------------------------------
    # PoE merge (no marginal subtract, no Δs correction).
    # ------------------------------------------------------------------
    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            xs, factor_eps, _ = self._pairwise_factor_predictions(
                eps_scalar_t_fn, long_x, scalar_t, enable_grad
            )
            full_eps = factor_eps[:, 1:-1].clone()
            left_cond = factor_eps[:, 0, :, :, self.overlap_size :]
            right_cond = factor_eps[:, -1, :, :, : self.overlap_size]
            full_eps[:, 0, :, :, : self.overlap_size] += self.guidance_scale * left_cond
            full_eps[:, -1, :, :, -self.overlap_size :] += self.guidance_scale * right_cond
            return self._merge_bridge_eps(full_eps, is_avg=False)

        return eps_t_fn

    # ------------------------------------------------------------------
    # Re-implemented from ProposalFinalBridgeWorker (same logic, no
    # sigma_data / coupling_strength / etc — pure pairwise factor call).
    # ------------------------------------------------------------------
    def _pairwise_factor_predictions(self, eps_scalar_t_fn, long_x, scalar_t, enable_grad):
        batch_size = long_x.shape[0]
        device = long_x.device
        dtype = long_x.dtype
        xs = self._split_bridge(long_x)
        left_t, right_t = self._fixed_t(batch_size, device, dtype, scalar_t)
        left_factor, right_factor = self._build_endpoint_factors(xs, left_t, right_t)
        factors = th.cat([left_factor.unsqueeze(1), xs, right_factor.unsqueeze(1)], dim=1)
        factor_eps = eps_scalar_t_fn(
            rearrange(factors, "b n c h w -> (b n) c h w"),
            scalar_t,
            enable_grad,
        )
        factor_eps = rearrange(
            factor_eps,
            "(b n) c h w -> b n c h w",
            n=self.num_img + 2,
        )
        sigma = self._sigma_for_batch(scalar_t, factors, batch_size)
        factor_x0 = factors - sigma * factor_eps
        return xs, factor_eps, factor_x0

    @staticmethod
    def _sigma_for_batch(scalar_t, reference, batch_size):
        if not th.is_tensor(scalar_t):
            sigma = th.tensor(float(scalar_t), device=reference.device, dtype=reference.dtype)
        else:
            sigma = scalar_t.to(device=reference.device, dtype=reference.dtype)
        if sigma.ndim == 0:
            return sigma.view(1, 1, 1, 1, 1)
        return sigma.view(batch_size, 1, 1, 1, 1)

    # ------------------------------------------------------------------
    # FK weight increment via ⟨ε_xy, ε_yz⟩ in overlap region per particle.
    # Each pair of adjacent factors (i, i+1) share an overlap of size
    # self.overlap_size on the right of i and left of i+1. The dot product
    # of their ε predictions in that overlap measures how much the two
    # conditional models agree on the bridge state — high agreement →
    # high weight.
    # ------------------------------------------------------------------
    def smc_weight_increment(self, long_x, scalar_t):
        """Returns negative log-weight increment per particle (B,). Lower x0 disagreement → higher weight. Caller multiplies by dt."""
        xs, factor_eps, factor_x0 = self._pairwise_factor_predictions(
            self._raw_eps_fn, long_x, scalar_t, enable_grad=False
        )
        # factor_x0 shape: (B, N+2, C, H, W).
        # Adjacent pairs: right overlap of x0[i] vs left overlap of x0[i+1].
        right_part = factor_x0[:, :-1, :, :, -self.overlap_size:]        # (B, N+1, C, H, overlap)
        left_part_next = factor_x0[:, 1:, :, :, :self.overlap_size]      # (B, N+1, C, H, overlap)
        # Squared disagreement per particle, normalized by overlap dimension
        disagree = ((right_part - left_part_next).float() ** 2).flatten(1).sum(dim=1)  # (B,)
        D = right_part[0].numel()
        return -disagree / D  # negative: lower disagreement → higher weight

    # ------------------------------------------------------------------
    # Systematic (stratified) resampling — same algorithm as
    # FKCPoEPipeline._systematic_resample.
    # ------------------------------------------------------------------
    @staticmethod
    def systematic_resample(log_w: th.Tensor) -> th.LongTensor:
        """log_w: (K,)  →  parent indices: (K,) Long."""
        K = log_w.shape[0]
        probs = F.softmax(log_w, dim=0)
        cdf = th.cumsum(probs, dim=0)
        u = th.rand(1, device=log_w.device)
        pos = (u + th.arange(K, device=log_w.device).float()) / K
        return th.searchsorted(cdf, pos).clamp(max=K - 1)


# ---------------------------------------------------------------------------
# Driver: K-particle SMC sampling. Replaces dc.sampling for the SMC method.
# ---------------------------------------------------------------------------
@th.no_grad()
def smc_sample(
    worker: ProposalSMCBridgeWorker,
    x_t: th.Tensor,
    n_step: int,
    ts_order: int,
    solver: str = "heun",
    is_tqdm: bool = True,
) -> th.Tensor:
    """Sample from worker using K-particle SMC with FK weight + resample + SNIS.

    Args:
        worker: ProposalSMCBridgeWorker.
        x_t: initial noise (B, C, H, W).
        n_step: number of ODE steps.
        ts_order: EDM time-step ordering parameter.
        solver: "heun" or "euler".
        is_tqdm: show progress bar.
    Returns:
        x_final: (B, C, H, W) best particle per batch via SNIS.
    """
    K = worker.num_particles
    B = x_t.shape[0]
    device = x_t.device
    dtype = x_t.dtype

    # Replicate to K particles per batch item. Shape becomes (B*K, C, H, W).
    # interleave so that index i*K + k is batch i's k-th particle.
    x = x_t.repeat_interleave(K, dim=0).contiguous()
    log_w = th.zeros(B * K, device=device, dtype=th.float32)

    sigmas = worker.rev_ts(n_step, ts_order).to(device=device, dtype=dtype)
    sigma_max = float(sigmas[0])
    t_lo = worker.t_resample_start * sigma_max
    t_hi = worker.t_resample_end * sigma_max

    loop = zip(sigmas[:-1], sigmas[1:])
    if is_tqdm:
        loop = tqdm(list(loop), desc=f"SMC sample K={K}")

    eps_fn = worker.eps_fn  # merged PoE eps (no marginal subtract, no Δs)

    for sigma_cur, sigma_next in loop:
        dt = float(sigma_cur - sigma_next)   # positive (sigma decreases)

        # ── ① FK weight: negative x0 overlap disagreement × dt ──────────
        inner = worker.smc_weight_increment(x, sigma_cur)                 # (B*K,) — negative disagree/D
        log_w = log_w + worker.beta * inner.float() * dt

        # ── ② Resample BEFORE ODE step (FKC-correct order) ─────────────
        if t_lo <= float(sigma_cur) <= t_hi:
            x_view = x.view(B, K, *x.shape[1:])
            log_w_view = log_w.view(B, K)
            new_x = th.empty_like(x_view)
            for b in range(B):
                idx = ProposalSMCBridgeWorker.systematic_resample(log_w_view[b])
                new_x[b] = x_view[b, idx]
            x = new_x.view(B * K, *x.shape[1:])
            log_w = th.zeros(B * K, device=device, dtype=th.float32)

        # ── ③ ODE step (Euler or Heun) ─────────────────────────────────
        eps_cur = eps_fn(x, sigma_cur)
        d_cur = eps_cur
        x_next = x + (sigma_next - sigma_cur) * d_cur

        if solver == "heun" and float(sigma_next) > 0:
            eps_next = eps_fn(x_next, sigma_next)
            d_next = eps_next
            x_next = x + 0.5 * (sigma_next - sigma_cur) * (d_cur + d_next)

        x = x_next

    # ── ④ SNIS final selection: best particle per batch ──────────────
    x_view = x.view(B, K, *x.shape[1:])
    log_w_view = log_w.view(B, K)
    best_idx = log_w_view.argmax(dim=1)                                   # (B,)
    x_final = x_view[th.arange(B, device=device), best_idx]
    return x_final
