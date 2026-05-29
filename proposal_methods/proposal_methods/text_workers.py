"""
Text-conditioned bridge workers for CBG-Diffusion.

Replaces fixed image endpoint strips with class-conditional CFG guidance.
- Patch 0 (left endpoint):  guided by left_class_idx via CFG
- Patch N-1 (right endpoint): guided by right_class_idx via CFG
- Middle patches: unconditional, free to interpolate

Three composition strategies are provided, mirroring the image-based workers:
  DiffCollageBridgeWorkerText  -- subtract implicit overlap marginals
  NaiveBridgeWorkerText        -- no marginal correction
  ProposalFinalBridgeWorkerText -- DiffCollage + Tweedie x0 consistency Δs
"""

import torch as th
from einops import rearrange

from diff_collage.base_worker import BaseWorker
from diff_collage.w_img import split_wimg, avg_merge_wimg


# ── class registry ────────────────────────────────────────────────────────────

TEXT_TO_CLASS = {
    "alp": 970,
    "cliff": 972,
    "coral_reef": 973,
    "geyser": 974,
    "lakeside": 975,
    "promontory": 976,
    "sandbar": 977,
    "seashore": 978,
    "valley": 979,
    "volcano": 980,
}


def text_to_class_idx(text):
    """Map a landscape class name to its ImageNet-1k class index."""
    key = text.lower().strip().replace(" ", "_")
    if key in TEXT_TO_CLASS:
        return TEXT_TO_CLASS[key]
    for name, idx in TEXT_TO_CLASS.items():
        if name in key:
            return idx
    raise ValueError(
        f"Unknown landscape class: {text!r}. "
        f"Available: {sorted(TEXT_TO_CLASS.keys())}"
    )


# ── EDM model helpers ─────────────────────────────────────────────────────────

def _edm_predict(net, xs, scalar_t, class_idx=None, enable_grad=False):
    """Run EDM model; return (eps, x0) in the same dtype as xs."""
    B = xs.shape[0]
    device = xs.device

    if class_idx is not None:
        class_labels = th.zeros(B, 1000, device=device)
        class_labels[:, class_idx] = 1.0
    else:
        class_labels = None

    if not th.is_tensor(scalar_t):
        scalar_t = th.tensor(float(scalar_t), device=device, dtype=xs.dtype)
    sigma = scalar_t.to(device)
    if sigma.ndim == 0:
        sigma = sigma.repeat(B)
    sigma_img = sigma.to(th.float32).view(-1, 1, 1, 1)

    ctx = th.enable_grad() if enable_grad else th.no_grad()
    with ctx:
        x0 = net(xs.to(th.float32), sigma.to(th.float32), class_labels)
        eps = (xs.to(th.float32) - x0) / sigma_img

    return eps.to(xs.dtype), x0.to(xs.dtype)


# ── base worker ───────────────────────────────────────────────────────────────

class _TextBridgeBase(BaseWorker):
    """
    Common infrastructure for text-conditioned bridge workers.

    Subclasses implement get_eps_t_fn().
    """

    def __init__(
        self,
        shape,
        net,
        num_img,
        left_class_idx,
        right_class_idx,
        overlap_size=32,
        guidance_scale=1.0,
        sigma_max=80.0,
        sigma_min=1e-3,
    ):
        c, h, w = shape
        assert overlap_size == w // 2, "overlap_size must equal w // 2"

        self.net = net
        self.num_img = num_img
        self.left_class_idx = left_class_idx
        self.right_class_idx = right_class_idx
        self.overlap_size = overlap_size
        self.guidance_scale = guidance_scale

        final_img_w = w * num_img - overlap_size * (num_img - 1)

        super().__init__(
            (c, h, final_img_w),
            self.get_eps_t_fn(),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _edm_eps(self, xs, scalar_t, class_idx=None, enable_grad=False):
        eps, _ = _edm_predict(self.net, xs, scalar_t, class_idx, enable_grad)
        return eps

    def _edm_x0(self, xs, scalar_t, class_idx=None, enable_grad=False):
        _, x0 = _edm_predict(self.net, xs, scalar_t, class_idx, enable_grad)
        return x0

    def _all_patch_eps(self, xs, scalar_t, enable_grad=False):
        """
        Compute per-patch eps with endpoint CFG and unconditional middle.

        Uses 3 model calls regardless of num_img:
          1. uncond forward on all patches (flattened)
          2. cond forward on patch 0 only (left_class)
          3. cond forward on patch N-1 only (right_class)

        xs: [B, N, C, H, W]
        returns: [B, N, C, H, W]
        """
        B, N = xs.shape[:2]

        # 1. Uncond for all patches
        xs_flat = rearrange(xs, "b n c h w -> (b n) c h w")
        eps_uncond_flat = self._edm_eps(xs_flat, scalar_t, class_idx=None, enable_grad=enable_grad)
        eps = rearrange(eps_uncond_flat, "(b n) c h w -> b n c h w", b=B)

        if self.guidance_scale == 0:
            return eps

        # 2. Left endpoint CFG (patch 0)
        eps_uncond_left = eps[:, 0].clone()
        eps_cond_left = self._edm_eps(
            xs[:, 0], scalar_t, class_idx=self.left_class_idx, enable_grad=enable_grad
        )
        eps[:, 0] = eps_uncond_left + self.guidance_scale * (eps_cond_left - eps_uncond_left)

        # 3. Right endpoint CFG (patch N-1)
        eps_uncond_right = eps[:, -1].clone()
        eps_cond_right = self._edm_eps(
            xs[:, -1], scalar_t, class_idx=self.right_class_idx, enable_grad=enable_grad
        )
        eps[:, -1] = eps_uncond_right + self.guidance_scale * (eps_cond_right - eps_uncond_right)

        return eps

    def _compute_overlap_eps(self, xs, scalar_t, enable_grad=False):
        """
        Compute eps for the right-half overlaps of patches 0..N-2.

        Used by DiffCollage to subtract implicit overlap marginals.
        Only patch 0's overlap uses left_class CFG; all others are uncond.

        xs: [B, N, C, H, W]
        returns: [B, N-1, C, H, overlap_size]
        """
        B = xs.shape[0]

        # Right-half overlaps for patches 0 to N-2
        overlaps = xs[:, :-1, :, :, -self.overlap_size:]  # [B, N-1, C, H, overlap]
        overlaps_flat = rearrange(overlaps, "b n c h w -> (b n) c h w")

        eps_uncond_flat = self._edm_eps(
            overlaps_flat, scalar_t, class_idx=None, enable_grad=enable_grad
        )
        eps_ovlp = rearrange(eps_uncond_flat, "(b n) c h w -> b n c h w", b=B)

        if self.guidance_scale != 0:
            eps_uncond_left_ovlp = eps_ovlp[:, 0].clone()
            eps_cond_left_ovlp = self._edm_eps(
                overlaps[:, 0], scalar_t, class_idx=self.left_class_idx, enable_grad=enable_grad
            )
            eps_ovlp[:, 0] = eps_uncond_left_ovlp + self.guidance_scale * (
                eps_cond_left_ovlp - eps_uncond_left_ovlp
            )

        return eps_ovlp

    def _all_patch_x0(self, xs, scalar_t, enable_grad=False):
        """
        Compute CFG x0 for all patches (same structure as _all_patch_eps).

        xs: [B, N, C, H, W]
        returns: [B, N, C, H, W]
        """
        B, N = xs.shape[:2]

        xs_flat = rearrange(xs, "b n c h w -> (b n) c h w")
        _, x0_uncond_flat = _edm_predict(
            self.net, xs_flat, scalar_t, class_idx=None, enable_grad=enable_grad
        )
        x0 = rearrange(x0_uncond_flat, "(b n) c h w -> b n c h w", b=B)

        if self.guidance_scale == 0:
            return x0

        # Left endpoint
        x0_uncond_left = x0[:, 0].clone()
        _, x0_cond_left = _edm_predict(
            self.net, xs[:, 0], scalar_t, class_idx=self.left_class_idx, enable_grad=enable_grad
        )
        x0[:, 0] = x0_uncond_left + self.guidance_scale * (x0_cond_left - x0_uncond_left)

        # Right endpoint
        x0_uncond_right = x0[:, -1].clone()
        _, x0_cond_right = _edm_predict(
            self.net, xs[:, -1], scalar_t, class_idx=self.right_class_idx, enable_grad=enable_grad
        )
        x0[:, -1] = x0_uncond_right + self.guidance_scale * (x0_cond_right - x0_uncond_right)

        return x0

    def _split_patches(self, long_x):
        B = long_x.shape[0]
        xs = split_wimg(long_x, self.num_img, rtn_overlap=False)
        return rearrange(xs, "(b n) c h w -> b n c h w", n=self.num_img)

    def _merge_eps(self, full_eps):
        whole_eps = rearrange(full_eps, "b n c h w -> (b n) c h w")
        return avg_merge_wimg(whole_eps, self.overlap_size, n=self.num_img, is_avg=False)

    def attach_fixed_ends(self, generated_long_x):
        # No image strips to attach with text conditioning.
        return generated_long_x

    def reset_fixed_end_noise(self):
        pass

    def get_eps_t_fn(self):
        raise NotImplementedError


# ── three worker variants ─────────────────────────────────────────────────────

class DiffCollageBridgeWorkerText(_TextBridgeBase):
    """
    DiffCollage bridge composition with text (class-conditional) endpoints.

    Computes:
        s_dc = sum_i s_i - sum_j s_overlap_j
    where s_i uses CFG for endpoint patches and uncond for middle patches.
    """

    def get_eps_t_fn(self):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            xs = self._split_patches(long_x)
            full_eps = self._all_patch_eps(xs, scalar_t, enable_grad)
            overlap_eps = self._compute_overlap_eps(xs, scalar_t, enable_grad)
            full_eps[:, :-1, :, :, -self.overlap_size:] -= overlap_eps
            return self._merge_eps(full_eps)

        return eps_t_fn


class NaiveBridgeWorkerText(_TextBridgeBase):
    """
    Naive product bridge with text (class-conditional) endpoints.

    No overlap marginal subtraction — the overlap is double-counted.
    Computes:
        s_naive = sum_i s_i
    """

    def get_eps_t_fn(self):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            xs = self._split_patches(long_x)
            full_eps = self._all_patch_eps(xs, scalar_t, enable_grad)
            return self._merge_eps(full_eps)

        return eps_t_fn


class ProposalFinalBridgeWorkerText(_TextBridgeBase):
    """
    Proposal-final bridge with text endpoints.

    Adds a Tweedie x0-consistency correction Δs on top of the DiffCollage base:
        s = s_dc + Δs
    where Δs pushes adjacent patches to agree on their denoised x0 at the overlap.
    """

    def __init__(
        self,
        shape,
        net,
        num_img,
        left_class_idx,
        right_class_idx,
        overlap_size=32,
        guidance_scale=1.0,
        sigma_max=80.0,
        sigma_min=1e-3,
        coupling_strength=0.25,
        correction_clip=4.0,
        sigma_data=0.5,
    ):
        self.coupling_strength = coupling_strength
        self.correction_clip = correction_clip
        self.sigma_data = sigma_data
        super().__init__(
            shape=shape,
            net=net,
            num_img=num_img,
            left_class_idx=left_class_idx,
            right_class_idx=right_class_idx,
            overlap_size=overlap_size,
            guidance_scale=guidance_scale,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    def _log_r_delta_eps(self, xs, x0, scalar_t):
        """
        Δs = grad log R via Tweedie x0 overlap consistency.

        For each adjacent pair (i, i+1), the residual between their x0
        estimates at the shared overlap pushes both patches toward agreement.
        """
        device = xs.device
        if not th.is_tensor(scalar_t):
            sigma = th.tensor(float(scalar_t), device=device, dtype=xs.dtype)
        else:
            sigma = scalar_t.to(device).to(xs.dtype)
        sigma = sigma.clamp_min(1e-3)
        if sigma.ndim == 0:
            sigma = sigma.view(1, 1, 1, 1)

        scale = self.coupling_strength * sigma / (sigma ** 2 + self.sigma_data ** 2)

        patch_delta = th.zeros_like(xs)
        for idx in range(self.num_img - 1):
            residual = (
                x0[:, idx, :, :, -self.overlap_size:]
                - x0[:, idx + 1, :, :, :self.overlap_size]
            )
            patch_delta[:, idx, :, :, -self.overlap_size:] += scale * residual
            patch_delta[:, idx + 1, :, :, :self.overlap_size] -= scale * residual

        patch_delta = patch_delta.clamp(-self.correction_clip, self.correction_clip)
        whole_delta = rearrange(patch_delta, "b n c h w -> (b n) c h w")
        return avg_merge_wimg(whole_delta, self.overlap_size, n=self.num_img, is_avg=True)

    def get_eps_t_fn(self):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            xs = self._split_patches(long_x)

            # DiffCollage base
            full_eps = self._all_patch_eps(xs, scalar_t, enable_grad)
            overlap_eps = self._compute_overlap_eps(xs, scalar_t, enable_grad)
            full_eps[:, :-1, :, :, -self.overlap_size:] -= overlap_eps
            merged_eps = self._merge_eps(full_eps)

            if self.coupling_strength == 0:
                return merged_eps

            # Δs correction
            x0 = self._all_patch_x0(xs, scalar_t, enable_grad)
            return merged_eps + self._log_r_delta_eps(xs, x0, scalar_t)

        return eps_t_fn
