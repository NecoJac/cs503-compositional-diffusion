"""
Stable Diffusion 1.5 bridge generation with DiffCollage patch composition.

Replaces the EDM 64×64 pixel-space backbone with SD1.5 in latent space.
Each latent patch is 64×64 (= 512×512 pixels). With overlap=32 latents
(= 256 pixels) and num_img=3 patches, the output is 512×1024 pixels.

Three composition methods mirror the CBG-Diffusion paper:
  naive         -- average overlapping noise predictions (MultiDiffusion)
  diffcollage   -- subtract implicit overlap marginals
  bridge_correction -- DiffCollage + Tweedie x0 overlap consistency Δs

Usage:
    from proposal_methods.sd_bridge import SDBridgeModel, generate_sd_bridge
    model = SDBridgeModel()
    image = generate_sd_bridge(model, "a lakeside landscape", "a volcano", method="diffcollage")
"""

import numpy as np
import PIL.Image
import torch as th
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


SD_DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
PATCH_LATENT = 64    # UNet designed for 64×64 latents = 512×512 image pixels
VAE_SCALE = 0.18215  # SD1.5 VAE scaling constant


# ── Model wrapper ─────────────────────────────────────────────────────────────

class SDBridgeModel:
    """Loads SD1.5 components and provides helpers for bridge generation."""

    def __init__(self, model_id=SD_DEFAULT_MODEL, device="cuda", dtype=th.float16):
        self.device = device
        self.dtype = dtype
        print(f"Loading SD model: {model_id}")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype).eval()
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device, dtype).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype).eval()
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        print("SD model loaded.")

    # ── text encoding ─────────────────────────────────────────────────────────

    def encode_text(self, prompts):
        """Encode text prompts to CLIP embeddings [B, 77, 768]."""
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        with th.no_grad():
            emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
        return emb.to(self.dtype)

    # ── VAE helpers ───────────────────────────────────────────────────────────

    def encode_image(self, pil_image, target_size=512):
        """Encode a PIL image to scaled latents [1, 4, H/8, W/8]."""
        img = pil_image.convert("RGB").resize((target_size, target_size), PIL.Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        tensor = th.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype)
        with th.no_grad():
            latent = self.vae.encode(tensor).latent_dist.sample() * VAE_SCALE
        return latent

    def decode_latents(self, latents):
        """Decode scaled latents to pixel image tensor in [-1, 1]."""
        with th.no_grad():
            images = self.vae.decode(latents / VAE_SCALE).sample
        return images.clamp(-1, 1)

    def latents_to_pil(self, latents):
        """Decode latents and convert to PIL Image."""
        images = self.decode_latents(latents)
        images = (images.float().cpu().permute(0, 2, 3, 1).numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return [PIL.Image.fromarray(img) for img in images]

    # ── UNet prediction helpers ───────────────────────────────────────────────

    def _t_tensor(self, t_scalar, batch_size):
        """Broadcast a scalar timestep to [B] on the correct device."""
        if not th.is_tensor(t_scalar):
            t_scalar = th.tensor(t_scalar, device=self.device)
        return t_scalar.to(self.device).long().view(1).expand(batch_size).contiguous()

    @th.no_grad()
    def predict_eps(self, latents, t, emb):
        """Run UNet: returns epsilon prediction [B, 4, H, W]."""
        B = latents.shape[0]
        t_b = self._t_tensor(t, B)
        return self.unet(latents, t_b, encoder_hidden_states=emb.expand(B, -1, -1)).sample

    @th.no_grad()
    def cfg_eps(self, latents, t, cond_emb, uncond_emb, guidance_scale):
        """CFG epsilon prediction with a single batched UNet call."""
        B = latents.shape[0]
        t_b = self._t_tensor(t, B * 2)
        batch_lat = th.cat([latents, latents])
        batch_emb = th.cat([uncond_emb.expand(B, -1, -1), cond_emb.expand(B, -1, -1)])
        eps_u, eps_c = self.unet(batch_lat, t_b, encoder_hidden_states=batch_emb).sample.chunk(2)
        return eps_u + guidance_scale * (eps_c - eps_u)

    def eps_to_x0(self, latents, eps, t):
        """Convert epsilon prediction to x0 estimate using DDIM formula."""
        t_idx = t.long().item() if th.is_tensor(t) else int(t)
        alpha_prod = self.scheduler.alphas_cumprod[t_idx].to(latents.device, latents.dtype)
        beta_prod = 1.0 - alpha_prod
        return (latents - beta_prod.sqrt() * eps) / alpha_prod.sqrt()


# ── CLIP embedding interpolation ─────────────────────────────────────────────

def _slerp_emb(a, b, t):
    """Spherical linear interpolation between two CLIP embedding tensors.

    Falls back to lerp when the vectors are nearly parallel.
    a, b: [1, 77, 768];  t: scalar in [0, 1]
    """
    a_n = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b_n = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_theta = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = cos_theta.acos()
    sin_theta = theta.sin()
    near_parallel = sin_theta.abs() < 1e-6
    slerp = (
        ((1.0 - t) * theta).sin() / sin_theta.clamp(min=1e-8) * a
        + (t * theta).sin() / sin_theta.clamp(min=1e-8) * b
    )
    return th.where(near_parallel, (1.0 - t) * a + t * b, slerp)


def _build_patch_embs(waypoint_embs, num_img):
    """Per-patch CLIP embeddings via piecewise slerp across waypoints.

    waypoint_embs: list of (B, seq, hidden) tensors, length >= 2.
      [left_emb, (optional middle embs...), right_emb]
    Waypoints are evenly spaced across the num_img windows; each segment
    between consecutive waypoints is slerp-interpolated independently.
    """
    if num_img == 1:
        return [waypoint_embs[0]]
    n_segs = len(waypoint_embs) - 1
    result = []
    for i in range(num_img):
        t = i / (num_img - 1)           # global position in [0, 1]
        seg_f = t * n_segs              # which segment (float)
        seg_idx = min(int(seg_f), n_segs - 1)
        seg_t = seg_f - seg_idx         # local position within segment [0, 1]
        result.append(_slerp_emb(waypoint_embs[seg_idx], waypoint_embs[seg_idx + 1], seg_t))
    return result


# ── Patch windowing helpers ───────────────────────────────────────────────────

def _window_positions(num_img, patch_size, stride):
    return [(i * stride, i * stride + patch_size) for i in range(num_img)]


def _split_windows(latents, num_img, patch_size, stride):
    """Extract overlapping windows from the long latent."""
    return [latents[:, :, :, start:end] for start, end in _window_positions(num_img, patch_size, stride)]


def _merge_windows(window_eps, corr_list, num_img, patch_size, stride, total_w, device, dtype):
    """
    Merge per-window noise predictions.

    corr_list: list of (start, end, eps_tensor) corrections to subtract
               (DiffCollage overlap marginals).  Pass [] for naive.
    """
    B, C = window_eps[0].shape[:2]
    H = window_eps[0].shape[2]
    noise_sum = th.zeros(B, C, H, total_w, device=device, dtype=dtype)
    count     = th.zeros(B, 1, H, total_w, device=device, dtype=dtype)

    for i, eps in enumerate(window_eps):
        s = i * stride
        noise_sum[:, :, :, s:s + patch_size] += eps
        count    [:, :, :, s:s + patch_size] += 1

    for start, end, eps in corr_list:
        noise_sum[:, :, :, start:end] -= eps
        count    [:, :, :, start:end] -= 1

    return noise_sum / count.clamp(min=1)


# ── Bridge generation ─────────────────────────────────────────────────────────

@th.no_grad()
def generate_sd_bridge(
    model: SDBridgeModel,
    left_prompt,           # str or list[str]; if list, batched B = len(list)
    right_prompt,          # str or list[str]; must match length of left_prompt
    method: str = "diffcollage",
    num_img: int = 3,
    overlap_latents: int = 32,
    n_steps: int = 50,
    guidance_scale: float = 7.5,
    coupling_strength: float = 0.05,
    correction_clip: float = 1.0,
    seed: int = None,
    middle_prompts: list = None,   # optional waypoint prompts between left and right
):
    """
    Generate a 512×(512 + (num_img-1)*overlap_latents*8) pixel bridge image.

    Each patch receives a slerp-interpolated CLIP embedding (α=0..1 left→right),
    giving every position a semantically graded conditioning signal.
    Returns (pil_image, latents).
    """
    patch = PATCH_LATENT
    stride = patch - overlap_latents
    total_w = patch + (num_img - 1) * stride
    device, dtype = model.device, model.dtype

    # Normalize prompts to lists for batching (B = len(left_prompt))
    if isinstance(left_prompt, str):
        left_prompt = [left_prompt]
        right_prompt = [right_prompt]
    B = len(left_prompt)
    assert len(right_prompt) == B, f"prompt list length mismatch: {B} vs {len(right_prompt)}"

    # Text embeddings (batched)
    uncond_emb = model.encode_text("")                                      # (1, seq, hidden)
    left_embs  = th.cat([model.encode_text(p) for p in left_prompt],  dim=0)  # (B, seq, hidden)
    right_embs = th.cat([model.encode_text(p) for p in right_prompt], dim=0)  # (B, seq, hidden)

    # Waypoint embeddings: [left, (middles...), right]
    mid_embs = [model.encode_text(p).expand(B, -1, -1) for p in (middle_prompts or [])]
    waypoint_embs = [left_embs] + mid_embs + [right_embs]
    patch_embs = _build_patch_embs(waypoint_embs, num_img)

    # Initial noise (B, 4, patch, total_w)
    if seed is not None:
        th.manual_seed(seed)
    latents = th.randn(B, 4, patch, total_w, device=device, dtype=dtype)

    # DDIM schedule
    model.scheduler.set_timesteps(n_steps)
    latents = latents * model.scheduler.init_noise_sigma

    for t in model.scheduler.timesteps:
        latents = _denoise_step(
            model, latents, t,
            uncond_emb, patch_embs,
            guidance_scale, num_img, overlap_latents, patch, stride, total_w,
            method, coupling_strength, correction_clip,
        )

    pil_images = model.latents_to_pil(latents)
    # Backward-compat: scalar return when single prompt (B == 1); list otherwise
    if len(pil_images) == 1:
        return pil_images[0], latents
    return pil_images, latents


def _denoise_step(
    model, latents, t,
    uncond_emb, patch_embs,
    guidance_scale, num_img, overlap, patch, stride, total_w,
    method, coupling_strength, correction_clip,
):
    """One denoising step with patch composition."""
    if method == "proposal_final":
        method = "bridge_correction"
    windows = _split_windows(latents, num_img, patch, stride)

    # ── Per-window CFG with slerp-interpolated embeddings ─────────────────────
    window_eps = [
        model.cfg_eps(w, t, patch_embs[i], uncond_emb, guidance_scale)
        for i, w in enumerate(windows)
    ]

    # ── Compose predictions ───────────────────────────────────────────────────
    if method == "naive":
        composed = _merge_windows(
            window_eps, [], num_img, patch, stride, total_w, model.device, model.dtype
        )

    elif method == "diffcollage":
        composed = _compose_diffcollage(
            model, windows, window_eps, t, uncond_emb, patch_embs,
            num_img, overlap, patch, stride, total_w, guidance_scale,
        )

    elif method == "product":
        composed = _compose_product(
            window_eps, num_img, patch, stride, total_w, model.device, model.dtype
        )

    elif method == "bridge_correction":
        composed = _compose_proposal(
            model, latents, windows, window_eps, t, uncond_emb, patch_embs,
            num_img, overlap, patch, stride, total_w, guidance_scale,
            coupling_strength, correction_clip,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Scheduler step on the full long latent ────────────────────────────────
    return model.scheduler.step(composed, t, latents).prev_sample


def _compose_diffcollage(
    model, windows, window_eps, t, uncond_emb, patch_embs,
    num_img, overlap, patch, stride, total_w, guidance_scale,
):
    """
    DiffCollage: subtract implicit overlap marginal predictions.

    s_dc = Σ s_i - Σ s_overlap_i

    Each overlap marginal uses the same slerp embedding as its parent window,
    so the correction is semantically consistent with the main prediction.
    """
    corr_list = []
    for i in range(num_img - 1):
        ovlp_latent = windows[i][:, :, :, -overlap:]
        ovlp_eps = model.cfg_eps(ovlp_latent, t, patch_embs[i], uncond_emb, guidance_scale)
        ovlp_start = i * stride + (patch - overlap)
        corr_list.append((ovlp_start, ovlp_start + overlap, ovlp_eps))

    return _merge_windows(
        window_eps, corr_list, num_img, patch, stride, total_w, model.device, model.dtype
    )


def _compose_product(window_eps, num_img, patch, stride, total_w, device, dtype):
    """
    Product-of-experts approximation: average window predictions.

    The true PoE score (unnormalized sum) doubles eps magnitude in overlap
    regions, which breaks DDIM's unit-Gaussian assumption and causes collapse.
    Averaging is the practical DDIM-compatible approximation.
    """
    B, C = window_eps[0].shape[:2]
    H = window_eps[0].shape[2]
    noise_sum = th.zeros(B, C, H, total_w, device=device, dtype=dtype)
    count     = th.zeros(B, 1, H, total_w, device=device, dtype=dtype)
    for i, eps in enumerate(window_eps):
        s = i * stride
        noise_sum[:, :, :, s:s + patch] += eps
        count    [:, :, :, s:s + patch] += 1
    return noise_sum / count.clamp(min=1)


def _compose_proposal(
    model, latents, windows, window_eps, t, uncond_emb, patch_embs,
    num_img, overlap, patch, stride, total_w, guidance_scale,
    coupling_strength, correction_clip,
):
    """
    Proposal-final: DiffCollage base + Tweedie x0 overlap consistency Δs.

    Δs: single-sided correction at each overlap; pushes adjacent x0 estimates
    toward agreement.  The previous symmetric ±r write was a no-op because
    ovlp_start == next_start whenever stride = patch - overlap.
    """
    composed = _compose_diffcollage(
        model, windows, window_eps, t, uncond_emb, patch_embs,
        num_img, overlap, patch, stride, total_w, guidance_scale,
    )

    if coupling_strength == 0:
        return composed

    x0_windows = [model.eps_to_x0(windows[i], window_eps[i], t) for i in range(num_img)]

    alpha_prod = model.scheduler.alphas_cumprod[t].to(latents.device, latents.dtype)
    scale = coupling_strength / alpha_prod.sqrt().clamp(min=1e-3)

    delta = th.zeros_like(latents)
    for i in range(num_img - 1):
        residual = x0_windows[i][:, :, :, -overlap:] - x0_windows[i + 1][:, :, :, :overlap]
        ovlp_start = i * stride + (patch - overlap)
        delta[:, :, :, ovlp_start:ovlp_start + overlap] += scale * residual

    delta = delta.clamp(-correction_clip, correction_clip)
    return composed + delta



@th.no_grad()
def generate_sd_bridge_smc(
    model: SDBridgeModel,
    left_prompt: str,
    right_prompt: str,
    num_img: int = 3,
    overlap_latents: int = 32,
    n_steps: int = 50,
    guidance_scale: float = 7.5,
    K: int = 4,
    beta: float = 1.0,
    t_resample_start: float = 0.0,
    t_resample_end: float = 0.8,
    coupling_strength: float = 0.0,
    correction_clip: float = 1.0,
    seed: int = None,
    middle_prompts: list = None,   # optional waypoint prompts between left and right
):
    """K-particle SMC for SD 1.5 bridge. Returns (best_pil, best_latents)."""
    patch = PATCH_LATENT
    stride = patch - overlap_latents
    total_w = patch + (num_img - 1) * stride
    overlap = overlap_latents
    device, dtype = model.device, model.dtype

    uncond_emb = model.encode_text("")
    left_emb = model.encode_text(left_prompt)
    right_emb = model.encode_text(right_prompt)
    mid_embs = [model.encode_text(p) for p in (middle_prompts or [])]
    patch_embs = _build_patch_embs([left_emb] + mid_embs + [right_emb], num_img)

    if seed is not None:
        th.manual_seed(seed)
    latents = th.randn(K, 4, patch, total_w, device=device, dtype=dtype)
    log_w = th.zeros(K, device=device, dtype=th.float32)

    model.scheduler.set_timesteps(n_steps)
    latents = latents * model.scheduler.init_noise_sigma

    T_total = len(model.scheduler.timesteps)
    t_lo_idx = max(0, int(t_resample_start * T_total))
    t_hi_idx = min(T_total - 1, int(t_resample_end * T_total))

    for t_idx, t in enumerate(model.scheduler.timesteps):
        windows = _split_windows(latents, num_img, patch, stride)
        window_eps_list = [
            model.cfg_eps(w, t, patch_embs[i], uncond_emb, guidance_scale)
            for i, w in enumerate(windows)
        ]

        # ── FK potential: negative Tweedie x0 overlap disagreement ──
        # x0 disagreement in latent space directly proxies seam MSE;
        # eps inner product is a weak signal; the final report discusses this limitation.
        x0_list = [
            model.eps_to_x0(windows[i], window_eps_list[i], t)
            for i in range(num_img)
        ]
        x0_disagree = sum(
            ((x0_list[i][..., -overlap:] - x0_list[i + 1][..., :overlap]) ** 2)
            .float().flatten(1).sum(dim=1)
            for i in range(num_img - 1)
        )
        overlap_dims = max(1, 4 * windows[0].shape[-2] * overlap * (num_img - 1))
        log_w = log_w - beta * x0_disagree / overlap_dims

        # ── SMC systematic resample BEFORE ODE step (FKC-correct order) ──
        if t_lo_idx <= t_idx <= t_hi_idx:
            probs = th.softmax(log_w, dim=0)
            cdf = th.cumsum(probs, dim=0)
            u = th.rand(1, device=device)
            pos = (u + th.arange(K, device=device).float()) / K
            indices = th.searchsorted(cdf, pos).clamp(max=K - 1)
            latents = latents[indices]
            window_eps_list = [eps[indices] for eps in window_eps_list]
            log_w = th.zeros(K, device=device, dtype=th.float32)

        # ── Naive PoE merge and DDIM step ──
        composed = _merge_windows(
            window_eps_list, [], num_img, patch, stride, total_w, device, dtype,
        )
        latents = model.scheduler.step(composed, t, latents).prev_sample

    # SNIS: pick best particle
    best_idx = int(log_w.argmax().item())
    best_lat = latents[best_idx : best_idx + 1]
    pil_images = model.latents_to_pil(best_lat)
    return pil_images[0], best_lat
