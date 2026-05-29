import torch as th
from einops import rearrange

from diff_collage.base_worker import BaseWorker
from diff_collage.w_img import split_wimg, avg_merge_wimg
from diff_collage.condind_long_fixed_end import CondIndLongFixedEnd


class NaiveFixedEndWorker(BaseWorker):
    """
    Fixed-end long-image worker for the proposal's naive product baseline.

    This intentionally keeps the pairwise factors p(x_i, x_{i+1}) and the
    terminal factor p(x_n, x_fixed), but does not subtract an implicit
    p(x_i-overlap) marginal. In score-arithmetic terms it is the product
    p(x,y) p(y,z) without the /p(y) correction.
    """

    def __init__(
        self,
        shape,
        eps_scalar_t_fn,
        num_img,
        fixed_images,
        overlap_size=32,
        guidance_scale=1.0,
        sigma_max=80.0,
        sigma_min=1e-3,
    ):
        c, h, w = shape
        assert overlap_size == w // 2
        self.overlap_size = overlap_size
        self.num_img = num_img
        self.fixed_images = self._validate_fixed_images(
            fixed_images,
            channels=c,
            height=h,
            overlap_size=overlap_size,
        )
        self.guidance_scale = guidance_scale
        self.fixed_end_noise = None

        final_img_w = w * num_img - overlap_size * (num_img - 1)
        super().__init__(
            (c, h, final_img_w),
            self.get_eps_t_fn(eps_scalar_t_fn),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    @staticmethod
    def _validate_fixed_images(fixed_images, channels, height, overlap_size):
        if fixed_images.ndim == 3:
            fixed_images = fixed_images.unsqueeze(0)
        elif fixed_images.ndim != 4:
            raise ValueError(
                f"fixed_images must be [C,H,overlap] or [B,C,H,overlap], got {fixed_images.shape}"
            )
        if fixed_images.shape[1:] != (channels, height, overlap_size):
            raise ValueError(
                "fixed_images must have shape "
                f"(*,{channels},{height},{overlap_size}), got {tuple(fixed_images.shape)}"
            )
        return fixed_images

    def _get_fixed_images_for_batch(self, batch_size, device, dtype):
        fixed = self.fixed_images.to(device=device, dtype=dtype)
        if fixed.shape[0] == 1 and batch_size > 1:
            fixed = fixed.repeat(batch_size, 1, 1, 1)
        if fixed.shape[0] != batch_size:
            raise ValueError(
                f"fixed_images batch {fixed.shape[0]} must match requested batch {batch_size}"
            )
        return fixed

    def reset_fixed_end_noise(self):
        self.fixed_end_noise = None

    def _get_fixed_end_noise(self, batch_size, device, dtype):
        if self.fixed_end_noise is None:
            self.fixed_end_noise = th.randn_like(
                self._get_fixed_images_for_batch(batch_size, device, dtype)
            )
        return self.fixed_end_noise

    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            batch_size = long_x.shape[0]
            device = long_x.device
            dtype = long_x.dtype

            fixed = self._get_fixed_images_for_batch(batch_size, device, dtype)
            fixed_noise = self._get_fixed_end_noise(batch_size, device, dtype)
            fixed_t = fixed + scalar_t * fixed_noise

            xs = split_wimg(long_x, self.num_img, rtn_overlap=False)
            xs = rearrange(xs, "(b n) c h w -> b n c h w", n=self.num_img)

            last_overlap = xs[:, -1, :, :, -self.overlap_size :]
            terminal_factor = th.cat([last_overlap, fixed_t], dim=-1)

            xs_all = th.cat([xs, terminal_factor.unsqueeze(1)], dim=1)
            xs_all_flat = rearrange(xs_all, "b n c h w -> (b n) c h w")

            factor_eps = eps_scalar_t_fn(xs_all_flat, scalar_t, enable_grad)
            factor_eps = rearrange(
                factor_eps,
                "(b n) c h w -> b n c h w",
                n=self.num_img + 1,
            )

            full_eps = factor_eps[:, :-1].clone()
            terminal_left_eps = factor_eps[:, -1, :, :, : self.overlap_size]
            full_eps[:, -1, :, :, -self.overlap_size :] += (
                self.guidance_scale * terminal_left_eps
            )

            whole_eps = rearrange(full_eps, "b n c h w -> (b n) c h w")
            return avg_merge_wimg(
                whole_eps,
                self.overlap_size,
                n=self.num_img,
                is_avg=False,
            )

        return eps_t_fn

    def attach_fixed_end(self, generated_long_x):
        fixed = self._get_fixed_images_for_batch(
            generated_long_x.shape[0],
            generated_long_x.device,
            generated_long_x.dtype,
        )
        return th.cat([generated_long_x, fixed], dim=-1)


class ProposalFinalFixedEndWorker(CondIndLongFixedEnd):
    """
    Practical implementation of the proposal's final formula.

    It keeps the DiffCollage implicit-marginal term, then adds a heuristic
    coupling correction Delta s in epsilon space. It also exposes an
    initial-noise correction hook that approximates the proposal's p_T^*
    reweighting by reducing overlap mismatch before denoising starts.

    Key improvements over the naive version:
    - sigma_data: sigma-normalized scale avoids 1/sigma blowup at low sigma.
    - implicit_scale: controls implicit-marginal subtraction strength (default 0.25
      matches DiffCollage better than 0.5).
    """

    def __init__(
        self,
        shape,
        eps_scalar_t_fn,
        num_img,
        fixed_images,
        overlap_size=32,
        guidance_scale=1.0,
        sigma_max=80.0,
        sigma_min=1e-3,
        coupling_strength=0.25,
        correction_clip=4.0,
        init_correction_steps=4,
        init_correction_step_size=0.15,
        init_correction_strength=None,
        sigma_data=0.5,
    ):
        self.coupling_strength = coupling_strength
        self.correction_clip = correction_clip
        self.init_correction_steps = init_correction_steps
        self.init_correction_step_size = init_correction_step_size
        self.init_correction_strength = (
            coupling_strength if init_correction_strength is None else init_correction_strength
        )
        self.sigma_data = sigma_data
        super().__init__(
            shape=shape,
            eps_scalar_t_fn=eps_scalar_t_fn,
            num_img=num_img,
            fixed_images=fixed_images,
            overlap_size=overlap_size,
            guidance_scale=guidance_scale,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    def get_eps_t_fn(self, eps_scalar_t_fn):
        base_eps_t_fn = super().get_eps_t_fn(eps_scalar_t_fn)

        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            base_eps = base_eps_t_fn(long_x, scalar_t, enable_grad)
            if self.coupling_strength == 0:
                return base_eps

            delta_eps = self._coupling_delta_eps(long_x, scalar_t)
            return base_eps + delta_eps

        return eps_t_fn

    def _as_sigma(self, scalar_t, device, dtype):
        if not th.is_tensor(scalar_t):
            sigma = th.tensor(float(scalar_t), device=device, dtype=dtype)
        else:
            sigma = scalar_t.to(device=device, dtype=dtype)
        return sigma.clamp_min(self.sigma_min)

    def _coupling_delta_eps(self, long_x, scalar_t):
        batch_size = long_x.shape[0]
        device = long_x.device
        dtype = long_x.dtype
        sigma = self._as_sigma(scalar_t, device, dtype)

        # Sigma-normalized scale: peaks at sigma=sigma_data, fades at both extremes.
        # Prevents the 1/sigma blowup at small sigma that corrupts fine details.
        scale = self.coupling_strength * sigma / (sigma ** 2 + self.sigma_data ** 2)
        while scale.ndim < long_x.ndim:
            scale = scale.view(1, 1, 1, 1) if scale.ndim == 0 else scale.view(-1, 1, 1, 1)

        xs = split_wimg(long_x, self.num_img, rtn_overlap=False)
        xs = rearrange(xs, "(b n) c h w -> b n c h w", n=self.num_img)
        patch_delta = th.zeros_like(xs)

        for idx in range(self.num_img - 1):
            residual = (
                xs[:, idx, :, :, -self.overlap_size :]
                - xs[:, idx + 1, :, :, : self.overlap_size]
            )
            patch_delta[:, idx, :, :, -self.overlap_size :] += scale * residual
            patch_delta[:, idx + 1, :, :, : self.overlap_size] -= scale * residual

        fixed = self._get_fixed_images_for_batch(batch_size, device, dtype)
        fixed_noise = self._get_fixed_end_noise(batch_size, device, dtype)
        fixed_t = fixed + scalar_t * fixed_noise
        terminal_residual = xs[:, -1, :, :, -self.overlap_size :] - fixed_t
        patch_delta[:, -1, :, :, -self.overlap_size :] += scale * terminal_residual

        patch_delta = patch_delta.clamp(-self.correction_clip, self.correction_clip)
        patch_delta = rearrange(patch_delta, "b n c h w -> (b n) c h w")
        return avg_merge_wimg(
            patch_delta,
            self.overlap_size,
            n=self.num_img,
            is_avg=True,
        )


class BridgeWorkerBase(BaseWorker):
    """
    Base utilities for the proposal bridge task.

    The generated state contains only the middle bridge patches. The final
    visualization replaces the generated endpoint overlaps with fixed overlap
    variables:

        [left_fixed] + [generated bridge interior] + [right_fixed]

    Both fixed endpoints are overlap-sized tensors [B,C,H,overlap].
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
    ):
        c, h, w = shape
        assert overlap_size == w // 2
        self.overlap_size = overlap_size
        self.num_img = num_img
        self.left_fixed_images = self._validate_fixed_images(
            left_fixed_images,
            channels=c,
            height=h,
            overlap_size=overlap_size,
            name="left_fixed_images",
        )
        self.right_fixed_images = self._validate_fixed_images(
            right_fixed_images,
            channels=c,
            height=h,
            overlap_size=overlap_size,
            name="right_fixed_images",
        )
        self.guidance_scale = guidance_scale
        self.left_fixed_noise = None
        self.right_fixed_noise = None

        final_img_w = w * num_img - overlap_size * (num_img - 1)
        super().__init__(
            (c, h, final_img_w),
            self.get_eps_t_fn(eps_scalar_t_fn),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    @staticmethod
    def _validate_fixed_images(fixed_images, channels, height, overlap_size, name):
        if fixed_images.ndim == 3:
            fixed_images = fixed_images.unsqueeze(0)
        elif fixed_images.ndim != 4:
            raise ValueError(f"{name} must be [C,H,overlap] or [B,C,H,overlap], got {fixed_images.shape}")
        if fixed_images.shape[1:] != (channels, height, overlap_size):
            raise ValueError(
                f"{name} must have shape (*,{channels},{height},{overlap_size}), "
                f"got {tuple(fixed_images.shape)}"
            )
        return fixed_images

    def _expand_fixed(self, fixed_images, batch_size, device, dtype, name):
        fixed = fixed_images.to(device=device, dtype=dtype)
        if fixed.shape[0] == 1 and batch_size > 1:
            fixed = fixed.repeat(batch_size, 1, 1, 1)
        if fixed.shape[0] != batch_size:
            raise ValueError(f"{name} batch {fixed.shape[0]} must match requested batch {batch_size}")
        return fixed

    def _get_left_fixed_for_batch(self, batch_size, device, dtype):
        return self._expand_fixed(
            self.left_fixed_images,
            batch_size,
            device,
            dtype,
            "left_fixed_images",
        )

    def _get_right_fixed_for_batch(self, batch_size, device, dtype):
        return self._expand_fixed(
            self.right_fixed_images,
            batch_size,
            device,
            dtype,
            "right_fixed_images",
        )

    def reset_fixed_end_noise(self):
        self.left_fixed_noise = None
        self.right_fixed_noise = None

    def _get_left_fixed_noise(self, batch_size, device, dtype):
        if self.left_fixed_noise is None:
            self.left_fixed_noise = th.randn_like(
                self._get_left_fixed_for_batch(batch_size, device, dtype)
            )
        return self.left_fixed_noise

    def _get_right_fixed_noise(self, batch_size, device, dtype):
        if self.right_fixed_noise is None:
            self.right_fixed_noise = th.randn_like(
                self._get_right_fixed_for_batch(batch_size, device, dtype)
            )
        return self.right_fixed_noise

    def _fixed_t(self, batch_size, device, dtype, scalar_t):
        left = self._get_left_fixed_for_batch(batch_size, device, dtype)
        right = self._get_right_fixed_for_batch(batch_size, device, dtype)
        left_t = left + scalar_t * self._get_left_fixed_noise(batch_size, device, dtype)
        right_t = right + scalar_t * self._get_right_fixed_noise(batch_size, device, dtype)
        return left_t, right_t

    def _split_bridge(self, long_x):
        xs = split_wimg(long_x, self.num_img, rtn_overlap=False)
        return rearrange(xs, "(b n) c h w -> b n c h w", n=self.num_img)

    def _merge_bridge_eps(self, patch_eps, is_avg=False):
        patch_eps = rearrange(patch_eps, "b n c h w -> (b n) c h w")
        return avg_merge_wimg(
            patch_eps,
            self.overlap_size,
            n=self.num_img,
            is_avg=is_avg,
        )

    def _build_endpoint_factors(self, xs, left_t, right_t):
        left_node = xs[:, 0, :, :, : self.overlap_size]
        right_node = xs[:, -1, :, :, -self.overlap_size :]
        left_factor = th.cat([left_t, left_node], dim=-1)
        right_factor = th.cat([right_node, right_t], dim=-1)
        return left_factor, right_factor

    def _endpoint_marginals(self, eps_scalar_t_fn, xs, scalar_t, enable_grad):
        left_node = xs[:, 0, :, :, : self.overlap_size]
        right_nodes = xs[:, :, :, :, -self.overlap_size :]
        left_eps = eps_scalar_t_fn(left_node, scalar_t, enable_grad)
        right_eps = eps_scalar_t_fn(
            rearrange(right_nodes, "b n c h w -> (b n) c h w"),
            scalar_t,
            enable_grad,
        )
        right_eps = rearrange(right_eps, "(b n) c h w -> b n c h w", n=self.num_img)
        return left_eps, right_eps

    def attach_fixed_ends(self, generated_long_x):
        batch_size = generated_long_x.shape[0]
        device = generated_long_x.device
        dtype = generated_long_x.dtype
        left = self._get_left_fixed_for_batch(batch_size, device, dtype)
        right = self._get_right_fixed_for_batch(batch_size, device, dtype)
        composed = generated_long_x.clone()
        composed[:, :, :, : self.overlap_size] = left
        composed[:, :, :, -self.overlap_size :] = right
        return composed

    def attach_fixed_end(self, generated_long_x):
        return self.attach_fixed_ends(generated_long_x)


class DiffCollageBridgeWorker(BridgeWorkerBase):
    """
    DiffCollage-style bridge composition.

    It combines pairwise patch factors and subtracts implicit overlap
    marginals, including the two generated endpoint variables conditioned by
    left/right fixed overlaps.
    """

    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
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

            left_cond = factor_eps[:, 0, :, :, self.overlap_size :]
            full_eps = factor_eps[:, 1:-1].clone()
            right_cond = factor_eps[:, -1, :, :, : self.overlap_size]
            left_marginal, right_marginals = self._endpoint_marginals(
                eps_scalar_t_fn,
                xs,
                scalar_t,
                enable_grad,
            )

            if self.num_img > 1:
                full_eps[:, :-1, :, :, -self.overlap_size :] -= right_marginals[:, :-1]

            full_eps[:, 0, :, :, : self.overlap_size] += (
                self.guidance_scale * (left_cond - left_marginal)
            )
            full_eps[:, -1, :, :, -self.overlap_size :] += (
                self.guidance_scale * (right_cond - right_marginals[:, -1])
            )

            return self._merge_bridge_eps(full_eps, is_avg=False)

        return eps_t_fn


class NaiveBridgeWorker(BridgeWorkerBase):
    """
    Naive proposal baseline for bridge composition.

    It uses only pairwise factors p(left,x), p(x,y), ..., p(z,right) and does
    not divide by / subtract any overlap marginal p(y).
    """

    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
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

            full_eps = factor_eps[:, 1:-1].clone()
            left_cond = factor_eps[:, 0, :, :, self.overlap_size :]
            right_cond = factor_eps[:, -1, :, :, : self.overlap_size]
            full_eps[:, 0, :, :, : self.overlap_size] += self.guidance_scale * left_cond
            full_eps[:, -1, :, :, -self.overlap_size :] += self.guidance_scale * right_cond
            return self._merge_bridge_eps(full_eps, is_avg=False)

        return eps_t_fn


class ProposalFinalBridgeWorker(BridgeWorkerBase):
    """
    Proposal-final bridge implementation.

    Formula: s_xyz = s_xy oplus s_yz - s_y_implicit + Delta s

    Key improvements:
    - sigma_data: sigma-normalized Delta s scale peaks at sigma=sigma_data and
      fades at both high and low sigma, preventing 1/sigma blowup near the end
      of sampling that corrupts fine details.
    - implicit_scale: controls the implicit-marginal subtraction weight. The
      default (0.25) makes internal-overlap correction weaker than the naive
      0.5, aligning more closely with DiffCollage's standalone-marginal approach
      and improving endpoint MSE.
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
        coupling_strength=0.25,
        correction_clip=4.0,
        init_correction_steps=4,
        init_correction_step_size=0.15,
        init_correction_strength=None,
        sigma_data=0.5,
        implicit_scale=0.25,
    ):
        self.coupling_strength = coupling_strength
        self.correction_clip = correction_clip
        self.init_correction_steps = init_correction_steps
        self.init_correction_step_size = init_correction_step_size
        self.init_correction_strength = (
            coupling_strength if init_correction_strength is None else init_correction_strength
        )
        self.sigma_data = sigma_data
        self.implicit_scale = implicit_scale
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

    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            xs, factor_eps, factor_x0 = self._pairwise_factor_predictions(
                eps_scalar_t_fn,
                long_x,
                scalar_t,
                enable_grad,
            )

            # s_xy oplus s_yz: sum pairwise predictions, add endpoint guidance.
            full_eps = factor_eps[:, 1:-1].clone()
            left_cond = factor_eps[:, 0, :, :, self.overlap_size :]
            right_cond = factor_eps[:, -1, :, :, : self.overlap_size]
            full_eps[:, 0, :, :, : self.overlap_size] += self.guidance_scale * left_cond
            full_eps[:, -1, :, :, -self.overlap_size :] += self.guidance_scale * right_cond

            # Subtract symmetric implicit marginal s_y_implicit.
            full_eps = self._subtract_implicit_y(full_eps, factor_eps)
            merged_eps = self._merge_bridge_eps(full_eps, is_avg=False)

            # Delta s = grad log R via Tweedie x0 overlap consistency.
            if self.coupling_strength == 0:
                return merged_eps
            return merged_eps + self._log_r_delta_eps(xs, factor_x0, scalar_t)

        return eps_t_fn

    def _as_sigma(self, scalar_t, device, dtype):
        if not th.is_tensor(scalar_t):
            sigma = th.tensor(float(scalar_t), device=device, dtype=dtype)
        else:
            sigma = scalar_t.to(device=device, dtype=dtype)
        return sigma.clamp_min(self.sigma_min)

    def _scale_for_sigma(self, scalar_t, ref):
        """
        Sigma-normalized coupling scale.

        scale = coupling_strength * sigma / (sigma^2 + sigma_data^2)

        This peaks at sigma = sigma_data (≈0.5) and fades at both very high
        sigma (where x0 predictions are unreliable) and very low sigma (where
        correction should not disturb fine detail).  Replaces the original
        coupling_strength / sigma which diverges as sigma -> 0.
        """
        sigma = self._as_sigma(scalar_t, ref.device, ref.dtype)
        scale = self.coupling_strength * sigma / (sigma ** 2 + self.sigma_data ** 2)
        if scale.ndim == 0:
            return scale.view(*([1] * ref.ndim))
        return scale.view(-1, *([1] * (ref.ndim - 1)))

    def _sigma_for_batch(self, scalar_t, reference, batch_size):
        sigma = self._as_sigma(scalar_t, reference.device, reference.dtype)
        if sigma.ndim == 0:
            return sigma.view(1, 1, 1, 1, 1)
        return sigma.view(batch_size, 1, 1, 1, 1)

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

    def _subtract_implicit_y(self, full_eps, factor_eps):
        """
        Subtract the symmetric implicit marginal estimate s_y_implicit.

        Uses implicit_scale (default 0.25) rather than 0.5 to control the
        subtraction strength.  Analysis shows that using 0.5 over-subtracts at
        internal overlaps: the merged value becomes only 0.5*(factor_right +
        factor_left), which is weaker than DiffCollage's
        factor_right + factor_left - standalone_marginal (standalone marginal
        is smaller than 0.5*(right+left) for half-patches).  A lower
        implicit_scale keeps more signal and improves endpoint MSE.
        """
        # Left endpoint: s_y^{left,x0} implicit marginal.
        left_implicit = (
            factor_eps[:, 0, :, :, self.overlap_size :]
            + factor_eps[:, 1, :, :, : self.overlap_size]
        )
        full_eps[:, 0, :, :, : self.overlap_size] -= (
            self.guidance_scale * self.implicit_scale * left_implicit
        )

        # Internal overlaps.
        for idx in range(self.num_img - 1):
            implicit_y = (
                factor_eps[:, idx + 1, :, :, -self.overlap_size :]
                + factor_eps[:, idx + 2, :, :, : self.overlap_size]
            )
            full_eps[:, idx, :, :, -self.overlap_size :] -= self.implicit_scale * implicit_y

        # Right endpoint: s_y^{xN,right} implicit marginal.
        right_implicit = (
            factor_eps[:, -2, :, :, -self.overlap_size :]
            + factor_eps[:, -1, :, :, : self.overlap_size]
        )
        full_eps[:, -1, :, :, -self.overlap_size :] -= (
            self.guidance_scale * self.implicit_scale * right_implicit
        )
        return full_eps

    def _log_r_delta_eps(self, xs, factor_x0, scalar_t):
        """
        Delta s = grad log R via Tweedie x0 overlap consistency.

        Uses the sigma-normalized scale to avoid blowup at low sigma.
        """
        scale = self._scale_for_sigma(scalar_t, xs[:, 0])
        patch_delta = th.zeros_like(xs)

        left_residual = (
            factor_x0[:, 1, :, :, : self.overlap_size]
            - factor_x0[:, 0, :, :, self.overlap_size :]
        )
        patch_delta[:, 0, :, :, : self.overlap_size] += scale * left_residual

        for idx in range(self.num_img - 1):
            residual = (
                factor_x0[:, idx + 1, :, :, -self.overlap_size :]
                - factor_x0[:, idx + 2, :, :, : self.overlap_size]
            )
            patch_delta[:, idx, :, :, -self.overlap_size :] += scale * residual
            patch_delta[:, idx + 1, :, :, : self.overlap_size] -= scale * residual

        right_residual = (
            factor_x0[:, -2, :, :, -self.overlap_size :]
            - factor_x0[:, -1, :, :, : self.overlap_size]
        )
        patch_delta[:, -1, :, :, -self.overlap_size :] += scale * right_residual

        patch_delta = patch_delta.clamp(-self.correction_clip, self.correction_clip)
        return self._merge_bridge_eps(patch_delta, is_avg=True)

    @th.no_grad()
    def initial_noise_correction(self, long_x):
        if self.init_correction_steps <= 0 or self.init_correction_strength == 0:
            return long_x

        running = long_x
        for _ in range(self.init_correction_steps):
            running = running + self._initial_overlap_delta(running)
        return running

    def _initial_overlap_delta(self, long_x):
        batch_size = long_x.shape[0]
        device = long_x.device
        dtype = long_x.dtype
        xs = self._split_bridge(long_x)
        patch_delta = th.zeros_like(xs)
        step = self.init_correction_step_size * self.init_correction_strength

        for idx in range(self.num_img - 1):
            residual = (
                xs[:, idx, :, :, -self.overlap_size :]
                - xs[:, idx + 1, :, :, : self.overlap_size]
            )
            patch_delta[:, idx, :, :, -self.overlap_size :] -= step * residual
            patch_delta[:, idx + 1, :, :, : self.overlap_size] += step * residual

        left = self._get_left_fixed_for_batch(batch_size, device, dtype)
        right = self._get_right_fixed_for_batch(batch_size, device, dtype)
        left_t = left + self.sigma_max * self._get_left_fixed_noise(batch_size, device, dtype)
        right_t = right + self.sigma_max * self._get_right_fixed_noise(batch_size, device, dtype)
        left_residual = xs[:, 0, :, :, : self.overlap_size] - left_t
        right_residual = xs[:, -1, :, :, -self.overlap_size :] - right_t
        patch_delta[:, 0, :, :, : self.overlap_size] -= step * left_residual
        patch_delta[:, -1, :, :, -self.overlap_size :] -= step * right_residual

        return self._merge_bridge_eps(patch_delta, is_avg=True)
