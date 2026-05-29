import torch as th
from einops import rearrange

from .base_worker import BaseWorker
from .w_img import split_wimg, avg_merge_wimg


class CondIndLongFixedEndCtrlSmc(BaseWorker):
    """
    Same as CondIndLong, but adds one extra conditional factor f(x5, x6_fixed)
    at the end, where x6_fixed is given per batch.

    fixed_images is assumed to be the size of x6 directly:
        [B, C, H, overlap_size]
    """

    def __init__(
        self,
        shape,
        eps_scalar_t_fn,
        num_img,
        fixed_images,             # [B, C, H, overlap]
        overlap_size=32,
        guidance_scale=1.0,
        sigma_max=80.0,
        use_smc=False,
        sigma_min=1e-3,
        smc_lambda=1.0,
        smc_k=1.0
    ):
        c, h, w = shape
        assert overlap_size == w // 2
        
        # ----------------- Check consistency of fixed images ----------------#
        if fixed_images.ndim == 3:
            # [C,H,overlap] -> [1,C,H,overlap]
            fixed_images = fixed_images.unsqueeze(0)
        elif fixed_images.ndim != 4:
            raise ValueError(
                f"fixed_images must be [C,H,overlap] or [B,C,H,overlap], got {fixed_images.shape}"
            )

        assert fixed_images.shape[1] == c, (
            f"Expected channel dim {c}, got {fixed_images.shape[1]}"
        )
        assert fixed_images.shape[2] == h, (
            f"Expected height {h}, got {fixed_images.shape[2]}"
        )
        assert fixed_images.shape[3] == overlap_size, (
            f"Expected last dim {overlap_size}, got {fixed_images.shape[3]}"
        )
        #--------------------------------------------------------------------#


        self.overlap_size = overlap_size
        self.num_img = num_img
        self.fixed_images = fixed_images


        self.guidance_scale = guidance_scale

        # Variables for Ctrl-SMC
        self.smc_lambda = smc_lambda
        self.smc_k = smc_k
        self.use_smc = use_smc
        self.e_prev = None

        final_img_w = w * num_img - overlap_size * (num_img - 1)

        # Init noise for factor x6
        self.fixed_end_noise = None

        
        super().__init__(
            (c, h, final_img_w),
            self.get_eps_t_fn(eps_scalar_t_fn),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    def _get_fixed_images_for_batch(self, B, device, dtype):
        fixed = self.fixed_images.to(device=device, dtype=dtype)

        if fixed.shape[0] == 1 and B > 1:
            fixed = fixed.repeat(B, 1, 1, 1)

        if fixed.shape[0] != B:
            raise ValueError(
                f"fixed_images batch {fixed.shape[0]} must match requested batch {B}"
            )

        return fixed

    # ----------------- Methods to reset 
    def reset_fixed_end_noise(self):
        self.fixed_end_noise = None

    def _get_fixed_end_noise(self, B, device, dtype):
        if self.fixed_end_noise is None:
            self.fixed_end_noise = th.randn_like(
                self._get_fixed_images_for_batch(B, device, dtype)
            )
        return self.fixed_end_noise

    def loss(self, x):
        x1, x2 = x[:-1], x[1:]
        return th.sum(
            (th.abs(x1[:, :, :, -self.overlap_size:] - x2[:, :, :, :self.overlap_size])) ** 2,
            dim=(1, 2, 3),
        )

    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            B = long_x.shape[0]
            device = long_x.device
            dtype = long_x.dtype

            
            x6_fixed = self._get_fixed_images_for_batch(B, device, dtype)
            z = self._get_fixed_end_noise(B, device, dtype) # Reuse same z so Heun deterministic on t and xt
            x6_t = x6_fixed + scalar_t * z

            # --------------------------------------------------
            # 1) Split generated long image into n full factors
            # --------------------------------------------------
            xs = split_wimg(long_x, self.num_img, rtn_overlap=False)   # [(B*n), C, H, W]
            xs = rearrange(xs, "(b n) c h w -> b n c h w", n=self.num_img)  # [B,n,C,H,W]

            # --------------------------------------------------
            # 2) Build extra conditional factor f(x5, x6)
            # --------------------------------------------------
            x5 = xs[:, -1, :, :, -self.overlap_size:]   # [B,C,H,overlap]
            extra_factor = th.cat([x5, x6_t], dim=-1)   # [B,C,H,W]

            # Append extra factor per batch
            xs_all = th.cat([xs, extra_factor.unsqueeze(1)], dim=1)   # [B,n+1,C,H,W]
            xs_all_flat = rearrange(xs_all, "b n c h w -> (b n) c h w")

            # --------------------------------------------------
            # 3) Full factor eps: f1,...,fn,f(x5,x6)
            # --------------------------------------------------
            full_eps = eps_scalar_t_fn(xs_all_flat, scalar_t, enable_grad)
            full_eps = rearrange(
                full_eps,
                "(b n) c h w -> b n c h w",
                n=self.num_img + 1
            )   # [B,n+1,C,H,W]

            # Save conditional factor and remove it from normal chain update
            cond_factor = full_eps[:, -1]      # [B,C,H,W]
            full_eps = full_eps[:, :-1]        # [B,n,C,H,W]

            # --------------------------------------------------
            # 4) Half eps for variable nodes
            # --------------------------------------------------
            half_in = xs_all_flat[:, :, :, -self.overlap_size:]   # [(B*(n+1)),C,H,overlap]
            half_eps = eps_scalar_t_fn(half_in, scalar_t, enable_grad)
            half_eps = rearrange(
                half_eps,
                "(b n) c h w -> b n c h w",
                n=self.num_img + 1
            )   # [B,n+1,C,H,overlap]

            # Remove useless x6 boundary
            half_eps = half_eps[:, :-1]        # [B,n,C,H,overlap]

            # Save x5 variable-node contribution, then set it to zero for normal update
            eps_x5 = half_eps[:, -1].clone()   # [B,C,H,overlap]
            half_eps[:, -1] = 0

            # --------------------------------------------------
            # 5) Normal chain update unchanged
            # --------------------------------------------------
            full_eps[:, :, :, :, -self.overlap_size:] = (
                full_eps[:, :, :, :, -self.overlap_size:] - half_eps
            )

            # --------------------------------------------------
            # 6) Independent conditional correction on last variable node
            # --------------------------------------------------
            cond_left = cond_factor[:, :, :, :self.overlap_size]   # left half of f(x5,x6)
            
            # --------------------------------------------------
            # 7) Ctrl-SMC (optional)
            # --------------------------------------------------
            if self.use_smc:
                e_t = cond_left - eps_x5

                if self.e_prev is None: # First call
                    self.e_prev = e_t.detach()

                s_slide = (e_t - self.e_prev) + self.smc_lambda * self.e_prev
                delta_e = -self.smc_k * th.sign(s_slide)
                e_hat = e_t + delta_e

                full_eps[:, -1, :, :, -self.overlap_size:] += (
                    self.guidance_scale * e_hat
                )

                self.e_prev = e_hat.detach()
            
            else:
                full_eps[:, -1, :, :, -self.overlap_size:] += (
                    self.guidance_scale * (cond_left - eps_x5)
                )
           
            # --------------------------------------------------
            # 8) Merge back
            # --------------------------------------------------
            whole_eps = rearrange(full_eps, "b n c h w -> (b n) c h w")

            return avg_merge_wimg(
                whole_eps,
                self.overlap_size,
                n=self.num_img,
                is_avg=False,
            )

        return eps_t_fn

    def attach_fixed_end(self, generated_long_x):
        """
        Append fixed x6 overlap to the generated long image.
        Since fixed_images are only overlap-sized, this appends that block directly.
        """
        B = generated_long_x.shape[0]
        fixed = self._get_fixed_images_for_batch(
            B,
            generated_long_x.device,
            generated_long_x.dtype,
        )
        
        return th.cat([generated_long_x, fixed], dim=-1)