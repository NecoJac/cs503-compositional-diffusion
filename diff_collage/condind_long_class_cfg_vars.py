import torch as th
from einops import rearrange

from .base_worker import BaseWorker
from .w_img import split_wimg, avg_merge_wimg


class CondIndLongClassCFG(BaseWorker):
    """
    Standard unconditional chain update:
        sum full patch factors - sum overlap variable factors

    Plus an independent class-guidance correction on each variable node:
        + guidance_scale * (eps_cond(x_i | c_i) - eps_uncond(x_i))

    Notes
    -----
    - This is a factor-graph-style approximation of adding unary class factors.
    - class_labels can be:
        * int                    -> same class for all patches
        * list[int] length num_img -> one class per patch
    - The last overlap does not correspond to an internal variable node,
      so its correction is set to zero.
    """

    def __init__(
        self,
        shape,
        edm_dual_eps_wrapper,   # instance of EDMDualEpsWrapper
        num_img,
        class_labels,
        overlap_size=32,
        guidance_scale=1.0,
        sigma_max=80.0,
        sigma_min=1e-3,
    ):
        c, h, w = shape
        assert overlap_size == w // 2, "This implementation assumes half-overlap."

        self.overlap_size = overlap_size
        self.num_img = num_img
        self.class_labels = class_labels
        self.guidance_scale = guidance_scale
        self.edm = edm_dual_eps_wrapper

        final_img_w = w * num_img - overlap_size * (num_img - 1)

        super().__init__(
            (c, h, final_img_w),
            self.get_eps_t_fn(),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    def get_eps_t_fn(self):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            # --------------------------------------------------
            # 1) Split long image into full patch factors
            # --------------------------------------------------
            xs_flat = split_wimg(long_x, self.num_img, rtn_overlap=False)   # [(B*n), C, H, W]
            xs = rearrange(xs_flat, "(b n) c h w -> b n c h w", n=self.num_img) # # [B, n, C, H, W]

            # --------------------------------------------------
            # 2) Unconditional full-factor eps
            # --------------------------------------------------
            full_eps = self.edm.eps_uncond(xs_flat, scalar_t, enable_grad=enable_grad)
            full_eps = rearrange(
                full_eps,
                "(b n) c h w -> b n c h w",
                n=self.num_img
            )  # [B, n, C, H, W]

            # --------------------------------------------------
            # 3) Variable-node strips 
            # --------------------------------------------------
            # xs: [B, n, C, H, W]

            # --------------------------------------------
            # Build ALL variable nodes from beginning to end
            # v0 = left boundary of first patch
            # v1..vn = right boundaries of each patch
            # --------------------------------------------
            var_in = th.cat(
                [
                    xs[:, 0:1, :, :, :self.overlap_size],   # [B,1,C,H,overlap]
                    xs[:, :, :, :, -self.overlap_size:]     # [B,n,C,H,overlap]
                ],
                dim=1
            )   # [B, n+1, C, H, overlap]

            var_in_flat = rearrange(var_in, "b m c h w -> (b m) c h w")

            # --------------------------------------------
            # 4) Unconditional variable-node eps on ALL variables
            # --------------------------------------------
            var_eps_uncond = self.edm.eps_uncond(
                var_in_flat, scalar_t, enable_grad=enable_grad
            )
            var_eps_uncond = rearrange(
                var_eps_uncond,
                "(b m) c h w -> b m c h w",
                m=self.num_img + 1
            )


            # For the usual chain subtraction, use only right-half-aligned variables
            chain_var_eps = var_eps_uncond[:, 1:].clone()   # [B,n,C,H,overlap]
            chain_var_eps[:, -1] = 0                        # Set to 0 for usual update

            # --------------------------------------------------
            # 5) Normal unconditional chain subtraction
            # --------------------------------------------------

            # Standard subtraction
            full_eps[:, :, :, :, -self.overlap_size:] -= chain_var_eps

            # --------------------------------------------------
            # 6) Class guidance on each variable node only
            # --------------------------------------------------
            if self.guidance_scale != 0:
                var_eps_cond = self.edm.eps_cond(
                    var_in_flat,
                    scalar_t,
                    class_labels=self.class_labels,   # Must be length n+1 if variable labels differ
                    enable_grad=enable_grad,
                )
                var_eps_cond = rearrange(
                    var_eps_cond,
                    "(b m) c h w -> b m c h w",
                    m=self.num_img + 1
                )

                var_delta = self.guidance_scale * (var_eps_cond - var_eps_uncond)

                # Left boundary correction to first patch
                full_eps[:, 0, :, :, :self.overlap_size] += var_delta[:, 0]

                # Correction for remaining patches
                full_eps[:, :, :, :, -self.overlap_size:] += var_delta[:, 1:]
    
            
            # --------------------------------------------------
            # 7) Merge back to long-image epsilon
            # --------------------------------------------------
            whole_eps = rearrange(full_eps, "b n c h w -> (b n) c h w")

            return avg_merge_wimg(
                whole_eps,
                self.overlap_size,
                n=self.num_img,
                is_avg=False,
            )

        return eps_t_fn