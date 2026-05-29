import torch as th
from einops import rearrange

from .base_worker import BaseWorker
from .w_img import split_wimg, avg_merge_wimg


import torch as th
from einops import rearrange

from .base_worker import BaseWorker
from .w_img import split_wimg, avg_merge_wimg


class CondIndLongClassCFG(BaseWorker):
    """
    DiffCollage chain with class conditioning applied consistently to:
      - full factor windows
      - half overlap strips

    Update:
        full_eps_cfg - half_eps_cfg

    where both full and half use the same conditional/CFG rule.

    class_labels:
      - int, or
      - list/tensor of length num_img
    """

    def __init__(
        self,
        shape,
        edm_dual_eps_wrapper,
        num_img,
        class_labels,
        overlap_size=32,
        guidance_scale=1.0,
        sigma_max=80.0,
        sigma_min=1e-3,
        use_cfg=True,   # if False: use pure conditional instead of CFG
    ):
        c, h, w = shape
        assert overlap_size == w // 2, "This implementation assumes half-overlap."

        self.overlap_size = overlap_size
        self.num_img = num_img
        self.class_labels = class_labels
        self.guidance_scale = guidance_scale
        self.edm = edm_dual_eps_wrapper
        self.use_cfg = use_cfg

        if not isinstance(class_labels, int):
            n_labels = len(class_labels)
            if n_labels != num_img:
                raise ValueError(
                    f"class_labels must have length num_img={num_img}, got {n_labels}"
                )

        final_img_w = w * num_img - overlap_size * (num_img - 1)

        super().__init__(
            (c, h, final_img_w),
            self.get_eps_t_fn(),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    def _guided_eps(self, xs, scalar_t, enable_grad=False):
        """
        Apply the SAME conditioning rule to any input tensor xs:
          - unconditional if guidance_scale == 0
          - CFG if use_cfg=True
          - pure conditional if use_cfg=False
        """
        eps_uncond = self.edm.eps_uncond(
            xs, scalar_t, enable_grad=enable_grad
        )

        if self.guidance_scale == 0:
            return eps_uncond

        eps_cond = self.edm.eps_cond(
            xs,
            scalar_t,
            class_labels=self.class_labels,
            enable_grad=enable_grad,
        )

        if self.use_cfg:
            return eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        else:
            return eps_cond

    def get_eps_t_fn(self):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            # --------------------------------------------------
            # 1) Split long image into full windows
            # --------------------------------------------------
            xs = split_wimg(
                long_x, self.num_img, rtn_overlap=False
            )  # ((b*n), c, h, w)

            # --------------------------------------------------
            # 2) Full-window eps with consistent guidance
            # --------------------------------------------------
            full_eps = self._guided_eps(
                xs, scalar_t, enable_grad=enable_grad
            )
            full_eps = rearrange(
                full_eps,
                "(b n) c h w -> n b c h w",
                n=self.num_img,
            )

            # --------------------------------------------------
            # 3) Half-window eps with SAME guidance rule
            # --------------------------------------------------
            half_xs = xs[:, :, :, -self.overlap_size:]   # ((b*n), c, h, w//2)

            half_eps = self._guided_eps(
                half_xs, scalar_t, enable_grad=enable_grad
            )
            half_eps = rearrange(
                half_eps,
                "(b n) c h w -> n b c h w",
                n=self.num_img,
            )

            # Last half-strip is not shared with a next factor
            half_eps[-1] = 0

            # --------------------------------------------------
            # 4) Standard DiffCollage subtraction
            # --------------------------------------------------
            full_eps[:, :, :, :, -self.overlap_size:] -= half_eps

            # --------------------------------------------------
            # 5) Merge back
            # --------------------------------------------------
            whole_eps = rearrange(
                full_eps,
                "n b c h w -> (b n) c h w"
            )

            return avg_merge_wimg(
                whole_eps,
                self.overlap_size,
                n=self.num_img,
                is_avg=False,
            )

        return eps_t_fn

'''
class CondIndLongClassCFG(BaseWorker):
    """
    Standard DiffCollage chain update, but with CFG applied only on full factor nodes.

    Score approximation:
        sum_i score_cfg(full_factor_i | c_i) - sum_j score_uncond(overlap_j)

    Notes
    -----
    - full factors = full 64x64 windows
    - overlap factors = right overlap strips, kept unconditional
    - class_labels must be:
        * int, or
        * list/tensor of length num_img
    """

    def __init__(
        self,
        shape,
        edm_dual_eps_wrapper,
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

        if not isinstance(class_labels, int):
            n_labels = len(class_labels)
            if n_labels != num_img:
                raise ValueError(
                    f"class_labels must have length num_img={num_img}, got {n_labels}"
                )

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
            # 1) Split long image into full factor windows
            # --------------------------------------------------
            xs = split_wimg(
                long_x, self.num_img, rtn_overlap=False
            )  # ((b*n), c, h, w)

            # --------------------------------------------------
            # 2) Full factor eps: unconditional + CFG correction
            # --------------------------------------------------
            full_eps_uncond = self.edm.eps_uncond(
                xs, scalar_t, enable_grad=enable_grad
            )

            if self.guidance_scale == 0:
                full_eps = full_eps_uncond
            else:
                full_eps_cond = self.edm.eps_cond(
                    xs,
                    scalar_t,
                    class_labels=self.class_labels,
                    enable_grad=enable_grad,
                )
                full_eps = full_eps_uncond + self.guidance_scale * (
                    full_eps_cond - full_eps_uncond
                )

            full_eps = rearrange(
                full_eps,
                "(b n) c h w -> n b c h w",
                n=self.num_img,
            )

            # --------------------------------------------------
            # 3) Half eps stays unconditional, exactly as usual
            # --------------------------------------------------
            half_xs = xs[:, :, :, -self.overlap_size:]   # ((b*n), c, h, w//2)

            half_eps = self.edm.eps_uncond(
                half_xs, scalar_t, enable_grad=enable_grad
            )
            half_eps = rearrange(
                half_eps,
                "(b n) c h w -> n b c h w",
                n=self.num_img,
            )

            # last overlap is not shared with a next factor
            half_eps[-1] = 0

            # subtract overlap contribution as in DiffCollage
            full_eps[:, :, :, :, -self.overlap_size:] -= half_eps

            # --------------------------------------------------
            # 4) Merge back
            # --------------------------------------------------
            whole_eps = rearrange(
                full_eps,
                "n b c h w -> (b n) c h w"
            )

            return avg_merge_wimg(
                whole_eps,
                self.overlap_size,
                n=self.num_img,
                is_avg=False,
            )

        return eps_t_fn
'''