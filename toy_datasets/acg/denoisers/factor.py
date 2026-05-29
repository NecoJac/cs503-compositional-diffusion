
import torch


class FactorGraphDenoiser:
    def __init__(self, denoiser_xy, denoiser_yc, denoiser_y):
        self.denoiser_xy = denoiser_xy
        self.denoiser_yc = denoiser_yc
        self.denoiser_y = denoiser_y

    def __call__(self, x_full: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x_full: [B, 3] = [x, y, c]
        sigma: scalar tensor or [B] / [1]
        returns: [B, 3] full denoised prediction
        """
        # Make sigma broadcastable to [B, 1]
        if sigma.ndim == 0:
            sigma_sq = sigma**2
        else:
            sigma_sq = sigma.view(-1, 1) ** 2

        # Split variables
        x = x_full[:, 0:1]   # [B,1]
        y = x_full[:, 1:2]   # [B,1]
        c = x_full[:, 2:3]   # [B,1]

        # Build factor inputs
        xy = torch.cat([x, y], dim=1)   # [B,2]
        yc = torch.cat([y, c], dim=1)   # [B,2]

        # Local denoisers
        D_xy = self.denoiser_xy(xy, sigma)   # [B,2]
        D_yc = self.denoiser_yc(yc, sigma)   # [B,2]
        D_y  = self.denoiser_y(y, sigma)     # [B,1]

        # Convert local denoisers -> local scores
        s_xy = (D_xy - xy) / sigma_sq        # [B,2]
        s_yc = (D_yc - yc) / sigma_sq        # [B,2]
        s_y  = (D_y  - y)  / sigma_sq        # [B,1]

        # Combine into full 3D score
        s_full = torch.zeros_like(x_full)    # [B,3]
        s_full[:, 0:1] = s_xy[:, 0:1]                        # x contribution
        s_full[:, 1:2] = s_xy[:, 1:2] + s_yc[:, 0:1] - s_y  # y contribution
        s_full[:, 2:3] = s_yc[:, 1:2]                        # c contribution

        # Convert combined score -> full denoiser
        D_full = x_full + sigma_sq * s_full
        return D_full
    


class FactorGraphDenoiserCondC:
    def __init__(self, denoiser_xy, denoiser_yc, denoiser_y, c_fixed, guidance_scale):
        self.denoiser_xy = denoiser_xy
        self.denoiser_yc = denoiser_yc
        self.denoiser_y = denoiser_y
        self.c_fixed = c_fixed
        self.guidance_scale = guidance_scale

    def __call__(self, x_full: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x_full: [B, 2] = [x, y]
        sigma: scalar tensor or [B] / [1]
        returns: [B, 2] full denoised prediction
        """
        B = x_full.shape[0]

        # Make sigma broadcastable to [B, 1]
        if sigma.ndim == 0:
            sigma_sq = sigma**2
        else:
            sigma_sq = sigma.view(-1, 1) ** 2

        # Split variables
        x = x_full[:, 0:1]   # [B,1]
        y = x_full[:, 1:2]   # [B,1]

        c = torch.full_like(y, self.c_fixed)
    

        # Build factor inputs
        xy = torch.cat([x, y], dim=1)   # [B,2]
        yc = torch.cat([y, c], dim=1)   # [B,2]

        # Local denoisers
        D_xy = self.denoiser_xy(xy, sigma)   # [B,2]
        D_yc = self.denoiser_yc(yc, sigma)   # [B,2]
        D_y  = self.denoiser_y(y, sigma)     # [B,1]

        # Convert local denoisers -> local scores
        s_xy = (D_xy - xy) / sigma_sq        # [B,2]
        s_yc = (D_yc - yc) / sigma_sq        # [B,2]
        s_y  = (D_y  - y)  / sigma_sq        # [B,1]

        # Combine into full 3D score
        s_full = torch.zeros_like(x_full)    # [B,3]
        s_full[:, 0:1] = s_xy[:, 0:1]                        # x contribution

        # CFG scaling
        s_full[:, 1:2] = s_xy[:, 1:2] +  self.guidance_scale * (s_yc[:, 0:1] - s_y) # y contribution                    # c contribution

        # Convert combined score -> full denoiser
        D_full = x_full + sigma_sq * s_full
        return D_full