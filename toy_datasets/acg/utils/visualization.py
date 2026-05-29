"""Visualization Utilities.

This module provides visualization functions for diffusion models,
including denoising fields, sampling trajectories, and data plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_2d_data(
    data: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    title: str = "",
    ax: Optional["Axes"] = None,
    figsize: Tuple[float, float] = (6, 6),
    cmap: str = "tab10",
    alpha: float = 0.7,
    s: float = 30,
) -> "Axes":
    """Plot 2D data points with optional labels.

    Parameters
    ----------
    data : torch.Tensor
        Data points of shape [N, 2].
    labels : torch.Tensor, optional
        Class labels of shape [N].
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple, optional
        Figure size if creating new figure.
    cmap : str, optional
        Colormap for labels.
    alpha : float, optional
        Point transparency.
    s : float, optional
        Point size.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    data_np = data.cpu().numpy()

    if labels is not None:
        labels_np = labels.cpu().numpy()
        scatter = ax.scatter(
            data_np[:, 0], data_np[:, 1], c=labels_np, cmap=cmap, alpha=alpha, s=s
        )
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(data_np[:, 0], data_np[:, 1], alpha=alpha, s=s)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return ax


def plot_denoising_field(
    denoiser,
    sigma: float,
    xlim: Tuple[float, float] = (-4, 4),
    ylim: Tuple[float, float] = (-4, 4),
    grid_size: int = 30,
    data: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    condition: Optional[int] = None,
    ax: Optional["Axes"] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: str = "",
) -> "Axes":
    """Visualize the denoising vector field in 2D.

    Shows arrows pointing from noisy points to their denoised predictions.

    Parameters
    ----------
    denoiser : Callable
        Denoiser function that takes (x, sigma) or (x, sigma, condition).
    sigma : float
        Noise level.
    xlim : tuple, optional
        X-axis limits.
    ylim : tuple, optional
        Y-axis limits.
    grid_size : int, optional
        Number of grid points per axis.
    data : torch.Tensor, optional
        Original data points to overlay [N, 2].
    labels : torch.Tensor, optional
        Labels for data points.
    condition : int, optional
        Condition to pass to denoiser.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple, optional
        Figure size if creating new figure.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Create grid
    x = torch.linspace(xlim[0], xlim[1], grid_size)
    y = torch.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Get denoised predictions
    sigma_tensor = torch.full((grid_points.shape[0],), sigma)

    if condition is not None:
        cond_tensor = torch.full((grid_points.shape[0],), condition, dtype=torch.long)
        denoised = denoiser(grid_points, sigma_tensor, cond_tensor)
    else:
        denoised = denoiser(grid_points, sigma_tensor)

    # Compute displacement vectors
    displacement = denoised - grid_points

    # Plot vector field
    grid_np = grid_points.cpu().numpy()
    disp_np = displacement.cpu().numpy()

    ax.quiver(
        grid_np[:, 0],
        grid_np[:, 1],
        disp_np[:, 0],
        disp_np[:, 1],
        alpha=0.5,
        scale=20,
    )

    # Plot data points if provided
    if data is not None:
        plot_2d_data(data, labels, ax=ax, alpha=0.8)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title or f"Denoising Field (σ={sigma})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return ax


def plot_sampling_trajectory(
    trajectories: torch.Tensor,
    data: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    ax: Optional["Axes"] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: str = "Sampling Trajectories",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> "Axes":
    """Plot sampling trajectories in 2D.

    Parameters
    ----------
    trajectories : torch.Tensor
        Sampling trajectories of shape [n_samples, n_steps, 2].
    data : torch.Tensor, optional
        Original data points to overlay [N, 2].
    labels : torch.Tensor, optional
        Labels for data points.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple, optional
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    xlim : tuple, optional
        X-axis limits.
    ylim : tuple, optional
        Y-axis limits.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    traj_np = trajectories.cpu().numpy()

    # Plot data points if provided
    if data is not None:
        plot_2d_data(data, labels, ax=ax, alpha=0.5, s=20)

    # Plot trajectories
    for i in range(traj_np.shape[0]):
        ax.plot(traj_np[i, :, 0], traj_np[i, :, 1], "b-", alpha=0.3, linewidth=0.5)
        ax.plot(traj_np[i, 0, 0], traj_np[i, 0, 1], "ro", markersize=3)  # Start
        ax.plot(traj_np[i, -1, 0], traj_np[i, -1, 1], "g*", markersize=5)  # End

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax

############################################################
# -------- Visualization for half-helix (x,y,c) data ------#
###########################################################

def plot_xyc_data(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    title: str = "2D Visualization: color = c",
    ax: Optional["Axes"] = None,
    figsize: Tuple[float, float] = (6, 5),
    cmap: str = "coolwarm",
    alpha: float = 0.8,
    s: float = 15,
    colorbar_label: str = "c",
) -> "Axes":
    """Plot a coupled ``(x, y, c)`` dataset with color-coded attribute ``c``."""

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        x.cpu().numpy(),
        y.cpu().numpy(),
        c=c.cpu().numpy(),
        cmap=cmap,
        alpha=alpha,
        s=s,
    )
    plt.colorbar(scatter, ax=ax, label=colorbar_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_xyc_sampling_trajectories(
    trajectories: torch.Tensor,
    *,
    ax: Optional["Axes"] = None,
    figsize: Tuple[float, float] = (6, 6),
    cmap: str = "coolwarm",
    alpha: float = 0.3,
    linewidth: float = 0.7,
    point_size: float = 50,
    edgecolors: str = "black",
    colorbar_label: str = "c (third coordinate)",
    title: str = "Trajectories (gray) + final samples colored by c",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> "Axes":
    """Plot ``(x, y, c)`` sampling trajectories with final samples colored by ``c``."""
 
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    traj_np = trajectories.cpu().numpy()

    for i in range(traj_np.shape[0]):
        ax.plot(
            traj_np[i, :, 0],
            traj_np[i, :, 1],
            color="gray",
            alpha=alpha,
            linewidth=linewidth,
        )

    scatter = ax.scatter(
        traj_np[:, -1, 0],
        traj_np[:, -1, 1],
        c=traj_np[:, -1, 2],
        cmap=cmap,
        s=point_size,
        edgecolors=edgecolors,
        linewidth=0.5,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(scatter, ax=ax, label=colorbar_label)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


plot_xyc_trajectory = plot_xyc_sampling_trajectories


def plot_conditional_generation_comparison(
    trajectories: Sequence[torch.Tensor],
    target_classes: Sequence[int],
    titles: Sequence[str],
    target_data: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    colors: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (14, 6),
    suptitle: str = "Conditional Generation: Same Initial -> Different Targets",
) -> "tuple[plt.Figure, Sequence[Axes]]":
    """Compare conditional generation trajectories for multiple classes."""
  

    if colors is None:
        colors = ("#3498DB", "#E74C3C")

    fig, axes = plt.subplots(1, len(trajectories), figsize=figsize)
    if len(trajectories) == 1:
        axes = [axes]

    target_np = target_data.cpu().numpy()
    labels_np = target_labels.cpu().numpy()

    for ax, traj, target_class, title in zip(axes, trajectories, target_classes, titles):
        traj_np = traj.cpu().numpy()
        ax.scatter(target_np[:, 0], target_np[:, 1], c="lightgray", alpha=0.3, s=10)

        mask = labels_np == target_class
        ax.scatter(
            target_np[mask, 0],
            target_np[mask, 1],
            c=colors[target_class],
            alpha=0.4,
            s=15,
            label=f"Target Class {target_class}",
        )

        for i in range(traj_np.shape[0]):
            ax.plot(traj_np[i, :, 0], traj_np[i, :, 1], "-", color="gray", alpha=0.3, linewidth=0.5)

        ax.scatter(
            traj_np[:, -1, 0],
            traj_np[:, -1, 1],
            c=colors[target_class],
            s=40,
            edgecolors="black",
            linewidth=0.5,
            label="Generated",
        )

        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()

    return fig, axes


def plot_cfg_scale_comparison(
    trajectories_by_scale: dict[float, torch.Tensor],
    target_data: torch.Tensor,
    target_labels: torch.Tensor,
    target_class: int,
    *,
    cfg_scales: Optional[Sequence[float]] = None,
    colors: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (18, 4.5),
    xlim: Tuple[float, float] = (-3, 4),
    ylim: Tuple[float, float] = (-2, 3),
    suptitle: Optional[str] = None,
) -> "tuple[plt.Figure, Sequence[Axes]]":
    """Plot sampling trajectories for multiple CFG scales."""
   

    if cfg_scales is None:
        cfg_scales = list(trajectories_by_scale.keys())
    if colors is None:
        colors = ("#3498DB", "#E74C3C")
    if suptitle is None:
        suptitle = f"CFG Scale Comparison (Target: Class {target_class})"

    fig, axes = plt.subplots(1, len(cfg_scales), figsize=figsize)
    if len(cfg_scales) == 1:
        axes = [axes]

    target_np = target_data.cpu().numpy()
    labels_np = target_labels.cpu().numpy()
    mask = labels_np == target_class

    for ax, cfg_scale in zip(axes, cfg_scales):
        traj_np = trajectories_by_scale[cfg_scale].cpu().numpy()

        ax.scatter(target_np[:, 0], target_np[:, 1], c="lightgray", alpha=0.2, s=8)
        ax.scatter(target_np[mask, 0], target_np[mask, 1], c=colors[target_class], alpha=0.3, s=12)

        for i in range(traj_np.shape[0]):
            ax.plot(traj_np[i, :, 0], traj_np[i, :, 1], "-", color="steelblue", alpha=0.25, linewidth=0.6)

        ax.scatter(
            traj_np[:, 0, 0],
            traj_np[:, 0, 1],
            c="white",
            s=25,
            edgecolors="gray",
            linewidth=0.5,
            zorder=5,
        )
        ax.scatter(
            traj_np[:, -1, 0],
            traj_np[:, -1, 1],
            c=colors[target_class],
            s=35,
            edgecolors="black",
            linewidth=0.5,
            zorder=6,
        )

        ax.set_title(f"CFG Scale = {cfg_scale}", fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.tight_layout()

    return fig, axes


def plot_vector_field(
    denoiser,
    sigma: float,
    condition: int,
    target_data: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    guidance_scale: float = 1.0,
    xlim: Tuple[float, float] = (-2, 3),
    ylim: Tuple[float, float] = (-1.5, 2),
    grid_size: int = 18,
    ax: Optional["Axes"] = None,
    figsize: Tuple[float, float] = (8, 6),
    colors: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> "Axes":
    """Plot a conditional denoising vector field in 2D."""
   

    if colors is None:
        colors = ("#3498DB", "#E74C3C")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    device = target_data.device

    x = torch.linspace(xlim[0], xlim[1], grid_size, device=device)
    y = torch.linspace(ylim[0], ylim[1], grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    sigma_tensor = torch.full((grid_points.shape[0],), sigma, device=device)
    cond_tensor = torch.full((grid_points.shape[0],), condition, dtype=torch.long, device=device)
    denoised = denoiser(
        grid_points,
        sigma_tensor,
        condition=cond_tensor,
        guidance_scale=guidance_scale,
    )
    displacement = denoised - grid_points

    target_np = target_data.cpu().numpy()
    labels_np = target_labels.cpu().numpy()
    mask_other = labels_np != condition
    mask_target = labels_np == condition

    ax.scatter(target_np[mask_other, 0], target_np[mask_other, 1], c="lightgray", alpha=0.3, s=8)
    ax.scatter(target_np[mask_target, 0], target_np[mask_target, 1], c=colors[condition], alpha=0.5, s=12)

    grid_np = grid_points.cpu().numpy()
    disp_np = displacement.cpu().numpy()
    ax.quiver(
        grid_np[:, 0],
        grid_np[:, 1],
        disp_np[:, 0],
        disp_np[:, 1],
        alpha=0.7,
        color="darkblue",
        scale=20,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if title is not None:
        ax.set_title(title)

    return ax


def plot_factor_vector_field(
    denoiser,
    sigma: float,
    target_data: torch.Tensor,
    c_fixed: float,
    xlim=(-2, 3),
    ylim=(-1.5, 2),
    grid_size=18,
    ax=None,
    colors=('#3498DB', '#E74C3C')
):
    """
    Plot denoising vector field for FactorGraphDenoiserCondC in (x, y).

    Args:
        denoiser: factor denoiser with fixed c already built in
        sigma: scalar noise level
        target_data: full data tensor of shape [N, 3] = [x, y, c]
        c_fixed: fixed conditioning value used inside the factor denoiser
        xlim, ylim: plotting limits
        grid_size: number of grid points per axis
        ax: matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    device = target_data.device

    # --------------------------
    # 1) Create grid in (x, y)
    # --------------------------
    x = torch.linspace(xlim[0], xlim[1], grid_size, device=device)
    y = torch.linspace(ylim[0], ylim[1], grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # [G, 2]

    # --------------------------
    # 2) Evaluate denoiser
    # --------------------------
    sigma_tensor = torch.full((grid_points.shape[0],), sigma, device=device)
    denoised = denoiser(grid_points, sigma_tensor)   # [G, 2]
    displacement = denoised - grid_points

    # --------------------------
    # 3) Plot target data
    # --------------------------
    target_np = target_data.detach().cpu().numpy()   # [N, 3]
    target_xy = target_np[:, :2]
    target_c  = target_np[:, 2]

    # Approximate compatible subset
    if c_fixed >= 0.5:
        mask_target = target_c > 0.5
        highlight_color = colors[1]
        label_name = "c ≈ 1"
    else:
        mask_target = target_c <= 0.5
        highlight_color = colors[0]
        label_name = "c ≈ 0"

    mask_other = ~mask_target

    ax.scatter(
        target_xy[mask_other, 0], target_xy[mask_other, 1],
        c='lightgray', alpha=0.3, s=8
    )

    ax.scatter(
        target_xy[mask_target, 0], target_xy[mask_target, 1],
        c=highlight_color, alpha=0.5, s=12, label=label_name
    )

    # --------------------------
    # 4) Plot vector field
    # --------------------------
    grid_np = grid_points.detach().cpu().numpy()
    disp_np = displacement.detach().cpu().numpy()

    ax.quiver(
        grid_np[:, 0], grid_np[:, 1],
        disp_np[:, 0], disp_np[:, 1],
        alpha=0.7, color='darkblue', scale=20
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Vector field, sigma={sigma:.3f}, fixed c={c_fixed}")

    return ax