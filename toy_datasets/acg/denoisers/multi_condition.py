"""Multi-Condition Denoiser (Composable Diffusion).

Implements multi-condition composition based on Product of Experts.

Theory (Composable Diffusion):
------------------------------
For multiple conditions c1, c2, ..., we can compose the conditional
distributions using:

    p(y|c1, c2, ...) ∝ p(y|c1) · p(y|c2) · ... / p(y)^(n-1)

In the denoiser space, this translates to:
    D_combo = D_uncond + Σ_i w_i · (D_ci - D_uncond)

This enables:
- AND semantics: generate samples matching ALL conditions
- Negation: generate samples matching some conditions but NOT others
- Weighted composition: different importance for each condition

Reference:
    Liu et al. 2022 - "Compositional Visual Generation with Composable
    Diffusion Models"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from acg.denoisers.base import Denoiser, GaussianKernelMixin


class MultiConditionDenoiser(Denoiser, GaussianKernelMixin):
    """Multi-condition denoiser for composable generation.

    Supports:
    - Single condition denoising
    - Multi-condition composition (AND/OR semantics)
    - Positive/negative conditions
    - Per-condition guidance scales

    Parameters
    ----------
    data : torch.Tensor
        Dataset of shape [N, D].
    conditions : torch.Tensor
        Condition labels of shape [N] (discrete).
    condition_type : str, optional
        Currently only "discrete" is supported.

    Examples
    --------
    >>> data, labels = make_quadrant_dataset()
    >>> denoiser = MultiConditionDenoiser(data, labels)
    >>> # Generate samples matching conditions 0 AND 2
    >>> samples = sample_heun(
    ...     denoiser, latents, sigmas,
    ...     conditions=[0, 2], guidance_scales=[3.0, 3.0]
    ... )
    """

    def __init__(
        self,
        data: torch.Tensor,
        conditions: torch.Tensor,
        condition_type: str = "discrete",
    ) -> None:
        """Initialize multi-condition denoiser."""
        self.data = data
        self.data_flat = data.flatten(1)
        self.conditions = conditions
        self.condition_type = condition_type
        self.n_points = data.shape[0]
        self._original_shape = data.shape[1:]

        # Pre-compute indices for each condition
        self.unique_conditions = torch.unique(conditions)
        self.condition_indices = {
            int(c): (conditions == c).nonzero(as_tuple=True)[0]
            for c in self.unique_conditions
        }
        self.n_classes = len(self.unique_conditions)

    def denoise_single(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: Optional[int] = None,
    ) -> torch.Tensor:
        """Single condition denoising.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input [B, D].
        sigma : torch.Tensor
            Noise level [B].
        condition : int, optional
            Condition label. None for unconditional.

        Returns
        -------
        torch.Tensor
            Denoised result.
        """
        original_shape = x.shape
        x_flat = x.flatten(1)

        if sigma.ndim == 0:
            sigma = sigma.expand(x_flat.shape[0])
        sigma = sigma.flatten()

        data_flat = self.data_flat.to(x.device)

        if condition is not None and condition in self.condition_indices:
            indices = self.condition_indices[condition].to(x.device)
            data_subset = data_flat[indices]
        else:
            data_subset = data_flat

        weights = self.compute_gaussian_weights(x_flat, data_subset, sigma)
        denoised_flat = torch.mm(weights, data_subset)

        return denoised_flat.view(original_shape)

    def denoise_multi_condition(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        conditions: List[int],
        guidance_scales: Optional[List[float]] = None,
        mode: str = "and",
    ) -> torch.Tensor:
        """Multi-condition composition.

        AND mode (Product of Experts):
            D_combo = D_uncond + Σ_i w_i · (D_ci - D_uncond)

        This is equivalent to: p(y|c1,c2,...) ∝ p(y|c1)·p(y|c2)·... / p(y)^(n-1)

        Parameters
        ----------
        x : torch.Tensor
            Noisy input.
        sigma : torch.Tensor
            Noise level.
        conditions : list of int
            List of condition labels.
        guidance_scales : list of float, optional
            Per-condition guidance scales. Default: 1.0 for all.
        mode : str, optional
            Composition mode: "and" (default) or "or".

        Returns
        -------
        torch.Tensor
            Composed denoised result.
        """
        if guidance_scales is None:
            guidance_scales = [1.0] * len(conditions)

        # Get unconditional prediction
        d_uncond = self.denoise_single(x, sigma, condition=None)

        if mode == "and":
            # AND: D_combo = D_uncond + Σ_i w_i · (D_ci - D_uncond)
            result = d_uncond.clone()
            for cond, scale in zip(conditions, guidance_scales):
                d_cond = self.denoise_single(x, sigma, condition=cond)
                result = result + scale * (d_cond - d_uncond)
            return result

        elif mode == "or":
            # OR: Simple average of conditional predictions
            result = torch.zeros_like(d_uncond)
            for cond, scale in zip(conditions, guidance_scales):
                d_cond = self.denoise_single(x, sigma, condition=cond)
                result = result + scale * d_cond
            return result / sum(guidance_scales)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'and' or 'or'.")

    def denoise_with_negation(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        positive_conditions: List[int],
        negative_conditions: List[int],
        positive_scales: Optional[List[float]] = None,
        negative_scales: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Denoising with positive and negative conditions.

        D = D_uncond + Σ w_i^+ · (D_ci^+ - D_uncond) - Σ w_j^- · (D_cj^- - D_uncond)

        Positive conditions: guide TOWARDS these classes.
        Negative conditions: guide AWAY from these classes.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input.
        sigma : torch.Tensor
            Noise level.
        positive_conditions : list of int
            Conditions to guide towards.
        negative_conditions : list of int
            Conditions to guide away from.
        positive_scales : list of float, optional
            Scales for positive conditions.
        negative_scales : list of float, optional
            Scales for negative conditions.

        Returns
        -------
        torch.Tensor
            Denoised result.
        """
        if positive_scales is None:
            positive_scales = [1.0] * len(positive_conditions)
        if negative_scales is None:
            negative_scales = [1.0] * len(negative_conditions)

        d_uncond = self.denoise_single(x, sigma, condition=None)
        result = d_uncond.clone()

        # Add positive conditions
        for cond, scale in zip(positive_conditions, positive_scales):
            d_cond = self.denoise_single(x, sigma, condition=cond)
            result = result + scale * (d_cond - d_uncond)

        # Subtract negative conditions
        for cond, scale in zip(negative_conditions, negative_scales):
            d_cond = self.denoise_single(x, sigma, condition=cond)
            result = result - scale * (d_cond - d_uncond)

        return result

    def denoise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[Union[int, List[int]]] = None,
        guidance_scales: Optional[Union[float, List[float]]] = None,
        mode: str = "and",
        **kwargs,
    ) -> torch.Tensor:
        """Unified denoising interface.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input.
        sigma : torch.Tensor
            Noise level.
        conditions : int or list of int, optional
            Single condition or list of conditions.
        guidance_scales : float or list of float, optional
            Guidance scale(s).
        mode : str, optional
            Composition mode for multiple conditions.

        Returns
        -------
        torch.Tensor
            Denoised result.
        """
        if conditions is None:
            return self.denoise_single(x, sigma, condition=None)

        if isinstance(conditions, int):
            # Single condition with optional guidance
            if guidance_scales is None or guidance_scales == 1.0:
                return self.denoise_single(x, sigma, condition=conditions)
            else:
                d_uncond = self.denoise_single(x, sigma, condition=None)
                d_cond = self.denoise_single(x, sigma, condition=conditions)
                scale = guidance_scales if isinstance(guidance_scales, float) else guidance_scales[0]
                return d_uncond + scale * (d_cond - d_uncond)

        # Multiple conditions
        if isinstance(guidance_scales, (int, float)):
            guidance_scales = [float(guidance_scales)] * len(conditions)

        return self.denoise_multi_condition(
            x, sigma, conditions, guidance_scales, mode
        )


class MultiAttributeDenoiser(Denoiser, GaussianKernelMixin):
    """Multi-attribute denoiser for data with multiple independent attributes.

    For data with attributes like (shape, color), allows conditioning on
    any subset of attributes.

    Parameters
    ----------
    data : torch.Tensor
        Dataset of shape [N, D].
    attributes : list of torch.Tensor
        List of attribute tensors, each of shape [N].
    attribute_names : list of str, optional
        Names for each attribute.

    Examples
    --------
    >>> data, shapes, colors = make_color_shape_dataset()
    >>> denoiser = MultiAttributeDenoiser(data, [shapes, colors], ["shape", "color"])
    >>> # Generate "blue triangle": shape=2, color=1
    >>> samples = sample_heun(
    ...     denoiser, latents, sigmas,
    ...     attr_conditions={"shape": 2, "color": 1}
    ... )
    """

    def __init__(
        self,
        data: torch.Tensor,
        attributes: List[torch.Tensor],
        attribute_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize multi-attribute denoiser."""
        self.data = data
        self.data_flat = data.flatten(1)
        self.attributes = attributes
        self.n_attributes = len(attributes)

        if attribute_names is None:
            attribute_names = [f"attr_{i}" for i in range(self.n_attributes)]
        self.attribute_names = attribute_names
        self.name_to_idx = {name: i for i, name in enumerate(attribute_names)}

        # Pre-compute indices for each attribute value
        self.attribute_indices: List[Dict[int, torch.Tensor]] = []
        for attr in attributes:
            unique_vals = torch.unique(attr)
            indices = {
                int(v): (attr == v).nonzero(as_tuple=True)[0]
                for v in unique_vals
            }
            self.attribute_indices.append(indices)

    def denoise_uncond(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Unconditional denoising."""
        original_shape = x.shape
        x_flat = x.flatten(1)

        if sigma.ndim == 0:
            sigma = sigma.expand(x_flat.shape[0])
        sigma = sigma.flatten()

        data_flat = self.data_flat.to(x.device)
        weights = self.compute_gaussian_weights(x_flat, data_flat, sigma)
        denoised_flat = torch.mm(weights, data_flat)

        return denoised_flat.view(original_shape)

    def denoise_single_attr(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        attr_idx: int,
        attr_val: int,
    ) -> torch.Tensor:
        """Single attribute conditional denoising."""
        original_shape = x.shape
        x_flat = x.flatten(1)

        if sigma.ndim == 0:
            sigma = sigma.expand(x_flat.shape[0])
        sigma = sigma.flatten()

        data_flat = self.data_flat.to(x.device)
        indices = self.attribute_indices[attr_idx].get(attr_val)

        if indices is not None:
            indices = indices.to(x.device)
            data_subset = data_flat[indices]
        else:
            data_subset = data_flat

        weights = self.compute_gaussian_weights(x_flat, data_subset, sigma)
        denoised_flat = torch.mm(weights, data_subset)

        return denoised_flat.view(original_shape)

    def denoise_multi_attr(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        attr_conditions: Dict[Union[int, str], int],
        guidance_scales: Optional[Dict[Union[int, str], float]] = None,
    ) -> torch.Tensor:
        """Multi-attribute conditional denoising (Composable).

        D = D_uncond + Σ_i w_i · (D_attr_i - D_uncond)

        Parameters
        ----------
        x : torch.Tensor
            Noisy input.
        sigma : torch.Tensor
            Noise level.
        attr_conditions : dict
            {attr_idx: attr_val} or {attr_name: attr_val}.
        guidance_scales : dict, optional
            {attr_idx: scale} or {attr_name: scale}.

        Returns
        -------
        torch.Tensor
            Denoised result.
        """
        if guidance_scales is None:
            guidance_scales = {}

        d_uncond = self.denoise_uncond(x, sigma)
        result = d_uncond.clone()

        for key, attr_val in attr_conditions.items():
            # Convert name to index if needed
            if isinstance(key, str):
                attr_idx = self.name_to_idx[key]
            else:
                attr_idx = key

            scale = guidance_scales.get(key, 1.0)
            d_attr = self.denoise_single_attr(x, sigma, attr_idx, attr_val)
            result = result + scale * (d_attr - d_uncond)

        return result

    def denoise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        attr_conditions: Optional[Dict[Union[int, str], int]] = None,
        guidance_scales: Optional[Dict[Union[int, str], float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Unified interface."""
        if attr_conditions is None or len(attr_conditions) == 0:
            return self.denoise_uncond(x, sigma)
        return self.denoise_multi_attr(x, sigma, attr_conditions, guidance_scales)
