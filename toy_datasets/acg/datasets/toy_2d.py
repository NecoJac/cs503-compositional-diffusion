"""2D Toy Datasets.

Simple 2D datasets for testing and visualizing conditional diffusion models.
Most datasets return ``(data, labels)`` tensors. Some structured generators,
such as :func:`make_half_helix`, return multiple tensors describing coupled
variables used in factorized experiments.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def make_concentric_circles(
    n_samples_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate concentric circles dataset.

    Each class is a ring with increasing radius.

    Parameters
    ----------
    n_samples_per_class : int
        Samples per class.
    n_classes : int
        Number of concentric circles.
    noise : float
        Gaussian noise standard deviation.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels : torch.Tensor
        Class labels of shape [N].
    """
    data_list = []
    labels_list = []

    for i in range(n_classes):
        radius = 1.0 + i * 0.8
        theta = torch.rand(n_samples_per_class) * 2 * math.pi
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        points = torch.stack([x, y], dim=1)
        points += torch.randn_like(points) * noise

        data_list.append(points)
        labels_list.append(torch.full((n_samples_per_class,), i, dtype=torch.long))

    return torch.cat(data_list), torch.cat(labels_list)


def make_spirals(
    n_samples_per_class: int = 100,
    n_classes: int = 2,
    noise: float = 0.1,
    n_turns: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate interleaved spirals dataset.

    Parameters
    ----------
    n_samples_per_class : int
        Samples per spiral.
    n_classes : int
        Number of spirals.
    noise : float
        Gaussian noise standard deviation.
    n_turns : float
        Number of spiral turns.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels : torch.Tensor
        Class labels of shape [N].
    """
    data_list = []
    labels_list = []

    for i in range(n_classes):
        t = torch.linspace(0, n_turns * 2 * math.pi, n_samples_per_class)
        offset = 2 * math.pi * i / n_classes

        r = t / (n_turns * 2 * math.pi) * 2
        x = r * torch.cos(t + offset)
        y = r * torch.sin(t + offset)
        points = torch.stack([x, y], dim=1)
        points += torch.randn_like(points) * noise

        data_list.append(points)
        labels_list.append(torch.full((n_samples_per_class,), i, dtype=torch.long))

    return torch.cat(data_list), torch.cat(labels_list)


def make_checkerboard(
    n_samples_per_class: int = 50,
    grid_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate checkerboard pattern dataset.

    Parameters
    ----------
    n_samples_per_class : int
        Samples per class.
    grid_size : int
        Size of the grid.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels : torch.Tensor
        Class labels of shape [N].
    """
    data_list = []
    labels_list = []

    for i in range(grid_size):
        for j in range(grid_size):
            label = (i + j) % 2
            x = torch.rand(n_samples_per_class) + i - grid_size / 2
            y = torch.rand(n_samples_per_class) + j - grid_size / 2
            points = torch.stack([x, y], dim=1)

            data_list.append(points)
            labels_list.append(torch.full((n_samples_per_class,), label, dtype=torch.long))

    return torch.cat(data_list), torch.cat(labels_list)


def make_gaussian_mixture(
    n_samples_per_class: int = 50,
    n_classes: int = 4,
    spread: float = 3.0,
    std: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Gaussian mixture dataset.

    Each class is a Gaussian cluster arranged in a circle.

    Parameters
    ----------
    n_samples_per_class : int
        Samples per class.
    n_classes : int
        Number of Gaussian clusters.
    spread : float
        Distance of clusters from center.
    std : float
        Standard deviation of each cluster.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels : torch.Tensor
        Class labels of shape [N].
    """
    data_list = []
    labels_list = []

    for i in range(n_classes):
        angle = 2 * math.pi * i / n_classes
        center_x = spread * math.cos(angle)
        center_y = spread * math.sin(angle)

        points = torch.randn(n_samples_per_class, 2) * std
        points[:, 0] += center_x
        points[:, 1] += center_y

        data_list.append(points)
        labels_list.append(torch.full((n_samples_per_class,), i, dtype=torch.long))

    return torch.cat(data_list), torch.cat(labels_list)


def make_two_moons(
    n_samples_per_class: int = 100,
    noise: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two moons dataset.

    Two interleaved half-circles.

    Parameters
    ----------
    n_samples_per_class : int
        Samples per moon.
    noise : float
        Gaussian noise standard deviation.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels : torch.Tensor
        Class labels of shape [N].
    """
    # First moon
    theta1 = torch.linspace(0, math.pi, n_samples_per_class)
    x1 = torch.cos(theta1)
    y1 = torch.sin(theta1)
    moon1 = torch.stack([x1, y1], dim=1)
    moon1 += torch.randn_like(moon1) * noise

    # Second moon (shifted and flipped)
    theta2 = torch.linspace(0, math.pi, n_samples_per_class)
    x2 = 1 - torch.cos(theta2)
    y2 = 0.5 - torch.sin(theta2)
    moon2 = torch.stack([x2, y2], dim=1)
    moon2 += torch.randn_like(moon2) * noise

    data = torch.cat([moon1, moon2])
    labels = torch.cat([
        torch.zeros(n_samples_per_class, dtype=torch.long),
        torch.ones(n_samples_per_class, dtype=torch.long),
    ])

    return data, labels


def make_half_helix(
    n_samples: int = 2000,
    y_min: float = -2.0,
    y_max: float = 2.0,
    x_noise: float = 0.1,
    c_noise: float = 0.05,
    seed: int | None = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a coupled ``(x, y, c)`` toy dataset.

    The dataset is defined by

    .. math::
        y \\sim \\mathrm{Uniform}(y_{min}, y_{max})
        x = \\sin(y) + \\varepsilon_x
        c = \\mathbb{1}[y > 0] + \\varepsilon_c

    It is useful for testing dependency structure and factor-graph style
    decompositions where ``x`` and ``c`` become conditionally independent
    given ``y``.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    y_min : float
        Lower bound for the latent variable ``y``.
    y_max : float
        Upper bound for the latent variable ``y``.
    x_noise : float
        Standard deviation of Gaussian noise added to ``x``.
    c_noise : float
        Standard deviation of Gaussian noise added to ``c``.
    seed : int or None
        Random seed for reproducibility. If ``None``, uses the current global
        RNG state.

    Returns
    -------
    x : torch.Tensor
        Noisy sine coordinate of shape ``[N]``.
    y : torch.Tensor
        Latent coordinate of shape ``[N]``.
    c : torch.Tensor
        Soft binary attribute of shape ``[N]``.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    y = torch.empty(n_samples).uniform_(y_min, y_max, generator=generator)
    x = torch.sin(y) + torch.randn(n_samples, generator=generator) * x_noise
    c = (y > 0).to(torch.float32) + torch.randn(n_samples, generator=generator) * c_noise

    return x, y, c


def make_swiss_roll_slices(
    n_samples_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Swiss roll slices (2D projection).

    Parameters
    ----------
    n_samples_per_class : int
        Samples per slice.
    n_classes : int
        Number of slices.
    noise : float
        Gaussian noise standard deviation.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels : torch.Tensor
        Class labels of shape [N].
    """
    data_list = []
    labels_list = []

    for i in range(n_classes):
        t_min = 1.5 * math.pi + i * 2 * math.pi / n_classes
        t_max = t_min + 2 * math.pi / n_classes

        t = torch.linspace(t_min, t_max, n_samples_per_class)
        x = t * torch.cos(t)
        y = t * torch.sin(t)
        points = torch.stack([x, y], dim=1) / 10  # Scale down
        points += torch.randn_like(points) * noise

        data_list.append(points)
        labels_list.append(torch.full((n_samples_per_class,), i, dtype=torch.long))

    return torch.cat(data_list), torch.cat(labels_list)


def make_mickey_mouse(
    n_samples_per_class: int = 100,
    noise: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Mickey Mouse dataset.

    Three circles: face + two ears.
    Classes: 0=face, 1=left ear, 2=right ear

    Parameters
    ----------
    n_samples_per_class : int
        Samples per part.
    noise : float
        Gaussian noise standard deviation.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels : torch.Tensor
        Class labels of shape [N].
    """
    data_list = []
    labels_list = []

    # Face (big circle at center)
    theta = torch.rand(n_samples_per_class) * 2 * math.pi
    x = torch.cos(theta)
    y = torch.sin(theta)
    face = torch.stack([x, y], dim=1)
    face += torch.randn_like(face) * noise
    data_list.append(face)
    labels_list.append(torch.zeros(n_samples_per_class, dtype=torch.long))

    # Left ear
    theta = torch.rand(n_samples_per_class) * 2 * math.pi
    x = 0.5 * torch.cos(theta) - 0.9
    y = 0.5 * torch.sin(theta) + 0.9
    left_ear = torch.stack([x, y], dim=1)
    left_ear += torch.randn_like(left_ear) * noise
    data_list.append(left_ear)
    labels_list.append(torch.ones(n_samples_per_class, dtype=torch.long))

    # Right ear
    theta = torch.rand(n_samples_per_class) * 2 * math.pi
    x = 0.5 * torch.cos(theta) + 0.9
    y = 0.5 * torch.sin(theta) + 0.9
    right_ear = torch.stack([x, y], dim=1)
    right_ear += torch.randn_like(right_ear) * noise
    data_list.append(right_ear)
    labels_list.append(torch.full((n_samples_per_class,), 2, dtype=torch.long))

    return torch.cat(data_list), torch.cat(labels_list)


def make_smiley_face(
    n_samples_per_part: int = 50,
    noise: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate smiley face dataset.

    Classes: 0=face outline, 1=left eye, 2=right eye, 3=mouth

    Parameters
    ----------
    n_samples_per_part : int
        Samples per part.
    noise : float
        Gaussian noise standard deviation.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels : torch.Tensor
        Class labels of shape [N].
    """
    data_list = []
    labels_list = []

    # Face outline
    theta = torch.rand(n_samples_per_part * 2) * 2 * math.pi
    x = 1.5 * torch.cos(theta)
    y = 1.5 * torch.sin(theta)
    face = torch.stack([x, y], dim=1)
    face += torch.randn_like(face) * noise
    data_list.append(face)
    labels_list.append(torch.zeros(n_samples_per_part * 2, dtype=torch.long))

    # Left eye
    theta = torch.rand(n_samples_per_part) * 2 * math.pi
    x = 0.2 * torch.cos(theta) - 0.5
    y = 0.2 * torch.sin(theta) + 0.5
    left_eye = torch.stack([x, y], dim=1)
    left_eye += torch.randn_like(left_eye) * noise
    data_list.append(left_eye)
    labels_list.append(torch.ones(n_samples_per_part, dtype=torch.long))

    # Right eye
    theta = torch.rand(n_samples_per_part) * 2 * math.pi
    x = 0.2 * torch.cos(theta) + 0.5
    y = 0.2 * torch.sin(theta) + 0.5
    right_eye = torch.stack([x, y], dim=1)
    right_eye += torch.randn_like(right_eye) * noise
    data_list.append(right_eye)
    labels_list.append(torch.full((n_samples_per_part,), 2, dtype=torch.long))

    # Mouth (arc)
    theta = torch.linspace(math.pi + 0.5, 2 * math.pi - 0.5, n_samples_per_part)
    x = 0.8 * torch.cos(theta)
    y = 0.8 * torch.sin(theta) - 0.2
    mouth = torch.stack([x, y], dim=1)
    mouth += torch.randn_like(mouth) * noise
    data_list.append(mouth)
    labels_list.append(torch.full((n_samples_per_part,), 3, dtype=torch.long))

    return torch.cat(data_list), torch.cat(labels_list)


def make_letter_shapes(
    letter: str = "A",
    n_samples: int = 200,
    noise: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate letter shape dataset.

    Each stroke is a separate class.

    Parameters
    ----------
    letter : str
        Letter to generate ("A", "X", "Y").
    n_samples : int
        Total samples.
    noise : float
        Gaussian noise standard deviation.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels : torch.Tensor
        Class labels of shape [N].
    """
    data_list = []
    labels_list = []

    if letter == "A":
        strokes = [
            [(0, 0), (0.5, 1.5)],     # Left diagonal
            [(0.5, 1.5), (1, 0)],     # Right diagonal
            [(0.25, 0.5), (0.75, 0.5)],  # Horizontal bar
        ]
    elif letter == "X":
        strokes = [
            [(0, 0), (1, 1.5)],
            [(1, 0), (0, 1.5)],
        ]
    elif letter == "Y":
        strokes = [
            [(0, 1.5), (0.5, 0.75)],
            [(1, 1.5), (0.5, 0.75)],
            [(0.5, 0.75), (0.5, 0)],
        ]
    else:
        raise ValueError(f"Unknown letter: {letter}")

    n_per_stroke = n_samples // len(strokes)

    for i, (start, end) in enumerate(strokes):
        t = torch.rand(n_per_stroke)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        points = torch.stack([x, y], dim=1)
        points += torch.randn_like(points) * noise

        # Center
        points -= torch.tensor([0.5, 0.75])

        data_list.append(points)
        labels_list.append(torch.full((n_per_stroke,), i, dtype=torch.long))

    return torch.cat(data_list), torch.cat(labels_list)


def make_quadrant_dataset(
    n_samples_per_class: int = 50,
    spread: float = 2.0,
    std: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate quadrant dataset for multi-condition experiments.

    4 quadrants with two attributes:
    - Attribute A: top/bottom (0/1)
    - Attribute B: left/right (0/1)

    Quadrant layout:
        Q0 (A=0, B=0): Top-Left
        Q1 (A=0, B=1): Top-Right
        Q2 (A=1, B=0): Bottom-Left
        Q3 (A=1, B=1): Bottom-Right

    Parameters
    ----------
    n_samples_per_class : int
        Samples per quadrant.
    spread : float
        Distance from center.
    std : float
        Cluster standard deviation.

    Returns
    -------
    data : torch.Tensor
        Points of shape [N, 2].
    labels_A : torch.Tensor
        Attribute A (top/bottom) of shape [N].
    labels_B : torch.Tensor
        Attribute B (left/right) of shape [N].
    """
    data_list = []
    labels_A_list = []
    labels_B_list = []

    centers = [
        (-spread, spread),   # Q0: top-left
        (spread, spread),    # Q1: top-right
        (-spread, -spread),  # Q2: bottom-left
        (spread, -spread),   # Q3: bottom-right
    ]

    attrs = [
        (0, 0),  # Q0: A=top, B=left
        (0, 1),  # Q1: A=top, B=right
        (1, 0),  # Q2: A=bottom, B=left
        (1, 1),  # Q3: A=bottom, B=right
    ]

    for (cx, cy), (a, b) in zip(centers, attrs):
        points = torch.randn(n_samples_per_class, 2) * std
        points[:, 0] += cx
        points[:, 1] += cy

        data_list.append(points)
        labels_A_list.append(torch.full((n_samples_per_class,), a, dtype=torch.long))
        labels_B_list.append(torch.full((n_samples_per_class,), b, dtype=torch.long))

    return (
        torch.cat(data_list),
        torch.cat(labels_A_list),
        torch.cat(labels_B_list),
    )
