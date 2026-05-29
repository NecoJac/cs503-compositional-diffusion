import csv
import json
import os
import pickle
from dataclasses import asdict, dataclass

import dnnlib
import numpy as np
import PIL.Image
import torch as th


DEFAULT_MODEL_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/"
    "edm-imagenet-64x64-cond-adm.pkl"
)

DEFAULT_IMAGE_ROOT = "data/imagenet_landscapes"

DEFAULT_CLASSES = [
    ("lakeside", 975),
    ("volcano", 980),
    ("alp", 970),
    ("coral_reef", 973),
]


@dataclass
class GenerationConfig:
    method: str
    network_pkl: str = DEFAULT_MODEL_URL
    image_root: str = DEFAULT_IMAGE_ROOT
    output_root: str = "proposal_outputs"
    device: str = "cuda"
    seed: int = 13
    batch_size: int = 1
    right_image_offset: int = 1
    n_step: int = 40
    num_img: int = 3
    overlap_size: int = 32
    ts_order: int = 5
    s_churn: float = 0.0
    guidance_scale: float = 1.0
    solver: str = "heun"
    coupling_strength: float = 0.25
    correction_clip: float = 4.0
    init_correction_steps: int = 4
    init_correction_step_size: float = 0.15
    sigma_data: float = 0.5
    implicit_scale: float = 0.25
    condition_type: str = "image"
    left_class_name: str = "lakeside"
    right_class_name: str = "volcano"
    left_class_idx: int = 975
    right_class_idx: int = 980
    # ── SMC method (proposal_smc) extra fields ────────────────────
    num_particles: int = 4
    beta_smc: float = 1.0
    t_resample_start: float = 0.0
    t_resample_end: float = 0.8


class EDMEpsWrapper:
    def __init__(self, net, class_idx=None):
        self.net = net
        self.class_idx = class_idx

    def __call__(self, xs, scalar_t, enable_grad=False):
        batch_size = xs.shape[0]
        device = xs.device

        if self.class_idx is not None:
            class_labels = th.zeros(batch_size, 1000, device=device)
            class_labels[:, self.class_idx] = 1.0
        else:
            class_labels = None

        if not th.is_tensor(scalar_t):
            scalar_t = th.tensor(scalar_t, device=device, dtype=xs.dtype)
        else:
            scalar_t = scalar_t.to(device)

        if scalar_t.ndim == 0:
            sigma = scalar_t.repeat(batch_size)
        elif scalar_t.ndim == 1:
            sigma = scalar_t if scalar_t.shape[0] == batch_size else scalar_t.repeat(batch_size)
        else:
            raise ValueError("scalar_t must be scalar or 1D")

        sigma_img = sigma.to(th.float32).view(-1, 1, 1, 1)
        context = th.enable_grad() if enable_grad else th.no_grad()
        with context:
            x0_hat = self.net(xs.to(th.float32), sigma.to(th.float32), class_labels)
            eps_hat = (xs.to(th.float32) - x0_hat) / sigma_img

        return eps_hat.to(xs.dtype)


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root(), path)


def load_edm_model(network_pkl, device="cuda"):
    with dnnlib.util.open_url(network_pkl) as f:
        return pickle.load(f)["ema"].to(device).eval()


def load_condition_image(path, side, device="cuda", image_size=64, overlap_size=32):
    if not os.path.exists(path):
        image_root = os.path.dirname(os.path.dirname(path))
        raise FileNotFoundError(
            f"Missing fixed conditioning image: {path}\n\n"
            f"{dataset_setup_message(image_root)}"
        )
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side}")

    img = PIL.Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), PIL.Image.LANCZOS)
    tensor = th.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    if side == "left":
        tensor = tensor[:, :, -overlap_size:]
    else:
        tensor = tensor[:, :, :overlap_size]
    return tensor.to(device)


def load_fixed_image(path, device="cuda", image_size=64, overlap_size=32):
    return load_condition_image(
        path,
        side="right",
        device=device,
        image_size=image_size,
        overlap_size=overlap_size,
    )


def load_fixed_images(paths, device="cuda", image_size=64, overlap_size=32):
    return th.stack(
        [
            load_fixed_image(
                path,
                device=device,
                image_size=image_size,
                overlap_size=overlap_size,
            )
            for path in paths
        ],
        dim=0,
    )


def load_bridge_conditions(
    left_paths,
    right_paths,
    device="cuda",
    image_size=64,
    overlap_size=32,
):
    if len(left_paths) != len(right_paths):
        raise ValueError(f"left_paths length {len(left_paths)} != right_paths length {len(right_paths)}")
    left = [
        load_condition_image(
            path,
            side="left",
            device=device,
            image_size=image_size,
            overlap_size=overlap_size,
        )
        for path in left_paths
    ]
    right = [
        load_condition_image(
            path,
            side="right",
            device=device,
            image_size=image_size,
            overlap_size=overlap_size,
        )
        for path in right_paths
    ]
    return th.stack(left, dim=0), th.stack(right, dim=0)


def fixed_image_path(image_root, class_name, class_idx, image_index):
    return os.path.join(
        image_root,
        f"class_{class_idx}_{class_name}",
        f"{image_index:06d}.jpg",
    )


def dataset_setup_message(image_root):
    return (
        "Prepare the bridge endpoint ImageNet landscape images first:\n"
        "  python scripts/download_imagenet_landscapes.py\n"
        "or pass an existing dataset folder with:\n"
        "  --image-root /path/to/imagenet_landscapes\n"
        f"Expected layout example:\n"
        f"  {os.path.join(image_root, 'class_975_lakeside', '000000.jpg')}"
    )


def validate_fixed_image_paths(paths, image_root):
    missing = [path for path in paths if not os.path.exists(path)]
    if not missing:
        return

    preview = "\n".join(f"  {path}" for path in missing[:5])
    extra = "" if len(missing) <= 5 else f"\n  ... and {len(missing) - 5} more"
    raise FileNotFoundError(
        "Missing fixed conditioning image(s):\n"
        f"{preview}{extra}\n\n"
        f"{dataset_setup_message(image_root)}"
    )


def save_tensor_image(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = (x.clamp(-1, 1) * 127.5 + 128).clamp(0, 255).to(th.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    PIL.Image.fromarray(x).save(path)


def tensor_to_pil(x):
    x = (x.clamp(-1, 1) * 127.5 + 128).clamp(0, 255).to(th.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    return PIL.Image.fromarray(x)


def upscale_image(image, scale):
    if scale <= 1:
        return image
    return image.resize(
        (image.width * scale, image.height * scale),
        PIL.Image.Resampling.NEAREST,
    )


def save_tensor_image_upscaled(x, path, scale):
    if scale <= 1:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    upscale_image(tensor_to_pil(x), scale).save(path)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_csv(path, row, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def config_to_jsonable(config):
    return asdict(config)
