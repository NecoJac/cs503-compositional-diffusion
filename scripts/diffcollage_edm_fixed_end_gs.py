import os
import math
import glob
import csv
import json
import pickle
from itertools import product

import numpy as np
import scipy.linalg
import torch as th
import dnnlib
import PIL.Image
import diff_collage as dc
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGENET_LANDSCAPES_ROOT = os.environ.get(
    "IMAGENET_LANDSCAPES_ROOT",
    os.path.join(PROJECT_ROOT, "data", "imagenet_landscapes"),
)


# ============================================================
# Model / generation utilities
# ============================================================

def load_edm_model(network_pkl, device="cuda"):
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)["ema"].to(device).eval()
    return net


class EDMEpsWrapper:
    def __init__(self, net, class_idx=None):
        self.net = net
        self.class_idx = class_idx

    def __call__(self, xs, scalar_t, enable_grad=False):
        B = xs.shape[0]
        device = xs.device

        if self.class_idx is not None:
            class_labels = th.zeros(B, 1000, device=device)
            class_labels[:, self.class_idx] = 1.0
        else:
            class_labels = None

        if not th.is_tensor(scalar_t):
            scalar_t = th.tensor(scalar_t, device=device, dtype=xs.dtype)
        else:
            scalar_t = scalar_t.to(device)

        if scalar_t.ndim == 0:
            sigma = scalar_t.repeat(B)
        elif scalar_t.ndim == 1:
            sigma = scalar_t if scalar_t.shape[0] == B else scalar_t.repeat(B)
        else:
            raise ValueError("scalar_t must be scalar or 1D")

        x_in = xs.to(th.float32)
        sigma_in = sigma.to(th.float32)
        sigma_img = sigma_in.view(-1, 1, 1, 1)

        context = th.enable_grad() if enable_grad else th.no_grad()
        with context:
            x0_hat = self.net(x_in, sigma_in, class_labels)
            eps_hat = (x_in - x0_hat) / sigma_img

        return eps_hat.to(xs.dtype)


def load_fixed_image(path, device="cuda", image_size=64, overlap_size=32):
    img = PIL.Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), PIL.Image.LANCZOS)
    img = th.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img = img * 2.0 - 1.0
    img = img[:, :, -overlap_size:]   # keep right half only
    return img.to(device)


def load_fixed_images(paths, device="cuda", image_size=64, overlap_size=32):
    images = [
        load_fixed_image(
            path,
            device=device,
            image_size=image_size,
            overlap_size=overlap_size,
        )
        for path in paths
    ]
    return th.stack(images, dim=0)


def chunk_list(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def make_trial_output_dir(
    output_root,
    guidance_scale,
    smc_lambda,
    smc_k,
    s_churn,
    n_step,
    solver,
    class_name,
):
    trial_dir = make_trial_dir_name(
        guidance_scale=guidance_scale,
        smc_lambda=smc_lambda,
        smc_k=smc_k,
        s_churn=s_churn,
        n_step=n_step,
        solver=solver,
    )
    class_dir = os.path.join(output_root, trial_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    return class_dir


def make_trial_dir_name(guidance_scale, smc_lambda, smc_k, s_churn, n_step, solver):
    return (
        f"gs_{guidance_scale}_lambda_{smc_lambda}_k_{smc_k}_"
        f"churn_{s_churn}_steps_{n_step}_solver_{solver}"
    )


def save_generated_samples(batch, output_dir, fixed_paths):
    images = batch.detach().cpu()
    if len(images) != len(fixed_paths):
        raise ValueError(
            f"Sample batch size {len(images)} does not match number of fixed paths {len(fixed_paths)}"
        )

    for image, fixed_path in zip(images, fixed_paths):
        base_name = os.path.splitext(os.path.basename(fixed_path))[0]
        image = image.clamp(-1, 1)
        image = ((image + 1.0) * 127.5).round().to(th.uint8)
        image = image.permute(1, 2, 0).numpy()
        PIL.Image.fromarray(image).save(os.path.join(output_dir, f"{base_name}.jpg"))


def get_generated_sample_output_path(output_dir, fixed_path):
    base_name = os.path.splitext(os.path.basename(fixed_path))[0]
    return os.path.join(output_dir, f"{base_name}.jpg")


def load_generated_sample(path, device="cpu"):
    img = PIL.Image.open(path).convert("RGB")
    img = th.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img = img * 2.0 - 1.0
    return img.to(device)


def load_generated_samples(paths, device="cpu"):
    images = [load_generated_sample(path, device=device) for path in paths]
    return th.stack(images, dim=0)


def append_trial_result_to_csv(csv_path, result):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "trial_dir",
        "output_root",
        "image_root",
        "device",
        "mean_fid",
        "std_fid",
        "stderr_fid",
        "ci95_fid",
        "mean_fid_all_landscapes",
        "std_fid_all_landscapes",
        "stderr_fid_all_landscapes",
        "ci95_fid_all_landscapes",
        "guidance_scale",
        "smc_lambda",
        "smc_k",
        "s_churn",
        "n_step",
        "solver",
        "use_smc",
        "batch_size",
        "num_passes",
        "num_classes",
        "class_names",
        "class_ids",
        "num_img",
        "overlap_size",
        "crop_size",
        "crop_stride",
        "crop_indices",
        "img_shape",
        "ref_mode",
        "num_real_total",
        "num_real_per_class",
        "ref_class_names",
        "per_class_fid",
        "ref_mode_all_landscapes",
        "num_real_total_all_landscapes",
        "ref_class_names_all_landscapes",
        "per_class_fid_all_landscapes",
    ]
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: result.get(key) for key in fieldnames})


def generate_long_image(
    net,
    fixed_images,
    device="cuda",
    batch_size=1,
    n_step=40,
    overlap_size=32,
    num_img=5,
    ts_order=5,
    img_shape=(3, 64, 64),
    s_churn=10.0,
    index_imagenet=None,
    guidance_scale=1.0,
    use_smc=True,
    smc_lambda=1.0,
    smc_k=1.0,
    solver="euler",
):
    eps_fn = EDMEpsWrapper(net, index_imagenet)

    worker = dc.CondIndLongFixedEndCtrlSmc(
        shape=img_shape,
        eps_scalar_t_fn=eps_fn,
        num_img=num_img,
        fixed_images=fixed_images,
        overlap_size=overlap_size,
        guidance_scale=guidance_scale,
        use_smc=use_smc,
        smc_lambda=smc_lambda,
        smc_k=smc_k,
    )

    sample_gen = dc.sampling(
        x=worker.generate_xT(batch_size).to(device),
        noise_fn=worker.noise,
        rev_ts=worker.rev_ts(n_step, ts_order),
        x0_pred_fn=worker.x0_fn,
        s_churn=s_churn,
        return_traj=False,
        solver=solver,
    )

    sample_full = worker.attach_fixed_end(sample_gen)
    worker.reset_fixed_end_noise()
    return sample_full


# ============================================================
# FID utilities
# ============================================================

def load_inception_detector(device="cuda"):
    detector_url = (
        "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"
        "versions/1/files/metrics/inception-2015-12-05.pkl"
    )
    with dnnlib.util.open_url(detector_url, verbose=True) as f:
        detector_net = pickle.load(f).to(device).eval()
    return detector_net


def to_uint8_images(x):
    """
    x: [B, C, H, W] in [-1, 1]
    returns uint8-like float tensor in [0,255] expected by the detector code
    """
    x = (x.clamp(-1, 1) * 127.5 + 128).clamp(0, 255)
    return x


@th.no_grad()
def compute_inception_features(images, detector_net, device="cuda", batch_size=64):
    """
    images: [N, C, H, W] in [-1,1] or already converted float [0,255]
    returns features [N, 2048]
    """
    feats = []
    N = images.shape[0]
    for start in range(0, N, batch_size):
        batch = images[start:start + batch_size].to(device)
        if batch.max() <= 1.5 and batch.min() >= -1.5:
            batch = to_uint8_images(batch)
        feat = detector_net(batch, return_features=True)
        feats.append(feat.to(th.float64).cpu())
    return th.cat(feats, dim=0)


def compute_stats_from_features(features):
    """
    features: [N, D] torch or numpy
    """
    if isinstance(features, th.Tensor):
        features = features.cpu().numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


# ============================================================
# Crop extraction
# ============================================================

def extract_right_crop_by_index(batch_long_imgs, crop_size=64, stride=32, crop_idx=0):
    """
    Extract one crop from the right side of each generated long image.

    batch_long_imgs: [B, C, H, W]
    Returns:
        crop: [B, C, crop_size, crop_size]

    Example with stride=32:
      crop_idx=0: rightmost 64x64
      crop_idx=1: shifted 32 px left
    """
    B, C, H, W = batch_long_imgs.shape
    if H < crop_size or W < crop_size:
        raise ValueError(f"Image too small for crop_size={crop_size}: got {(B, C, H, W)}")

    x1 = W - crop_idx * stride
    x0 = x1 - crop_size
    if x0 < 0:
        raise ValueError(
            f"Cannot extract crop {crop_idx}: x0={x0} < 0. "
            f"Try a smaller crop_idx or stride."
        )

    crop = batch_long_imgs[:, :, :, x0:x1]
    if crop.shape[-1] != crop_size:
        raise RuntimeError(f"Bad crop width: got {crop.shape}")
    return crop


def extract_right_crops_by_indices(batch_long_imgs, crop_size=64, stride=32, crop_indices=(0,)):
    crops = [
        extract_right_crop_by_index(
            batch_long_imgs,
            crop_size=crop_size,
            stride=stride,
            crop_idx=crop_idx,
        )
        for crop_idx in crop_indices
    ]
    return th.cat(crops, dim=0)


# ============================================================
# Real-image loading for class references
# ============================================================

def load_real_images_from_folder(folder, max_images=None):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = sorted(paths)

    if max_images is not None:
        paths = paths[:max_images]

    images = []
    for p in paths:
        img = PIL.Image.open(p).convert("RGB").resize((64, 64), PIL.Image.LANCZOS)
        arr = np.array(img)
        x = th.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        x = x * 2.0 - 1.0
        images.append(x)

    if len(images) == 0:
        raise ValueError(f"No images found in folder: {folder}")

    return th.stack(images, dim=0)  # [N,3,64,64]


def precompute_pooled_reference_stats(
    class_list,
    image_root,
    detector_net,
    device="cuda",
    max_real_per_class=None,
):
    """
    Pool all real images from the provided landscape classes into one
    reference distribution.
    """
    all_real_imgs = []
    ref_class_names = []
    for name, cid in class_list:
        folder = os.path.join(image_root, f"class_{cid}_{name}")
        real_imgs = load_real_images_from_folder(folder, max_images=max_real_per_class)
        print(f"[ref] {name}: {real_imgs.shape[0]} real images")
        all_real_imgs.append(real_imgs)
        ref_class_names.append(name)

    all_real_imgs = th.cat(all_real_imgs, dim=0)
    real_feats = compute_inception_features(all_real_imgs, detector_net, device=device)
    mu_ref, sigma_ref = compute_stats_from_features(real_feats)
    print(f"[ref] pooled landscape images: {all_real_imgs.shape[0]}")
    return {
        "mu": mu_ref,
        "sigma": sigma_ref,
        "num_real_total": int(all_real_imgs.shape[0]),
        "ref_mode": "pooled_landscape",
        "ref_class_names": json.dumps(ref_class_names),
    }


def precompute_class_reference_stats(
    class_list,
    image_root,
    detector_net,
    device="cuda",
    max_real_per_class=None,
):
    """
    Precompute one real-image reference distribution per class.
    """
    ref_stats_by_name = {}
    num_real_per_class = {}
    ref_class_names = []

    for name, cid in class_list:
        folder = os.path.join(image_root, f"class_{cid}_{name}")
        real_imgs = load_real_images_from_folder(folder, max_images=max_real_per_class)
        print(f"[class-ref] {name}: {real_imgs.shape[0]} real images")

        real_feats = compute_inception_features(real_imgs, detector_net, device=device)
        mu_ref, sigma_ref = compute_stats_from_features(real_feats)
        ref_stats_by_name[name] = {
            "mu": mu_ref,
            "sigma": sigma_ref,
            "num_real": int(real_imgs.shape[0]),
            "ref_mode": "class_specific",
            "ref_class_name": name,
        }
        num_real_per_class[name] = int(real_imgs.shape[0])
        ref_class_names.append(name)

    return {
        "ref_stats_by_name": ref_stats_by_name,
        "num_real_total": int(sum(num_real_per_class.values())),
        "num_real_per_class": json.dumps(num_real_per_class),
        "ref_mode": "class_specific",
        "ref_class_names": json.dumps(ref_class_names),
    }


def discover_class_list_from_image_root(image_root):
    class_list = []
    for entry in sorted(os.listdir(image_root)):
        full_path = os.path.join(image_root, entry)
        if not os.path.isdir(full_path):
            continue
        if not entry.startswith("class_"):
            continue

        parts = entry.split("_", 2)
        if len(parts) != 3:
            print(f"[warn] skipping unexpected folder name: {entry}")
            continue

        _, class_id_str, class_name = parts
        try:
            class_id = int(class_id_str)
        except ValueError:
            print(f"[warn] skipping folder with non-integer class id: {entry}")
            continue

        class_list.append((class_name, class_id))

    if not class_list:
        raise ValueError(f"No class_* folders found in image root: {image_root}")

    return class_list


def summarize_fid_dict(per_class_fid):
    per_class_fid_values = np.array(list(per_class_fid.values()), dtype=np.float64)
    mean_fid = float(np.mean(per_class_fid_values))
    std_fid = float(np.std(per_class_fid_values, ddof=1)) if len(per_class_fid_values) > 1 else 0.0
    stderr_fid = float(std_fid / np.sqrt(len(per_class_fid_values))) if len(per_class_fid_values) > 1 else 0.0
    ci95_fid = float(1.96 * stderr_fid) if len(per_class_fid_values) > 1 else 0.0
    return {
        "mean_fid": mean_fid,
        "std_fid": std_fid,
        "stderr_fid": stderr_fid,
        "ci95_fid": ci95_fid,
    }


# ============================================================
# Evaluation for one hyperparameter combination
# ============================================================

@th.no_grad()
def evaluate_param_combo(
    net,
    detector_net,
    class_list,
    image_root,
    device="cuda",
    num_passes=10,
    batch_size=1,
    n_step=80,
    overlap_size=32,
    num_img=5,
    img_shape=(3, 64, 64),
    s_churn=10.0,
    guidance_scale=1.0,
    smc_lambda=1.0,
    smc_k=0.01,
    use_smc=True,
    solver="euler",
    crop_size=64,
    crop_stride=32,
    crop_indices=(0, 1),
    class_ref_stats=None,
    pooled_all_landscapes_ref_stats=None,
    output_root="image_outputs",
):
    """
    Returns:
        result dict with average FID and per-class FID
    """
    if class_ref_stats is None:
        raise ValueError("class_ref_stats must be precomputed before evaluation.")
    if pooled_all_landscapes_ref_stats is None:
        raise ValueError("pooled_all_landscapes_ref_stats must be precomputed before evaluation.")

    per_class_fid = {}
    per_class_fid_all_landscapes = {}

    for name, idx in class_list:
        print(f"\n[class={name}] collecting samples...")
        class_output_dir = make_trial_output_dir(
            output_root=output_root,
            guidance_scale=guidance_scale,
            smc_lambda=smc_lambda,
            smc_k=smc_k,
            s_churn=s_churn,
            n_step=n_step,
            solver=solver,
            class_name=name,
        )

        class_fixed_paths = [
        os.path.join(image_root, f"class_{idx}_{name}", f"{i:06d}.jpg")
        for i in range(num_passes)
        ]

        gen_crops_all = []
        existing_generated_paths = []
        missing_fixed_paths = []

        for fixed_path in class_fixed_paths:
            generated_output_path = get_generated_sample_output_path(class_output_dir, fixed_path)
            if os.path.exists(generated_output_path):
                existing_generated_paths.append(generated_output_path)
            else:
                missing_fixed_paths.append(fixed_path)

        print(
            f"[class={name}] found {len(existing_generated_paths)} existing generated images "
            f"and {len(missing_fixed_paths)} missing images for num_passes={num_passes}"
        )

        for generated_paths_batch in chunk_list(existing_generated_paths, batch_size):
            existing_samples = load_generated_samples(generated_paths_batch, device="cpu")
            crops = extract_right_crops_by_indices(
                existing_samples,
                crop_size=crop_size,
                stride=crop_stride,
                crop_indices=crop_indices,
            )
            gen_crops_all.append(crops.cpu())

        for fixed_paths_batch in chunk_list(missing_fixed_paths, batch_size):
            fixed_images = load_fixed_images(
                fixed_paths_batch,
                device=device,
                image_size=64,
                overlap_size=overlap_size,
            )

            sample = generate_long_image(
                net=net,
                fixed_images=fixed_images,
                device=device,
                batch_size=len(fixed_paths_batch),
                n_step=n_step,
                overlap_size=overlap_size,
                num_img=num_img,
                img_shape=img_shape,
                s_churn=s_churn,
                index_imagenet=idx,
                guidance_scale=guidance_scale,
                use_smc=use_smc,
                smc_lambda=smc_lambda,
                smc_k=smc_k,
                solver=solver,
            )

            save_generated_samples(
                batch=sample,
                output_dir=class_output_dir,
                fixed_paths=fixed_paths_batch,
            )

            crops = extract_right_crops_by_indices(
                sample,
                crop_size=crop_size,
                stride=crop_stride,
                crop_indices=crop_indices,
            )
            gen_crops_all.append(crops.cpu())

        if not gen_crops_all:
            raise ValueError(f"No generated crops were collected for class={name}")

        gen_crops_all = th.cat(gen_crops_all, dim=0)
        print(f"[class={name}] generated crop count: {gen_crops_all.shape[0]}")

        gen_feats = compute_inception_features(gen_crops_all, detector_net, device=device)
        mu_gen, sigma_gen = compute_stats_from_features(gen_feats)

        class_specific_ref = class_ref_stats["ref_stats_by_name"][name]
        fid = calculate_fid_from_inception_stats(
            mu_gen,
            sigma_gen,
            class_specific_ref["mu"],
            class_specific_ref["sigma"],
        )
        per_class_fid[name] = fid
        print(f"[class={name}] class-specific FID = {fid:.4f}")

        fid_all_landscapes = calculate_fid_from_inception_stats(
            mu_gen,
            sigma_gen,
            pooled_all_landscapes_ref_stats["mu"],
            pooled_all_landscapes_ref_stats["sigma"],
        )
        per_class_fid_all_landscapes[name] = fid_all_landscapes
        print(f"[class={name}] pooled all-landscapes FID = {fid_all_landscapes:.4f}")

    class_specific_summary = summarize_fid_dict(per_class_fid)
    pooled_all_landscapes_summary = summarize_fid_dict(per_class_fid_all_landscapes)
    return {
        "mean_fid": class_specific_summary["mean_fid"],
        "std_fid": class_specific_summary["std_fid"],
        "stderr_fid": class_specific_summary["stderr_fid"],
        "ci95_fid": class_specific_summary["ci95_fid"],
        "mean_fid_all_landscapes": pooled_all_landscapes_summary["mean_fid"],
        "std_fid_all_landscapes": pooled_all_landscapes_summary["std_fid"],
        "stderr_fid_all_landscapes": pooled_all_landscapes_summary["stderr_fid"],
        "ci95_fid_all_landscapes": pooled_all_landscapes_summary["ci95_fid"],
        "per_class_fid": json.dumps(per_class_fid),
        "per_class_fid_all_landscapes": json.dumps(per_class_fid_all_landscapes),
        "guidance_scale": guidance_scale,
        "smc_lambda": smc_lambda,
        "smc_k": smc_k,
        "s_churn": s_churn,
        "n_step": n_step,
        "solver": solver,
        "use_smc": use_smc,
        "batch_size": batch_size,
        "num_passes": num_passes,
        "num_classes": len(class_list),
        "class_names": json.dumps([name for name, _ in class_list]),
        "class_ids": json.dumps([idx for _, idx in class_list]),
        "num_img": num_img,
        "overlap_size": overlap_size,
        "crop_size": crop_size,
        "crop_stride": crop_stride,
        "crop_indices": json.dumps(list(crop_indices)),
        "img_shape": json.dumps(list(img_shape)),
        "ref_mode": class_ref_stats["ref_mode"],
        "num_real_total": class_ref_stats["num_real_total"],
        "num_real_per_class": class_ref_stats["num_real_per_class"],
        "ref_class_names": class_ref_stats["ref_class_names"],
        "ref_mode_all_landscapes": pooled_all_landscapes_ref_stats["ref_mode"],
        "num_real_total_all_landscapes": pooled_all_landscapes_ref_stats["num_real_total"],
        "ref_class_names_all_landscapes": pooled_all_landscapes_ref_stats["ref_class_names"],
        "image_root": image_root,
        "output_root": output_root,
        "trial_dir": make_trial_dir_name(
            guidance_scale=guidance_scale,
            smc_lambda=smc_lambda,
            smc_k=smc_k,
            s_churn=s_churn,
            n_step=n_step,
            solver=solver,
        ),
    }


# ============================================================
# Grid search
# ============================================================

def run_grid_search():
    device = "cuda"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_root = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
    network_pkl = f"{model_root}/edm-imagenet-64x64-cond-adm.pkl"
    net = load_edm_model(network_pkl, device=device)
    detector_net = load_inception_detector(device=device)

    class_list = [
        ("lakeside", 975),
        ("alp", 970),
        ("coral_reef", 973),
    ]
    
    image_root = IMAGENET_LANDSCAPES_ROOT
    output_root = os.path.join(script_dir, "image_outputs")
    results_csv_path = os.path.join(output_root, "grid_search_results_class_specific.csv")
    

    # Precompute real reference stats once.
    class_ref_stats = precompute_class_reference_stats(
        class_list=class_list,
        image_root=image_root,
        detector_net=detector_net,
        device=device,
        max_real_per_class=None,
    )

    all_landscape_class_list = discover_class_list_from_image_root(image_root)
    pooled_all_landscapes_ref_stats = precompute_pooled_reference_stats(
        class_list=all_landscape_class_list,
        image_root=image_root,
        detector_net=detector_net,
        device=device,
        max_real_per_class=None,
    )

    # Search space
    guidance_scales = [2]
    smc_lambdas = [0]
    smc_ks = [0]
    s_churns = [0]
    n_steps_list = [10, 20, 40, 60, 80]
    solvers = ["heun", "euler"]  # or ["euler", "heun"]

    results = []

    all_combos = list(product(
        guidance_scales,
        smc_lambdas,
        smc_ks,
        s_churns,
        n_steps_list,
        solvers,
    ))

    for guidance_scale, smc_lambda, smc_k, s_churn, n_step, solver in all_combos:
        print("\n" + "=" * 80)
        print(
            f"Trial: gs={guidance_scale}, lambda={smc_lambda}, "
            f"k={smc_k}, churn={s_churn}, steps={n_step}, solver={solver}"
        )
        if smc_k == 0:
            use_smc = False
        else:
            use_smc = True

        result = evaluate_param_combo(
            net=net,
            detector_net=detector_net,
            class_list=class_list,
            image_root=image_root,
            device=device,
            num_passes=30,            
            batch_size=10,
            n_step=n_step,
            overlap_size=32,
            num_img=5,
            img_shape=(3, 64, 64),
            s_churn=s_churn,
            guidance_scale=guidance_scale,
            smc_lambda=smc_lambda,
            smc_k=smc_k,
            use_smc=use_smc,
            solver=solver,
            crop_size=64,
            crop_stride=32,
            crop_indices=(0, 1),
            class_ref_stats=class_ref_stats,
            pooled_all_landscapes_ref_stats=pooled_all_landscapes_ref_stats,
            output_root=output_root,
        )

        results.append(result)
        result["device"] = device
        append_trial_result_to_csv(results_csv_path, result)
        print(f"Mean class-specific FID: {result['mean_fid']:.4f}")
        print(f"Mean pooled all-landscapes FID: {result['mean_fid_all_landscapes']:.4f}")

    results = sorted(results, key=lambda x: x["mean_fid"])

    print("\n" + "#" * 80)
    print("Top results:")
    for r in results[:10]:
        print(
            f"mean_fid={r['mean_fid']:.4f} | "
            f"gs={r['guidance_scale']} | "
            f"lambda={r['smc_lambda']} | "
            f"k={r['smc_k']} | "
            f"churn={r['s_churn']} | "
            f"steps={r['n_step']} | "
            f"solver={r['solver']} | "
            f"per_class={r['per_class_fid']} | "
            f"per_class_all_landscapes={r['per_class_fid_all_landscapes']}"
        )


if __name__ == "__main__":
    run_grid_search()
