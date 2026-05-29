import argparse
import os

import torch as th

import diff_collage as dc

from proposal_methods.common import (
    DEFAULT_CLASSES,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL_URL,
    EDMEpsWrapper,
    GenerationConfig,
    config_to_jsonable,
    fixed_image_path,
    load_bridge_conditions,
    load_edm_model,
    resolve_project_path,
    save_tensor_image,
    validate_fixed_image_paths,
    write_json,
)
from proposal_methods.workers import DiffCollageBridgeWorker, NaiveBridgeWorker, ProposalFinalBridgeWorker
from proposal_methods.smc_worker import ProposalSMCBridgeWorker, smc_sample
from proposal_methods.text_workers import (
    DiffCollageBridgeWorkerText,
    NaiveBridgeWorkerText,
    ProposalFinalBridgeWorkerText,
    text_to_class_idx,
)


METHODS = ("diffcollage", "naive", "bridge_correction", "proposal_smc")


def normalize_method(method):
    """Accept the old experiment name while writing new runs with the paper name."""
    if method == "proposal_final":
        return "bridge_correction"
    return method


def build_worker(method, eps_fn, left_fixed_images, right_fixed_images, config):
    method = normalize_method(method)
    kwargs = dict(
        shape=(3, 64, 64),
        eps_scalar_t_fn=eps_fn,
        num_img=config.num_img,
        left_fixed_images=left_fixed_images,
        right_fixed_images=right_fixed_images,
        overlap_size=config.overlap_size,
        guidance_scale=config.guidance_scale,
    )

    if method == "diffcollage":
        return DiffCollageBridgeWorker(**kwargs)
    if method == "naive":
        return NaiveBridgeWorker(**kwargs)
    if method == "bridge_correction":
        return ProposalFinalBridgeWorker(
            **kwargs,
            coupling_strength=config.coupling_strength,
            correction_clip=config.correction_clip,
            init_correction_steps=config.init_correction_steps,
            init_correction_step_size=config.init_correction_step_size,
            sigma_data=config.sigma_data,
            implicit_scale=config.implicit_scale,
        )
    if method == "proposal_smc":
        return ProposalSMCBridgeWorker(
            **kwargs,
            num_particles=config.num_particles,
            beta=config.beta_smc,
            t_resample_start=config.t_resample_start,
            t_resample_end=config.t_resample_end,
        )
    raise ValueError(f"Unknown method: {method}")


def build_worker_text(method, net, left_class_idx, right_class_idx, config):
    method = normalize_method(method)
    kwargs = dict(
        shape=(3, 64, 64),
        net=net,
        num_img=config.num_img,
        left_class_idx=left_class_idx,
        right_class_idx=right_class_idx,
        overlap_size=config.overlap_size,
        guidance_scale=config.guidance_scale,
    )

    if method == "diffcollage":
        return DiffCollageBridgeWorkerText(**kwargs)
    if method == "naive":
        return NaiveBridgeWorkerText(**kwargs)
    if method == "bridge_correction":
        return ProposalFinalBridgeWorkerText(
            **kwargs,
            coupling_strength=config.coupling_strength,
            correction_clip=config.correction_clip,
            sigma_data=config.sigma_data,
        )
    raise ValueError(f"Unknown method: {method}")


@th.no_grad()
def generate_batch(net, fixed_images, class_idx, config, return_bridge=False):
    left_fixed_images, right_fixed_images = fixed_images
    eps_fn = EDMEpsWrapper(net, class_idx)
    method = normalize_method(config.method)
    worker = build_worker(method, eps_fn, left_fixed_images, right_fixed_images, config)

    x_t = worker.generate_xT(left_fixed_images.shape[0]).to(config.device)
    if hasattr(worker, "initial_noise_correction"):
        x_t = worker.initial_noise_correction(x_t)

    if method == "proposal_smc":
        sample_gen = smc_sample(
            worker, x_t,
            n_step=config.n_step,
            ts_order=config.ts_order,
            solver=config.solver,
        )
    else:
        sample_gen = dc.sampling(
            x=x_t,
            noise_fn=worker.noise,
            rev_ts=worker.rev_ts(config.n_step, config.ts_order),
            x0_pred_fn=worker.x0_fn,
            s_churn=config.s_churn,
            return_traj=False,
            solver=config.solver,
        )
    sample_full = worker.attach_fixed_ends(sample_gen)
    if hasattr(worker, "reset_fixed_end_noise"):
        worker.reset_fixed_end_noise()
    if return_bridge:
        return sample_full, sample_gen
    return sample_full


@th.no_grad()
def generate_batch_text(net, left_class_idx, right_class_idx, config, return_bridge=False):
    method = normalize_method(config.method)
    worker = build_worker_text(method, net, left_class_idx, right_class_idx, config)

    x_t = worker.generate_xT(config.batch_size).to(config.device)

    sample_gen = dc.sampling(
        x=x_t,
        noise_fn=worker.noise,
        rev_ts=worker.rev_ts(config.n_step, config.ts_order),
        x0_pred_fn=worker.x0_fn,
        s_churn=config.s_churn,
        return_traj=False,
        solver=config.solver,
    )
    sample_full = worker.attach_fixed_ends(sample_gen)
    if return_bridge:
        return sample_full, sample_gen
    return sample_full


def parse_args():
    parser = argparse.ArgumentParser(description="Run one inference-time proposal method.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--network-pkl", default=DEFAULT_MODEL_URL)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", default="proposal_outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--class-name", default="lakeside")
    parser.add_argument("--class-idx", type=int, default=975)
    parser.add_argument("--image-indices", default="0")
    parser.add_argument("--right-image-offset", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=40)
    parser.add_argument("--num-img", type=int, default=3)
    parser.add_argument("--overlap-size", type=int, default=32)
    parser.add_argument("--ts-order", type=int, default=5)
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--solver", choices=("heun", "euler"), default="heun")
    parser.add_argument("--coupling-strength", type=float, default=0.25)
    parser.add_argument("--correction-clip", type=float, default=4.0)
    parser.add_argument("--init-correction-steps", type=int, default=4)
    parser.add_argument("--init-correction-step-size", type=float, default=0.15)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--implicit-scale", type=float, default=0.25)
    parser.add_argument("--all-default-classes", action="store_true")
    # Text condition arguments
    parser.add_argument(
        "--condition-type",
        choices=("image", "text"),
        default="image",
        help="Use fixed image endpoints (image) or class-conditional CFG endpoints (text).",
    )
    parser.add_argument("--left-class-name", default="lakeside",
                        help="Left endpoint class name for text condition mode.")
    parser.add_argument("--left-class-idx", type=int, default=975,
                        help="Left endpoint class index for text condition mode.")
    parser.add_argument("--right-class-name", default="volcano",
                        help="Right endpoint class name for text condition mode.")
    parser.add_argument("--right-class-idx", type=int, default=980,
                        help="Right endpoint class index for text condition mode.")
    # SMC method (proposal_smc) flags
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument("--beta-smc", type=float, default=1.0)
    parser.add_argument("--t-resample-start", type=float, default=0.0)
    parser.add_argument("--t-resample-end", type=float, default=0.8)
    return parser.parse_args()


def chunked(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def bridge_pairs_for_class(image_root, class_name, class_idx, image_indices, right_image_offset):
    pairs = []
    for left_index in image_indices:
        right_index = left_index + right_image_offset
        pairs.append(
            (
                left_index,
                right_index,
                fixed_image_path(image_root, class_name, class_idx, left_index),
                fixed_image_path(image_root, class_name, class_idx, right_index),
            )
        )
    return pairs


def _make_config(args, method):
    method = normalize_method(method)
    return GenerationConfig(
        method=method,
        network_pkl=args.network_pkl,
        image_root=resolve_project_path(args.image_root),
        output_root=resolve_project_path(args.output_root),
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        right_image_offset=args.right_image_offset,
        n_step=args.n_step,
        num_img=args.num_img,
        overlap_size=args.overlap_size,
        ts_order=args.ts_order,
        s_churn=args.s_churn,
        guidance_scale=args.guidance_scale,
        solver=args.solver,
        coupling_strength=args.coupling_strength,
        correction_clip=args.correction_clip,
        init_correction_steps=args.init_correction_steps,
        init_correction_step_size=args.init_correction_step_size,
        sigma_data=args.sigma_data,
        implicit_scale=args.implicit_scale,
        condition_type=args.condition_type,
        left_class_name=args.left_class_name,
        left_class_idx=args.left_class_idx,
        right_class_name=args.right_class_name,
        right_class_idx=args.right_class_idx,
        num_particles=args.num_particles,
        beta_smc=args.beta_smc,
        t_resample_start=args.t_resample_start,
        t_resample_end=args.t_resample_end,
    )


def main():
    args = parse_args()
    th.manual_seed(args.seed)

    image_root = resolve_project_path(args.image_root)
    output_root = resolve_project_path(args.output_root)
    method = normalize_method(args.method)
    config = _make_config(args, method)

    net = load_edm_model(args.network_pkl, device=args.device)
    method_dir = os.path.join(output_root, method)
    write_json(os.path.join(method_dir, "config.json"), config_to_jsonable(config))

    if args.condition_type == "text":
        # Text condition mode: CFG-guided class endpoints, no fixed images needed.
        left_class_idx = args.left_class_idx or text_to_class_idx(args.left_class_name)
        right_class_idx = args.right_class_idx or text_to_class_idx(args.right_class_name)
        image_indices = [int(item.strip()) for item in args.image_indices.split(",") if item.strip()]
        for sample_idx in image_indices:
            th.manual_seed(args.seed + sample_idx)
            sample = generate_batch_text(net, left_class_idx, right_class_idx, config)
            # sample shape: [B, C, H, W]; save each item
            for b, img in enumerate(sample):
                name = f"left_{args.left_class_name}_right_{args.right_class_name}_{sample_idx:06d}"
                out_path = os.path.join(
                    method_dir,
                    f"{args.left_class_name}_to_{args.right_class_name}",
                    f"{name}.png",
                )
                save_tensor_image(img, out_path)
                print(f"[{method}] saved {out_path}")
        return

    # Image condition mode (default).
    classes = DEFAULT_CLASSES if args.all_default_classes else [(args.class_name, args.class_idx)]
    image_indices = [int(item.strip()) for item in args.image_indices.split(",") if item.strip()]
    all_pairs = [
        pair
        for class_name, class_idx in classes
        for pair in bridge_pairs_for_class(
            image_root,
            class_name,
            class_idx,
            image_indices,
            args.right_image_offset,
        )
    ]
    all_paths = [path for _, _, left_path, right_path in all_pairs for path in (left_path, right_path)]
    validate_fixed_image_paths(all_paths, image_root)

    for class_name, class_idx in classes:
        pairs = bridge_pairs_for_class(
            image_root,
            class_name,
            class_idx,
            image_indices,
            args.right_image_offset,
        )
        for batch_pairs in chunked(pairs, args.batch_size):
            left_paths = [item[2] for item in batch_pairs]
            right_paths = [item[3] for item in batch_pairs]
            fixed_images = load_bridge_conditions(
                left_paths,
                right_paths,
                device=args.device,
                overlap_size=args.overlap_size,
            )
            samples = generate_batch(net, fixed_images, class_idx, config)
            for sample, (left_index, right_index, _, _) in zip(samples, batch_pairs):
                name = f"left_{left_index:06d}_right_{right_index:06d}"
                out_path = os.path.join(method_dir, class_name, f"{name}.png")
                save_tensor_image(sample, out_path)
                print(f"[{method}] saved {out_path}")


if __name__ == "__main__":
    main()
