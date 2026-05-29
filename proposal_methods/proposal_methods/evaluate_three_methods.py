import argparse
import csv
import os
import shutil

import numpy as np
import PIL.Image
import torch as th

from diff_collage.w_img import split_wimg

from proposal_methods.common import (
    DEFAULT_CLASSES,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL_URL,
    GenerationConfig,
    fixed_image_path,
    load_bridge_conditions,
    load_edm_model,
    resolve_project_path,
    save_tensor_image,
    save_tensor_image_upscaled,
    tensor_to_pil,
    upscale_image,
    validate_fixed_image_paths,
    write_json,
)
from proposal_methods.generate_method import (
    bridge_pairs_for_class,
    generate_batch,
    generate_batch_text,
)


METHODS = ("diffcollage", "naive", "bridge_correction")

# Default class pairs for text condition evaluation.
DEFAULT_TEXT_PAIRS = [
    ("lakeside", 975, "volcano", 980),
    ("alp", 970, "coral_reef", 973),
]


def bridge_seam_mse(raw_bridge, left_fixed, right_fixed, num_img, overlap_size):
    raw_bridge = raw_bridge.unsqueeze(0)
    left_fixed = left_fixed.unsqueeze(0)
    right_fixed = right_fixed.unsqueeze(0)
    patches = split_wimg(raw_bridge, num_img, rtn_overlap=False)
    patches = patches.view(1, num_img, *patches.shape[1:])

    left_endpoint = th.mean((left_fixed - patches[:, 0, :, :, :overlap_size]) ** 2).item()
    right_endpoint = th.mean((patches[:, -1, :, :, -overlap_size:] - right_fixed) ** 2).item()

    values = []
    for idx in range(num_img - 1):
        left = patches[:, idx, :, :, -overlap_size:]
        right = patches[:, idx + 1, :, :, :overlap_size]
        values.append(th.mean((left - right) ** 2).item())
    all_values = [left_endpoint] + values + [right_endpoint]
    internal_mean = float(np.mean(values)) if values else 0.0
    return {
        "seam_mse_mean": float(np.mean(all_values)),
        "seam_mse_max": float(np.max(all_values)),
        "internal_seam_mse_mean": internal_mean,
        "left_endpoint_mse": float(left_endpoint),
        "right_endpoint_mse": float(right_endpoint),
        "endpoint_mse_mean": float(np.mean([left_endpoint, right_endpoint])),
    }


def visible_boundary_mse(image, num_img, overlap_size, strip_width=4):
    """
    Measure abrupt visible changes around the window-start boundaries.

    The old internal overlap metric compares two views cut from the already
    merged long image, so it is usually zero by construction.  This metric
    instead compares narrow strips on the two sides of visible boundary
    positions in the rendered long image.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)

    height = image.shape[-2]
    patch_width = height
    stride = patch_width - overlap_size
    values = []
    for idx in range(1, num_img):
        boundary = idx * stride
        left_start = max(0, boundary - strip_width)
        right_end = min(image.shape[-1], boundary + strip_width)
        if boundary <= left_start or right_end <= boundary:
            continue
        left = image[:, :, :, left_start:boundary]
        right = image[:, :, :, boundary:right_end]
        width = min(left.shape[-1], right.shape[-1])
        if width == 0:
            continue
        values.append(th.mean((left[:, :, :, -width:] - right[:, :, :, :width]) ** 2).item())

    return {
        "visible_boundary_mse_mean": float(np.mean(values)) if values else 0.0,
        "visible_boundary_mse_max": float(np.max(values)) if values else 0.0,
    }


def text_seam_mse(raw_bridge, num_img, overlap_size):
    """Compute only internal seam MSE; no fixed endpoints for text condition."""
    raw_bridge = raw_bridge.unsqueeze(0)
    patches = split_wimg(raw_bridge, num_img, rtn_overlap=False)
    patches = patches.view(1, num_img, *patches.shape[1:])

    values = []
    for idx in range(num_img - 1):
        left = patches[:, idx, :, :, -overlap_size:]
        right = patches[:, idx + 1, :, :, :overlap_size]
        values.append(th.mean((left - right) ** 2).item())
    internal_mean = float(np.mean(values)) if values else 0.0
    return {
        "internal_seam_mse_mean": internal_mean,
        "seam_mse_mean": internal_mean,
        "seam_mse_max": float(np.max(values)) if values else 0.0,
    }


def make_grid(rows, col_labels, row_labels, path):
    if not rows:
        return

    cell_w, cell_h = rows[0][0].size
    label_h = 28
    label_w = 96
    canvas = PIL.Image.new(
        "RGB",
        (label_w + cell_w * len(col_labels), label_h + cell_h * len(rows)),
        "white",
    )

    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(col_labels):
            draw.text((label_w + col * cell_w + 6, 7), label, fill=(0, 0, 0))
        for row, label in enumerate(row_labels):
            draw.text((6, label_h + row * cell_h + 7), label, fill=(0, 0, 0))
    except Exception:
        pass

    for row_idx, images in enumerate(rows):
        for col_idx, image in enumerate(images):
            canvas.paste(image, (label_w + col_idx * cell_w, label_h + row_idx * cell_h))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual and numeric comparison for three methods.")
    parser.add_argument("--network-pkl", default=DEFAULT_MODEL_URL)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", default="proposal_outputs/evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--image-indices", default="0,1")
    parser.add_argument("--right-image-offset", type=int, default=1)
    parser.add_argument("--grid-max-pairs-per-class", type=int, default=1)
    parser.add_argument("--vis-scale", type=int, default=4)
    parser.add_argument("--classes", default="lakeside:975,volcano:980,alp:970")
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
    # Condition type
    parser.add_argument(
        "--condition-type",
        choices=("image", "text"),
        default="image",
        help="Use fixed image endpoints (image) or class-conditional CFG endpoints (text).",
    )
    # Text-mode arguments
    parser.add_argument(
        "--text-pairs",
        default=None,
        help=(
            "Class pairs for text mode: "
            "'lakeside:975+volcano:980,alp:970+coral_reef:973'. "
            "Defaults to built-in pairs if omitted."
        ),
    )
    parser.add_argument("--num-text-samples", type=int, default=2,
                        help="Number of samples per text class pair.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--class-name", default="lakeside")
    parser.add_argument("--class-idx", type=int, default=975)
    parser.add_argument("--left-class-name", default="lakeside")
    parser.add_argument("--left-class-idx", type=int, default=975)
    parser.add_argument("--right-class-name", default="volcano")
    parser.add_argument("--right-class-idx", type=int, default=980)
    return parser.parse_args()


def parse_classes(value):
    if value == "default":
        return DEFAULT_CLASSES
    classes = []
    for item in value.split(","):
        name, idx = item.split(":")
        classes.append((name.strip(), int(idx)))
    return classes


def parse_text_pairs(value):
    """Parse 'lakeside:975+volcano:980,alp:970+coral_reef:973' into list of 4-tuples."""
    if value is None:
        return DEFAULT_TEXT_PAIRS
    pairs = []
    for pair_str in value.split(","):
        left_str, right_str = pair_str.strip().split("+")
        left_name, left_idx = left_str.strip().split(":")
        right_name, right_idx = right_str.strip().split(":")
        pairs.append((left_name.strip(), int(left_idx), right_name.strip(), int(right_idx)))
    return pairs


def summarize_metrics(metrics, group_fields, metric_names):
    groups = {}
    for row in metrics:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    summaries = []
    for key, rows in sorted(groups.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["num_samples"] = len(rows)
        for name in metric_names:
            summary[f"{name}_mean"] = float(np.mean([row[name] for row in rows]))
        summaries.append(summary)
    return summaries


def write_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_text_evaluation(args, net, output_root):
    text_pairs = parse_text_pairs(args.text_pairs)
    image_indices = list(range(args.num_text_samples))

    rows = []
    row_labels = []
    metrics = []
    grid_sample_paths = []

    for left_name, left_idx, right_name, right_idx in text_pairs:
        pair_label = f"{left_name}->{right_name}"
        for pair_idx, sample_idx in enumerate(image_indices):
            row_images = []
            for method in METHODS:
                th.manual_seed(args.seed + sample_idx)
                config = GenerationConfig(
                    method=method,
                    network_pkl=args.network_pkl,
                    image_root=resolve_project_path(args.image_root),
                    output_root=output_root,
                    device=args.device,
                    seed=args.seed + sample_idx,
                    batch_size=1,
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
                    condition_type="text",
                    left_class_name=left_name,
                    left_class_idx=left_idx,
                    right_class_name=right_name,
                    right_class_idx=right_idx,
                )
                sample, raw_bridge = generate_batch_text(
                    net, left_idx, right_idx, config, return_bridge=True
                )
                sample = sample[0]
                raw_bridge = raw_bridge[0]
                sample_path = os.path.join(
                    output_root,
                    method,
                    f"{left_name}_to_{right_name}",
                    f"sample_{sample_idx:06d}.png",
                )
                save_tensor_image(sample, sample_path)
                if args.vis_scale > 1:
                    sample_x_path = os.path.splitext(sample_path)[0] + f"_x{args.vis_scale}.png"
                    save_tensor_image_upscaled(sample, sample_x_path, args.vis_scale)

                if pair_idx < args.grid_max_pairs_per_class:
                    row_images.append(tensor_to_pil(sample))
                    grid_sample_paths.append((
                        sample_path,
                        os.path.join(output_root, "grid_samples", f"{left_name}_to_{right_name}",
                                     method, os.path.basename(sample_path)),
                    ))

                seam = text_seam_mse(raw_bridge, args.num_img, args.overlap_size)
                seam.update(visible_boundary_mse(sample, args.num_img, args.overlap_size))
                metric_row = {
                    "method": method,
                    "left_class": left_name,
                    "right_class": right_name,
                    "sample_idx": sample_idx,
                    "sample_path": sample_path,
                }
                metric_row.update(seam)
                metrics.append(metric_row)
                print(
                    f"[eval/text] {method} {pair_label} #{sample_idx}: "
                    f"internal_seam={seam['internal_seam_mse_mean']:.6f}"
                )

            if row_images:
                rows.append(row_images)
                row_labels.append(f"{pair_label} #{sample_idx}")

    grid_path = os.path.join(output_root, "comparison_grid.png")
    make_grid(rows, list(METHODS), row_labels, grid_path)
    if args.vis_scale > 1:
        scaled_rows = [[upscale_image(img, args.vis_scale) for img in row] for row in rows]
        make_grid(scaled_rows, list(METHODS), row_labels,
                  os.path.join(output_root, f"comparison_grid_x{args.vis_scale}.png"))

    for src, dst in grid_sample_paths:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    text_metric_fields = [
        "method", "left_class", "right_class", "sample_idx",
        "internal_seam_mse_mean", "seam_mse_mean", "seam_mse_max",
        "visible_boundary_mse_mean", "visible_boundary_mse_max", "sample_path",
    ]
    write_rows(os.path.join(output_root, "metrics.csv"), metrics, text_metric_fields)

    text_metric_names = [
        "internal_seam_mse_mean",
        "seam_mse_mean",
        "seam_mse_max",
        "visible_boundary_mse_mean",
        "visible_boundary_mse_max",
    ]
    write_rows(
        os.path.join(output_root, "summary_metrics.csv"),
        summarize_metrics(metrics, ["method"], text_metric_names),
        ["method", "num_samples"] + [f"{n}_mean" for n in text_metric_names],
    )
    print(f"[eval/text] saved grid: {grid_path}")
    print(f"[eval/text] saved metrics: {os.path.join(output_root, 'metrics.csv')}")


def _run_image_evaluation(args, net, output_root):
    image_root = resolve_project_path(args.image_root)
    image_indices = [int(item.strip()) for item in args.image_indices.split(",") if item.strip()]
    classes = parse_classes(args.classes)
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

    rows = []
    row_labels = []
    metrics = []
    grid_sample_paths = []

    for class_name, class_idx in classes:
        pairs = bridge_pairs_for_class(
            image_root,
            class_name,
            class_idx,
            image_indices,
            args.right_image_offset,
        )
        for pair_idx, (left_index, right_index, left_path, right_path) in enumerate(pairs):
            fixed_images = load_bridge_conditions(
                [left_path],
                [right_path],
                device=args.device,
                overlap_size=args.overlap_size,
            )

            row_images = []
            for method in METHODS:
                th.manual_seed(args.seed + left_index)
                config = GenerationConfig(
                    method=method,
                    network_pkl=args.network_pkl,
                    image_root=image_root,
                    output_root=output_root,
                    device=args.device,
                    seed=args.seed,
                    batch_size=1,
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
                    condition_type="image",
                )
                samples, raw_bridges = generate_batch(
                    net,
                    fixed_images,
                    class_idx,
                    config,
                    return_bridge=True,
                )
                sample = samples[0]
                raw_bridge = raw_bridges[0]
                sample_path = os.path.join(
                    output_root,
                    method,
                    class_name,
                    f"left_{left_index:06d}_right_{right_index:06d}.png",
                )
                save_tensor_image(sample, sample_path)
                sample_x_path = os.path.splitext(sample_path)[0] + f"_x{args.vis_scale}.png"
                save_tensor_image_upscaled(sample, sample_x_path, args.vis_scale)
                if pair_idx < args.grid_max_pairs_per_class:
                    row_images.append(tensor_to_pil(sample))
                    grid_sample_paths.append(
                        (
                            sample_path,
                            os.path.join(
                                output_root,
                                "grid_samples",
                                class_name,
                                method,
                                os.path.basename(sample_path),
                            ),
                            sample_x_path,
                            os.path.join(
                                output_root,
                                "grid_samples_x" + str(args.vis_scale),
                                class_name,
                                method,
                                os.path.basename(sample_x_path),
                            ),
                        )
                    )

                metric_row = {
                    "method": method,
                    "class_name": class_name,
                    "class_idx": class_idx,
                    "left_image_index": left_index,
                    "right_image_index": right_index,
                    "sample_path": sample_path,
                }
                left_fixed, right_fixed = fixed_images
                metric_row.update(
                    bridge_seam_mse(
                        raw_bridge,
                        left_fixed[0],
                        right_fixed[0],
                        args.num_img,
                        args.overlap_size,
                    )
                )
                metric_row.update(visible_boundary_mse(sample, args.num_img, args.overlap_size))
                metrics.append(metric_row)
                print(
                    f"[eval] {method} {class_name}/{left_index:06d}->{right_index:06d}: "
                    f"seam={metric_row['seam_mse_mean']:.6f}, "
                    f"endpoints={metric_row['endpoint_mse_mean']:.6f}, "
                    f"visible_boundary={metric_row['visible_boundary_mse_mean']:.6f}"
                )

            if row_images:
                rows.append(row_images)
                row_labels.append(f"{class_name} {left_index:06d}->{right_index:06d}")

    grid_path = os.path.join(output_root, "comparison_grid.png")
    make_grid(rows, METHODS, row_labels, grid_path)
    grid_x_path = os.path.join(output_root, f"comparison_grid_x{args.vis_scale}.png")
    if args.vis_scale > 1:
        scaled_rows = [
            [upscale_image(image, args.vis_scale) for image in row]
            for row in rows
        ]
        make_grid(scaled_rows, METHODS, row_labels, grid_x_path)
    for src, dst, src_x, dst_x in grid_sample_paths:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        if args.vis_scale > 1:
            os.makedirs(os.path.dirname(dst_x), exist_ok=True)
            shutil.copy2(src_x, dst_x)

    metrics_path = os.path.join(output_root, "metrics.csv")
    metric_fieldnames = [
        "method",
        "class_name",
        "class_idx",
        "left_image_index",
        "right_image_index",
        "seam_mse_mean",
        "seam_mse_max",
        "internal_seam_mse_mean",
        "left_endpoint_mse",
        "right_endpoint_mse",
        "endpoint_mse_mean",
        "visible_boundary_mse_mean",
        "visible_boundary_mse_max",
        "sample_path",
    ]
    write_rows(metrics_path, metrics, metric_fieldnames)

    metric_names = [
        "seam_mse_mean",
        "seam_mse_max",
        "internal_seam_mse_mean",
        "left_endpoint_mse",
        "right_endpoint_mse",
        "endpoint_mse_mean",
        "visible_boundary_mse_mean",
        "visible_boundary_mse_max",
    ]
    summary_fields = ["method", "num_samples"] + [f"{name}_mean" for name in metric_names]
    class_summary_fields = ["class_name", "class_idx", "method", "num_samples"] + [
        f"{name}_mean" for name in metric_names
    ]

    summary_path = os.path.join(output_root, "summary_metrics.csv")
    class_summary_path = os.path.join(output_root, "class_summary_metrics.csv")
    write_rows(
        summary_path,
        summarize_metrics(metrics, ["method"], metric_names),
        summary_fields,
    )
    write_rows(
        class_summary_path,
        summarize_metrics(metrics, ["class_name", "class_idx", "method"], metric_names),
        class_summary_fields,
    )

    write_json(
        os.path.join(output_root, "evaluation_config.json"),
        vars(args) | {"image_root_resolved": image_root, "output_root_resolved": output_root},
    )
    print(f"[eval] saved grid: {grid_path}")
    if args.vis_scale > 1:
        print(f"[eval] saved upscaled grid: {grid_x_path}")
    print(f"[eval] saved grid samples: {os.path.join(output_root, 'grid_samples')}")
    print(f"[eval] saved metrics: {metrics_path}")
    print(f"[eval] saved summary: {summary_path}")
    print(f"[eval] saved class summary: {class_summary_path}")


def main():
    args = parse_args()
    output_root = resolve_project_path(args.output_root)
    net = load_edm_model(args.network_pkl, device=args.device)

    if args.condition_type == "text":
        _run_text_evaluation(args, net, output_root)
    else:
        _run_image_evaluation(args, net, output_root)


if __name__ == "__main__":
    main()
