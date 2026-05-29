"""
Evaluate text-conditioned bridge generation.

Usage example:
    python -m proposal_methods.evaluate_text_bridge \\
        --text-pairs "lakeside:volcano" "lakeside:alp" "volcano:alp" \\
        --output-root proposal_outputs/text_bridge \\
        --num-img 3 --n-step 40 --guidance-scale 1.5

Available landscape class names:
    alp, cliff, coral_reef, geyser, lakeside,
    promontory, sandbar, seashore, valley, volcano
"""

import argparse
import csv
import os
import shutil

import numpy as np
import PIL.Image
import torch as th

from diff_collage.w_img import split_wimg

from proposal_methods.common import (
    DEFAULT_MODEL_URL,
    load_edm_model,
    resolve_project_path,
    save_tensor_image,
    save_tensor_image_upscaled,
    tensor_to_pil,
    upscale_image,
    write_json,
)
from proposal_methods.text_workers import (
    TEXT_TO_CLASS,
    DiffCollageBridgeWorkerText,
    NaiveBridgeWorkerText,
    ProposalFinalBridgeWorkerText,
    text_to_class_idx,
)

import diff_collage as dc


METHODS = ("diffcollage", "naive", "bridge_correction")
PATCH_SHAPE = (3, 64, 64)


# ── worker factory ────────────────────────────────────────────────────────────

def build_worker(method, net, left_class_idx, right_class_idx, config):
    if method == "proposal_final":
        method = "bridge_correction"
    kwargs = dict(
        shape=PATCH_SHAPE,
        net=net,
        num_img=config["num_img"],
        left_class_idx=left_class_idx,
        right_class_idx=right_class_idx,
        overlap_size=config["overlap_size"],
        guidance_scale=config["guidance_scale"],
    )
    if method == "diffcollage":
        return DiffCollageBridgeWorkerText(**kwargs)
    if method == "naive":
        return NaiveBridgeWorkerText(**kwargs)
    if method == "bridge_correction":
        return ProposalFinalBridgeWorkerText(
            **kwargs,
            coupling_strength=config["coupling_strength"],
            correction_clip=config["correction_clip"],
        )
    raise ValueError(f"Unknown method: {method}")


# ── generation ────────────────────────────────────────────────────────────────

@th.no_grad()
def generate_one(net, method, left_class_idx, right_class_idx, config):
    """Generate one bridge sample; returns (full_image_tensor, raw_bridge_tensor)."""
    worker = build_worker(method, net, left_class_idx, right_class_idx, config)
    x_t = worker.generate_xT(1).to(config["device"])

    sample_gen = dc.sampling(
        x=x_t,
        noise_fn=worker.noise,
        rev_ts=worker.rev_ts(config["n_step"], config["ts_order"]),
        x0_pred_fn=worker.x0_fn,
        s_churn=config["s_churn"],
        return_traj=False,
        solver=config["solver"],
    )
    full = worker.attach_fixed_ends(sample_gen)
    return full[0], sample_gen[0]


# ── metrics ───────────────────────────────────────────────────────────────────

def internal_seam_mse(raw_bridge, num_img, overlap_size):
    """MSE between adjacent patch overlaps (measures transition smoothness)."""
    raw = raw_bridge.unsqueeze(0)
    patches = split_wimg(raw, num_img, rtn_overlap=False)
    patches = patches.view(1, num_img, *patches.shape[1:])

    values = []
    for idx in range(num_img - 1):
        left = patches[:, idx, :, :, -overlap_size:]
        right = patches[:, idx + 1, :, :, :overlap_size]
        values.append(th.mean((left - right) ** 2).item())

    return {
        "internal_seam_mse_mean": float(np.mean(values)) if values else 0.0,
        "internal_seam_mse_max": float(np.max(values)) if values else 0.0,
    }


# ── grid helpers ──────────────────────────────────────────────────────────────

def make_grid(rows, col_labels, row_labels, path):
    if not rows:
        return
    cell_w, cell_h = rows[0][0].size
    label_h, label_w = 28, 120
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


# ── CSV helpers ───────────────────────────────────────────────────────────────

def write_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(metrics, group_fields, metric_names):
    groups = {}
    for row in metrics:
        key = tuple(row[f] for f in group_fields)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        s = {f: v for f, v in zip(group_fields, key)}
        s["num_samples"] = len(rows)
        for name in metric_names:
            s[f"{name}_mean"] = float(np.mean([r[name] for r in rows]))
        summaries.append(s)
    return summaries


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate three bridge methods with text (class) conditioning."
    )
    parser.add_argument(
        "--text-pairs",
        nargs="+",
        default=["lakeside:volcano"],
        metavar="LEFT:RIGHT",
        help='Text pairs in "left_class:right_class" format, e.g. "lakeside:volcano"',
    )
    parser.add_argument("--network-pkl", default=DEFAULT_MODEL_URL)
    parser.add_argument("--output-root", default="proposal_outputs/text_bridge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Number of samples per (pair, method) to generate")
    parser.add_argument("--vis-scale", type=int, default=4)
    parser.add_argument("--n-step", type=int, default=40)
    parser.add_argument("--num-img", type=int, default=3)
    parser.add_argument("--overlap-size", type=int, default=32)
    parser.add_argument("--ts-order", type=int, default=5)
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--solver", choices=("heun", "euler"), default="heun")
    parser.add_argument("--coupling-strength", type=float, default=0.25)
    parser.add_argument("--correction-clip", type=float, default=4.0)
    return parser.parse_args()


def parse_text_pairs(text_pairs):
    pairs = []
    for item in text_pairs:
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Text pair must be 'left:right', got: {item!r}"
            )
        left_text, right_text = parts[0].strip(), parts[1].strip()
        left_idx = text_to_class_idx(left_text)
        right_idx = text_to_class_idx(right_text)
        pairs.append((left_text, left_idx, right_text, right_idx))
    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_root = resolve_project_path(args.output_root)

    config = {
        "num_img": args.num_img,
        "overlap_size": args.overlap_size,
        "guidance_scale": args.guidance_scale,
        "n_step": args.n_step,
        "ts_order": args.ts_order,
        "s_churn": args.s_churn,
        "solver": args.solver,
        "coupling_strength": args.coupling_strength,
        "correction_clip": args.correction_clip,
        "device": args.device,
        "seed": args.seed,
    }

    pairs = parse_text_pairs(args.text_pairs)
    print(f"Text pairs: {[(l, r) for l, _, r, _ in pairs]}")
    print(f"Available classes: {sorted(TEXT_TO_CLASS.keys())}")

    net = load_edm_model(args.network_pkl, device=args.device)
    write_json(os.path.join(output_root, "config.json"), vars(args) | {"pairs_resolved": [
        {"left": l, "left_idx": li, "right": r, "right_idx": ri}
        for l, li, r, ri in pairs
    ]})

    all_metrics = []
    grid_rows = []
    grid_row_labels = []

    for left_text, left_idx, right_text, right_idx in pairs:
        pair_label = f"{left_text}→{right_text}"
        print(f"\n── {pair_label} (class {left_idx}→{right_idx}) ──")

        for repeat_idx in range(args.n_repeats):
            th.manual_seed(args.seed + repeat_idx)
            row_images = []

            for method in METHODS:
                sample, raw_bridge = generate_one(
                    net, method, left_idx, right_idx, config
                )

                sample_path = os.path.join(
                    output_root, method,
                    f"{left_text}_to_{right_text}",
                    f"sample_{repeat_idx:03d}.png",
                )
                save_tensor_image(sample, sample_path)
                save_tensor_image_upscaled(sample, sample_path.replace(".png", f"_x{args.vis_scale}.png"), args.vis_scale)

                row_images.append(tensor_to_pil(sample))

                m = internal_seam_mse(raw_bridge, args.num_img, args.overlap_size)
                m.update({
                    "method": method,
                    "left_text": left_text,
                    "left_class_idx": left_idx,
                    "right_text": right_text,
                    "right_class_idx": right_idx,
                    "repeat_idx": repeat_idx,
                    "sample_path": sample_path,
                })
                all_metrics.append(m)

                print(
                    f"  [{method}] seam_mean={m['internal_seam_mse_mean']:.4f} "
                    f"seam_max={m['internal_seam_mse_max']:.4f}"
                )

            grid_rows.append(row_images)
            grid_row_labels.append(f"{pair_label} #{repeat_idx}")

    # Grids
    grid_path = os.path.join(output_root, "comparison_grid.png")
    make_grid(grid_rows, list(METHODS), grid_row_labels, grid_path)
    grid_x_path = os.path.join(output_root, f"comparison_grid_x{args.vis_scale}.png")
    if args.vis_scale > 1:
        scaled_rows = [
            [upscale_image(img, args.vis_scale) for img in row]
            for row in grid_rows
        ]
        make_grid(scaled_rows, list(METHODS), grid_row_labels, grid_x_path)

    # Metrics CSV
    metric_names = ["internal_seam_mse_mean", "internal_seam_mse_max"]
    fieldnames = [
        "method", "left_text", "left_class_idx",
        "right_text", "right_class_idx", "repeat_idx",
        "internal_seam_mse_mean", "internal_seam_mse_max", "sample_path",
    ]
    write_rows(os.path.join(output_root, "metrics.csv"), all_metrics, fieldnames)

    summary_fields = ["method", "num_samples"] + [f"{n}_mean" for n in metric_names]
    write_rows(
        os.path.join(output_root, "summary_metrics.csv"),
        summarize_metrics(all_metrics, ["method"], metric_names),
        summary_fields,
    )

    pair_summary_fields = (
        ["left_text", "right_text", "method", "num_samples"]
        + [f"{n}_mean" for n in metric_names]
    )
    write_rows(
        os.path.join(output_root, "pair_summary_metrics.csv"),
        summarize_metrics(all_metrics, ["left_text", "right_text", "method"], metric_names),
        pair_summary_fields,
    )

    print(f"\n[done] grid: {grid_path}")
    if args.vis_scale > 1:
        print(f"[done] upscaled grid: {grid_x_path}")
    print(f"[done] metrics: {os.path.join(output_root, 'metrics.csv')}")


if __name__ == "__main__":
    main()
