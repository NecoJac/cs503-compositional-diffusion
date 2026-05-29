"""
Evaluate SD1.5 bridge generation with three composition methods.

Usage:
    python -m proposal_methods.evaluate_sd_bridge \\
        --text-pairs "a lakeside landscape:a volcanic landscape" \\
                     "a snowy mountain:a sandy beach" \\
        --output-root proposal_outputs/sd_bridge \\
        --num-img 3 --n-steps 50 --guidance-scale 7.5
"""

import argparse
import csv
import os

import numpy as np
import PIL.Image
import torch as th

from proposal_methods.common import resolve_project_path, write_json
from proposal_methods.methodA_sd_bridge import (
    SD_DEFAULT_MODEL,
    SDBridgeModel,
    generate_sd_bridge,
    generate_sd_bridge_smc,
)


METHODS = ("diffcollage", "naive", "product", "bridge_correction")


# ── Metrics ───────────────────────────────────────────────────────────────────

def seam_mse(pil_image, num_img, overlap_pixels):
    """MSE across visible window-start boundaries in pixel space."""
    import numpy as np
    arr = np.array(pil_image).astype(np.float32) / 255.0  # [H, W, 3]
    W = arr.shape[1]
    patch_px = 512
    stride_px = patch_px - overlap_pixels
    seams = []
    half = max(1, min(16, overlap_pixels // 8))
    for i in range(1, num_img):
        boundary = i * stride_px
        if boundary - half < 0 or boundary + half > W:
            continue
        left_strip  = arr[:, boundary - half: boundary, :]
        right_strip = arr[:, boundary: boundary + half, :]
        seams.append(np.mean((left_strip - right_strip) ** 2))
    return {
        "seam_mse_mean": float(np.mean(seams)) if seams else 0.0,
        "seam_mse_max":  float(np.max(seams))  if seams else 0.0,
    }


# ── Grid ──────────────────────────────────────────────────────────────────────

def make_grid(rows, col_labels, row_labels, path):
    if not rows:
        return
    cell_w, cell_h = rows[0][0].size
    label_h, label_w = 28, 150
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
    for ri, images in enumerate(rows):
        for ci, image in enumerate(images):
            canvas.paste(image, (label_w + ci * cell_w, label_h + ri * cell_h))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    canvas.save(path)


def make_boundary_grid(rows, col_labels, row_labels, path, num_img, overlap_pixels, zoom=4):
    if not rows:
        return

    patch_px = 512
    stride_px = patch_px - overlap_pixels
    half_w = 48
    crops = []
    for images in rows:
        crop_row = []
        for image in images:
            parts = []
            for idx in range(1, num_img):
                boundary = idx * stride_px
                left = max(0, boundary - half_w)
                right = min(image.width, boundary + half_w)
                crop = image.crop((left, 0, right, image.height))
                if zoom > 1:
                    crop = crop.resize((crop.width * zoom, crop.height * zoom), PIL.Image.NEAREST)
                parts.append(crop)
            if parts:
                combined = PIL.Image.new(
                    "RGB",
                    (sum(part.width for part in parts), max(part.height for part in parts)),
                    "white",
                )
                x = 0
                for part in parts:
                    combined.paste(part, (x, 0))
                    x += part.width
                crop_row.append(combined)
        crops.append(crop_row)

    make_grid(crops, col_labels, row_labels, path)


# ── CSV ───────────────────────────────────────────────────────────────────────

def write_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def summarize(metrics, group_fields, metric_names):
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
    p = argparse.ArgumentParser()
    p.add_argument("--text-pairs", nargs="+",
                   default=["a calm lakeside landscape:a volcanic eruption"],
                   metavar="LEFT:RIGHT",
                   help='Pairs like "a lakeside:a volcano"')
    p.add_argument("--middle-prompts", nargs="*", default=None,
                   metavar="PROMPT",
                   help='Optional waypoint prompts between left and right, e.g. "a forest" "a meadow"')
    p.add_argument("--model-id", default=SD_DEFAULT_MODEL)
    p.add_argument("--output-root", default="proposal_outputs/sd_bridge")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--num-img", type=int, default=3,
                   help="Number of overlapping windows (final width = 512 + (num_img-1)*256 px)")
    p.add_argument("--overlap-latents", type=int, default=32,
                   help="Overlap in latent pixels (32 = 256 image pixels)")
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--coupling-strength", type=float, default=0.05)
    p.add_argument(
        "--proposal-couplings",
        default=None,
        help="Comma-separated coupling sweep for bridge_correction, e.g. '0,0.002,0.005,0.01'.",
    )
    p.add_argument("--correction-clip", type=float, default=1.0)
    p.add_argument("--vis-scale", type=int, default=1)
    p.add_argument("--smc-only", action="store_true",
                   help="Replace method_runs with single proposal_smc run.")
    p.add_argument("--smc-K", type=int, default=4)
    p.add_argument("--smc-beta", type=float, default=1.0)
    p.add_argument("--smc-resample-start", type=float, default=0.0)
    p.add_argument("--smc-resample-end", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch multiple prompt pairs per generation call for GPU efficiency.")
    p.add_argument(
        "--demo-degrade-baselines",
        action="store_true",
        help=(
            "Demo-only mode: intentionally run naive/diffcollage with worse "
            "sampling settings. Results are marked as degraded in labels, "
            "config, and CSV; do not use as a fair comparison."
        ),
    )
    p.add_argument("--demo-baseline-n-steps", type=int, default=20)
    p.add_argument("--demo-baseline-guidance-scale", type=float, default=12.0)
    p.add_argument(
        "--demo-diffcollage-guidance-scale",
        type=float,
        default=None,
        help=(
            "Optional demo-only guidance scale for degraded DiffCollage. "
            "Defaults to --demo-baseline-guidance-scale."
        ),
    )
    p.add_argument(
        "--demo-naive-guidance-scale",
        type=float,
        default=None,
        help=(
            "Optional demo-only guidance scale for degraded naive. "
            "Defaults to --demo-baseline-guidance-scale."
        ),
    )
    return p.parse_args()


def parse_pairs(text_pairs):
    pairs = []
    for item in text_pairs:
        parts = item.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Pair must be 'left:right', got: {item!r}")
        pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def method_runs(args):
    if getattr(args, "smc_only", False):
        return [{
            "label": f"smc_K{args.smc_K}_b{args.smc_beta}_c{args.coupling_strength}",
            "base_method": "proposal_smc",
            "coupling": args.coupling_strength,
            "n_steps": args.n_steps,
            "guidance_scale": args.guidance_scale,
            "demo_degraded": False,
        }]
    baseline_steps = args.demo_baseline_n_steps if args.demo_degrade_baselines else args.n_steps
    default_baseline_guidance = (
        args.demo_baseline_guidance_scale
        if args.demo_degrade_baselines
        else args.guidance_scale
    )
    diffcollage_guidance = (
        args.demo_diffcollage_guidance_scale
        if args.demo_degrade_baselines and args.demo_diffcollage_guidance_scale is not None
        else default_baseline_guidance
    )
    naive_guidance = (
        args.demo_naive_guidance_scale
        if args.demo_degrade_baselines and args.demo_naive_guidance_scale is not None
        else default_baseline_guidance
    )
    baseline_suffix = "_DEGRADED" if args.demo_degrade_baselines else ""
    runs = [
        {
            "label": f"diffcollage{baseline_suffix}",
            "base_method": "diffcollage",
            "coupling": args.coupling_strength,
            "n_steps": baseline_steps,
            "guidance_scale": diffcollage_guidance,
            "demo_degraded": args.demo_degrade_baselines,
        },
        {
            "label": f"naive{baseline_suffix}",
            "base_method": "naive",
            "coupling": args.coupling_strength,
            "n_steps": baseline_steps,
            "guidance_scale": naive_guidance,
            "demo_degraded": args.demo_degrade_baselines,
        },
    ]
    if args.proposal_couplings:
        for raw in args.proposal_couplings.split(","):
            coupling = float(raw.strip())
            runs.append(
                {
                    "label": f"bridge_correction_c{coupling:g}",
                    "base_method": "bridge_correction",
                    "coupling": coupling,
                    "n_steps": args.n_steps,
                    "guidance_scale": args.guidance_scale,
                    "demo_degraded": False,
                }
            )
    else:
        runs.append(
            {
                "label": "bridge_correction",
                "base_method": "bridge_correction",
                "coupling": args.coupling_strength,
                "n_steps": args.n_steps,
                "guidance_scale": args.guidance_scale,
                "demo_degraded": False,
            }
        )
    return runs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_root = resolve_project_path(args.output_root)
    pairs = parse_pairs(args.text_pairs)

    overlap_px = args.overlap_latents * 8  # 32 latents × 8 = 256 pixels
    final_w = 512 + (args.num_img - 1) * overlap_px
    print(f"Output size: 512 × {final_w} px  ({args.num_img} windows, {overlap_px}px overlap)")
    print(f"Pairs: {pairs}")
    print(f"Middle conditioning: slerp(left, right, α) per patch")

    model = SDBridgeModel(model_id=args.model_id, device=args.device)
    runs = method_runs(args)
    write_json(
        os.path.join(output_root, "config.json"),
        vars(args)
        | {
            "demo_warning": (
                "DEMO ONLY: naive/diffcollage were intentionally degraded."
                if args.demo_degrade_baselines
                else None
            ),
            "method_runs": runs,
        },
    )

    all_metrics = []
    grid_rows = []
    grid_labels = []

    col_labels = [
        (
            f"{run['label']}\nsteps={run['n_steps']} cfg={run['guidance_scale']}"
            + (" DEMO-DEGRADED" if run["demo_degraded"] else "")
        )
        for run in runs
    ]

    boundary_dir = os.path.join(output_root, "boundary_grids")
    os.makedirs(boundary_dir, exist_ok=True)

    for pair_idx, (left_prompt, right_prompt) in enumerate(pairs):
        pair_label = f"{left_prompt[:20]}…→{right_prompt[:20]}…"
        print(f"\n── {pair_label} ──")

        for rep in range(args.n_repeats):
            row_images = []

            for run in runs:
                label = run["label"]
                method = run["base_method"]
                coupling = run["coupling"]
                print(f"  [{label}] generating...", end="", flush=True)
                if method == "proposal_smc":
                    pil_img, _ = generate_sd_bridge_smc(
                        model,
                        left_prompt=left_prompt,
                        right_prompt=right_prompt,
                        num_img=args.num_img,
                        overlap_latents=args.overlap_latents,
                        n_steps=run["n_steps"],
                        guidance_scale=run["guidance_scale"],
                        K=getattr(args, "smc_K", 4),
                        beta=getattr(args, "smc_beta", 1.0),
                        t_resample_start=getattr(args, "smc_resample_start", 0.0),
                        t_resample_end=getattr(args, "smc_resample_end", 0.8),
                        coupling_strength=coupling,
                        correction_clip=args.correction_clip,
                        seed=args.seed + rep,
                        middle_prompts=args.middle_prompts,
                    )
                else:
                    pil_img, _ = generate_sd_bridge(
                        model,
                        left_prompt=left_prompt,
                        right_prompt=right_prompt,
                        method=method,
                        num_img=args.num_img,
                        overlap_latents=args.overlap_latents,
                        n_steps=run["n_steps"],
                        guidance_scale=run["guidance_scale"],
                        coupling_strength=coupling,
                        correction_clip=args.correction_clip,
                        seed=args.seed + rep,
                        middle_prompts=args.middle_prompts,
                    )

                out_path = os.path.join(
                    output_root, label,
                    f"{left_prompt[:30]}_to_{right_prompt[:30]}",
                    f"sample_{rep:03d}.png",
                ).replace(" ", "_")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                pil_img.save(out_path)

                if args.vis_scale > 1:
                    scaled = pil_img.resize(
                        (pil_img.width * args.vis_scale, pil_img.height * args.vis_scale),
                        PIL.Image.NEAREST,
                    )
                    scaled.save(out_path.replace(".png", f"_x{args.vis_scale}.png"))

                row_images.append(pil_img)

                m = seam_mse(pil_img, args.num_img, overlap_px)
                m.update({
                    "method": label,
                    "base_method": method,
                    "coupling_strength": coupling,
                    "effective_n_steps": run["n_steps"],
                    "effective_guidance_scale": run["guidance_scale"],
                    "demo_degraded": run["demo_degraded"],
                    "left_prompt": left_prompt,
                    "right_prompt": right_prompt,
                    "repeat": rep,
                    "sample_path": out_path,
                })
                all_metrics.append(m)
                print(f" seam={m['seam_mse_mean']:.4f}")

            row_label = f"{left_prompt[:15]}→{right_prompt[:15]} #{rep}"
            grid_rows.append(row_images)
            grid_labels.append(row_label)

            # Save boundary grid for this pair immediately (one file per pair)
            safe_label = f"{pair_idx:02d}_{left_prompt[:20]}_to_{right_prompt[:20]}".replace(" ", "_")
            make_boundary_grid(
                [row_images],
                col_labels,
                [row_label],
                os.path.join(boundary_dir, f"{safe_label}_rep{rep:02d}.png"),
                args.num_img,
                overlap_px,
            )

    # Save overall comparison grid (all pairs in one image)
    grid_path = os.path.join(output_root, "comparison_grid.png")
    make_grid(grid_rows, col_labels, grid_labels, grid_path)
    if args.vis_scale > 1:
        scaled_rows = [
            [img.resize((img.width * args.vis_scale, img.height * args.vis_scale), PIL.Image.NEAREST) for img in row]
            for row in grid_rows
        ]
        make_grid(scaled_rows, col_labels, grid_labels,
                  os.path.join(output_root, f"comparison_grid_x{args.vis_scale}.png"))

    # Save metrics
    fieldnames = [
        "method", "base_method", "coupling_strength",
        "effective_n_steps", "effective_guidance_scale", "demo_degraded",
        "left_prompt", "right_prompt", "repeat",
        "seam_mse_mean", "seam_mse_max", "sample_path",
    ]
    write_rows(os.path.join(output_root, "metrics.csv"), all_metrics, fieldnames)

    metric_names = ["seam_mse_mean", "seam_mse_max"]
    write_rows(
        os.path.join(output_root, "summary_metrics.csv"),
        summarize(all_metrics, ["method", "demo_degraded"], metric_names),
        ["method", "demo_degraded", "num_samples", "seam_mse_mean_mean", "seam_mse_max_mean"],
    )

    print(f"\n[done] grid: {grid_path}")
    print(f"[done] boundary grids: {boundary_dir}/")
    print(f"[done] metrics: {os.path.join(output_root, 'metrics.csv')}")


if __name__ == "__main__":
    main()
