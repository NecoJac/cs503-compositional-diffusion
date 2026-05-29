# README Proposal: DiffCollage Bridge Experiments

This folder contains an inference-time prototype for the proposal idea:
compose pairwise bridge distributions and compare correction strategies without
training a new diffusion model. The original DiffCollage source under
`diff_collage/` is left unchanged; all new code lives under `proposal_methods/`.

## Scope

The current implementation is a bridge / overlap-composition prototype. It is
not yet the full `cs503_style_proposal_revised.tex` image-editing setup with
content/style constraints, LPIPS, SSIM, CLIP, and FID.

Each sample uses two endpoint images from ImageNet landscape classes:

```text
left image -> right 32px strip
right image -> left 32px strip
```

The sampler generates bridge patches between those two fixed endpoint strips.
For visualization, the fixed endpoint strips replace the generated endpoint
overlaps; they are not concatenated twice.

With the default patch size `64 x 64` and overlap `32`:

```text
visible width = 64 + (num_img - 1) * (64 - 32)
num_img=3 -> 64 x 128
num_img=5 -> 64 x 192
```

The diffusion backbone is the EDM ImageNet 64x64 conditional ADM checkpoint:

```text
https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl
```

## Methods

### `diffcollage`

This is the DiffCollage-style factor-graph baseline. It composes adjacent
pairwise factors and subtracts implicit overlap marginals:

```text
p(x,y,z) approx p(x,y) p(y,z) / p(y)
```

In code, it predicts scores/noise for pairwise patch factors and subtracts the
model prediction on overlapping half-patches.

### `naive`

This is the uncorrected pairwise product baseline:

```text
p_naive(x,y,z) proportional to p(x,y) p(y,z)
```

It does not divide by or subtract the marginal overlap term `p(y)`. This often
creates stronger local pressure but can double-count overlap information.

### `bridge_correction`

This is the formula-following implementation from `proposal_formula.html`:

```text
s_xyz = s_xy oplus s_yz - s_y_implicit + Delta s
```

The implementation no longer starts from the DiffCollage worker. It directly
builds the pairwise bridge factors `P(x,y)` and `P(y,z)`, then applies the two
proposal terms:

- `s_y_implicit`: the symmetric implicit marginal estimator
  `0.5 * (s_y^{xy} + s_y^{yz})`, using the current noisy bridge state as the
  practical one-sample conditional approximation suggested in the HTML.
- `Delta s`: the `grad log R` correction. Since the HTML does not give a
  closed-form `R` for the pretrained EDM patch model, the code uses the
  HTML's Tweedie idea: estimate `x0` from adjacent pairwise factors and push
  their overlap predictions toward consistency.
- optional initial-noise correction, approximating the proposal's `p_T^*`
  reweighting.

Important caveat: this is now aligned with the HTML formula skeleton, but the
true `R(x,y,z)` is still approximated because it is not directly available from
the frozen EDM checkpoint.

### Remaining Differences From The HTML Formula

The current `bridge_correction` follows the executable score-composition skeleton
from `proposal_formula.html`, but it is still not a mathematically exact
implementation of every object in the HTML. The main remaining gaps are:

- `Delta s = grad log R`

  HTML status: the HTML gives the theoretical definition
  `R = p(x,z | y) / (p(x | y) p(z | y))`, but it does not give a directly
  computable estimator for `p(x,z | y)`, `p(x | y)`, or `p(z | y)` from the
  frozen EDM checkpoint.

  Current resolution: the code uses the HTML's Tweedie direction. It estimates
  `x0` from adjacent pairwise denoising predictions and treats disagreement in
  their overlap `x0` estimates as a proxy energy for `grad log R`.

- `s_y_implicit`

  HTML status: the HTML defines the implicit marginal as a conditional
  expectation, for example `E_{x|y}[s_y^{xy}]`, and mentions practical
  approximations such as short-chain Langevin or using the current state.

  Current resolution: the code uses the cheapest current-state approximation:
  `s_y_implicit = 0.5 * (s_y^{xy} + s_y^{yz})` evaluated at the current noisy
  bridge state. This keeps inference fast and training-free, but it is a
  one-sample proxy rather than a Monte Carlo expectation.

- `p_T^*` initial-noise reweighting

  HTML status: the HTML describes the corrected initial distribution
  `p_T^*`, and later discussion points toward importance weighting / SMC-style
  handling, but does not provide a simple closed-form sampler for this codebase.

  Current resolution: the code provides optional deterministic initial-overlap
  correction through `--init-correction-steps`. This is a lightweight proxy for
  reducing obvious endpoint/overlap mismatch before denoising starts, not a
  full importance-weighted or SMC implementation.

- Score space versus EDM epsilon space

  HTML status: the HTML writes formulas in score notation. The EDM codebase
  exposes denoising / epsilon-style predictions through the existing sampler.

  Current resolution: the implementation applies the same score arithmetic in
  epsilon/noise-prediction space. At fixed sigma this is linearly related to
  score space, so it is a reasonable prototype choice. If a stronger learned
  `grad log R` estimator is added, the sign and sigma scaling should be
  recalibrated carefully.

- From `x-y-z` to a bridge chain

  HTML status: the central derivation is written for three variables
  `x, y, z`.

  Current resolution: the code applies the same local rule to a chain of
  overlapping 64x64 patches, plus left and right fixed endpoint strips. Each
  adjacent overlap is treated as a local `y`. This is the bridge-composition
  prototype, not yet the full image-editing setting.

- Conditioning type

  HTML status: the HTML/proposal direction also motivates broader constraints,
  including text/style/content-style editing in the revised proposal.

  Current resolution: this implementation still uses image-strip endpoint
  conditioning. Text conditioning, style conditioning, and content-preservation
  losses are left for the next stage.

## Key Files

- `proposal_methods/workers.py`: method implementations.
- `proposal_methods/generate_method.py`: run one method.
- `proposal_methods/evaluate_three_methods.py`: run all three methods and write
  grids plus CSV metrics.
- `proposal_methods/README_RESULTS.md`: completed-run metrics and analysis.
- `scripts/download_imagenet_landscapes.py`: prepare endpoint images.
- `proposal_methods/sbatch/*.run`: Slurm launch scripts.
- `proposal_outputs/evaluation/`: completed short-bridge `num_img=3` outputs.
- `proposal_outputs/evaluation_tuned/`: completed long-bridge `num_img=5`
  outputs.

## Environment

All paths below use `<username>` as a placeholder. On Izar, replace it with
your cluster username, or let the scripts infer it from `$USER`.

Recommended path variables:

```bash
export PROJECT_ROOT=/path/to/cs503-compositional-diffusion
export DIFFUSION_SCRATCH_ROOT=/scratch/izar/<username>/diffusion
```

If your checkout is in a different location, set `PROJECT_ROOT` to that path
before launching any sbatch script.

Create and activate the conda environment, then install dependencies:

```bash
conda create -n cbg_diffusion python=3.10 -y
conda activate cbg_diffusion

cd "$PROJECT_ROOT"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install --user -r proposal_methods/requirements.txt
```

`pyspng` is intentionally not required. It often fails to build on the cluster
because the package cannot find `spng/spng.h`, and these scripts use PIL/imageio
paths instead.

Check CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Data

The endpoint data lives inside the project by default:

```text
data/imagenet_landscapes/
```

Expected layout:

```text
data/imagenet_landscapes/class_975_lakeside/000000.jpg
data/imagenet_landscapes/class_980_volcano/000000.jpg
data/imagenet_landscapes/class_970_alp/000000.jpg
```

The preparation script stores the small selected image subset in the repository
but uses scratch for HuggingFace cache and temporary files:

```text
/scratch/izar/<username>/diffusion/hf_cache
/scratch/izar/<username>/diffusion/tmp
```

Prepare data on the login node or with Slurm:

```bash
cd "$PROJECT_ROOT"
python scripts/download_imagenet_landscapes.py --max-per-class 10
```

or:

```bash
CONDA_ENV=cbg_diffusion MAX_PER_CLASS=10 \
sbatch proposal_methods/sbatch/prepare_imagenet_landscapes.run
```

`MAX_PER_CLASS=10` is enough for bridge pairs `0->1` through `8->9`.

## Running On Izar

Before submitting jobs, either submit from the project root or set
`PROJECT_ROOT` explicitly:

```bash
export PROJECT_ROOT=/path/to/cs503-compositional-diffusion
export DIFFUSION_SCRATCH_ROOT=/scratch/izar/<username>/diffusion
cd "$PROJECT_ROOT"
```

The sbatch scripts default to:

```bash
PROJECT_ROOT=/path/to/cs503-compositional-diffusion
DIFFUSION_SCRATCH_ROOT=/scratch/izar/$USER/diffusion
```

If your username is not reflected correctly in `$USER`, or your repository is
not under `/home/$USER`, override these variables at submission time:

```bash
PROJECT_ROOT=/path/to/cs503-compositional-diffusion DIFFUSION_SCRATCH_ROOT=/scratch/izar/<username>/diffusion \
sbatch proposal_methods/sbatch/run_evaluation.run
```

All provided jobs request one GPU:

```text
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

The cluster may reject non-GPU jobs with `QOSMinGRES`, so the data preparation
job also requests one GPU even though the work is mostly CPU/network I/O.

### Full Tuned Evaluation

The current `run_evaluation.run` uses the shorter tuned `num_img=3` setting.
This makes patch artifacts easier to inspect than the longer `num_img=5`
bridge:

```bash
cd "$PROJECT_ROOT"
sbatch proposal_methods/sbatch/run_evaluation.run
```

Equivalent command:

```bash
python -m proposal_methods.evaluate_three_methods \
  --image-indices 0,1,2,3,4,5,6,7,8 \
  --classes lakeside:975,volcano:980,alp:970 \
  --image-root data/imagenet_landscapes \
  --output-root proposal_outputs/evaluation_n3_tuned \
  --grid-max-pairs-per-class 3 \
  --num-img 3 \
  --n-step 80 \
  --solver heun \
  --guidance-scale 0.7 \
  --coupling-strength 0.03 \
  --init-correction-steps 0
```

This produces:

```text
3 classes x 9 bridge pairs x 3 methods = 81 samples
```

With 64x64 patches and 32px overlap, `num_img=3` produces `64 x 128` images
instead of the `64 x 192` images from `num_img=5`. This shorter setting is a
useful sanity check: if patch boundaries are much less visible at `n=3`, then
the artifacts are likely amplified by the longer bridge chain.

For a longer bridge comparison, change `--num-img 3` back to `--num-img 5` and
use a separate output directory such as `proposal_outputs/evaluation_tuned`.

### Smoke Test

For a quick run, use one class and one pair:

```bash
python -m proposal_methods.evaluate_three_methods \
  --image-indices 0 \
  --classes lakeside:975 \
  --image-root data/imagenet_landscapes \
  --output-root proposal_outputs/eval_smoke \
  --grid-max-pairs-per-class 1 \
  --num-img 3 \
  --n-step 20
```

### Single Method Jobs

These run one method on the default class list and save to
`proposal_outputs/single_method/`:

```bash
sbatch proposal_methods/sbatch/run_diffcollage.run
sbatch proposal_methods/sbatch/run_naive.run
sbatch proposal_methods/sbatch/run_bridge_correction.run
```

## Outputs

Evaluation output folders contain:

- `comparison_grid.png`: compact side-by-side comparison.
- `comparison_grid_x4.png`: enlarged grid for visual inspection.
- `grid_samples/`: standalone copies of images shown in the grid.
- `grid_samples_x4/`: enlarged standalone grid samples.
- `metrics.csv`: one row per generated image.
- `summary_metrics.csv`: method-level averages.
- `class_summary_metrics.csv`: per-class, per-method averages.
- `evaluation_config.json`: exact run configuration.

The most useful visual file is usually:

```text
proposal_outputs/evaluation_tuned/comparison_grid_x4.png
```

## Metrics

Current metrics are overlap-debug proxies, not final image-quality metrics.
They are computed on the raw generated bridge before fixed endpoint strips are
pasted into the saved visualization.

- `left_endpoint_mse`: mismatch between left fixed strip and bridge start.
- `right_endpoint_mse`: mismatch between bridge end and right fixed strip.
- `endpoint_mse_mean`: average of the two endpoint mismatches.
- `seam_mse_mean`: average over endpoint and internal seam mismatches.
- `seam_mse_max`: worst seam mismatch.
- `internal_seam_mse_mean`: currently `0.0` because the saved bridge is merged
  before this metric is computed; it should not be used as a quality signal in
  the current implementation.

Interpretation:

- Lower endpoint/seam MSE means the bridge aligns more tightly with fixed
  endpoint strips.
- Lower MSE does not necessarily mean better image quality. A method can
  over-optimize seams and produce blurry or gray transitions.

## Results

Detailed completed-run metrics and analysis are recorded in:

```text
proposal_methods/README_RESULTS.md
```

Current completed outputs:

- `proposal_outputs/evaluation/`: short bridge, `num_img=3`, native image size
  `64 x 128`.
- `proposal_outputs/evaluation_tuned/`: long bridge, `num_img=5`, native image
  size `64 x 192`.

Overall summary:

| Run | Method | Samples | Seam Mean | Seam Max | Endpoint Mean |
|---|---|---:|---:|---:|---:|
| `num_img=3` | `diffcollage` | 27 | 0.123393 | 0.322802 | 0.246786 |
| `num_img=3` | `naive` | 27 | 0.1296 | 0.3329 | 0.2592 |
| `num_img=3` | `bridge_correction` | 27 | 0.1520 | 0.3988 | 0.3041 |
| `num_img=5` | `diffcollage` | 27 | 0.0807 | 0.3141 | 0.2420 |
| `num_img=5` | `naive` | 27 | 0.0876 | 0.3369 | 0.2628 |
| `num_img=5` | `bridge_correction` | 27 | 0.1036 | 0.4056 | 0.3107 |

Takeaway: after switching `bridge_correction` to the HTML-style direct
composition skeleton, the current proxy implementation does not outperform
DiffCollage on the seam/endpoint metrics. The likely reason is that
`s_y_implicit` and `Delta s = grad log R` are still approximations rather than
the exact theoretical quantities.

Visual takeaway: generated images can look blocky because this is a local
patch-composition prototype using a `64 x 64` ImageNet diffusion model. The
sampler composes overlapping local windows but has no global long-image prior.

## Recommended Next Experiments

### 1. Replace Image Endpoints With Text Conditions

The current endpoint images can be very different even within the same ImageNet
class. This makes the bridge task hard for reasons unrelated to the proposal:
the model may be asked to connect two visually incompatible crops.

The next experiment should use text as the two endpoint conditions, for example:

```text
left condition:  "a calm lakeside landscape"
right condition: "a snowy mountain landscape"
```

This may be less visually constrained than hard image strips and could make the
composition behavior easier to interpret. The risk is instability: text
conditions are weaker and may not anchor the endpoints as precisely as image
overlaps. A useful experiment is therefore to compare:

- image endpoint conditions;
- text endpoint conditions;
- mixed text + image endpoint conditions.

### 2. Add Better Metrics

The current seam MSE mainly measures endpoint consistency. It does not measure
whether the image looks good or whether the intended condition is satisfied.

Add metrics that capture both quality and condition alignment:

- perceptual quality: FID or a lightweight no-reference quality proxy;
- text alignment: CLIP similarity between generated image and endpoint prompts;
- visual coherence: CLIP image-image similarity between neighboring bridge
  regions, or LPIPS between adjacent generated regions;
- diversity: variance across multiple seeds for the same condition pair;
- seam diagnostics: keep endpoint MSE, but report it separately from quality.

For current bridge experiments, the most important change is to stop treating a
low seam MSE as the main success criterion. A method can win seam MSE while
looking worse.

### 3. Test Different CFG / Guidance Variants

The framework should allow swapping in different CFG-style guidance strategies.
This is useful because the proposal is about inference-time composition, and CFG
is one of the main inference-time control mechanisms.

Candidate variants:

- vanilla CFG / current class-conditional guidance;
- weaker or stronger `--guidance-scale`;
- text CFG if moving to text-conditioned diffusion;
- control-based CFG variants from TTC-style code if they can be adapted to the
  DiffCollage sampling interface;
- correction scheduling, such as low-sigma fade-out so correction does not
  destroy fine details near the end of sampling.

The immediate parameter ablation to run is:

```bash
--num-img 5 \
--n-step 80 \
--guidance-scale 0.8 \
--coupling-strength 0.05 \
--init-correction-steps 1 \
--init-correction-step-size 0.05
```

Then compare against the tuned run:

```bash
--guidance-scale 0.7 \
--coupling-strength 0.03 \
--init-correction-steps 0
```

### 4. Direct Composition Without Using DiffCollage As The Base

The final methodological target is not just to tune DiffCollage. The goal is to
compose the pairwise factors directly, rather than starting from the original
DiffCollage worker and then adding a correction.

This part is now implemented in `bridge_correction`. The method directly follows
the HTML score skeleton:

```text
s_xyz = s_xy oplus s_yz - s_y_implicit + Delta s
```

So the DiffCollage worker is no longer the starting point for the final method.
In that engineering sense, the method has moved from:

```text
s = s_diffcollage + Delta s
```

to direct pairwise composition:

```text
s = s_xy oplus s_yz - s_y_implicit + Delta s
```

What remains as future work is not removing DiffCollage as the base worker;
that has already been done. The remaining work is to improve the estimators for
the two quantities that the HTML defines theoretically:

- `s_y_implicit`: currently a current-state one-sample approximation to the
  conditional expectation.
- `Delta s = grad log R`: currently a Tweedie `x0` overlap-consistency proxy
  rather than a true estimator of the coupling factor.

The ideal correction term would correspond to:

```text
p_corr(x,y,z) proportional to p(x,y) p(y,z) exp(-E_corr(x,y,z))
```

or in score form:

```text
s_corr = s_xy oplus s_yz - s_y_implicit + grad log R
```

This is the cleaner version of the proposal: directly compose `P(x,y)` and
`P(y,z)`, then use the HTML correction term to compensate for bias, without
calling the original DiffCollage worker as a base method.

## Relation To The Revised Proposal

The revised proposal describes a broader image-editing project:

- source content preservation;
- target style or text alignment;
- optional auxiliary constraints;
- metrics such as LPIPS, SSIM, CLIP similarity, and FID.

The current code is a smaller prototype. It tests the bridge-variable
composition idea in a controlled DiffCollage-style overlap setting. This is
useful because it isolates the mathematical question:

```text
How do p(x,y) and p(y,z) behave when composed at inference time?
```

However, it does not yet fully instantiate the revised proposal. The path from
this prototype to the revised proposal is:

1. Replace hard image-overlap endpoints with text/style/content conditions.
2. Add metrics for condition satisfaction and perceptual quality.
3. Make CFG/guidance strategies swappable inside the same composition
   framework.
4. Improve the current direct `s_xy oplus s_yz - s_y_implicit + grad log R`
   implementation with better `R` estimation or learned/proxy correction.
5. Evaluate on an image-editing benchmark rather than only ImageNet landscape
   bridges.

In short: the current implementation is a proof-of-concept for the bridge
composition mechanism; the revised proposal is the target application and
evaluation setting.

## Known Limitations

- This is a prototype for bridge composition, not the final image-editing task
  described in `cs503_style_proposal_revised.tex`.
- The EDM ImageNet 64x64 model limits native image quality and resolution.
- Current metrics mainly measure endpoint consistency, not perceptual quality.
- `bridge_correction` can overfit seam consistency if correction is too strong.
