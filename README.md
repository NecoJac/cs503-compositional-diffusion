# Test-Time Compositional Control for Diffusion Models

This repository contains the code for our EPFL 2026 Spring CS-503 Visual Intelligence project:

**Test-Time Compositional Control for Diffusion Models**  
Bridge-Correction and SMC / FKC-PoE for Panoramic Text-Prompt Composition

Training-free composition of pairwise diffusion bridge distributions at inference time, with a focus on **Sequential Monte Carlo (SMC / FKC-PoE)** sampling on the Stable Diffusion 1.5 backend. This branch ports the Feynman-Kac-Corrector Product-of-Experts algorithm (Skreta et al., ICML 2025) to SD 1.5 text-conditioned panoramic bridge generation and benchmarks it against DiffCollage and Naive PoE baselines.

---

## Links

- Project code: [NecoJac/cs503-compositional-diffusion](https://github.com/NecoJac/cs503-compositional-diffusion)
- Website source code: [NecoJac/cs503-compositional-diffusion-webpage](https://github.com/NecoJac/cs503-compositional-diffusion-webpage)
- Website: [cs503-compositional-diffusion-webpage](https://necojac.github.io/cs503-compositional-diffusion-webpage/)
- Course: [EPFL CS-503 Visual Intelligence, Spring 2026](https://edu.epfl.ch/coursebook/en/visual-intelligence-CS-503)

## Authors and Contributions

- [Hantao Zhang](https://github.com/kanydao)  &mdash; developed the bridge-correction theoretical
              formalism (DiffCollage as $R\equiv 1$ projection, identification of $\nabla\log R$),
              adapted the FKC-PoE sampler to the SD 1.5 bridge composition setting, and led the
              50+ configuration sweep design.
- [Yunyi Chen](https://github.com/C-Yunyi)  &mdash; implemented the bridge-worker framework on top of
              Stable Diffusion 1.5; managed window slicing, slerp prompt interpolation, and the
              per-method composition rules.
- [Shengze Jiang](https://github.com/NecoJac) &mdash; co-implemented the bridge-worker framework;
              integrated the Tweedie corrector and DDIM scheduler; executed the 50+ configuration
              sweep and the three-prompt panorama runs.
- [Xinran Wang](https://github.com/AmbitiousOcean)  &mdash; designed the seam-MSE / seam-max evaluation
              protocol, curated the prompt set, and produced the qualitative comparison grids.
- All four authors discussed scope decisions and jointly wrote this report, the slides,
              and the webpage.

## Table of Contents

- [Overview](#overview)
- [Methods](#methods)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Quickstart](#quickstart)
- [Output Structure](#output-structure)
- [Metrics](#metrics)
- [Results](#results)
- [Conclusions](#conclusions)
- [Known Limitations](#known-limitations)
- [External Code](#external-code)

---

## Overview

Given two text endpoint prompts, the system generates a smooth panoramic bridge between them by composing overlapping Stable Diffusion 1.5 windows:

```
"left prompt" → [window 0] → [window 1] → ... → "right prompt"
```

Each adjacent pair of windows shares a 256 px (32 latent) overlap. Methods differ in how they handle this overlap during score composition. All methods are **training-free**.

**Backend:** Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`), 512×512 px per window.

| `num_img` | Output width |
|-----------|-------------|
| 3 | 512 + 2×256 = 1024 px |
| 5 | 512 + 4×256 = 1536 px |

---

## Methods

Method names used in the webpage and report map to the code keys below:

| Display name | Code key |
|--------------|----------|
| DiffCollage | `diffcollage` |
| Naive PoE | `naive` |
| Bridge Correction | `bridge_correction` |
| SMC / FKC-PoE | `proposal_smc` |

### DiffCollage (`diffcollage`) — Factor-Graph Baseline

Composes adjacent pairwise factors and subtracts the implicit overlap marginal:

```
p(x,y,z) ≈ p(x,y) · p(y,z) / p(y)
```

### Naive PoE (`naive`) — Uncorrected Product

Multiplies adjacent pairwise factors without marginal correction. Double-counts the overlap but is fast and simple.

### Bridge Correction (`bridge_correction`) — Direct Pairwise Composition

Directly implements the score composition formula:

```
s_xyz = s_xy ⊕ s_yz − s_y_implicit + Δs
```

- **`s_y_implicit`**: symmetric implicit marginal `0.5 · (s_y^{xy} + s_y^{yz})`.
- **`Δs`**: Tweedie x0 overlap-consistency correction (`grad log R` proxy).

### SMC / FKC-PoE (`proposal_smc`) — K-particle SMC

Faithful port of the Feynman-Kac-Corrector PoE algorithm (Skreta+ 2025, ICML Prop. 3.3).

Each prompt-pair is sampled through `K` parallel particles. Per DDIM step:
- Each particle runs naïve PoE merge of adjacent window ε-predictions.
- Log-weight of particle k is incremented by:

```
dlog_w_k = β · ⟨ε_right_overlap, ε_left_next_overlap⟩ / overlap_dims
```

  summed over `num_img − 1` adjacent window pairs.
- Systematic resampling inside `[0, t_resample_end × T]`.
- Final image: SNIS (argmax-log-weight particle).

Best variance result: seam MSE std **0.0128** (`K=8, β=1.0, overlap_latents=48, seed=17`).

---

## Project Structure

```
cs503-compositional-diffusion/
├── proposal_methods/              # Core implementation
│   ├── smc_worker.py              # K-particle SMC / FKC-PoE sampler
│   ├── sd_bridge.py               # SD 1.5 bridge: VAE encoding, UNet stepping
│   ├── methodA_sd_bridge.py       # SD 1.5 bridge implementation with Bridge Correction
│   ├── text_workers.py            # Text-conditioned workers (DiffCollage/Naive PoE/Bridge Correction)
│   ├── workers.py                 # Image-conditioned workers (EDM backend)
│   ├── evaluate_sd_bridge.py      # SD 1.5 evaluation entry point
│   ├── evaluate_three_methods.py  # EDM multi-method evaluation
│   ├── evaluate_text_bridge.py    # Text-conditioned EDM evaluation
│   ├── common.py                  # Shared utilities
│   ├── generate_method.py         # Single-method runner
│   ├── requirements.txt
│   ├── README_Proposal.md         # Method theory & theoretical gaps
│   ├── README_RESULTS.md          # Completed run metrics
│   └── sbatch/                    # Slurm job scripts
├── diff_collage/                  # DiffCollage library + extensions
├── dnnlib/                        # NVIDIA EDM model utilities
├── torch_utils/                   # NVIDIA EDM sampling utilities
├── src/customguidance/            # Custom CFG guidance package (SD3/Flux backend)
│   ├── guidance/                  # Guidance method implementations
│   ├── pipeline/                  # SD3 / Flux custom pipelines
│   ├── evaluation/                # FID, CLIP, BLIP metrics
│   └── configs/                   # Hyperparameter grids
├── scripts/                       # Python runners & evaluation scripts
│   ├── inference.py               # Single-image inference
│   ├── benchmark.py               # Multi-method benchmark
│   ├── hyperparameters.py         # Hyperparameter grid search
│   └── download_imagenet_landscapes.py
├── bash/                          # Shell launch scripts
│   ├── _run_smc_only.sh           # SMC-only 22-pair evaluation
│   ├── _run_smc_v2.sh             # SMC v2 sweep
│   ├── _run_num5.sh               # 5-window panoramic bridge
│   ├── run_text_sweep.sh          # 22-pair text sweep (baselines + Bridge Correction)
│   ├── run_all.sh                 # EDM image+text evaluation
│   └── _sweep_gpu*.sh             # Per-GPU hyperparameter sweeps
├── toy_datasets/                  # 3-variable toy diffusion experiments
│   ├── acg/                       # ACG diffusion module
│   └── notebooks/                 # Jupyter experiment notebooks
├── data/
│   └── imagenet_landscapes/       # ImageNet landscape endpoint images
├── pyproject.toml                 # Package setup (customguidance)
└── proposal_outputs/              # Generated outputs (not tracked by git)
```

---

## Environment Setup

```bash
conda create -n diffusion_ttc python=3.10 -y
conda activate diffusion_ttc

# PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies for SD 1.5 bridge
pip install -r proposal_methods/requirements.txt
pip install diffusers transformers accelerate

# Optional: customguidance package (SD3/Flux backend)
pip install -e .
```

Verify CUDA:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

---

## Quickstart

### SMC Bridge (main use case)

Single text-pair with SMC:
```bash
python -m proposal_methods.evaluate_sd_bridge \
  --text-pairs "a calm lakeside landscape:a volcanic eruption" \
  --output-root proposal_outputs/smc_demo \
  --num-img 3 --overlap-latents 48 --n-steps 50 --guidance-scale 7.5 \
  --smc-only --smc-K 8 --smc-beta 1.0 --smc-resample-end 0.8 \
  --seed 17 --device cuda
```

22-pair benchmark (baselines + Bridge Correction):
```bash
bash bash/run_text_sweep.sh cuda
```

22-pair benchmark (SMC only):
```bash
bash bash/_run_smc_only.sh
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `--text-pairs "A:B"` | Left and right endpoint prompts, separated by `:` |
| `--num-img` | Number of windows; width = 512 + (N−1)×256 px |
| `--overlap-latents` | Overlap in latent space (32 = 256 image px) |
| `--smc-only` | Use SMC/FKC sampler |
| `--smc-K` | Number of SMC particles |
| `--smc-beta` | FK potential strength |
| `--smc-resample-end` | Fraction of steps with active resampling (0.8 recommended) |
| `--guidance-scale` | CFG scale |
| `--proposal-couplings` | Coupling sweep for `bridge_correction`, e.g. `0.01,0.05` |

### EDM Backend (image-conditioned)

```bash
python scripts/download_imagenet_landscapes.py --max-per-class 10

python -m proposal_methods.evaluate_three_methods \
  --image-indices 0,1,2,3,4,5,6,7,8 \
  --classes lakeside:975,volcano:980,alp:970 \
  --image-root data/imagenet_landscapes \
  --output-root proposal_outputs/evaluation_n3 \
  --grid-max-pairs-per-class 3 \
  --num-img 3 --n-step 80 --solver heun \
  --guidance-scale 0.7 --coupling-strength 0.03
```

### Running on a Cluster (Slurm)

```bash
export PROJECT_ROOT=/path/to/cs503-compositional-diffusion
export DIFFUSION_SCRATCH_ROOT=/scratch/$USER/diffusion

sbatch proposal_methods/sbatch/run_bridge_correction.run
sbatch proposal_methods/sbatch/run_evaluation.run
```

---

## Output Structure

```
proposal_outputs/<run_name>/
├── comparison_grid.png        # Side-by-side: [DiffCollage | Naive PoE | Bridge Correction | SMC / FKC-PoE]
├── boundary_grid.png          # Zoomed seam regions (×4)
├── <method>/
│   └── <prompt_pair>/
│       └── sample_000.png
├── metrics.csv                # Per-image metrics
├── summary_metrics.csv        # Method-level averages
└── config.json                # Run configuration
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| `seam_mse_mean` | Average MSE across all window-boundary seams |
| `seam_mse_max` | Worst single-seam MSE |
| `left_endpoint_mse` | Bridge start vs. left fixed strip (EDM only) |
| `right_endpoint_mse` | Bridge end vs. right fixed strip (EDM only) |

Lower seam MSE = tighter boundary alignment. Low MSE does not guarantee visual quality.

---

## Results

### SD 1.5 — Text-Prompt Panorama Sweep

Settings: frozen Stable Diffusion 1.5, DDIM 50 steps, CFG `7.5`, two-prompt output size `512×1024`, three-prompt output size `512×1536`. Baselines and Bridge Correction use `overlap_latents=32`; SMC / FKC-PoE uses `overlap_latents=48`.

| Method | Config | Seam MSE mean ↓ | Seam MSE max ↓ | Seam MSE std ↓ |
|--------|--------|-----------------|----------------|----------------|
| Naive PoE (`naive`) | seed=42 | 0.0187 | 0.0236 | 0.0163 |
| DiffCollage (`diffcollage`) | seed=42 | **0.0175** | **0.0208** | 0.0158 |
| **Bridge Correction (`bridge_correction`)** | `c=0.01`, seed=42 | 0.0184 | 0.0223 | 0.0165 |
| Bridge Correction (`bridge_correction`) | `c=0.05`, seed=42 | 0.0628 | 0.0996 | 0.0259 |
| **SMC / FKC-PoE (`proposal_smc`)** | `K=8`, `β=1.0`, seed=17, overlap=48 | 0.0198 | 0.0237 | **0.0128** |
| SMC / FKC-PoE (`proposal_smc`) | `K=8`, `β=0.1`, seed=17, overlap=48 | 0.0211 | 0.0257 | 0.0127 |

Two things to read here. First, tuned DiffCollage is already the best method on mean seam MSE (`0.0175`), so reproducing strong baselines closes much of the apparent gap. Second, SMC / FKC-PoE trades a little mean performance for much lower per-prompt variance (`0.0128`, about `-22%` versus baselines near `0.016`).

Additional SMC sweeps over `K`, `β`, seeds, and overlap settings are summarized in the project webpage and report.

### Qualitative Comparisons

The webpage reports two-prompt panoramas comparing Naive PoE, DiffCollage, Bridge Correction, and SMC / FKC-PoE on four prompt pairs, plus three-prompt panoramas comparing Bridge Correction and SMC / FKC-PoE. The qualitative takeaway matches the quantitative one: Bridge Correction can sharpen transitions in a single trajectory, while SMC tends to select more compatible bridge trajectories on harder transitions.

---

## Conclusions

We re-derived DiffCollage as the `R ≡ 1` projection of the true panoramic joint and identified the missing `∇ log R` correction term. This led to two training-free correctors for SD 1.5 panoramic composition: single-trajectory Bridge Correction and multi-trajectory SMC / FKC-PoE with overlap-compatibility Feynman-Kac weights.

Our strongest finding is that careful baseline reproduction matters. Tuned DiffCollage reaches `0.0175` seam MSE mean, about `-40%` versus the published `0.0307` reference number, so much of the apparent headroom for new methods disappears once baselines are tuned.

SMC / FKC-PoE's main win is not lower mean seam MSE, but lower per-prompt variance: `0.0128`, about `-22%` versus baselines near `0.016`. Future work should therefore benchmark reliability and variance, not just mean seam smoothness, before claiming improvements in particle-based correctors.

---

## Known Limitations

- **Seam MSE only**: The headline metric is a local smoothness measure on pixel strips around visible window boundaries. It does not measure prompt satisfaction, global perceptual quality, or semantic coherence.
- **FK reward saturation**: The SMC reward uses ε-agreement on overlap regions. It saturates at `β ≥ 10` and starts hurting, so a stronger semantic-space reward such as CLIP-overlap compatibility or a learned consistency critic is needed.
- **Protocol mismatch**: SMC was tuned with a different seed (`17` vs. `42`) and overlap (`48` vs. `32`) than the baselines, so the mean seam-MSE comparison is not fully apples-to-apples.
- **Bridge Correction brittleness**: The Tweedie corrector has a narrow `c` sweet spot. Increasing `c` from `0.01` to `0.05` collapses seam MSE from `0.0184` to `0.0628`.
- **Scope**: Experiments are on SD 1.5 panoramic generation only. SDXL, non-latent diffusion backbones, and non-panorama composition tasks are not tested.

---

## External Code

| Folder | Source |
|--------|--------|
| `diff_collage/` | [sbyebss/DiffCollage](https://github.com/sbyebss/DiffCollage) + custom extensions |
| `dnnlib/`, `torch_utils/` | [NVlabs/edm](https://github.com/NVlabs/edm) |
| `proposal_methods/smc_worker.py` | Feynman-Kac Correctors (Skreta et al., ICML 2025, Prop. 3.3) |
