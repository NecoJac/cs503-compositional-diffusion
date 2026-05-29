# README Results: Bridge Composition Experiments

This note records the completed bridge-generation evaluations currently on
disk. Lower metric values are better for all MSE metrics below.

## Completed Runs

### Long Bridge, `num_img=5`

Output directory:

```text
proposal_outputs/evaluation_tuned/
```

Native image size:

```text
64 x 192
```

Configuration:

```text
classes: lakeside:975, volcano:980, alp:970
pairs per class: 9
methods: diffcollage, naive, bridge_correction
n_step: 80
solver: heun
guidance_scale: 0.7
coupling_strength: 0.03
init_correction_steps: 0
```

Overall metrics:

| method | samples | seam MSE mean | seam MSE max | endpoint MSE mean |
|---|---:|---:|---:|---:|
| diffcollage | 27 | 0.080665 | 0.314117 | 0.241996 |
| naive | 27 | 0.087595 | 0.336861 | 0.262784 |
| bridge_correction | 27 | 0.103556 | 0.405631 | 0.310667 |

Per-class endpoint MSE:

| class | diffcollage | naive | bridge_correction |
|---|---:|---:|---:|
| alp | 0.200548 | 0.230423 | 0.261328 |
| lakeside | 0.271170 | 0.254813 | 0.340427 |
| volcano | 0.254271 | 0.303115 | 0.330246 |

### Short Bridge, `num_img=3`

Output directory:

```text
proposal_outputs/evaluation/
```

Native image size:

```text
64 x 128
```

Overall metrics:

| method | samples | seam MSE mean | seam MSE max | endpoint MSE mean |
|---|---:|---:|---:|---:|
| diffcollage | 27 | 0.123393 | 0.322802 | 0.246786 |
| naive | 27 | 0.129611 | 0.332868 | 0.259223 |
| bridge_correction | 27 | 0.152030 | 0.398765 | 0.304060 |

Per-class endpoint MSE:

| class | diffcollage | naive | bridge_correction |
|---|---:|---:|---:|
| alp | 0.257426 | 0.233740 | 0.314738 |
| lakeside | 0.232234 | 0.250161 | 0.309698 |
| volcano | 0.250698 | 0.293767 | 0.287744 |

## Analysis

The current formula-based `bridge_correction` does not improve the reported
seam/endpoint MSE on these completed runs. `diffcollage` is best overall on
both `num_img=5` and `num_img=3`, while `naive` is usually second. The current
`bridge_correction` is worse on average after switching to the HTML-style formula
skeleton:

```text
s_xyz = s_xy oplus s_yz - s_y_implicit + Delta s
```

This is not necessarily a proof that the proposal idea is bad. It mainly says
that the current proxies for the two hard theoretical terms are not yet good:

- `s_y_implicit` is only a current-state one-sample approximation, not a true
  conditional expectation.
- `Delta s = grad log R` is only a Tweedie `x0` overlap-consistency proxy, not
  a true estimator of the coupling factor
  `R = p(x,z | y) / (p(x | y) p(z | y))`.

The result is that `bridge_correction` can over-correct local overlaps without
improving global visual coherence. In the current metrics, that appears as
higher endpoint/seam MSE; visually, it can appear as blocky or tiled structure.

## Why The Images Look Blocky

The blockiness is expected for this prototype, although it is not desired for
the final goal.

- The backbone is an EDM ImageNet `64 x 64` model. It has no native long-image
  or panorama prior.
- The long image is built from overlapping `64 x 64` local patches. With
  overlap `32`, `num_img=5` creates a `64 x 192` image from five local windows;
  `num_img=3` creates a `64 x 128` image from three local windows.
- Each local factor only sees a `64 x 64` crop. The sampler never has a single
  global model that understands the whole `128` or `192` pixel-wide scene.
- The fixed endpoint strips are hard image conditions. If the two endpoint
  crops are visually incompatible, the middle bridge is forced to satisfy local
  overlaps rather than discover a globally natural scene.
- The current `Delta s` proxy optimizes local overlap consistency. It does not
  directly optimize perceptual realism, object continuity, or semantic
  coherence across the whole bridge.
- The current metrics mostly measure endpoint/seam consistency. They do not
  penalize patch-wise semantic changes strongly enough, so a method can look
  blocky even when the overlap metric is acceptable.

In short, the current method is a local patch-composition prototype. It is
useful for testing score arithmetic, but block artifacts are a natural failure
mode until the method uses better global constraints, better `R` estimation, or
a backbone that supports larger images.

## Practical Takeaways

- Use `diffcollage` as the strongest current baseline for these bridge metrics.
- Treat `bridge_correction` as a formula-aligned prototype, not as the best visual
  method yet.
- Keep the `num_img=3` run for sanity checks, because shorter bridges reduce
  the number of patch transitions.
- Add perceptual metrics before making claims about visual quality. Good next
  candidates are LPIPS / CLIP image similarity for neighboring regions, CLIP
  text alignment if moving to text conditions, and a no-reference image-quality
  proxy.
- The next method work should focus on improving `s_y_implicit` and `Delta s`,
  not just retuning `coupling_strength`.
