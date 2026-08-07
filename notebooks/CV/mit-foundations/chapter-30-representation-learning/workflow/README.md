# Workflow — How this chapter was produced

Companion notebook for Chapter 30 (Representation Learning), following the
Ch 17/18/19/27/28/29 pattern. 5 figures on the book's toy colored-shapes dataset.
First **deep-learning** chapter of the set: the autoencoder and contrastive encoders
are trained with PyTorch (device-agnostic — CUDA when available; the maintainer
validates under `torch.dev.gpu`).

## Dataset — the book's procedural colored shapes
Generated exactly (the book's is procedural too, not a fixed image set): 3 shapes
(circle / square / triangle) x 8 colour classes, random size / position / rotation,
plus a small per-image colour perturbation. Two independent factors — **shape** and
**colour** — for a representation to disentangle. 32x32 RGB.

## Figures reproduced (5)
| Fig | Content |
|---|---|
| 30.4 | colored-shapes dataset samples |
| 30.5a | autoencoder: reconstructions + nearest neighbours in embedding space |
| 30.5b | 1-NN probe of shape vs colour at each layer |
| 30.11 | K-means iterations on 2-D data, k=5 |
| 30.14 | contrastive (InfoNCE): the augmentation decides colour- vs shape-invariance |

## Implementation
- **Autoencoder** (the book's architecture): **six conv layers** (32->16->16->8->8->4->4,
  channels 24/48/64/96/128/128) + a **128-d** linear bottleneck + mirror deconv decoder;
  MSE reconstruction, Adam, **20000 steps**. Downsampling stops at 4x4 so the deep features
  keep the spatial detail that distinguishes the shapes (downsampling to 1x1 destroys it
  and reverses the 30.5b trend). `feats()` exposes activations at pixels / conv1..conv6.
- **1-NN probe** (30.5b): cosine 1-NN from train features -> predict a test image's
  shape / colour, per layer.
- **K-means** (30.11): pure numpy block-coordinate descent (assign to nearest code
  vector; move each code vector to its assigned mean), snapshots of the first iterations.
- **Contrastive** (30.14): a small conv encoder to a 2-D unit-circle embedding, trained
  with **InfoNCE** (temperature 0.2). Two augmentation regimes select the invariance:
  - `T_c` = random-resized crop only -> colour preserved -> embedding organizes by COLOUR;
  - `T_s` = colour jitter (per-channel gain + channel permutation + brightness) with a
    small spatial shift -> colour destroyed, shape kept -> embedding organizes by SHAPE.

## Numerical checks
- 30.5b: the book's **crossing**, reproduced exactly — shape 1-NN rises 78% -> **99%** with
  depth while colour falls 79% -> **58%** (book: 99% / 58%).
- 30.14: crop-only embedding: colour 1-NN 84%, shape 35%; colour-jitter embedding:
  shape 74%, colour 14% (~chance). The augmentation demonstrably decides the invariance.

## Validation
Device-agnostic; runs on CPU or GPU. The 20000-step autoencoder trains in ~8 min on a
Quadro P1000 (the tiny model/batch underutilizes the GPU) — longer on CPU. The AE weights
were trained on GPU and the AE figures (30.5a/b) rendered from them; the AE-independent
figures (30.4/30.11/30.14) run on CPU. The notebook runs end to end under `torch.dev.gpu`.
```bash
docker compose run --rm torch.dev.gpu \
  bash -c "python scripts/execute_notebook.py \
    CV/mit-foundations/chapter-30-representation-learning/index.ipynb"
```

## Honest caveats
- The 30.5b crossing needs the book's architecture (depth + 4x4 spatial) AND a real
  training budget (20000 steps here). A too-shallow model, or one downsampled to a single
  vector, does **not** show it — with those, the trend is flat or reversed.
- Contrastive `T_s`: cropping leaves *partial* shapes that are ambiguous, so shape
  fails to cluster; using colour-jitter + a small shift (full shape visible) is what
  makes the shape-invariant embedding separate cleanly.
