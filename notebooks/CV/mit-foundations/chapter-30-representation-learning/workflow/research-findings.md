# Research findings — Ch 30 Representation Learning

Source: https://visionbook.mit.edu/representation_learning.html (Torralba, Isola, Freeman).

## What the chapter covers
Learning embeddings that expose the factors of a scene, mostly without labels:
autoencoders (bottleneck reconstruction, ~PCA), predictive/pretext & self-supervised
tasks (colorization, inpainting, next-frame), clustering / K-means (discrete codes),
and contrastive learning (InfoNCE; alignment + uniformity on the hypersphere).

## The book's figures (from the chapter HTML)
| Fig | src | content |
|---|---|---|
| 30.1 | rep_learning_schematic.png | data -> abstract representation (diagram) |
| 30.2/30.3 | autoencoder_diagram / _learning_diagram.png | AE architecture (diagram) |
| 30.4 | shapes_dataset_random_samples.png | colored-shapes samples |
| 30.5a | AE_NN_viz.png | nearest neighbours in AE embedding |
| 30.5b | AE_NN_probe.png | 1-NN accuracy per layer (shape up, colour down) |
| 30.6-30.9 | predictive_learning / obj_detectors / imputation ... | pretext tasks (mostly diagrams/results) |
| 30.10 | clustering_f_diagram.png | clustering as discrete map (diagram) |
| 30.11 | kmeans_ex_step1..4.jpg | K-means iterations, k=5 |
| 30.12/30.13 | contrastive_learning_diagram / _colorization.png | contrastive setup (diagram) |
| 30.14 | align_unif_results_shapes_dataset.png | colour- vs shape-sensitive contrastive embeddings |

## Dataset spec (from the text)
64,000 images; circles/triangles/squares; randomized size/position/rotation; colour
from one of 8 classes + small perturbation. (We use fewer images at 32x32 for speed —
the generator matches the description.)

## Book experiment details
- Autoencoder: 6-conv encoder + 6-conv decoder, 128-d bottleneck, 20k SGD (Adam) steps,
  batch 128. (We use a lighter 3-conv / 32-d / ~1.5k-step version to stay CPU-runnable.)
- Contrastive: same encoder but M=2 for direct 2-D plotting. T_c = crop (colour-
  sensitive); T_s = crop + hue/brightness/saturation shift (shape-sensitive).
- K-means: 2-D toy data, k=5, four snapshots.

## What is computational vs diagram
Computable (chosen for the notebook): 30.4 dataset, 30.5a/b autoencoder embedding,
30.11 K-means, 30.14 contrastive. Diagrams (schematics of architectures / pretext
tasks) are left to the book: 30.1/30.2/30.3/30.10/30.12/30.13 and the pretext-result
figures 30.6-30.9.

## Notes / gotchas learned
- **Deep-learning chapter** -> needs PyTorch training. Book uses 20k steps on 64k images;
  that is minutes-to-slow on CPU. Kept the notebook CPU-runnable (~285 s) with smaller
  data + step counts; the maintainer validator is GPU (`torch.dev.gpu`), where it is fast.
- **30.5b crossing** (shape 99% / colour 58%) is an emergent property of the fully-trained
  deep AE; a short-trained or over-downsampled AE does NOT reproduce it (aggressive
  downsampling to 1x1 even reverses it — deep global features lose the spatial detail that
  distinguishes triangle/square). We show the correct *direction* honestly.
- **30.14 shape branch**: crop makes shapes partial/ambiguous -> shape doesn't cluster.
  Fix = colour-jitter (+ tiny spatial shift) with NO crop, so the full shape stays visible;
  then shape 1-NN jumps ~38% -> ~74% while colour drops to chance.
