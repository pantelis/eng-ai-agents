# Prompts — Ch 30 Representation Learning

## 1. Digest
> Extract the computational core: the colored-shapes toy dataset; the autoencoder
> embedding (reconstructions, nearest neighbours, per-layer 1-NN probe of shape vs
> colour); K-means; contrastive InfoNCE where the augmentation (crop vs colour-jitter)
> selects colour- vs shape-invariance. List which figures are diagrams vs computable.

## 2. Build
> Reproduce on the book's colored-shapes data: dataset samples (30.4); a small conv
> autoencoder -> reconstructions + NN viz (30.5a) + per-layer probe (30.5b); K-means
> iterations k=5 (30.11); contrastive to a 2-D unit-circle embedding with T_c (crop ->
> colour) vs T_s (colour-jitter -> shape) (30.14). Device-agnostic torch; register as
> torch.dev.gpu.

## 3. Fixes
> Keep it CPU-runnable: small data (32px, ~3k imgs), ~1.5k AE steps. The 30.5b crossing
> needs the book's 20k-step training - show the DIRECTION honestly, don't fake it, and
> don't downsample the AE to 1x1 (that reverses the trend). For contrastive T_s, drop
> the crop (partial shapes are ambiguous) and use colour-jitter + small shift so the full
> shape stays visible -> shape clusters cleanly (shape 1-NN ~74%, colour ~chance).

## Lessons
- Verify the CLAIM: contrastive must show colour-invariance (colour 1-NN -> chance) OR
  shape-invariance (shape 1-NN >> chance) depending on the augmentation - quantify both.
- Deep-learning chapters cost real training time on CPU; validate device-agnostically,
  keep default steps modest, and lean on the GPU validator for the heavier runs.
