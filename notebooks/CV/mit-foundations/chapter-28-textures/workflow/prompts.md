# Prompts — Ch 28 Textures

## 1. Digest
> Extract the math: texture as statistics; Heeger-Bergen steerable-pyramid
> histogram matching (analysis + synthesis + colour PCA); Efros-Leung
> nonparametric patch synthesis and the context-window effect. List figures/textures.

## 2. Build
> Reproduce: tiling (28.1), Heeger-Bergen synthesis + iterations + one-subband
> matching (28.5/6/8/9), Efros-Leung context-size effect + result (28.11/12).
> Use the book's stone_wall.jpg and efros1a.jpg (circles).

## 3. Fixes
> Per-channel Heeger-Bergen -> rainbow noise (channels decorrelate). Synthesise
> luminance (stone wall is gray; note the book uses colour PCA). Efros-Leung: the
> unfilled NaNs poison the SSD -> nan_to_num the context window; keep sizes small
> (~60-96 px) so it finishes in ~30s. Use finest Laplacian band for the heavy-
> tailed subband demo.

## Lessons
- Real synthesis algorithms: verify the OUTPUT looks like a texture, not noise.
- Efros-Leung is slow; vectorise the SSD with sliding_window_view and cap sizes.
