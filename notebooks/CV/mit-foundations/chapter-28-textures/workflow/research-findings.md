# Research findings — Ch 28 Textures

Digest of visionbook.mit.edu/textures.html.

## What is a texture
An image made of many similar-looking elements; a texture representation captures
the STATISTICS of its elements, not the elements individually. (No formal
random-field definition given.)

## Heeger-Bergen (parametric)
- Decompose a reference with a steerable pyramid (6 orientations, 3 scales).
- Representation = the 18 subband histograms + low-pass residual histogram + input
  pixel histogram.
- Synthesis: start from white noise; iteratively histogram-match (a pointwise
  monotonic map) in both the transform (subband) and pixel domains; reconstruct.
- Colour: PCA-decorrelate the colour channels first.

## Efros-Leung (nonparametric, MRF)
- Model the distribution of a pixel as dependent on its neighbourhood.
- Grow pixel-by-pixel: find sample neighbourhoods with SSD to the current context
  below a threshold, randomly pick one, copy its centre pixel.
- Context window size: small preserves local detail but loses layout; large
  enforces the regular arrangement.

## Figures
28.1 infinite texture (tiling); 28.4 textons; 28.5 analysis/synthesis pipeline;
28.6 HB iterations; 28.7 HB architecture; 28.8 one-subband matching; 28.9 two HB
examples; 28.10 Efros-Leung schematic; 28.11 context-size effect; 28.12 Efros
results.

## Example textures
stone wall, plums, zebra (HB colour examples); circles/dots and pebbles
(Efros-Leung). Notebook uses stone_wall.jpg + efros1a.jpg (circles).
