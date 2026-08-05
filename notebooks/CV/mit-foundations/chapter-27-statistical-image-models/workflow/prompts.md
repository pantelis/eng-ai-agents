# Prompts — Ch 27 Statistical Image Models

## 1. Digest the chapter
> Extract the math: the 1/f power law; histograms of intensities and derivatives
> (heavy-tailed generalized-Laplacian fits); Gaussian/wavelet priors; Wiener,
> coring, and non-local-means denoising. List every figure and example image.

## 2. Build the notebook
> ~10 figures covering the arc: power law (real images vs noise), sampling clouds
> from 1/f, independent-pixel failure, correlation decay, derivative histograms,
> generalized Laplacian, band-pass histograms, Wiener/coring/NLM denoising. Use
> the book's MIT-dome + wheel photos and a skimage natural; compute the rest.

## 3. Fixes
> Wiener with a unit-amplitude 1/f prior over-blurs (crossover at w~5). Write the
> Wiener gain via the signal=noise crossover w0: H = 1/(1+(w/w0)^{2a}), w0~45.
> NLM: pass the known sigma (estimate_sigma needs PyWavelets, absent in container).
> numpy 2.0: use np.ptp(x), not x.ptp().

## Lessons
- A number beside every figure: derivative kurtosis ~40, denoising RMSE drops.
- Check the real book figure image for rendering conventions.
