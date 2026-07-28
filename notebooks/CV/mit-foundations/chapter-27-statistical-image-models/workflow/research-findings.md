# Research findings — Ch 27 Statistical Image Models

Digest of visionbook.mit.edu/stat_image_models_revised.html.

## Power law (natural image statistics)
- Fourier magnitude ~ 1/w^a, w = sqrt(u^2+v^2), a ~ 1-1.5. Angular-averaged
  spectrum of natural images hugs 1/w^1.5; white noise is flat.
- Stationary => covariance C is circulant, diagonalised by the FFT; eigenvalues 1/w^a.

## Models
- Independent-pixel: p(l) = prod p(l[n,m]); sampling keeps the histogram, destroys
  structure (only works for unstructured images like a star field).
- Gaussian prior: p(l) ~ exp(-1/2 l^T C^-1 l); sampling with matched Fourier
  magnitude + random phase => cloud-like images.
- Wavelet marginal: p(l) = prod_k prod p_k(l_k); each band-pass subband is
  generalized-Laplacian.

## Heavy-tailed derivatives
- Generalized Laplacian: p(x) ~ exp(-|x/s|^r). r=2 Gaussian, r=1 Laplacian,
  r in [0.4,0.8] natural images (sharp peak, heavy tails).
- Band-pass Gaussian noise stays Gaussian; natural images become heavy-tailed.

## Denoising
- Wiener (Gaussian/1/f prior): L(w) = S(w)/(S(w)+sigma^2) L_g(w), S=A/w^{2a}.
- Wavelet coring (Laplacian prior + Gaussian noise): MAP soft-thresholds small
  coefficients to zero, keeps large ones.
- Non-local means: l_NLM[n] = sum_k w[n,k] l[k], w ~ exp(-|patch_k - patch_n|^2/h^2);
  nonparametric MRF exploiting self-similarity.

## Example images
Book: recurring street scene (colour), MIT dome, wheel, hair, spheres, clouds,
stars, Mondrian, dead-leaves. Notebook: book MIT-dome + wheel + skimage cat +
synthetic (1/f clouds, star field).
