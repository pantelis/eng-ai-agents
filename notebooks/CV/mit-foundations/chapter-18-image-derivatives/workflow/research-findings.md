# Research findings — Ch 18 Image Derivatives

Digest of <https://visionbook.mit.edu/derivatives.html>, recording the math the
notebook reproduces. Section numbers follow the book.

## Discretizing the derivative
- Two-tap `d0 = [1,-1]`: `ell[n]-ell[n-1]`. DFT `D0[u] = 1 - exp(-2pi j u/N)`,
  magnitude `2 sin(pi u/N)`; half-sample phase shift.
- Centred `d1 = [1,0,-1]/2`: `(ell[n+1]-ell[n-1])/2`. DFT `D1[u] = j sin(2pi u/N)`;
  no shift, rolls off earlier (smoother). `[1,0,-1] = [1,-1] * [1,1]`.
- Ideal derivative multiplies each frequency by `j omega` (magnitude linear in u).

## Gradient, directional derivative
- `grad ell = (d ell/dx, d ell/dy)`, a per-pixel vector.
- Directional: `d ell/dt = cos(th) dx + sin(th) dy` — a linear combination of the
  two derivative images, no new convolution.

## Gaussian derivatives (Hermite)
- Commutativity: `(d/dx ell) * g = ell * (d/dx g)` — differentiate the smooth
  kernel instead of the noisy image.
- First order: `g_x = -x/sigma^2 * g`.
- Order n: `g_{x^n} = (-1/(sigma*sqrt2))^n H_n(x/(sigma*sqrt2)) g`, with Hermite
  recursion `H_n = 2x H_{n-1} - 2(n-1) H_{n-2}` (`H_0=1, H_1=2x, H_2=4x^2-2`).
- Composition adds orders and variances (like the plain Gaussian).

## Derivative-of-binomial
- `d_n = b_n * [1,-1]` (b_n = Pascal row): `d0=[1,-1]`, `d1=[1,0,-1]`,
  `d2=[1,1,-1,-1]`, ... all sum to 0 (DC gain 0), smoother as n grows.

## Roberts / Sobel
- Roberts cross (2x2 diagonal): `[[1,0],[0,-1]]`, `[[0,1],[-1,0]]`.
- Sobel_x = `[1,0,-1] (x) * [1,2,1]^T (y)` = `[[1,0,-1],[2,0,-2],[1,0,-1]]`;
  separable, most isotropic and noise-tolerant. `Sobel_x[u,v] = D1[u]*B2[v]`.

## Laplacian
- `lap ell = d2/dx2 + d2/dy2`, rotationally invariant.
- Five-point stencil `[[0,1,0],[1,-4,1],[0,1,0]] = [1,-2,1]^T * [1,-2,1]`.
- Laplacian-of-Gaussian (Mexican hat): `lap g = (x^2+y^2-2 sigma^2)/sigma^4 * g`.
  Edges at zero-crossings; used in SIFT.

## Sharpening (unsharp masking)
- `sharpen = 2 I - b_{2,2}` (DC gain 1). Boosts high frequencies; iterate for more.

## Retinex
- `ell = r * l`; in log, `log ell = log r + log l`.
- Reflectance edges are SHARP (large log-gradients); illumination is SMOOTH
  (small gradients). Threshold the log-gradient, keep the large part as
  reflectance, integrate back (Poisson), take the remainder as illumination.
- Recovers reflectance up to a global DC constant.

## Early visual system
- `h = -lap g + lambda g` (lambda=2, sigma=5) matches the human contrast
  sensitivity function; explains the Vasarely illusion (centre-surround).

## Figures (book -> notebook)

| Book | Notebook |
|---|---|
| 18.2 x/y derivative of photo | book MIT-dome (colour) fig18_01 |
| 18.4 |DFT| d0/d1 vs ideal | fig18_02 |
| 18.6 noise vs Gaussian derivative | book stop_noise (colour) fig18_03 |
| 18.9 Gaussian + derivative orders | fig18_04 |
| 18.8 multiscale on zebra | book zebra fig18_05 |
| 18.13 derivative-of-binomial | fig18_06 |
| 18.14 |DFT| d0/d1/Roberts/Sobel | fig18_07 |
| 18.15 directional derivatives | synthetic disc fig18_08 |
| 18.16 Laplacian-of-Gaussian | fig18_09 |
| 18.18 wheel Laplacian | book wheel fig18_10 |
| 18.23 unsharp masking | book boat (colour) fig18_11 |
| 18.25/26 Retinex | synthetic Mondrian fig18_12 |

## Book example images (loaded live, not redistributed)
mit_der_a.jpg (colour), stop_noise.jpg (colour), gausian_zebra_c_2.jpg (gray),
wheel256.jpg (gray), boat_sharp0.jpg (colour). Retinex uses a synthetic Mondrian
(the book's Retinex input is itself synthetic).
