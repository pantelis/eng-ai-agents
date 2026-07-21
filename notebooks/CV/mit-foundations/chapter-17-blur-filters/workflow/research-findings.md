# Research findings — Ch 17 Blur Filters

Digest of the chapter at <https://visionbook.mit.edu/blurring_2.html>, recording
the exact math and figure list the notebook reproduces. Section numbers follow
the book.

## Theme

Blur filters are **low-pass** linear filters: each output pixel is a weighted
average of a neighbourhood. Averaging suppresses zero-mean noise but also
attenuates genuine high-frequency detail — the chapter's central trade-off. The
three families differ only in their frequency response.

## 17.2 Box filter

- Kernel: `box_{N,M}[n,m] = 1` for `-N<=n<=N, -M<=m<=M`, else `0`.
- **DC gain** = sum of coefficients; normalise by `(2N+1)(2M+1)` for unit gain
  (brightness-preserving).
- **Separable**: `box_{N,M} = box_N^x * box_M^y`. Setting `N=0` or `M=0` gives a
  1-D (horizontal / vertical) blur.
- **Limitation**: the DTFT of a length-L box is a Dirichlet / aliased-sinc, so
  the frequency response is **non-monotonic** (side lobes) — some high
  frequencies pass with more gain than lower ones. Also, box * box = triangle
  (not a box).

## 17.3 Gaussian filter

- Continuous 1-D: `g(x;s) = 1/sqrt(2*pi*s^2) * exp(-x^2 / (2 s^2))`.
- Continuous 2-D: `g(x,y;s) = 1/(2*pi*s^2) * exp(-(x^2+y^2)/(2 s^2))`.
- **Discretise** by sampling `exp(-(n^2+m^2)/(2 s^2))` and renormalising; samples
  within `+-3s` suffice (radius = ceil(3s)).
- **Separability**: the Gaussian is the *only* circularly symmetric separable
  kernel; `g(x,y)=g(x)g(y)`, reducing cost `O(N^2) -> O(2N)`.
- **Fourier**: `G(w;s) = exp(-w^2 s^2 / 2)` — a Gaussian, **monotonically
  decreasing** (no side lobes). Wider in space <=> narrower in frequency.
- **Composition**: `g(s1)*g(s2) = g(s3)` with `s3^2 = s1^2 + s2^2`.
- Also: solution to the heat equation; central-limit attractor. These hold
  exactly only in the continuous case; the sampled kernel satisfies them
  approximately.

## 17.4 Binomial filters

- `b_n` = `[1,1]` convolved with itself `n` times = row `n` of Pascal's triangle
  (`b_2 = [1,2,1]`, `b_3 = [1,3,3,1]`, ...).
- **DC gain** = `2^n` (normalise by `2^n`); **variance** `s_n^2 = n/4`.
- **Composition**: `b_n * b_m = b_{n+m}`, `s_n^2 + s_m^2 = s_{n+m}^2` (discrete
  analogue of the Gaussian property).
- **Fourier**: `B_{2n}(u) = (2 + 2 cos(2*pi*u/N))^n` — zero-phase, **monotonic**.
  Convolving with the highest-frequency wave `[1,-1,1,-1,...]` yields exactly
  zero: `[1,2,1]/4` applied to alternating signs gives `(1-2+1)/4 = 0`. A box
  `[1,1,1]/3` leaves `(1-1+1)/3 = 1/3`.
- **2-D** by separability: `b_{2,2} = [1,2,1]^T [1,2,1] = [[1,2,1],[2,4,2],[1,2,1]]`,
  DC gain 16.
- By the **central limit theorem**, `b_n / 2^n -> Gaussian(var = n/4)` — binomials
  are the standard cheap integer Gaussian approximation (used in image pyramids).

## Figures (book -> notebook mapping)

| Book fig | Content | Notebook figure |
|---|---|---|
| 17.1 | noisy vs blurred image | book stop_256 + 5x5 box (fig17_01) |
| 17.2 | square / horizontal / vertical box | skimage box blurs (fig17_02) |
| 17.3 | 1-D box_1 kernel + rippled DFT | box_1=[1,1,1] + DFT (fig17_03) |
| 17.4 | Zebra at Gaussian s=2,4,8 | book zebra s=2 -> 4 -> 8 (fig17_04) |
| 17.5 | Lincoln block portrait vs blur | book blocky Lincoln + blur (fig17_06) |
| (props) | Gaussian monotonic response + variance-add | fig17_05 |
| 17.7 | 1-D binomial [1,2,1] + monotonic DFT | fig17_07 |
| 17.8 | Boat: checkerboard noise, box vs binomial | book boat + 1px checker (fig17_08) |
| (CLT) | binomial -> Gaussian convergence | fig17_09 |

## Book example images (loaded live from visionbook.mit.edu, not redistributed)

- Stop sign: `blur_filters/stop_256_noise_3.jpg`
- Zebra: `spatial_filters/gausian_zebra_c_2.jpg` (published already at s=2)
- Lincoln (Harmon & Julesz 1971): `blur_filters/Jules_Lincoln_1971.jpg`
- Boat: `blur_filters/boat_d_binomial.jpg` (book's cleaned boat, used as the clean base)

Only fig17_02 uses a `skimage.data.camera` sample.
