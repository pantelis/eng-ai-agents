# Research findings — Ch 19 Temporal Filters

Digest of <https://visionbook.mit.edu/temporal_filters_v2.html>. Section numbers
follow the book.

## Video as a space-time volume
- Sequence `l(x,y,t)` (discrete `l[n,m,t]`, size N x M x P).
- Globally translating image: `l(x,y,t) = l0(x - vx t, y - vy t)`.
- In an x-t slice: static point = vertical line; point at velocity v = line of
  slope 1/v. Cross-sections of the volume reveal motion as oriented streaks
  (the EPI / photo-finish idea).

## Fourier (space-time)
- `L(wx,wy,wt) = L0(wx,wy) * delta(wt + vx wx + vy wy)` — all energy lies on the
  plane `wt + vx wx + vy wy = 0`.
- A moving 1D pulse is a slanted band in x-t; its 2D FT is a sinc ridge on the
  line `wt + v wx = 0` (vertical for v=0, tilting with speed).

## Temporal derivatives / brightness constancy
- Discrete temporal difference: `l[n,m,t] - l[n,m,t-1]`.
- For a translating image: `dl/dt = -vx dl0/dx - vy dl0/dy`, i.e.
  `dl/dt + vx dl/dx + vy dl/dy = 0` (the optical-flow constraint).

## Spatiotemporal Gaussian & derivatives
- `g(x,y,t;s,st) = 1/((2pi)^1.5 s^2 st) exp(-(x^2+y^2)/2s^2) exp(-t^2/2st^2)`.
- Separable, non-causal. Gradient `grad g = (-x/s^2, -y/s^2, -t/st^2) g`.
- **Velocity-skewed** kernel: `g(x - vx t, y - vy t, t)` — blurs along a velocity.

## Velocity filters
- **Velocity-matched blur**: convolving with the skewed Gaussian keeps objects at
  (vx,vy) sharp, smears the rest.
- **Velocity-nulling filter**: `h = gt + vx gx + vy gy = grad g . (1, vx, vy)`.
  For a sequence moving at exactly (vx,vy), `l * h = 0` (from brightness
  constancy). Removes objects at that velocity, passes everything else. Nulling
  v=0 removes static content.

## IIR / causality (brief)
- Example: `l_out[t] = l_in[t] + a l_out[t-1]`, impulse response `a^t u[t]`,
  stable iff |a| < 1. Causal (past only) / non-causal (needs future) / anti-causal.
- Gaussian filters are non-causal; must be truncated + shifted for streaming.

## Figures (book -> notebook)
| Book | Notebook |
|---|---|
| 19.1 space-time volume of pedestrians | real vtest.avi frames + x-t slice (fig19_01/02) |
| 19.2 moving pulse + space-time DFT | fig19_03 |
| 19.3 spatiotemporal Gaussian kernels | standard vs velocity-skewed (fig19_04) |
| 19.4 temporal blur | velocity-matched blur (fig19_05) |
| 19.5/19.6 spatiotemporal derivatives | g_t/g_x/g_y slices + g_t frame response (fig19_06/07) |
| 19.7 nulling filter | x-t slices + frames (fig19_08/09) |

## Example media
- Book: colour pedestrian video (90 frames, 128x128) — NOT redistributed.
- Notebook: real OpenCV vtest.avi (56 colour frames, 120x160, static camera,
  people walking) via assets/ped_seq.npz; plus a synthetic 1D moving pulse for 19.2.
