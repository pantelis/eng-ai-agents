# Chapter 6 — Figure-by-figure progress

Tracking the 18 figures planned for the Chapter 6 (Lenses) companion notebook. Source: [Foundations of Computer Vision §6](https://visionbook.mit.edu/lenses.html), Torralba, Isola, Freeman (MIT Press, 2024).

Status: ⬜ not started · 🟡 in progress · ✅ committed

## 6.1 — Introduction

| Status | Book ref | Output file | Strategy |
|---|---|---|---|
| ✅ | Fig 6.1 | `images/fig06_01.png` | Synthesize. Dog scene imaged through small pinhole, large pinhole, and lens. Three panels showing the brightness/sharpness trade-off. |
| ⬜ | Fig 6.2 | `images/fig06_02.png` | Synthesize the effect. Same three-panel rig as 6.1 applied to a synthetic scene that stands in for the book's Gumby photo demo. |

## 6.2 — Lensmaker's Formula

| Status | Book ref | Output file | Strategy |
|---|---|---|---|
| ✅ | Fig 6.3 (a) | `images/fig06_03a.png` | Synthesize. Snell's law diagram — flat interface, normal, incoming and refracted ray with angle arcs. |
| — | Fig 6.3 (b) | — | Skip. Physical photo (straw in water). Acknowledged in prose. |
| ✅ | Fig 6.4 (a) | `images/fig06_04a.png` | Synthesize. Thin-lens geometry — object point, lens, image point, ray fan converging. |
| ✅ | Fig 6.4 (b) | `images/fig06_04b.png` | Synthesize. 6.4 (a) re-labeled with the four ray angles of Table 6.1. |
| ✅ | Fig 6.5 | `images/fig06_05.png` | Synthesize. Spherical lens surface as a circular arc, $\theta_S$ at height $c$. |
| ✅ | Fig 6.6 | `images/fig06_06.png` | Synthesize. Off-axis source. Lens rotated by $\theta_R$, on-axis point $P_0$ and off-axis point $P_1$ imaged. (Book places 6.6 in §6.2 — the off-axis case extends the lensmaker derivation; do not re-file under §6.3.) |

## 6.3 — Imaging with Lenses

| Status | Book ref | Output file | Strategy |
|---|---|---|---|
| — | Fig 6.7 | — | Skip. Physical laser-pointer photo. Acknowledged in prose. |
| ✅ | Fig 6.8 (a–c) | `images/fig06_08abc.png` | Synthesize as one combined 3-panel figure. (a) thick lens center ray; (b) thin lens center ray; (c) lens as pinhole. |
| ✅ | Fig 6.9 (a–e) | `images/fig06_09.png` | Synthesize. Five-panel conjugate points — object at infinity, beyond $2f$, at $2f$, between $f$ and $2f$, at $f$. |

## 6.3.1 — Depth of Field

| Status | Book ref | Output file | Strategy |
|---|---|---|---|
| ✅ | Fig 6.10 | `images/fig06_10.png` | Synthesize. Depth of field for a thin lens — circle of confusion vs object distance. |
| ✅ | Fig 6.11 | `images/fig06_11.png` | Synthesize. Variables for the depth-of-field calculation — two apertures, similar-triangle geometry. |
| ✅ | Fig 6.12 | `images/fig06_12.png` | Synthesize. Photographic depth of field vs aperture (depth-dependent blur). |

> **Resolved 2026-06-24:** the Fig 6.9(e) anomaly is fixed — renamed to Fig 6.11, saving to `fig06_11.png` (§6.3.1); orphan PNGs cleaned up.

## 6.3.2 — Concave Lenses

| Status | Book ref | Output file | Strategy |
|---|---|---|---|
| ✅ | Fig 6.13 (a–c) | `images/fig06_13.png` | Synthesize. Convex and concave thin-lens behavior — converging real focus, diverging virtual focus, and a tilted parallel bundle through the diverging lens. |

## 6.3.3 — Lenses in a Telescope

| Status | Book ref | Output file | Strategy |
|---|---|---|---|
| ✅ | Fig 6.14 (a, b) | `images/fig06_14ab.png` | Synthesize. Galilean telescope — two-lens angular magnification $M = f_1/f_2$. |
| — | Fig 6.15 | — | Skip. Physical photo (cardboard telescope recreation). Acknowledged in prose. |
| — | Fig 6.16 | — | Skip. Physical photo (moon through the telescope + Galileo's lunar drawings). Acknowledged in prose. |
