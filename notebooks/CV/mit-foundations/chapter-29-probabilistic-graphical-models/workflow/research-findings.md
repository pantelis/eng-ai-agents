# Research findings — Ch 29 Probabilistic Graphical Models

Source: https://visionbook.mit.edu/graphical_models.html (Torralba, Isola, Freeman).

## What the chapter covers
Directed and undirected graphical models; conditional independence; cliques and the
Hammersley-Clifford theorem; Markov chains and 2-D Markov random fields; Bayesian
inference / posterior estimation; **belief propagation** (sum-product message passing)
on trees and loopy graphs; and the **stereo** application. It closes by relating MRF
energies to neural networks.

## The book's figures (from the chapter HTML)
| Fig | src | content |
|---|---|---|
| 29.1 | x1bx2bx3.png | three dependent variables (factorization) |
| 29.2 | cliques1/2.png | clique / maximal clique |
| 29.3 | mrf.png | 2-D MRF: hidden x grid + observation y per node |
| 29.4 | leaf1/leaf2.jpg | image to be segmented + MRF node visualization |
| 29.6-29.9 | 3bpc2, bpmotivator2, bpdiscrete1, bpi.png | BP message-passing rules |
| 29.10-29.11 | man*, rootman*.png | BP update schedules |
| 29.12 | l/r canoe*.jpg | stereo pair + analyzed scanline region |
| 29.13 | stereomodel2.jpg | graphical model for stereo disparity |
| 29.14 | bpcanoes5.png | BP on the stereo scanline: evidence, messages, posterior |
| 29.15 | numerical2.png | worked numerical BP example (exact potentials) |

Most figures are **diagrams** (nodes, cliques, message rules). The computational
content lives in a few places, which this notebook targets:

## What is computationally reproducible (chosen for the notebook)
1. **29.3 MRF prior** — sampled by Gibbs sampling. Nothing to "match" pixel-wise; the
   point is that the prior favours smooth label fields, shown across coupling strengths.
2. **29.15 exact BP** — the figure prints the EXACT binary potentials
   (psi12 = [[1,.9],[.9,1]], psi23 = [[.1,1],[1,.1]], phi2 = [[1,.1],[.1,1]], y2 = 0).
   These reproduce exactly and are checked against brute force.
3. **29.4 segmentation** — the leaf is the book's "image to be segmented"; a two-label
   MRF with a colour data term + loopy BP is the natural computation (the book shows
   the model, not a computed result, so we compute one).
4. **29.14 stereo** — bpcanoes5.png shows a 1-D BP along a scanline with panels for the
   two cameras, the local evidence (position x depth), the two message directions, and
   the posterior. Directly reproducible from the canoe scanline crops.

## Stereo notes (the fiddly part)
- The canoe crops (lcanoe2 / rcanoe2) are **only approximately rectified**: their crop
  origins differ by a large constant horizontal offset (~77 px, found by edge NCC), and
  the marked scanline sits at slightly different rows (L ~233, R ~242). We match rows
  225 / 234 (just above each baked-in black scanline line) and search disparity in a
  band around the global offset.
- The blue hull is nearly textureless, so raw SSD "slides"; **normalized cross-
  correlation** over a window is far more stable. A **truncated-linear** smoothness lets
  BP smooth within a surface while still allowing the depth step at the canoe/grass edge.
- Sign check: the right pixel sits at `x - d` (disparity > 0), confirmed by lower median
  matching cost than `x + d`.

## Decisions
- 5 figures, not the whole chapter — the many pure-diagram figures (cliques, schedules)
  are not computational and are left to the book.
- Everything runs **offline** from bundled assets + synthetic data (no network at run
  time).
