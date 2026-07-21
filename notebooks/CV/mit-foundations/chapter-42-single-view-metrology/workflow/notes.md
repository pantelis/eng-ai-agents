# Chapter 42 notes — gotchas and design decisions

## The cross-ratio argument order trap

The generic helper `cross_ratio(p1, p2, p3, p4)` returns
$$\mathrm{CR} = \frac{|p_3 - p_1|\,|p_4 - p_2|}{|p_3 - p_2|\,|p_4 - p_1|}.$$

The book's Algorithm 2 equation rearranges to
$$\frac{h_{\text{desk}}}{h_{\text{bookshelf}}} = \frac{|b - g_b|\,|t_b - v_3|}{|b - v_3|\,|t_b - g_b|}.$$

Those are NOT the same; mapping `cross_ratio(g_b, t_b, b, v3)` to that ratio
expression doesn't actually work out. Easier and safer to inline the four
distances directly per the book's equation, instead of trying to pick the
right argument permutation for the generic helper.

The first implementation attempt forgot this and recovered desk height
**313 cm** (vs ground-truth 76 cm) — a factor of ~4 wrong, exactly what you'd
expect from inverting the wrong cross-ratio direction. Bug caught immediately
because we had a synthetic scene with known ground truth.

## Why three vanishing points perfectly recover K

The chapter says "three orthogonality constraints, four unknowns in $\mathbf{W}$,
SVD gives a one-parameter family scaled by $\mathbf{W}_{33}=1$." On a real
photograph the recovered K is approximate (vanishing-point estimation is
noisy). On our synthetic scene the vanishing points are exact (we computed
them via $\mathbf{K}\mathbf{R}\mathbf{D}$), so the SVD's smallest singular
value is exactly zero and the recovered $\mathbf{K}$ matches `K_true` to
machine precision. The published output shows 0.000% relative error.

## Cholesky needs positive-definiteness

`torch.linalg.cholesky(W)` will throw if $\mathbf{W}$ has a negative
eigenvalue. On synthetic data the SVD might return $-\mathbf{w}$ instead of
$\mathbf{w}$ (the sign of the null vector is arbitrary). We sign-flip
$\mathbf{W}$ if $W_{00} < 0$ before Cholesky to handle this.

## look_at convention

`look_at(eye, target, up)` returns $(\mathbf{R}, \text{eye})$ such that the
camera-from-world transform is $\mathbf{P}_c = \mathbf{R}(\mathbf{P}_w - \text{eye})$.
The rows of $\mathbf{R}$ are the camera basis (right, up, forward) in world
coordinates. This convention is the standard "look-at" used in 3D graphics
APIs (OpenGL, etc.), so it composes cleanly with intrinsics $\mathbf{K}$:
$$\mathbf{p}_{\text{img}} = \mathbf{K} \mathbf{P}_c.$$

If you ever see vanishing points coming out flipped, double-check the up
vector convention — Y-up vs Z-up swaps sign on the vertical vanishing point.

## What we left to a future revision

- **A real photograph-based version of Algorithm 2/3.** Same algorithm,
  contributor supplies an AI-generated office image with known dimensions and
  manually clicks the four corner pixels. Pedagogically equivalent to what
  we ship; just a visual richer.
- **Figure 42.18 (axis calibration tick marks).** Algorithmically simple
  (repeated cross-ratio applications) but visually requires a separate
  diagram. Not in critical path.
- **Algorithm 5 (3D point localisation).** Covered conceptually by Fig 42.20
  but no end-to-end "click a point, read its world coordinates" cell.
