from __future__ import annotations

import math
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
NOTEBOOK = ROOT / "index.ipynb"
README = ROOT / "README.md"

SEED = 7
HEIGHT = 120
WIDTH = 180
FOCAL_LENGTH = 84.0
BASELINE = 0.18
PATCH_SIZE = 7
MAX_DISPARITY = 32


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbf8",
            "axes.edgecolor": "#3f3f46",
            "axes.grid": True,
            "grid.color": "#d6d3d1",
            "grid.alpha": 0.45,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "savefig.dpi": 160,
        }
    )


def depth_to_disparity(depth: np.ndarray, focal_length: float, baseline: float) -> np.ndarray:
    return focal_length * baseline / np.maximum(depth, 1e-6)


def disparity_to_depth(disparity: np.ndarray, focal_length: float, baseline: float) -> np.ndarray:
    return focal_length * baseline / np.maximum(disparity, 1e-6)


def create_synthetic_scene(
    height: int = HEIGHT,
    width: int = WIDTH,
    focal_length: float = FOCAL_LENGTH,
    baseline: float = BASELINE,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]

    depth = np.full((height, width), 8.0, dtype=np.float32)
    region_map = np.full((height, width), "background", dtype=object)

    sky = y < 28
    depth[sky] = 10.0
    region_map[sky] = "sky"

    near_box = (x > 18) & (x < 72) & (y > 26) & (y < 90)
    depth[near_box] = 2.1
    region_map[near_box] = "near_box"

    mid_circle = (x - 118) ** 2 + (y - 52) ** 2 < 20**2
    depth[mid_circle] = 3.5
    region_map[mid_circle] = "mid_circle"

    far_ramp = (x > 86) & (x < 160) & (y > 70) & (y < 108)
    ramp_depth = 3.2 + 2.0 * ((x[far_ramp] - 86) / (160 - 86))
    depth[far_ramp] = np.minimum(depth[far_ramp], ramp_depth.astype(np.float32))
    region_map[far_ramp] = "far_ramp"

    occluder = (x > 82) & (x < 95) & (y > 18) & (y < 95)
    depth[occluder] = 1.6
    region_map[occluder] = "occluder"

    repeated = (x > 120) & (x < 172) & (y > 18) & (y < 62)
    depth[repeated] = 4.2
    region_map[repeated] = "repeated_pattern"

    textureless = (x > 18) & (x < 68) & (y > 90) & (y < 114)
    depth[textureless] = 4.7
    region_map[textureless] = "textureless"

    left = (
        0.28
        + 0.18 * np.sin(x / 9.0)
        + 0.13 * np.cos(y / 11.0)
        + 0.08 * np.sin((x + 1.4 * y) / 13.0)
        + 0.03 * rng.standard_normal((height, width))
    )

    left[near_box] = 0.50 + 0.18 * (((x[near_box] // 6) + (y[near_box] // 6)) % 2) + 0.05 * np.sin(
        y[near_box] / 3.0
    )
    left[mid_circle] = 0.22 + 0.55 * np.exp(-((x[mid_circle] - 118) ** 2 + (y[mid_circle] - 52) ** 2) / 250.0)
    left[far_ramp] = 0.25 + 0.35 * ((x[far_ramp] - 86) / (160 - 86)) + 0.08 * np.sin(y[far_ramp] / 4.0)
    left[occluder] = 0.88 - 0.08 * (((y[occluder] - 18) // 8) % 2)
    left[repeated] = 0.35 + 0.22 * (((x[repeated] - 120) // 5) % 2) + 0.05 * np.cos(y[repeated] / 2.0)
    left[textureless] = 0.63

    left = np.clip(left, 0.0, 1.0).astype(np.float32)
    disparity = depth_to_disparity(depth, focal_length, baseline).astype(np.float32)
    right, visible_mask, occlusion_mask = create_rectified_stereo_pair(left, depth, disparity)

    valid_mask = (disparity > 0.0) & (visible_mask > 0)
    return {
        "left": left,
        "right": right,
        "depth": depth,
        "disparity": disparity,
        "valid_mask": valid_mask,
        "visible_mask": visible_mask,
        "occlusion_mask": occlusion_mask,
        "region_map": region_map,
    }


def create_rectified_stereo_pair(
    left: np.ndarray,
    depth: np.ndarray,
    disparity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = left.shape
    right = np.full_like(left, np.nan, dtype=np.float32)
    z_buffer = np.full_like(left, np.inf, dtype=np.float32)
    visible_mask = np.zeros_like(left, dtype=bool)
    occlusion_mask = np.zeros_like(left, dtype=bool)

    for yy in range(height):
        for xx in range(width):
            disp = int(round(float(disparity[yy, xx])))
            xr = xx - disp
            if xr < 0 or xr >= width:
                occlusion_mask[yy, xx] = True
                continue
            if depth[yy, xx] < z_buffer[yy, xr]:
                if np.isfinite(z_buffer[yy, xr]):
                    occlusion_mask[yy, xx] = True
                right[yy, xr] = left[yy, xx]
                z_buffer[yy, xr] = depth[yy, xx]
                visible_mask[yy, xx] = True

    for yy in range(height):
        row = right[yy]
        finite = np.isfinite(row)
        if finite.any():
            idx = np.where(finite)[0]
            right[yy] = np.interp(np.arange(width), idx, row[idx]).astype(np.float32)
        else:
            right[yy] = 0.0

    right = np.clip(right, 0.0, 1.0)
    return right, visible_mask.astype(np.uint8), occlusion_mask.astype(np.uint8)


def extract_patch(image: np.ndarray, center: tuple[int, int], patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    yy, xx = center
    return image[yy - radius : yy + radius + 1, xx - radius : xx + radius + 1]


def compute_ssd(patch1: np.ndarray, patch2: np.ndarray) -> float:
    diff = patch1 - patch2
    return float(np.sum(diff * diff))


def block_matching_disparity(
    left: np.ndarray,
    right: np.ndarray,
    patch_size: int,
    max_disparity: int,
) -> np.ndarray:
    height, width = left.shape
    radius = patch_size // 2
    disparity = np.zeros((height, width), dtype=np.float32)

    for yy in range(radius, height - radius):
        for xx in range(radius + max_disparity, width - radius):
            left_patch = extract_patch(left, (yy, xx), patch_size)
            best_cost = math.inf
            best_disp = 0
            for disp in range(max_disparity + 1):
                xr = xx - disp
                if xr - radius < 0:
                    break
                right_patch = extract_patch(right, (yy, xr), patch_size)
                cost = compute_ssd(left_patch, right_patch)
                if cost < best_cost:
                    best_cost = cost
                    best_disp = disp
            disparity[yy, xx] = float(best_disp)
    return disparity


def compute_disparity_error(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.abs(pred - gt) * valid_mask


def compute_bad_pixel_ratio(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray,
) -> float:
    denom = max(float(valid_mask.sum()), 1.0)
    bad = ((np.abs(pred - gt) > threshold) & (valid_mask > 0)).sum()
    return float(bad / denom)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGES / name, bbox_inches="tight", dpi=180)
    plt.close(fig)


def generate_figure_stereo_setup() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-2.2, 10.0)
    ax.set_ylim(-3.2, 4.5)
    ax.axis("off")

    left_cam = np.array([0.0, 0.0])
    right_cam = np.array([3.0, 0.0])
    point = np.array([7.0, 2.5])

    for cam, color, label in [
        (left_cam, "#0f766e", "Left camera"),
        (right_cam, "#b45309", "Right camera"),
    ]:
        tri = np.array([cam + [-0.35, -0.45], cam + [-0.35, 0.45], cam + [0.45, 0.0], cam + [-0.35, -0.45]])
        ax.plot(tri[:, 0], tri[:, 1], color=color, lw=2.2)
        ax.text(cam[0], cam[1] - 0.7, label, ha="center", color=color, fontweight="bold")

    ax.plot([left_cam[0], point[0]], [left_cam[1], point[1]], color="#0f766e", lw=2)
    ax.plot([right_cam[0], point[0]], [right_cam[1], point[1]], color="#b45309", lw=2)
    ax.scatter(*point, s=90, color="#7c2d12", zorder=3)
    ax.text(point[0] + 0.15, point[1] + 0.2, "3D point P", color="#7c2d12")

    left_plane_x = 1.15
    right_plane_x = 4.15
    ax.plot([left_plane_x, left_plane_x], [-2.5, 2.7], color="#134e4a", lw=2.2)
    ax.plot([right_plane_x, right_plane_x], [-2.5, 2.7], color="#9a3412", lw=2.2)

    left_proj = np.array([left_plane_x, point[1] * 0.42])
    right_proj = np.array([right_plane_x, point[1] * 0.24])
    ax.scatter(*left_proj, color="#134e4a", s=55)
    ax.scatter(*right_proj, color="#9a3412", s=55)
    ax.text(left_proj[0] + 0.1, left_proj[1] + 0.15, "$x_L$", color="#134e4a")
    ax.text(right_proj[0] + 0.1, right_proj[1] + 0.15, "$x_R$", color="#9a3412")

    ax.annotate(
        "",
        xy=right_cam + [0, -1.6],
        xytext=left_cam + [0, -1.6],
        arrowprops=dict(arrowstyle="<->", lw=1.8, color="#3f3f46"),
    )
    ax.text(1.5, -1.3, "Baseline $B$", ha="center", color="#3f3f46")
    ax.text(6.0, -2.3, "$Z = fB/d$", fontsize=20, color="#111827", fontweight="bold")
    ax.text(
        5.95,
        -2.85,
        "This recreates the book's simple stereo geometry: one 3D point projects\n"
        "to two horizontal image locations, and their disparity reveals depth.",
        fontsize=10.5,
        color="#334155",
    )
    ax.set_title("Stereo Geometry and the Origin of Disparity", pad=10, fontweight="bold")
    save_figure(fig, "01-stereo-setup.png")


def generate_figure_epipolar() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    titles = ["Unrectified pair: search along an epipolar line", "Rectified pair: search collapses to one scanline"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontweight="bold", fontsize=13)

    left_points = np.array([[2.0, 5.2], [3.2, 3.6], [4.1, 1.8]])
    right_points = np.array([[6.9, 5.0], [6.1, 3.3], [5.4, 1.5]])

    for i, (lp, rp) in enumerate(zip(left_points, right_points, strict=True)):
        color = ["#0f766e", "#9333ea", "#ea580c"][i]
        axes[0].scatter(*lp, s=60, color=color)
        axes[0].plot([lp[0], rp[0]], [lp[1], rp[1]], linestyle="--", color=color, lw=1.8)
        axes[0].scatter(*rp, s=60, color=color, marker="s")
        axes[0].text(lp[0] - 0.25, lp[1] + 0.25, f"$p_{i+1}$", color=color)
        axes[0].text(rp[0] + 0.1, rp[1] + 0.2, f"$p'_{i+1}$", color=color)

    for row_y, disp, color in [(5.0, 2.2, "#0f766e"), (3.3, 1.5, "#9333ea"), (1.6, 0.9, "#ea580c")]:
        axes[1].plot([1.0, 9.0], [row_y, row_y], color="#cbd5e1", lw=2)
        axes[1].scatter(3.4, row_y, s=60, color=color)
        axes[1].scatter(3.4 + disp, row_y, s=60, color=color, marker="s")
        axes[1].annotate(
            "",
            xy=(3.4 + disp, row_y - 0.28),
            xytext=(3.4, row_y - 0.28),
            arrowprops=dict(arrowstyle="<->", lw=1.5, color=color),
        )
        axes[1].text(3.4 + disp / 2, row_y - 0.75, f"$d={disp:.1f}$ px", ha="center", color=color)

    axes[1].text(
        0.9,
        6.1,
        "Rectification keeps corresponding points on the same row,\n"
        "which makes stereo search one-dimensional.",
        color="#334155",
        fontsize=10.5,
    )
    save_figure(fig, "02-epipolar-and-rectification.png")


def generate_figure_disparity_depth() -> None:
    disparities = np.linspace(1.0, 36.0, 300)
    depths = disparity_to_depth(disparities, FOCAL_LENGTH, BASELINE)
    sample_disp = np.array([4.0, 8.0, 16.0, 28.0])
    sample_depth = disparity_to_depth(sample_disp, FOCAL_LENGTH, BASELINE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(disparities, depths, color="#0f766e", lw=2.7)
    axes[0].scatter(sample_disp, sample_depth, color="#b91c1c", s=55, zorder=3)
    for disp, depth in zip(sample_disp, sample_depth, strict=True):
        axes[0].text(disp + 0.5, depth + 0.1, f"({disp:.0f}, {depth:.2f}m)", fontsize=9)
    axes[0].set_xlabel("Disparity d (pixels)")
    axes[0].set_ylabel("Depth Z (scene units)")
    axes[0].set_title("Inverse Depth Relationship", fontweight="bold")

    axes[1].bar(["far", "mid", "near"], [4.0, 10.0, 22.0], color=["#94a3b8", "#f59e0b", "#0f766e"], width=0.6)
    axes[1].set_ylim(0, 25)
    axes[1].set_ylabel("Illustrative disparity (pixels)")
    axes[1].set_title("Nearer points move more between views", fontweight="bold")
    axes[1].text(
        0.03,
        0.95,
        "$Z = fB / d$\nLarge disparity -> small depth\nSmall disparity -> large depth",
        transform=axes[1].transAxes,
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    save_figure(fig, "03-disparity-depth-relationship.png")


def generate_figure_synthetic_pair(scene: dict[str, np.ndarray]) -> None:
    left = scene["left"]
    right = scene["right"]
    disparity = scene["disparity"]
    yy, xx = 48, 56
    disp = int(round(float(disparity[yy, xx])))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, image, title in [
        (axes[0], left, "Left view"),
        (axes[1], right, "Right view"),
    ]:
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    axes[0].scatter([xx], [yy], s=70, facecolors="none", edgecolors="#ef4444", linewidths=2)
    axes[1].scatter([xx - disp], [yy], s=70, facecolors="none", edgecolors="#ef4444", linewidths=2)
    axes[0].text(xx + 4, yy - 6, f"match source\n d={disp}px", color="#ef4444", fontsize=10)
    axes[1].text(xx - disp + 4, yy - 6, "same world point", color="#ef4444", fontsize=10)
    save_figure(fig, "04-synthetic-stereo-pair.png")


def run_reference_estimator(scene: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    start = time.perf_counter()
    pred = block_matching_disparity(scene["left"], scene["right"], PATCH_SIZE, MAX_DISPARITY)
    runtime = time.perf_counter() - start

    valid_mask = scene["valid_mask"].astype(np.float32)
    error = compute_disparity_error(pred, scene["disparity"], valid_mask)
    mae = float(error.sum() / max(valid_mask.sum(), 1.0))
    bad_1px = compute_bad_pixel_ratio(pred, scene["disparity"], 1.0, valid_mask)
    return {
        "pred": pred,
        "error": error,
        "mae": mae,
        "bad_1px": bad_1px,
        "runtime": runtime,
    }


def generate_figure_block_matching(scene: dict[str, np.ndarray], metrics: dict[str, np.ndarray | float]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    maps = [
        (scene["disparity"], "Ground-truth disparity", "viridis"),
        (metrics["pred"], f"Estimated disparity\npatch={PATCH_SIZE}, max d={MAX_DISPARITY}", "viridis"),
        (metrics["error"], f"Absolute error\nMAE={metrics['mae']:.2f}px", "magma"),
    ]

    for ax, (image, title, cmap) in zip(axes, maps, strict=True):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_figure(fig, "05-block-matching-results.png")


def parameter_sweep(scene: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    patch_sizes = [3, 5, 7, 9]
    max_disparities = [16, 24, 32, 40]
    mae_grid = np.zeros((len(patch_sizes), len(max_disparities)), dtype=np.float32)
    bad_grid = np.zeros_like(mae_grid)
    time_grid = np.zeros_like(mae_grid)
    valid_mask = scene["valid_mask"].astype(np.float32)

    for i, patch_size in enumerate(patch_sizes):
        for j, max_disp in enumerate(max_disparities):
            start = time.perf_counter()
            pred = block_matching_disparity(scene["left"], scene["right"], patch_size, max_disp)
            elapsed = time.perf_counter() - start
            err = compute_disparity_error(pred, scene["disparity"], valid_mask)
            mae_grid[i, j] = err.sum() / max(valid_mask.sum(), 1.0)
            bad_grid[i, j] = compute_bad_pixel_ratio(pred, scene["disparity"], 1.0, valid_mask)
            time_grid[i, j] = elapsed

    return {
        "patch_sizes": np.array(patch_sizes),
        "max_disparities": np.array(max_disparities),
        "mae_grid": mae_grid,
        "bad_grid": bad_grid,
        "time_grid": time_grid,
    }


def generate_figure_parameters_and_failures(
    scene: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray | float],
    sweep: dict[str, np.ndarray],
) -> None:
    pred = np.asarray(metrics["pred"])
    error = np.asarray(metrics["error"])
    region_map = scene["region_map"]

    texture_mask = region_map == "textureless"
    repeated_mask = region_map == "repeated_pattern"
    boundary_mask = scene["occlusion_mask"] > 0
    valid_mask = scene["valid_mask"] > 0

    failure_scores = []
    for mask in [texture_mask, repeated_mask, boundary_mask]:
        local_mask = mask & valid_mask
        if local_mask.any():
            failure_scores.append(float(error[local_mask].mean()))
        else:
            failure_scores.append(0.0)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0])

    ax_mae = fig.add_subplot(gs[0, 0])
    im1 = ax_mae.imshow(sweep["mae_grid"], cmap="viridis")
    ax_mae.set_title("Mean absolute disparity error", fontweight="bold")
    ax_mae.set_xticks(range(len(sweep["max_disparities"])), sweep["max_disparities"])
    ax_mae.set_yticks(range(len(sweep["patch_sizes"])), sweep["patch_sizes"])
    ax_mae.set_xlabel("Max disparity")
    ax_mae.set_ylabel("Patch size")
    fig.colorbar(im1, ax=ax_mae, fraction=0.046, pad=0.04)

    ax_bad = fig.add_subplot(gs[0, 1])
    im2 = ax_bad.imshow(sweep["bad_grid"], cmap="magma", vmin=0.0, vmax=max(0.25, float(sweep["bad_grid"].max())))
    ax_bad.set_title("Bad-pixel ratio (> 1 px)", fontweight="bold")
    ax_bad.set_xticks(range(len(sweep["max_disparities"])), sweep["max_disparities"])
    ax_bad.set_yticks(range(len(sweep["patch_sizes"])), sweep["patch_sizes"])
    ax_bad.set_xlabel("Max disparity")
    ax_bad.set_ylabel("Patch size")
    fig.colorbar(im2, ax=ax_bad, fraction=0.046, pad=0.04)

    ax_bar = fig.add_subplot(gs[0, 2])
    labels = ["Textureless", "Repeated", "Occlusion"]
    colors = ["#94a3b8", "#f59e0b", "#dc2626"]
    ax_bar.bar(labels, failure_scores, color=colors)
    ax_bar.set_ylabel("Local MAE (pixels)")
    ax_bar.set_title("Classic failure modes remain hardest", fontweight="bold")
    ax_bar.tick_params(axis="x", rotation=12)

    crops = [
        ((92, 18, 22, 48), "Textureless region"),
        ((22, 122, 34, 44), "Repeated pattern"),
        ((25, 76, 48, 34), "Occlusion boundary"),
    ]

    for idx, (spec, title) in enumerate(crops):
        ax = fig.add_subplot(gs[1, idx])
        y0, x0, h, w = spec
        crop = scene["left"][y0 : y0 + h, x0 : x0 + w]
        crop_err = error[y0 : y0 + h, x0 : x0 + w]
        ax.imshow(crop, cmap="gray", vmin=0.0, vmax=1.0)
        ax.imshow(crop_err, cmap="inferno", alpha=0.65)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    save_figure(fig, "06-parameter-and-failure-cases.png")


def notebook_markdown_cells() -> list[nbf.NotebookNode]:
    intro = textwrap.dedent(
        """
        # Chapter 40: Stereo Vision

        This notebook builds a compact, reproducible stereo-vision pipeline around the main conceptual arc of MIT *Foundations of Computer Vision* Chapter 40: intuitive stereo cues, binocular geometry, disparity, epipolar constraints, rectification, correspondence, validation, and failure modes.

        ## Connection to the MIT Vision Book

        The conceptual source for this notebook is Chapter 40 of the MIT Vision Book: [Stereo Vision](https://visionbook.mit.edu/3d_scene_understanding_stereo.html). Rather than copying textbook figures, the notebook recreates the core ideas with original code-generated diagrams, synthetic scenes, and quantitative experiments.

        The notebook mirrors the chapter's understanding path:

        1. Intuition first: what extra information a second view provides.
        2. Geometry next: why correspondence is constrained and why disparity implies inverse depth.
        3. Computation next: how local block matching turns that geometry into a disparity map.
        4. Validation last: when the method works, when it fails, and why.
        """
    ).strip()

    sections = [
        (
            "## Intuition: What information does a second view provide?\n\n"
            "This section recreates the intuition behind the book's simple stereo setup figure. "
            "A single 3D point projects to different horizontal positions in the left and right cameras, and the horizontal offset becomes a cue for depth."
        ),
        "![Stereo setup](images/01-stereo-setup.png)",
        (
            "## Epipolar Geometry and Rectification\n\n"
            "Here we reproduce the book's epipolar-search idea. For arbitrary camera poses, a match in the second view must lie on an epipolar line. After rectification, that search becomes one-dimensional along a scanline."
        ),
        "![Epipolar geometry and rectification](images/02-epipolar-and-rectification.png)",
        (
            "## Why is disparity a proxy for inverse depth?\n\n"
            "For a rectified stereo pair with focal length $f$ and baseline $B$, the textbook depth formula is\n\n"
            "$$Z = \\frac{fB}{d}$$\n\n"
            "where $Z$ is depth and $d = x_L - x_R$ is disparity. Nearby points generate large disparities, while faraway points generate small ones."
        ),
        "![Disparity-depth relationship](images/03-disparity-depth-relationship.png)",
        (
            "## Synthetic Rectified Stereo Pair\n\n"
            "The experiments below use synthetic data first so ground-truth depth and disparity are known exactly. This makes the validation honest and reproducible, and it keeps the pipeline aligned with the chapter's geometry-first emphasis."
        ),
        "![Synthetic stereo pair](images/04-synthetic-stereo-pair.png)",
        (
            "## Block Matching\n\n"
            "The stereo correspondence problem asks: for each left-image patch, where is the matching patch in the right image? In rectified stereo that search stays on the same row. Local block matching works best when there is enough local texture to disambiguate the match."
        ),
        "![Block matching results](images/05-block-matching-results.png)",
        (
            "## Validation, Parameter Tradeoffs, and Failure Cases\n\n"
            "This section makes the chapter's caveats computationally explicit. We report mean absolute disparity error and bad-pixel ratio, then show how patch size and search range change the results. Finally, we isolate three classic failure modes: textureless regions, repeated patterns, and occlusion boundaries."
        ),
        "![Parameter sweep and failure cases](images/06-parameter-and-failure-cases.png)",
        (
            "## Limits of This Demonstration\n\n"
            "This notebook is deliberately compact. It demonstrates the geometry and mechanics of stereo matching, not a production-quality stereo system. It omits learned cost volumes, left-right consistency checks, subpixel refinement, regularization, and robust photometric modeling."
        ),
    ]

    return [nbf.v4.new_markdown_cell(intro)] + [nbf.v4.new_markdown_cell(text) for text in sections]


def notebook_code_cells() -> list[nbf.NotebookNode]:
    cell1 = textwrap.dedent(
        """
        # Parameters (overridden by papermill when executed through the repo tooling)
        output_dir = "."
        images_dir = "./images"
        """
    ).strip()

    cell2 = textwrap.dedent(
        """
        import math
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        torch.manual_seed(7)
        np.random.seed(7)

        OUTPUT_DIR = Path(output_dir)
        IMAGES_DIR = Path(images_dir)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "#fbfbf8",
                "axes.edgecolor": "#3f3f46",
                "axes.grid": True,
                "grid.color": "#d6d3d1",
                "grid.alpha": 0.45,
                "grid.linestyle": "--",
                "font.size": 11,
            }
        )

        DEVICE = torch.device("cpu")
        FOCAL_LENGTH = 84.0
        BASELINE = 0.18
        HEIGHT = 120
        WIDTH = 180
        PATCH_SIZE = 7
        MAX_DISPARITY = 32
        """
    ).strip()

    cell3 = textwrap.dedent(
        """
        def depth_to_disparity(depth, focal_length, baseline):
            return focal_length * baseline / torch.clamp(depth, min=1e-6)


        def disparity_to_depth(disparity, focal_length, baseline):
            return focal_length * baseline / torch.clamp(disparity, min=1e-6)


        def create_synthetic_scene(height=HEIGHT, width=WIDTH, focal_length=FOCAL_LENGTH, baseline=BASELINE):
            y, x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=DEVICE),
                torch.arange(width, dtype=torch.float32, device=DEVICE),
                indexing="ij",
            )

            depth = torch.full((height, width), 8.0, dtype=torch.float32, device=DEVICE)
            left = (
                0.28
                + 0.18 * torch.sin(x / 9.0)
                + 0.13 * torch.cos(y / 11.0)
                + 0.08 * torch.sin((x + 1.4 * y) / 13.0)
                + 0.03 * torch.randn((height, width), device=DEVICE)
            )

            near_box = (x > 18) & (x < 72) & (y > 26) & (y < 90)
            mid_circle = (x - 118) ** 2 + (y - 52) ** 2 < 20**2
            far_ramp = (x > 86) & (x < 160) & (y > 70) & (y < 108)
            occluder = (x > 82) & (x < 95) & (y > 18) & (y < 95)
            repeated = (x > 120) & (x < 172) & (y > 18) & (y < 62)
            textureless = (x > 18) & (x < 68) & (y > 90) & (y < 114)

            depth[y < 28] = 10.0
            depth[near_box] = 2.1
            depth[mid_circle] = 3.5
            depth[far_ramp] = torch.minimum(
                depth[far_ramp],
                3.2 + 2.0 * ((x[far_ramp] - 86.0) / (160.0 - 86.0)),
            )
            depth[occluder] = 1.6
            depth[repeated] = 4.2
            depth[textureless] = 4.7

            left[near_box] = 0.50 + 0.18 * ((((x[near_box] // 6) + (y[near_box] // 6)) % 2)) + 0.05 * torch.sin(
                y[near_box] / 3.0
            )
            left[mid_circle] = 0.22 + 0.55 * torch.exp(-((x[mid_circle] - 118) ** 2 + (y[mid_circle] - 52) ** 2) / 250.0)
            left[far_ramp] = 0.25 + 0.35 * ((x[far_ramp] - 86.0) / (160.0 - 86.0)) + 0.08 * torch.sin(
                y[far_ramp] / 4.0
            )
            left[occluder] = 0.88 - 0.08 * ((((y[occluder] - 18) // 8) % 2))
            left[repeated] = 0.35 + 0.22 * ((((x[repeated] - 120) // 5) % 2)) + 0.05 * torch.cos(
                y[repeated] / 2.0
            )
            left[textureless] = 0.63
            left = torch.clamp(left, 0.0, 1.0)

            disparity = depth_to_disparity(depth, focal_length, baseline)
            right, visible_mask, occlusion_mask = create_rectified_stereo_pair(left, depth, disparity)
            valid_mask = (disparity > 0) & visible_mask

            region_map = {
                "textureless": textureless,
                "repeated_pattern": repeated,
                "occlusion_boundary": occlusion_mask,
            }
            return {
                "left": left,
                "right": right,
                "depth": depth,
                "disparity": disparity,
                "valid_mask": valid_mask,
                "visible_mask": visible_mask,
                "occlusion_mask": occlusion_mask,
                "region_map": region_map,
            }


        def create_rectified_stereo_pair(left, depth, disparity):
            height, width = left.shape
            right = torch.full_like(left, float("nan"))
            z_buffer = torch.full_like(left, float("inf"))
            visible_mask = torch.zeros_like(left, dtype=torch.bool)
            occlusion_mask = torch.zeros_like(left, dtype=torch.bool)

            for yy in range(height):
                for xx in range(width):
                    disp = int(torch.round(disparity[yy, xx]).item())
                    xr = xx - disp
                    if xr < 0 or xr >= width:
                        occlusion_mask[yy, xx] = True
                        continue
                    if depth[yy, xx] < z_buffer[yy, xr]:
                        right[yy, xr] = left[yy, xx]
                        z_buffer[yy, xr] = depth[yy, xx]
                        visible_mask[yy, xx] = True

            right_np = right.cpu().numpy()
            for yy in range(height):
                row = right_np[yy]
                finite = np.isfinite(row)
                if finite.any():
                    idx = np.where(finite)[0]
                    right_np[yy] = np.interp(np.arange(width), idx, row[idx])
                else:
                    right_np[yy] = 0.0
            right = torch.from_numpy(right_np).to(DEVICE, dtype=torch.float32)
            return torch.clamp(right, 0.0, 1.0), visible_mask, occlusion_mask


        def extract_patch(image, center, patch_size):
            radius = patch_size // 2
            yy, xx = center
            return image[yy - radius : yy + radius + 1, xx - radius : xx + radius + 1]


        def compute_ssd(patch1, patch2):
            diff = patch1 - patch2
            return torch.sum(diff * diff)


        def block_matching_disparity(left, right, patch_size, max_disparity):
            height, width = left.shape
            radius = patch_size // 2
            disparity = torch.zeros_like(left)
            for yy in range(radius, height - radius):
                for xx in range(radius + max_disparity, width - radius):
                    left_patch = extract_patch(left, (yy, xx), patch_size)
                    best_cost = float("inf")
                    best_disp = 0
                    for disp in range(max_disparity + 1):
                        xr = xx - disp
                        if xr - radius < 0:
                            break
                        right_patch = extract_patch(right, (yy, xr), patch_size)
                        cost = compute_ssd(left_patch, right_patch).item()
                        if cost < best_cost:
                            best_cost = cost
                            best_disp = disp
                    disparity[yy, xx] = float(best_disp)
            return disparity


        def compute_disparity_error(pred, gt, valid_mask):
            return torch.abs(pred - gt) * valid_mask.float()


        def compute_bad_pixel_ratio(pred, gt, threshold, valid_mask):
            bad = ((torch.abs(pred - gt) > threshold) & valid_mask).float().sum()
            total = valid_mask.float().sum().clamp(min=1.0)
            return (bad / total).item()
        """
    ).strip()

    cell4 = textwrap.dedent(
        """
        def save_current_figure(name):
            plt.savefig(IMAGES_DIR / name, dpi=180, bbox_inches="tight")
            plt.show()
            plt.close()


        scene = create_synthetic_scene()
        print(
            "Synthetic scene created:",
            {
                "image_shape": tuple(scene["left"].shape),
                "disparity_range": (
                    float(scene["disparity"][scene["valid_mask"]].min().item()),
                    float(scene["disparity"][scene["valid_mask"]].max().item()),
                ),
                "depth_range": (
                    float(scene["depth"].min().item()),
                    float(scene["depth"].max().item()),
                ),
            },
        )
        """
    ).strip()

    cell5 = textwrap.dedent(
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(plt.imread(IMAGES_DIR / "01-stereo-setup.png"))
        ax.axis("off")
        plt.title("Recreated stereo setup intuition")
        plt.show()
        """
    ).strip()

    cell6 = textwrap.dedent(
        """
        fig, ax = plt.subplots(figsize=(10, 4.4))
        ax.imshow(plt.imread(IMAGES_DIR / "02-epipolar-and-rectification.png"))
        ax.axis("off")
        plt.title("Epipolar constraint and why rectification matters")
        plt.show()
        """
    ).strip()

    cell7 = textwrap.dedent(
        """
        disparities = torch.linspace(1.0, 36.0, 300)
        depths = disparity_to_depth(disparities, FOCAL_LENGTH, BASELINE)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(disparities.cpu(), depths.cpu(), color="#0f766e", lw=2.7)
        axes[0].set_xlabel("Disparity d (pixels)")
        axes[0].set_ylabel("Depth Z")
        axes[0].set_title("Inverse relationship from the textbook formula")

        sample_disp = torch.tensor([4.0, 8.0, 16.0, 28.0])
        sample_depth = disparity_to_depth(sample_disp, FOCAL_LENGTH, BASELINE)
        axes[1].bar(["far", "mid", "near"], [4.0, 10.0, 22.0], color=["#94a3b8", "#f59e0b", "#0f766e"])
        axes[1].set_ylabel("Illustrative disparity")
        axes[1].set_title("Why nearby points move more")
        save_current_figure("03-disparity-depth-relationship.png")
        """
    ).strip()

    cell8 = textwrap.dedent(
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        axes[0].imshow(scene["left"].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("Left rectified image")
        axes[0].axis("off")

        axes[1].imshow(scene["right"].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title("Right rectified image")
        axes[1].axis("off")
        save_current_figure("04-synthetic-stereo-pair.png")
        """
    ).strip()

    cell9 = textwrap.dedent(
        """
        start = time.perf_counter()
        pred_disparity = block_matching_disparity(scene["left"], scene["right"], PATCH_SIZE, MAX_DISPARITY)
        runtime_seconds = time.perf_counter() - start

        error_map = compute_disparity_error(pred_disparity, scene["disparity"], scene["valid_mask"])
        mae = (
            error_map.sum() / scene["valid_mask"].float().sum().clamp(min=1.0)
        ).item()
        bad_1px = compute_bad_pixel_ratio(pred_disparity, scene["disparity"], threshold=1.0, valid_mask=scene["valid_mask"])

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, image, title, cmap in [
            (axes[0], scene["disparity"], "Ground-truth disparity", "viridis"),
            (axes[1], pred_disparity, f"Estimated disparity\\npatch={PATCH_SIZE}, max d={MAX_DISPARITY}", "viridis"),
            (axes[2], error_map, f"Absolute error\\nMAE={mae:.2f}px", "magma"),
        ]:
            im = ax.imshow(image.cpu(), cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_current_figure("05-block-matching-results.png")

        print({"mae": mae, "bad_pixel_ratio_1px": bad_1px, "runtime_seconds": runtime_seconds})
        """
    ).strip()

    cell10 = textwrap.dedent(
        """
        patch_sizes = [3, 5, 7, 9]
        max_disparities = [16, 24, 32, 40]
        mae_grid = torch.zeros((len(patch_sizes), len(max_disparities)))
        bad_grid = torch.zeros_like(mae_grid)
        runtime_grid = torch.zeros_like(mae_grid)

        for i, patch_size in enumerate(patch_sizes):
            for j, max_disp in enumerate(max_disparities):
                start = time.perf_counter()
                pred = block_matching_disparity(scene["left"], scene["right"], patch_size, max_disp)
                runtime_grid[i, j] = time.perf_counter() - start
                err = compute_disparity_error(pred, scene["disparity"], scene["valid_mask"])
                mae_grid[i, j] = err.sum() / scene["valid_mask"].float().sum().clamp(min=1.0)
                bad_grid[i, j] = compute_bad_pixel_ratio(pred, scene["disparity"], threshold=1.0, valid_mask=scene["valid_mask"])

        texture_mask = scene["region_map"]["textureless"] & scene["valid_mask"]
        repeated_mask = scene["region_map"]["repeated_pattern"] & scene["valid_mask"]
        boundary_mask = scene["occlusion_mask"] & scene["valid_mask"]
        failure_mae = []
        for mask in [texture_mask, repeated_mask, boundary_mask]:
            if mask.any():
                failure_mae.append(error_map[mask].mean().item())
            else:
                failure_mae.append(0.0)

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0])

        ax_mae = fig.add_subplot(gs[0, 0])
        im1 = ax_mae.imshow(mae_grid.cpu(), cmap="viridis")
        ax_mae.set_title("Mean absolute disparity error")
        ax_mae.set_xticks(range(len(max_disparities)), max_disparities)
        ax_mae.set_yticks(range(len(patch_sizes)), patch_sizes)
        ax_mae.set_xlabel("Max disparity")
        ax_mae.set_ylabel("Patch size")
        fig.colorbar(im1, ax=ax_mae, fraction=0.046, pad=0.04)

        ax_bad = fig.add_subplot(gs[0, 1])
        im2 = ax_bad.imshow(bad_grid.cpu(), cmap="magma")
        ax_bad.set_title("Bad-pixel ratio (> 1 px)")
        ax_bad.set_xticks(range(len(max_disparities)), max_disparities)
        ax_bad.set_yticks(range(len(patch_sizes)), patch_sizes)
        ax_bad.set_xlabel("Max disparity")
        ax_bad.set_ylabel("Patch size")
        fig.colorbar(im2, ax=ax_bad, fraction=0.046, pad=0.04)

        ax_bar = fig.add_subplot(gs[0, 2])
        ax_bar.bar(["Textureless", "Repeated", "Occlusion"], failure_mae, color=["#94a3b8", "#f59e0b", "#dc2626"])
        ax_bar.set_title("Failure-mode MAE")
        ax_bar.set_ylabel("MAE (pixels)")

        for idx, (y0, x0, h, w, title) in enumerate(
            [
                (92, 18, 22, 48, "Textureless region"),
                (22, 122, 34, 44, "Repeated pattern"),
                (25, 76, 48, 34, "Occlusion boundary"),
            ]
        ):
            ax = fig.add_subplot(gs[1, idx])
            crop = scene["left"][y0 : y0 + h, x0 : x0 + w].cpu()
            crop_err = error_map[y0 : y0 + h, x0 : x0 + w].cpu()
            ax.imshow(crop, cmap="gray", vmin=0.0, vmax=1.0)
            ax.imshow(crop_err, cmap="inferno", alpha=0.65)
            ax.set_title(title)
            ax.axis("off")

        save_current_figure("06-parameter-and-failure-cases.png")

        summary = {
            "patch_sizes": patch_sizes,
            "max_disparities": max_disparities,
            "mae_grid": mae_grid.tolist(),
            "bad_grid": bad_grid.tolist(),
            "runtime_grid_seconds": runtime_grid.tolist(),
            "failure_mae": failure_mae,
        }
        summary
        """
    ).strip()

    cell11 = textwrap.dedent(
        """
        metrics = {
            "mean_absolute_disparity_error": mae,
            "bad_pixel_ratio_at_1px": bad_1px,
            "depth_mae": (
                torch.abs(
                    disparity_to_depth(torch.clamp(pred_disparity, min=1.0), FOCAL_LENGTH, BASELINE)
                    - scene["depth"]
                )[scene["valid_mask"]].mean().item()
            ),
            "runtime_seconds": runtime_seconds,
        }
        metrics
        """
    ).strip()

    return [nbf.v4.new_code_cell(code) for code in [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10, cell11]]


def generate_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = notebook_markdown_cells() + notebook_code_cells()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    NOTEBOOK.write_text(nbf.writes(nb), encoding="utf-8")


def generate_readme(metrics: dict[str, np.ndarray | float], sweep: dict[str, np.ndarray]) -> None:
    text = f"""# Chapter 40: Stereo Vision

This chapter folder contains an educational stereo-vision notebook that recreates the core ideas from MIT *Foundations of Computer Vision* Chapter 40 using original code-generated figures and a deterministic synthetic stereo scene.

## What the notebook demonstrates

- Why a second view reveals depth through disparity.
- How binocular geometry leads to the rectified stereo depth formula `Z = fB / d`.
- Why epipolar geometry constrains correspondence search.
- How simple local block matching estimates a disparity map.
- How to validate stereo output with ground-truth disparity and failure-case analysis.

## Relation to the MIT Vision Book

The notebook is grounded in [Chapter 40: Stereo Vision](https://visionbook.mit.edu/3d_scene_understanding_stereo.html). It recreates the chapter's main figure themes without copying textbook images:

- Stereo camera setup and triangulation intuition
- Epipolar geometry and rectification
- Disparity-depth relationship
- Pixel-level stereo correspondence on a rectified pair
- Quantitative validation and known stereo failure modes

## Main concepts covered

- Stereo vision intuition
- Binocular geometry
- Disparity
- Depth from disparity
- Epipolar geometry
- Rectified stereo correspondence
- Simple SSD block matching
- Disparity-to-depth reconstruction
- Parameter tradeoffs
- Validation metrics
- Failure cases and limitations

## Generated figures

- `images/01-stereo-setup.png`
- `images/02-epipolar-and-rectification.png`
- `images/03-disparity-depth-relationship.png`
- `images/04-synthetic-stereo-pair.png`
- `images/05-block-matching-results.png`
- `images/06-parameter-and-failure-cases.png`

## How to run

From the repository root:

```bash
python notebooks/CV/2026/spring/final/chapter-40-stereo-vision/build_assets.py
make execute-notebook NOTEBOOK=CV/2026/spring/final/chapter-40-stereo-vision/index.ipynb
```

The first command regenerates the PNG assets and notebook structure. The second uses the repository's notebook execution flow inside the configured container environment.

## Expected outputs

- A synthetic left/right rectified stereo pair
- Ground-truth disparity and depth maps
- A block-matching disparity estimate
- Quantitative metrics and a parameter sweep
- Visualized failure cases for textureless, repeated, and occluded regions

## Validation metrics

- Mean absolute disparity error: `{metrics["mae"]:.3f}` pixels
- Bad-pixel ratio at 1 pixel: `{metrics["bad_1px"]:.3f}`
- Reference runtime for the default setting: `{metrics["runtime"]:.3f}` seconds

Best parameter-sweep MAE in the generated grid: `{float(np.min(sweep["mae_grid"])):.3f}` pixels

## Limitations

- The scene is synthetic and intentionally simple so that ground truth is known.
- The correspondence algorithm is local SSD block matching, not a production stereo method.
- The notebook does not implement subpixel refinement, global regularization, or learned stereo networks.
"""
    README.write_text(text, encoding="utf-8")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    set_style()
    scene = create_synthetic_scene()
    generate_figure_stereo_setup()
    generate_figure_epipolar()
    generate_figure_disparity_depth()
    generate_figure_synthetic_pair(scene)
    metrics = run_reference_estimator(scene)
    sweep = parameter_sweep(scene)
    generate_figure_block_matching(scene, metrics)
    generate_figure_parameters_and_failures(scene, metrics, sweep)
    generate_notebook()
    generate_readme(metrics, sweep)
    print(
        {
            "notebook": str(NOTEBOOK),
            "readme": str(README),
            "images": sorted(path.name for path in IMAGES.glob("*.png")),
            "mae": round(float(metrics["mae"]), 4),
            "bad_1px": round(float(metrics["bad_1px"]), 4),
        }
    )


if __name__ == "__main__":
    main()
