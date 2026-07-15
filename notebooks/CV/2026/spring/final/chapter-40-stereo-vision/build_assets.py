from __future__ import annotations

import json
import textwrap
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = CHAPTER_DIR / "index.ipynb"
README_PATH = CHAPTER_DIR / "README.md"


def markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip() + "\n",
    }


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip() + "\n",
    }


def notebook_dict(cells: list[dict[str, object]]) -> dict[str, object]:
    notebook_cells = []
    for idx, cell in enumerate(cells):
        notebook_cells.append({**cell, "id": f"cell-{idx:02d}"})
    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_notebook() -> dict[str, object]:
    md0 = """
    # Chapter 40: Stereo Vision

    ## 1. Introduction and MIT Book context

    This notebook stays close to one compact Chapter 40 story: a **rectified stereo pair**,
    **disparity as a depth cue**, and an **educational SSD block matcher** that we can inspect
    quantitatively. The goal is to connect a small number of MIT *Foundations of Computer Vision*
    reference figures to one runnable synthetic experiment, not to cover every stereo topic in
    the chapter.

    **MIT Vision Book reference — Figure 40.1: Stereo anaglyph motivation.**

    Source: MIT *Foundations of Computer Vision*, Chapter 40.

    ![MIT Vision Book Figure 40.1](assets/mit-book/figure-40-01-titanic.png)
    """

    code0 = """
    import json
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from IPython.display import display

    SEED = 7
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)

    OUTPUT_DIR = Path(globals().get("output_dir", "./output"))
    IMAGES_DIR = Path(globals().get("images_dir", "./images"))
    ASSETS_DIR = Path("assets/mit-book")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbf8",
            "axes.edgecolor": "#3f3f46",
            "figure.figsize": (6, 4),
            "font.size": 11,
            "image.cmap": "gray",
            "axes.grid": False,
        }
    )

    FOCAL_LENGTH_PX = 72.0
    BASELINE_UNITS = 0.24
    HEIGHT = 96
    WIDTH = 144
    DEFAULT_PATCH_SIZE = 7
    DEFAULT_MAX_DISPARITY = 12
    COST_TYPE = "ssd"
    DISPARITY_CMAP = "viridis"
    ERROR_CMAP = "magma"


    def save_figure(fig, name):
        path = IMAGES_DIR / name
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        display(fig)
        plt.close(fig)
        print(f"Saved figure: {path}")
        assert path.exists(), f"Expected figure was not created: {path}"
        assert path.stat().st_size > 0, f"Generated figure is empty: {path}"
        return path


    def depth_to_disparity(depth, focal_length, baseline):
        return focal_length * baseline / torch.clamp(depth, min=1e-6)


    def disparity_to_depth(disparity, focal_length, baseline):
        return focal_length * baseline / torch.clamp(disparity, min=1e-6)


    def fill_invalid_rows(image, valid_mask):
        image_np = image.numpy().copy()
        valid_np = valid_mask.numpy()
        for yy in range(image_np.shape[0]):
            row_valid = valid_np[yy]
            if row_valid.all():
                continue
            if row_valid.any():
                xs = np.where(row_valid)[0]
                image_np[yy] = np.interp(np.arange(image_np.shape[1]), xs, image_np[yy, xs])
            else:
                image_np[yy] = 0.0
        return torch.from_numpy(image_np).to(image.dtype)


    def matcher_support_mask(shape, patch_size, max_disparity):
        assert patch_size > 0 and patch_size % 2 == 1
        height, width = shape
        radius = patch_size // 2
        mask = torch.zeros((height, width), dtype=torch.bool)
        y0 = radius
        y1 = height - radius
        x0 = radius + max_disparity
        x1 = width - radius
        if y0 < y1 and x0 < x1:
            mask[y0:y1, x0:x1] = True
        return mask


    def render_right_from_layers(layers, focal_length, baseline, width):
        height = layers[0]["depth"].shape[0]
        right_sparse = torch.full((height, width), float("nan"), dtype=torch.float32)
        right_depth = torch.full((height, width), float("inf"), dtype=torch.float32)
        right_region = torch.full((height, width), -1, dtype=torch.int64)
        for layer_index, layer in enumerate(layers):
            ys, xs = torch.nonzero(layer["mask"], as_tuple=True)
            for yy, xx in zip(ys.tolist(), xs.tolist(), strict=True):
                depth_value = layer["depth"][yy, xx].item()
                disp = focal_length * baseline / max(depth_value, 1e-6)
                xr = int(round(xx - disp))
                if xr < 0 or xr >= width:
                    continue
                if depth_value < right_depth[yy, xr]:
                    right_depth[yy, xr] = depth_value
                    right_sparse[yy, xr] = layer["intensity"][yy, xx]
                    right_region[yy, xr] = layer_index
        right_valid_mask = torch.isfinite(right_sparse)
        right_filled = fill_invalid_rows(torch.nan_to_num(right_sparse, nan=0.0), right_valid_mask)
        return right_sparse, right_filled, right_depth, right_region, right_valid_mask


    def create_synthetic_scene(height=HEIGHT, width=WIDTH, focal_length=FOCAL_LENGTH_PX, baseline=BASELINE_UNITS):
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )

        background_depth = 8.5 - 1.1 * (x / width) + 0.35 * torch.sin(y / 18.0)
        background_intensity = (
            0.30
            + 0.12 * torch.sin(x / 6.0)
            + 0.11 * torch.cos(y / 7.0)
            + 0.06 * torch.sin((x + 1.4 * y) / 11.0)
        ).clamp(0.0, 1.0)

        near_box_mask = (x > 18) & (x < 58) & (y > 20) & (y < 70)
        near_box_depth = torch.full_like(background_depth, 2.0)
        near_box_intensity = (
            0.42
            + 0.24 * ((((x // 5) + (y // 5)) % 2).float())
            + 0.05 * torch.sin(y / 4.0)
        ).clamp(0.0, 1.0)

        mid_circle_mask = (x - 88) ** 2 + (y - 38) ** 2 < 17**2
        mid_circle_depth = torch.full_like(background_depth, 3.8)
        mid_circle_intensity = (
            0.22 + 0.55 * torch.exp(-((x - 88) ** 2 + (y - 38) ** 2) / 140.0)
        ).clamp(0.0, 1.0)

        ramp_mask = (x > 76) & (x < 132) & (y > 60) & (y < 90)
        ramp_depth = torch.full_like(background_depth, 5.3)
        ramp_depth[ramp_mask] = 3.2 + 2.1 * ((x[ramp_mask] - 76.0) / (132.0 - 76.0))
        ramp_intensity = (
            0.22
            + 0.32 * ((x - 76.0) / (132.0 - 76.0)).clamp(0.0, 1.0)
            + 0.08 * torch.sin(y / 4.5)
        ).clamp(0.0, 1.0)

        textureless_mask = (x > 18) & (x < 62) & (y > 72) & (y < 92)
        textureless_depth = torch.full_like(background_depth, 4.9)
        textureless_intensity = torch.full_like(background_depth, 0.63)

        repeated_mask = (x > 96) & (x < 138) & (y > 16) & (y < 52)
        repeated_depth = torch.full_like(background_depth, focal_length * baseline / 8.0)
        repeated_intensity = (
            0.34
            + 0.22 * ((((x - 96) // 4) % 2).float())
            + 0.04 * ((((y - 16) // 6) % 2).float())
        ).clamp(0.0, 1.0)

        occluder_mask = (x > 66) & (x < 76) & (y > 14) & (y < 82)
        occluder_depth = torch.full_like(background_depth, 1.8)
        occluder_intensity = (0.86 - 0.10 * ((((y - 14) // 6) % 2).float())).clamp(0.0, 1.0)

        layers = [
            {"name": "background", "mask": torch.ones_like(background_depth, dtype=torch.bool), "depth": background_depth, "intensity": background_intensity},
            {"name": "ramp", "mask": ramp_mask, "depth": ramp_depth, "intensity": ramp_intensity},
            {"name": "textureless", "mask": textureless_mask, "depth": textureless_depth, "intensity": textureless_intensity},
            {"name": "repeated_pattern", "mask": repeated_mask, "depth": repeated_depth, "intensity": repeated_intensity},
            {"name": "mid_circle", "mask": mid_circle_mask, "depth": mid_circle_depth, "intensity": mid_circle_intensity},
            {"name": "near_box", "mask": near_box_mask, "depth": near_box_depth, "intensity": near_box_intensity},
            {"name": "occluder", "mask": occluder_mask, "depth": occluder_depth, "intensity": occluder_intensity},
        ]

        left = torch.zeros((height, width), dtype=torch.float32)
        depth = torch.full((height, width), float("inf"), dtype=torch.float32)
        region_index = torch.full((height, width), -1, dtype=torch.int64)
        for layer_idx, layer in enumerate(layers):
            closer = layer["mask"] & (layer["depth"] < depth)
            left[closer] = layer["intensity"][closer]
            depth[closer] = layer["depth"][closer]
            region_index[closer] = layer_idx

        disparity = depth_to_disparity(depth, focal_length, baseline)
        right_sparse, right, right_depth, right_region, right_valid_mask = render_right_from_layers(
            layers, focal_length, baseline, width
        )

        visible_mask = torch.zeros_like(left, dtype=torch.bool)
        occlusion_mask = torch.zeros_like(left, dtype=torch.bool)
        right_correspondence_valid = torch.zeros_like(left, dtype=torch.bool)
        for yy in range(height):
            for xx in range(width):
                disp = float(disparity[yy, xx].item())
                xr = int(round(xx - disp))
                if xr < 0 or xr >= width:
                    occlusion_mask[yy, xx] = True
                    continue
                right_correspondence_valid[yy, xx] = bool(right_valid_mask[yy, xr].item())
                same_depth = abs(float(right_depth[yy, xr].item()) - float(depth[yy, xx].item())) < 1e-4
                same_region = int(right_region[yy, xr].item()) == int(region_index[yy, xx].item())
                if right_valid_mask[yy, xr] and same_depth and same_region:
                    visible_mask[yy, xx] = True
                else:
                    occlusion_mask[yy, xx] = True

        region_masks = {layer["name"]: region_index == idx for idx, layer in enumerate(layers)}
        return {
            "left": left,
            "right": right,
            "depth": depth,
            "disparity": disparity,
            "visible_mask": visible_mask,
            "occlusion_mask": occlusion_mask,
            "right_valid_mask": right_valid_mask,
            "right_correspondence_valid": right_correspondence_valid,
            "region_masks": region_masks,
        }


    def extract_patch(image, center, patch_size):
        radius = patch_size // 2
        yy, xx = center
        return image[yy - radius : yy + radius + 1, xx - radius : xx + radius + 1]


    def matching_cost(patch_a, patch_b, cost_type="ssd"):
        diff = patch_a - patch_b
        if cost_type == "ssd":
            return torch.sum(diff * diff)
        if cost_type == "sad":
            return torch.sum(torch.abs(diff))
        raise ValueError(f"Unsupported cost type: {cost_type}")


    def block_matching_disparity(left, right, patch_size, max_disparity, cost_type="ssd"):
        assert patch_size > 0 and patch_size % 2 == 1
        assert left.shape == right.shape
        pred = torch.full_like(left, float("nan"))
        min_cost = torch.full_like(left, float("nan"))
        valid_mask = torch.zeros_like(left, dtype=torch.bool)
        height, width = left.shape
        radius = patch_size // 2
        for yy in range(radius, height - radius):
            for xx in range(radius + max_disparity, width - radius):
                left_patch = extract_patch(left, (yy, xx), patch_size)
                best_cost = None
                best_disp = None
                for disp in range(max_disparity + 1):
                    xr = xx - disp
                    if xr - radius < 0:
                        break
                    right_patch = extract_patch(right, (yy, xr), patch_size)
                    cost = float(matching_cost(left_patch, right_patch, cost_type=cost_type).item())
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_disp = disp
                if best_disp is not None:
                    pred[yy, xx] = float(best_disp)
                    min_cost[yy, xx] = float(best_cost)
                    valid_mask[yy, xx] = True
        return pred, valid_mask, min_cost


    def disparity_cost_curve(left, right, point, patch_size, max_disparity, cost_type="ssd"):
        yy, xx = point
        left_patch = extract_patch(left, point, patch_size)
        disparities = []
        costs = []
        for disp in range(max_disparity + 1):
            xr = xx - disp
            right_patch = extract_patch(right, (yy, xr), patch_size)
            disparities.append(disp)
            costs.append(float(matching_cost(left_patch, right_patch, cost_type=cost_type).item()))
        return torch.tensor(disparities, dtype=torch.float32), torch.tensor(costs, dtype=torch.float32)


    def build_evaluation_mask(scene, matcher_valid_mask, support_mask=None):
        mask = torch.isfinite(scene["disparity"]) & scene["visible_mask"] & matcher_valid_mask
        mask = mask & scene["right_correspondence_valid"]
        if support_mask is not None:
            mask = mask & support_mask
        assert mask.any(), "Evaluation mask unexpectedly empty."
        return mask


    def compute_error_maps(pred_disparity, scene):
        disparity_error = torch.abs(pred_disparity - scene["disparity"])
        pred_depth = torch.full_like(pred_disparity, float("nan"))
        positive = torch.isfinite(pred_disparity) & (pred_disparity > 0.25)
        pred_depth[positive] = disparity_to_depth(pred_disparity[positive], FOCAL_LENGTH_PX, BASELINE_UNITS)
        depth_error = torch.abs(pred_depth - scene["depth"])
        return disparity_error, pred_depth, depth_error


    def summarize_metrics(pred_disparity, scene, evaluation_mask, runtime_seconds):
        disparity_error, pred_depth, depth_error = compute_error_maps(pred_disparity, scene)
        disp_values = disparity_error[evaluation_mask]
        depth_values = depth_error[evaluation_mask & torch.isfinite(depth_error)]
        metrics = {
            "disparity_mae_px": float(disp_values.mean().item()),
            "disparity_median_ae_px": float(disp_values.median().item()),
            "disparity_rmse_px": float(torch.sqrt(torch.mean(disp_values ** 2)).item()),
            "bad_pixel_ratio_gt_1px": float((disp_values > 1.0).float().mean().item()),
            "depth_mae_scene_units": float(depth_values.mean().item()),
            "depth_rmse_scene_units": float(torch.sqrt(torch.mean(depth_values ** 2)).item()),
            "valid_pixel_count": int(evaluation_mask.sum().item()),
            "valid_pixel_fraction": float(evaluation_mask.float().mean().item()),
            "runtime_seconds": float(runtime_seconds),
        }
        return metrics, disparity_error, pred_depth, depth_error


    expected_reference_assets = [
        ASSETS_DIR / "figure-40-01-titanic.png",
        ASSETS_DIR / "figure-40-03-random-dot-stereogram.png",
        ASSETS_DIR / "figure-40-05-triangularization-stereo.png",
        ASSETS_DIR / "figure-40-07-intensity-matching-failure.png",
    ]
    missing_assets = [str(path) for path in expected_reference_assets if not path.exists()]
    assert not missing_assets, f"Missing MIT reference assets: {missing_assets}"
    print(f"Using local MIT reference assets from {ASSETS_DIR}")
    """

    md2 = """
    ## 2. Disparity and depth intuition

    **MIT Vision Book reference — Figure 40.3: Random-dot stereogram as a disparity cue.**

    Source: MIT *Foundations of Computer Vision*, Chapter 40.

    ![MIT Vision Book Figure 40.3](assets/mit-book/figure-40-03-random-dot-stereogram.png)

    In a rectified stereo pair, corresponding points shift horizontally. The disparity is

    $$
    d = x_L - x_R
    $$

    and the standard stereo depth relation is

    $$
    Z = \\frac{fB}{d}.
    $$

    Nearby points produce larger disparities, while far points produce smaller ones.
    """

    code2 = """
    near_depth = torch.tensor(2.2)
    far_depth = torch.tensor(7.2)
    near_disparity = float(depth_to_disparity(near_depth, FOCAL_LENGTH_PX, BASELINE_UNITS).item())
    far_disparity = float(depth_to_disparity(far_depth, FOCAL_LENGTH_PX, BASELINE_UNITS).item())

    disparity_axis = torch.linspace(1.0, near_disparity + 2.0, 300)
    depth_axis = disparity_to_depth(disparity_axis, FOCAL_LENGTH_PX, BASELINE_UNITS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)

    left_row_y = 1.3
    right_row_y = 0.45
    x_left_near = 7.0
    x_left_far = 4.6
    scale = 0.18
    x_right_near = x_left_near - near_disparity * scale
    x_right_far = x_left_far - far_disparity * scale

    axes[0].hlines([left_row_y, right_row_y], 1.0, 8.0, color="#9ca3af", linewidth=2.0)
    axes[0].text(0.5, left_row_y, "Left image row", va="center", color="#374151")
    axes[0].text(0.45, right_row_y, "Right image row", va="center", color="#374151")

    axes[0].scatter([x_left_far, x_left_near], [left_row_y, left_row_y], color="#111827", s=46, zorder=3)
    axes[0].scatter([x_right_far, x_right_near], [right_row_y, right_row_y], color="#b91c1c", s=46, zorder=3)
    axes[0].text(x_left_far, left_row_y + 0.18, "far", ha="center", color="#111827")
    axes[0].text(x_left_near, left_row_y + 0.18, "near", ha="center", color="#111827")

    axes[0].annotate("", xy=(x_left_far, 0.88), xytext=(x_right_far, 0.88), arrowprops=dict(arrowstyle="<->", lw=1.8, color="#0f766e"))
    axes[0].annotate("", xy=(x_left_near, 0.05), xytext=(x_right_near, 0.05), arrowprops=dict(arrowstyle="<->", lw=1.8, color="#0f766e"))
    axes[0].text((x_left_far + x_right_far) / 2, 1.02, f"$d_{{far}}={far_disparity:.1f}$ px", ha="center", color="#0f766e")
    axes[0].text((x_left_near + x_right_near) / 2, 0.19, f"$d_{{near}}={near_disparity:.1f}$ px", ha="center", color="#0f766e")
    axes[0].set_xlim(0.1, 8.4)
    axes[0].set_ylim(-0.25, 1.8)
    axes[0].axis("off")
    axes[0].set_title("Horizontal disparity in rectified stereo")

    axes[1].plot(disparity_axis.numpy(), depth_axis.numpy(), color="#0f766e", lw=2.5)
    axes[1].scatter([near_disparity, far_disparity], [near_depth.item(), far_depth.item()], color="#b91c1c", s=55)
    axes[1].text(near_disparity + 0.15, near_depth.item() + 0.08, "near")
    axes[1].text(far_disparity + 0.15, far_depth.item() + 0.08, "far")
    axes[1].set_xlabel("Disparity d (pixels)")
    axes[1].set_ylabel("Depth Z (synthetic scene units)")
    axes[1].set_title("Depth decreases as disparity increases")

    save_figure(fig, "01-disparity-depth-intuition.png")
    print(json.dumps({"near_disparity_px": near_disparity, "far_disparity_px": far_disparity}, indent=2))
    """

    md3 = """
    ## 3. Synthetic rectified stereo scene

    **MIT Vision Book reference — Figure 40.5: Simple parallel stereo geometry and depth from disparity.**

    Source: MIT *Foundations of Computer Vision*, Chapter 40.

    ![MIT Vision Book Figure 40.5](assets/mit-book/figure-40-05-triangularization-stereo.png)

    The synthetic scene is rectified by construction, so every valid correspondence stays on
    the same image row. We use the sign convention

    $$
    d = x_L - x_R
    $$

    so a positive disparity means the corresponding right-image location lies to the left.
    """

    code3 = """
    scene = create_synthetic_scene()
    scene_summary = {
        "ground_truth_disparity_range_px": (
            float(scene["disparity"].min().item()),
            float(scene["disparity"].max().item()),
        ),
        "visible_pixel_count": int(scene["visible_mask"].sum().item()),
        "right_valid_pixel_count": int(scene["right_valid_mask"].sum().item()),
    }
    print(json.dumps(scene_summary, indent=2))

    disparity_vmin = float(scene["disparity"].min().item())
    disparity_vmax = float(scene["disparity"].max().item())

    labeled_points = {
        "near": (44, 38),
        "middle": (75, 92),
        "far": (14, 124),
    }
    label_colors = {"near": "#b91c1c", "middle": "#0f766e", "far": "#1d4ed8"}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    axes[0].imshow(scene["left"].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Left image")
    axes[1].imshow(scene["right"].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Right image")
    disp_im = axes[2].imshow(scene["disparity"].numpy(), cmap=DISPARITY_CMAP, vmin=disparity_vmin, vmax=disparity_vmax)
    axes[2].set_title("Ground-truth disparity")
    for label, (yy, xx) in labeled_points.items():
        color = label_colors[label]
        axes[0].scatter(xx, yy, color=color, s=28, edgecolors="white", linewidths=0.6)
        axes[0].text(xx + 2, yy - 2, label, color=color, fontsize=9, weight="bold")
        axes[2].scatter(xx, yy, color=color, s=28, edgecolors="white", linewidths=0.6)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(disp_im, ax=axes[2], fraction=0.046, pad=0.04, label="Disparity (pixels)")
    save_figure(fig, "02-synthetic-stereo-pair.png")
    """

    md4 = """
    ## 4. Single-patch correspondence

    **MIT Vision Book reference — Figure 40.7: Intensity matching can fail even when the geometry is simple.**

    Source: MIT *Foundations of Computer Vision*, Chapter 40.

    ![MIT Vision Book Figure 40.7](assets/mit-book/figure-40-07-intensity-matching-failure.png)

    In a rectified stereo pair, a left-image patch only needs to search horizontally in the
    right image. The diagnostic below shows the query location, the horizontal search line,
    enlarged patches, and the SSD cost as a function of candidate disparity.
    """

    code4 = """
    query_point = (42, 42)
    radius = DEFAULT_PATCH_SIZE // 2
    support_mask_default = matcher_support_mask(scene["left"].shape, DEFAULT_PATCH_SIZE, DEFAULT_MAX_DISPARITY)
    assert support_mask_default[query_point], "The chosen query point must lie inside the matcher-valid support."

    disparities, costs = disparity_cost_curve(
        scene["left"],
        scene["right"],
        query_point,
        DEFAULT_PATCH_SIZE,
        DEFAULT_MAX_DISPARITY,
        cost_type=COST_TYPE,
    )
    best_idx = int(torch.argmin(costs).item())
    predicted_disp = float(disparities[best_idx].item())
    gt_disp = float(scene["disparity"][query_point].item())

    yy, xx = query_point
    best_point = (yy, int(round(xx - predicted_disp)))
    query_patch = extract_patch(scene["left"], query_point, DEFAULT_PATCH_SIZE)
    best_patch = extract_patch(scene["right"], best_point, DEFAULT_PATCH_SIZE)
    side_by_side = torch.cat([query_patch, torch.full((DEFAULT_PATCH_SIZE, 2), 0.95), best_patch], dim=1)

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.8), constrained_layout=True)
    axes[0].imshow(scene["left"].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].add_patch(plt.Rectangle((xx - radius, yy - radius), DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE, fill=False, edgecolor="#ef4444", linewidth=2))
    axes[0].set_title("Left query location")

    axes[1].imshow(scene["right"].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].hlines(yy, xx - DEFAULT_MAX_DISPARITY - 2, xx + 2, color="#0f766e", linewidth=2)
    axes[1].scatter(xx - gt_disp, yy, color="#b91c1c", s=45, label="GT")
    axes[1].scatter(xx - predicted_disp, yy, color="#2563eb", s=45, marker="s", label="Pred")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].set_title("Right-image search line")

    axes[2].imshow(side_by_side.numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[2].set_title("Enlarged patches")
    axes[2].set_xlabel("query patch | best right patch", labelpad=8)

    axes[3].plot(disparities.numpy(), costs.numpy(), color="#0f766e", lw=2.5)
    axes[3].axvline(gt_disp, color="#b91c1c", linestyle="--", label=f"GT={gt_disp:.2f}px")
    axes[3].axvline(predicted_disp, color="#2563eb", linestyle=":", label=f"Pred={predicted_disp:.0f}px")
    axes[3].set_xlabel("Candidate disparity (pixels)")
    axes[3].set_ylabel("SSD cost")
    axes[3].set_title("Matching cost versus disparity")
    axes[3].legend(fontsize=8)

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    save_figure(fig, "03-patch-matching-diagnostic.png")
    print(
        json.dumps(
            {
                "query_point": query_point,
                "ground_truth_disparity_px": gt_disp,
                "predicted_disparity_px": predicted_disp,
                "integer_match_correct": bool(int(round(gt_disp)) == int(round(predicted_disp))),
            },
            indent=2,
        )
    )
    """

    md5 = """
    ## 5. Dense disparity estimation

    The dense matcher still uses local SSD, but now applies it at every interior pixel inside
    the valid support. Border pixels and invalid right-image correspondences are excluded from
    the reported metrics.
    """

    code5 = """
    start = time.perf_counter()
    pred_disparity, matcher_valid_mask, min_cost_map = block_matching_disparity(
        scene["left"],
        scene["right"],
        DEFAULT_PATCH_SIZE,
        DEFAULT_MAX_DISPARITY,
        cost_type=COST_TYPE,
    )
    runtime_seconds = time.perf_counter() - start

    evaluation_mask = build_evaluation_mask(scene, matcher_valid_mask, support_mask=support_mask_default)
    metrics, disparity_error_map, pred_depth_map, depth_error_map = summarize_metrics(
        pred_disparity,
        scene,
        evaluation_mask,
        runtime_seconds,
    )
    print(json.dumps(metrics, indent=2))

    disparity_vmin = float(scene["disparity"][evaluation_mask].min().item())
    disparity_vmax = float(scene["disparity"][evaluation_mask].max().item())
    pred_masked = np.ma.masked_where(~matcher_valid_mask.numpy(), pred_disparity.numpy())
    error_masked = np.ma.masked_where(~evaluation_mask.numpy(), disparity_error_map.numpy())
    disparity_cmap = plt.get_cmap(DISPARITY_CMAP).copy()
    disparity_cmap.set_bad("#d4d4d8")
    error_cmap = plt.get_cmap(ERROR_CMAP).copy()
    error_cmap.set_bad("#d4d4d8")
    error_vmax = float(torch.quantile(disparity_error_map[evaluation_mask], 0.98).item())

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    gt_im = axes[0].imshow(scene["disparity"].numpy(), cmap=DISPARITY_CMAP, vmin=disparity_vmin, vmax=disparity_vmax)
    axes[0].set_title("Ground-truth disparity")
    pred_im = axes[1].imshow(pred_masked, cmap=disparity_cmap, vmin=disparity_vmin, vmax=disparity_vmax)
    axes[1].set_title("Estimated disparity")
    err_im = axes[2].imshow(error_masked, cmap=error_cmap, vmin=0.0, vmax=error_vmax)
    axes[2].set_title(f"Absolute error (gray = invalid)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(gt_im, ax=axes[0], fraction=0.046, pad=0.04, label="Disparity (pixels)")
    fig.colorbar(pred_im, ax=axes[1], fraction=0.046, pad=0.04, label="Disparity (pixels)")
    fig.colorbar(err_im, ax=axes[2], fraction=0.046, pad=0.04, label="Error (pixels)")
    save_figure(fig, "04-dense-disparity-result.png")
    """

    md5b = """
    The estimated disparity mostly tracks the synthetic ground truth, but the error map makes
    the weak spots visible: occluded or out-of-support pixels are shown in gray, while the
    remaining hot spots cluster around repeated texture and sharp depth transitions.
    """

    md6 = """
    ## 6. Parameter tradeoffs

    We keep the study small: patch size affects ambiguity versus boundary blur, and max
    disparity affects both coverage and runtime. All comparisons use a common evaluation mask.
    """

    code6 = """
    patch_sizes = [3, 5, 7, 9]
    max_disparities = [8, 10, 12, 16]
    common_support_mask = matcher_support_mask(scene["left"].shape, max(patch_sizes), max(max_disparities))
    common_evaluation_mask = build_evaluation_mask(scene, common_support_mask, support_mask=common_support_mask)
    print(json.dumps({"common_evaluated_pixels": int(common_evaluation_mask.sum().item())}, indent=2))

    patch_mae = []
    patch_bad = []
    for patch_size in patch_sizes:
        pred, valid_mask, _ = block_matching_disparity(
            scene["left"], scene["right"], patch_size, DEFAULT_MAX_DISPARITY, cost_type=COST_TYPE
        )
        mask = build_evaluation_mask(scene, valid_mask, support_mask=common_evaluation_mask)
        disp_error, _, _ = compute_error_maps(pred, scene)
        values = disp_error[mask]
        patch_mae.append(float(values.mean().item()))
        patch_bad.append(float((values > 1.0).float().mean().item()))

    range_mae = []
    range_runtime = []
    for max_disp in max_disparities:
        start = time.perf_counter()
        pred, valid_mask, _ = block_matching_disparity(
            scene["left"], scene["right"], DEFAULT_PATCH_SIZE, max_disp, cost_type=COST_TYPE
        )
        elapsed = time.perf_counter() - start
        mask = build_evaluation_mask(scene, valid_mask, support_mask=common_evaluation_mask)
        disp_error, _, _ = compute_error_maps(pred, scene)
        values = disp_error[mask]
        range_mae.append(float(values.mean().item()))
        range_runtime.append(float(elapsed))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    patch_ax2 = axes[0].twinx()
    range_ax2 = axes[1].twinx()

    patch_line_1 = axes[0].plot(patch_sizes, patch_mae, marker="o", color="#0f766e", label="MAE (px)")
    patch_line_2 = patch_ax2.plot(patch_sizes, patch_bad, marker="s", color="#b91c1c", label="Bad-pixel ratio > 1 px")
    axes[0].set_xlabel("Patch size")
    axes[0].set_ylabel("MAE (px)", color="#0f766e")
    patch_ax2.set_ylabel("Bad-pixel ratio", color="#b91c1c")
    axes[0].set_title("Patch size tradeoff")
    axes[0].legend(patch_line_1 + patch_line_2, [line.get_label() for line in patch_line_1 + patch_line_2], fontsize=8, loc="upper left")

    range_line_1 = axes[1].plot(max_disparities, range_mae, marker="o", color="#0f766e", label="MAE (px)")
    range_line_2 = range_ax2.plot(max_disparities, range_runtime, marker="s", color="#7c3aed", label="Runtime (s)")
    axes[1].set_xlabel("Max disparity")
    axes[1].set_ylabel("MAE (px)", color="#0f766e")
    range_ax2.set_ylabel("Runtime (s)", color="#7c3aed")
    axes[1].set_title("Search-range tradeoff")
    axes[1].legend(range_line_1 + range_line_2, [line.get_label() for line in range_line_1 + range_line_2], fontsize=8, loc="upper left")
    save_figure(fig, "05-parameter-tradeoffs.png")

    print(
        json.dumps(
            {
                "patch_sizes": patch_sizes,
                "patch_mae": patch_mae,
                "patch_bad_pixel_ratio": patch_bad,
                "max_disparities": max_disparities,
                "range_mae": range_mae,
                "range_runtime_s": range_runtime,
            },
            indent=2,
        )
    )
    """

    md6b = """
    The small tradeoff study stays intentionally narrow: larger patches reduce some local
    ambiguity but blur disparity boundaries, while broader search ranges improve coverage only
    up to the point where the extra compute stops buying noticeable error reduction.
    """

    md7 = """
    ## 7. Failure cases

    Textureless regions, repeated patterns, and depth boundaries remain difficult even in this
    controlled setting. We show one crop and its local error for each failure mode.
    """

    code7 = """
    depth_edges = torch.zeros_like(scene["depth"], dtype=torch.bool)
    depth_edges[:, 1:] |= torch.abs(scene["depth"][:, 1:] - scene["depth"][:, :-1]) > 0.45
    depth_edges[1:, :] |= torch.abs(scene["depth"][1:, :] - scene["depth"][:-1, :]) > 0.45
    boundary_mask = F.max_pool2d(depth_edges.float()[None, None], kernel_size=7, stride=1, padding=3)[0, 0].bool()
    boundary_mask = boundary_mask & evaluation_mask

    region_specs = [
        ("Textureless", (74, 20, 18, 38)),
        ("Repeated", (18, 98, 30, 34)),
        ("Boundary", (24, 50, 36, 34)),
    ]

    error_cmap = plt.get_cmap(ERROR_CMAP).copy()
    error_cmap.set_bad("#d4d4d8")
    crop_error_values = []
    for _, (y0, x0, h, w) in region_specs:
        ys = slice(y0, y0 + h)
        xs = slice(x0, x0 + w)
        crop_mask = evaluation_mask[ys, xs]
        if crop_mask.any():
            crop_error_values.append(disparity_error_map[ys, xs][crop_mask])
    err_vmax = float(torch.quantile(torch.cat(crop_error_values), 0.98).item())

    region_results = []
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 9.2), constrained_layout=True)
    repeated_mae = None
    for row, (name, (y0, x0, h, w)) in enumerate(region_specs):
        ys = slice(y0, y0 + h)
        xs = slice(x0, x0 + w)
        crop_mask = evaluation_mask[ys, xs]
        count = int(crop_mask.sum().item())
        mae_text = "N/A"
        mae_value = None
        if count > 0:
            mae_value = float(disparity_error_map[ys, xs][crop_mask].mean().item())
            mae_text = f"{mae_value:.2f}px"
        if name == "Repeated":
            repeated_mae = mae_value
        region_results.append({"region": name, "pixel_count": count, "mae_px": mae_value})
        axes[row, 0].imshow(scene["left"][ys, xs].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(f"{name} crop")
        error_artist = axes[row, 1].imshow(
            np.ma.masked_where(~crop_mask.numpy(), disparity_error_map[ys, xs].numpy()),
            cmap=error_cmap,
            vmin=0.0,
            vmax=err_vmax,
        )
        axes[row, 1].set_title(f"{name} error (MAE={mae_text})")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

    textured_reference_crop = disparity_error_map[28:50, 22:44][evaluation_mask[28:50, 22:44]]
    textured_reference_mae = float(textured_reference_crop.mean().item())
    assert repeated_mae is not None and repeated_mae > textured_reference_mae, "Repeated-pattern crop should be measurably harder than a textured interior crop."

    fig.colorbar(error_artist, ax=axes[:, 1], fraction=0.046, pad=0.02, label="Absolute disparity error (px)")
    save_figure(fig, "06-failure-cases.png")
    print(json.dumps(region_results, indent=2))
    """

    md7b = """
    The three crops expose distinct failure modes. The textureless patch has weak local cues,
    the repeated pattern now creates a genuine horizontal ambiguity, and the boundary crop
    shows the usual support-window blur where one patch straddles two depths.
    """

    md8 = """
    ## 8. Limitations, reproducibility, and summary

    This is still an **educational baseline**:

    - the stereo pair is rectified by construction;
    - disparity is estimated at integer-pixel precision only;
    - the matching cost is local SSD, with no global regularization;
    - truly occluded pixels are excluded from ordinary metrics because they do not have valid
      right-image correspondences;
    - repeated texture, textureless regions, and disparity boundaries remain difficult.

    Reproducibility:

    ```bash
    jupyter nbconvert \\
      --to notebook \\
      --execute index.ipynb \\
      --output index.executed.ipynb \\
      --ExecutePreprocessor.timeout=600
    ```
    """

    code8 = """
    expected_image_names = [
        "01-disparity-depth-intuition.png",
        "02-synthetic-stereo-pair.png",
        "03-patch-matching-diagnostic.png",
        "04-dense-disparity-result.png",
        "05-parameter-tradeoffs.png",
        "06-failure-cases.png",
    ]
    expected_paths = [IMAGES_DIR / name for name in expected_image_names]
    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, f"Missing generated images: {missing}"
    generated_pngs = sorted(path.name for path in IMAGES_DIR.glob("*.png"))
    assert generated_pngs == expected_image_names, f"Unexpected PNG set under images/: {generated_pngs}"

    for path in expected_paths:
        assert path.stat().st_size > 0, f"Generated image is empty: {path}"

    try:
        from PIL import Image

        image_sizes = {path.name: Image.open(path).size for path in expected_paths}
    except Exception:
        image_sizes = {}

    readme_text = Path("README.md").read_text()
    readme_image_names = sorted(
        line.strip().split("images/")[-1].strip("`")
        for line in readme_text.splitlines()
        if "images/" in line and line.strip().startswith("- ")
    )
    assert readme_image_names == expected_image_names, f"README image names do not match generated files: {readme_image_names}"

    print(json.dumps({"generated_images": expected_image_names, "image_sizes": image_sizes}, indent=2))
    """

    cells = [
        markdown_cell(md0),
        code_cell(code0),
        markdown_cell(md2),
        code_cell(code2),
        markdown_cell(md3),
        code_cell(code3),
        markdown_cell(md4),
        code_cell(code4),
        markdown_cell(md5),
        code_cell(code5),
        markdown_cell(md5b),
        markdown_cell(md6),
        code_cell(code6),
        markdown_cell(md6b),
        markdown_cell(md7),
        code_cell(code7),
        markdown_cell(md7b),
        markdown_cell(md8),
        code_cell(code8),
    ]
    return notebook_dict(cells)


def build_readme() -> str:
    return textwrap.dedent(
        """
        # Chapter 40: Stereo Vision

        This chapter folder contains a compact executable notebook for MIT *Foundations of Computer Vision*, Chapter 40. It follows one focused story: disparity as a depth cue, a synthetic rectified stereo pair, local SSD block matching, quantitative evaluation, parameter tradeoffs, and failure cases.

        ## MIT Book references used

        The notebook now uses local offline copies of four official MIT Chapter 40 figures:

        - `assets/mit-book/figure-40-01-titanic.png`
        - `assets/mit-book/figure-40-03-random-dot-stereogram.png`
        - `assets/mit-book/figure-40-05-triangularization-stereo.png`
        - `assets/mit-book/figure-40-07-intensity-matching-failure.png`

        Git history for this chapter did not contain local MIT reference images, so these assets were restored from the official MIT chapter page to remove remote notebook-image dependencies. Figures 40.5 and 40.7 are the better fit for the notebook's rectified-stereo geometry and local matching story than the earlier 40.12 and 40.13 choices.

        ## Generated figures

        - `images/01-disparity-depth-intuition.png`
        - `images/02-synthetic-stereo-pair.png`
        - `images/03-patch-matching-diagnostic.png`
        - `images/04-dense-disparity-result.png`
        - `images/05-parameter-tradeoffs.png`
        - `images/06-failure-cases.png`

        ## How to run

        ```bash
        python notebooks/CV/2026/spring/final/chapter-40-stereo-vision/build_assets.py
        jupyter nbconvert --to notebook --execute notebooks/CV/2026/spring/final/chapter-40-stereo-vision/index.ipynb --output index.executed.ipynb --ExecutePreprocessor.timeout=600
        ```

        ## Scope

        The notebook assumes a rectified stereo setup and uses an educational integer-pixel SSD matcher. It does not implement calibration, full rectification, essential/fundamental matrix estimation, learned stereo, or global optimization.
        """
    ).strip() + "\n"


def main() -> None:
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=1))
    README_PATH.write_text(build_readme())
    print(f"Wrote notebook: {NOTEBOOK_PATH}")
    print(f"Wrote README: {README_PATH}")


if __name__ == "__main__":
    main()
