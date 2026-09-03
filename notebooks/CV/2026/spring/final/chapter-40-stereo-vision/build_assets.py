from __future__ import annotations

import json
import textwrap
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = CHAPTER_DIR / "index.ipynb"
README_PATH = CHAPTER_DIR / "README.md"
ASSETS_README_PATH = CHAPTER_DIR / "assets/mit-book/README.md"


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
    return {
        "cells": [{**cell, "id": f"cell-{idx:02d}"} for idx, cell in enumerate(cells)],
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


def image_block(
    paths: list[str],
    alt_text: str,
    caption: str,
    label: str,
    widths: list[str] | None = None,
) -> str:
    if widths is None:
        widths = ["100%"] if len(paths) == 1 else [f"{max(30, 96 // len(paths))}%" for _ in paths]
    images = "\n".join(
        f'<img src="{path}" alt="{alt_text}" style="width: {width}; max-width: 100%; height: auto;" />'
        for path, width in zip(paths, widths, strict=True)
    )
    return textwrap.dedent(
        f"""
        <div style="text-align: center; margin: 1rem 0;">
          <div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; align-items: flex-start;">
            {images}
          </div>
          <p><em>{label} — {caption}</em></p>
        </div>
        """
    ).strip()


def build_notebook() -> dict[str, object]:
    def gen_block(filename: str, caption: str, label: str = "Generated experiment") -> str:
        return image_block([f"images/{filename}"], filename, caption, label)

    mit_40_01 = image_block(
        ["assets/mit-book/figure-40-01-titanic.png"],
        "MIT Vision Book Figure 40.1",
        "Stereo anaglyph of the Titanic and the red/cyan viewing setup that turns left-right displacement into a 3D percept.",
        "MIT Vision Book Figure 40.1 — original reference image",
    )
    mit_40_02 = image_block(
        ["assets/mit-book/figure-40-02-boats.png"],
        "MIT Vision Book Figure 40.2",
        "Two boat-distance constructions: one using a horizon reference and one using triangulation between two observation points.",
        "MIT Vision Book Figure 40.2 — original reference image",
    )
    mit_40_03 = image_block(
        ["assets/mit-book/figure-40-03-random-dot-stereogram.png"],
        "MIT Vision Book Figure 40.3",
        "Random-dot stereogram showing that disparity alone can create a depth percept.",
        "MIT Vision Book Figure 40.3 — original reference image",
    )
    mit_40_04 = image_block(
        ["assets/mit-book/figure-40-04-anaglyph-camera.png"],
        "MIT Vision Book Figure 40.4",
        "An anaglyph pinhole camera built from two pinholes, color filters, and a projection plane.",
        "MIT Vision Book Figure 40.4 — original reference image",
        widths=["48%"],
    )
    mit_40_05 = image_block(
        ["assets/mit-book/figure-40-05-triangulation-stereo.png"],
        "MIT Vision Book Figure 40.5",
        "Rectified stereo geometry with focal length, baseline, image coordinates, and a triangulated 3D point.",
        "MIT Vision Book Figure 40.5 — original reference image",
    )
    mit_40_06 = image_block(
        [
            "assets/mit-book/figure-40-06-office-left.jpg",
            "assets/mit-book/figure-40-06-office-right.jpg",
        ],
        "MIT Vision Book Figure 40.6",
        "A real stereo office pair with corresponding features and their displacements highlighted.",
        "MIT Vision Book Figure 40.6 — original reference images",
        widths=["48%", "48%"],
    )
    mit_40_07 = image_block(
        ["assets/mit-book/figure-40-07-intensity-matching-failure.jpg"],
        "MIT Vision Book Figure 40.7",
        "Disparity-space ambiguity under raw intensity matching, smoothing, and ground-truth comparison.",
        "MIT Vision Book Figure 40.7 — original reference image",
    )
    mit_40_08 = image_block(
        [
            "assets/mit-book/figure-40-08-points-left.jpg",
            "assets/mit-book/figure-40-08-points-right.jpg",
            "assets/mit-book/figure-40-08-depth.jpg",
        ],
        "MIT Vision Book Figure 40.8",
        "Feature detections on the office pair and the interpolated depth result.",
        "MIT Vision Book Figure 40.8 — original reference images",
        widths=["31%", "31%", "31%"],
    )
    mit_40_09 = image_block(
        [
            "assets/mit-book/figure-40-09-oriented-features.jpg",
            "assets/mit-book/figure-40-09-sift-descriptor.jpg",
        ],
        "MIT Vision Book Figure 40.9",
        "Orientation-based local descriptors and the spatial pooling idea behind SIFT-style matching.",
        "MIT Vision Book Figure 40.9 — original reference images",
        widths=["40%", "56%"],
    )
    mit_40_10 = image_block(
        ["assets/mit-book/figure-40-10-correspondence-ambiguity.png"],
        "MIT Vision Book Figure 40.10",
        "Candidate-match ambiguity for a point viewed under arbitrary camera geometry.",
        "MIT Vision Book Figure 40.10 — original reference image",
    )
    mit_40_11 = image_block(
        ["assets/mit-book/figure-40-11-epipolar-ray.png"],
        "MIT Vision Book Figure 40.11",
        "The viewing ray from camera 1 projects to an epipolar line in camera 2.",
        "MIT Vision Book Figure 40.11 — original reference image",
    )
    mit_40_12 = image_block(
        ["assets/mit-book/figure-40-12-epipolar-geometry.png"],
        "MIT Vision Book Figure 40.12",
        "Epipolar plane, epipolar lines, and epipoles for a stereo pair.",
        "MIT Vision Book Figure 40.12 — original reference image",
    )
    mit_40_13 = image_block(
        ["assets/mit-book/figure-40-13-epipolar-game.png"],
        "MIT Vision Book Figure 40.13",
        "Epipolar-line intuition game matching camera arrangements to line families.",
        "MIT Vision Book Figure 40.13 — original reference image",
    )
    mit_40_14 = image_block(
        ["assets/mit-book/figure-40-14-stereo-cnn-block-diagram.jpg"],
        "MIT Vision Book Figure 40.14",
        "Two-stage CNN stereo pipeline: feature extraction, cost volume, cost aggregation, and disparity estimate.",
        "MIT Vision Book Figure 40.14 — original reference image",
    )

    md0 = """
    # Chapter 40: Stereo Vision

    This notebook preserves the MIT Vision Book chapter order while turning the chapter into a
    more visual and executable stereo-vision lesson. The guiding rule is simple:

    - keep the **MIT Vision Book Figure 40.x** images when the original figure is pedagogically useful;
    - add **generated experiments** when executable geometry, matching, failure analysis, or
      evaluation helps the reader understand the chapter more deeply;
    - label everything honestly so a generated notebook figure is never confused with the
      original textbook figure.

    The notebook is organized around repeated teaching steps:

    1. concept introduction;
    2. relevant MIT figure;
    3. mathematical formulation;
    4. executable experiment;
    5. generated visualization;
    6. quantitative interpretation;
    7. failure cases and limitations.
    """

    code0 = """
    import json
    import math
    import re
    import time
    from pathlib import Path

    import matplotlib.colors as mcolors
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw

    SEED = 7
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    DEVICE = torch.device("cpu")

    NOTEBOOK_DIR = Path.cwd().resolve()
    if not (NOTEBOOK_DIR / "index.ipynb").exists():
        repo_candidate = NOTEBOOK_DIR / "notebooks" / "CV" / "2026" / "spring" / "final" / "chapter-40-stereo-vision"
        if repo_candidate.exists() and (repo_candidate / "index.ipynb").exists():
            NOTEBOOK_DIR = repo_candidate.resolve()
        else:
            matches = list(NOTEBOOK_DIR.rglob("index.ipynb"))
            chapter_matches = [path.parent for path in matches if path.parent.name == "chapter-40-stereo-vision"]
            assert chapter_matches, f"Could not resolve notebook directory from {Path.cwd()}"
            NOTEBOOK_DIR = chapter_matches[0].resolve()

    CHAPTER_DIR = NOTEBOOK_DIR
    OUTPUT_DIR = NOTEBOOK_DIR / "output"
    IMAGES_DIR = NOTEBOOK_DIR / "images"
    ASSETS_DIR = NOTEBOOK_DIR / "assets/mit-book"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfa",
            "axes.edgecolor": "#374151",
            "font.size": 11,
            "axes.grid": False,
            "image.cmap": "gray",
        }
    )

    FOCAL_LENGTH_PX = 72.0
    BASELINE_UNITS = 0.24
    HEIGHT = 96
    WIDTH = 144
    DEFAULT_PATCH_SIZE = 7
    DEFAULT_MAX_DISPARITY = 12
    BAD_PIXEL_THRESHOLD = 1.0
    EDGE_COLOR = "#0f766e"
    ACCENT_RED = "#b91c1c"
    ACCENT_BLUE = "#2563eb"
    ACCENT_GOLD = "#d97706"
    EXPECTED_IMAGE_NAMES = [
        "01-stereo-cues.png",
        "02-rectified-stereo-geometry.png",
        "03-depth-vs-disparity.png",
        "04-depth-error-sensitivity.png",
        "05-baseline-sensitivity.png",
        "06-pixel-vs-patch-matching.png",
        "07-cost-volume-slices.png",
        "08-winner-takes-all-disparity.png",
        "09-patch-size-sweep.png",
        "10-max-disparity-sweep.png",
        "11-textureless-failure.png",
        "12-repetitive-pattern-failure.png",
        "13-occlusion-and-left-right-check.png",
        "14-subpixel-refinement.png",
        "15-epipolar-constraint.png",
        "16-before-after-rectification.png",
        "17-disparity-depth-error-maps.png",
        "18-runtime-accuracy-tradeoff.png",
    ]
    NOTEBOOK_OWNED_OBSOLETE_IMAGES = [
        "01-disparity-depth-intuition.png",
        "02-synthetic-stereo-pair.png",
        "03-patch-matching-diagnostic.png",
        "04-dense-disparity-result.png",
        "05-parameter-tradeoffs.png",
        "06-failure-cases.png",
        "generated_figures_contact_sheet.png",
    ]
    EXPECTED_REFERENCE_ASSETS = [
        ASSETS_DIR / "figure-40-01-titanic.png",
        ASSETS_DIR / "figure-40-02-boats.png",
        ASSETS_DIR / "figure-40-03-random-dot-stereogram.png",
        ASSETS_DIR / "figure-40-04-anaglyph-camera.png",
        ASSETS_DIR / "figure-40-05-triangulation-stereo.png",
        ASSETS_DIR / "figure-40-06-office-left.jpg",
        ASSETS_DIR / "figure-40-06-office-right.jpg",
        ASSETS_DIR / "figure-40-07-intensity-matching-failure.jpg",
        ASSETS_DIR / "figure-40-08-points-left.jpg",
        ASSETS_DIR / "figure-40-08-points-right.jpg",
        ASSETS_DIR / "figure-40-08-depth.jpg",
        ASSETS_DIR / "figure-40-09-oriented-features.jpg",
        ASSETS_DIR / "figure-40-09-sift-descriptor.jpg",
        ASSETS_DIR / "figure-40-10-correspondence-ambiguity.png",
        ASSETS_DIR / "figure-40-11-epipolar-ray.png",
        ASSETS_DIR / "figure-40-12-epipolar-geometry.png",
        ASSETS_DIR / "figure-40-13-epipolar-game.png",
        ASSETS_DIR / "figure-40-14-stereo-cnn-block-diagram.jpg",
    ]
    missing_assets = [str(path) for path in EXPECTED_REFERENCE_ASSETS if not path.exists()]
    assert not missing_assets, f"Missing MIT reference assets: {missing_assets}"
    for owned_name in EXPECTED_IMAGE_NAMES + NOTEBOOK_OWNED_OBSOLETE_IMAGES:
        owned_path = IMAGES_DIR / owned_name
        if owned_path.exists():
            owned_path.unlink()

    def save_figure(fig, name, dpi=180):
        path = IMAGES_DIR / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        assert path.exists() and path.stat().st_size > 0, f"Failed to save figure: {path}"
        print(f"Saved figure: {path}")
        return path

    def image_stats(path):
        info = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            "format": None,
            "width": None,
            "height": None,
            "channels": None,
            "decodable": False,
            "finite": False,
            "variance": None,
            "near_white_ratio": None,
            "near_black_ratio": None,
            "alpha_zero_ratio": None,
            "tiny_or_strip": False,
        }
        if not path.exists():
            return info
        try:
            with Image.open(path) as image:
                info["format"] = image.format
                info["width"], info["height"] = image.size
                array = np.asarray(image)
                info["channels"] = 1 if array.ndim == 2 else int(array.shape[2])
                info["decodable"] = True
                finite = np.isfinite(array.astype(np.float64))
                info["finite"] = bool(finite.all())
                luminance = array.astype(np.float64)
                if array.ndim == 3:
                    luminance = luminance[..., :3].mean(axis=2)
                if luminance.size > 0:
                    info["variance"] = float(np.var(luminance))
                    info["near_white_ratio"] = float(np.mean(luminance >= 250.0))
                    info["near_black_ratio"] = float(np.mean(luminance <= 5.0))
                if array.ndim == 3 and array.shape[2] == 4:
                    info["alpha_zero_ratio"] = float(np.mean(array[..., 3] == 0))
                else:
                    info["alpha_zero_ratio"] = 0.0
                info["tiny_or_strip"] = bool(
                    min(info["width"], info["height"]) <= 8 or info["width"] / max(info["height"], 1) > 12 or info["height"] / max(info["width"], 1) > 12
                )
        except Exception:
            return info
        return info

    def normalize_cell_source(cell):
        source = cell.get("source", "")
        if isinstance(source, list):
            return "".join(source)
        return source

    def find_generation_cell_index(filename, notebook_cells):
        marker = f'"{filename}"'
        for idx, cell in enumerate(notebook_cells):
            source = normalize_cell_source(cell)
            if cell.get("cell_type") == "code" and marker in source and "save_figure(" in source:
                return idx
        return None

    def collect_notebook_image_audit():
        notebook = json.loads((NOTEBOOK_DIR / "index.ipynb").read_text())
        cells = notebook["cells"]
        audit_rows = []
        figure_generation = {name: find_generation_cell_index(name, cells) for name in EXPECTED_IMAGE_NAMES}
        figure_display = {}
        for idx, cell in enumerate(cells):
            source = normalize_cell_source(cell)
            refs = []
            refs.extend(re.findall(r'<img\\s+[^>]*src="([^"]+)"', source))
            refs.extend(re.findall(r'!\\[[^\\]]*\\]\\(([^)]+)\\)', source))
            if "save_figure(" in source:
                for name in EXPECTED_IMAGE_NAMES:
                    if f'"{name}"' in source:
                        refs.append(f"images/{name}")
                        figure_display[name] = idx
            for ref in refs:
                resolved = (NOTEBOOK_DIR / ref).resolve() if not Path(ref).is_absolute() else Path(ref)
                stats = image_stats(resolved)
                filename = Path(ref).name
                generated_cell = figure_generation.get(filename)
                displayed_cell = figure_display.get(filename, idx)
                audit_rows.append(
                    {
                        "cell_index": idx,
                        "caption": re.sub(r"\\s+", " ", source.strip())[:160],
                        "referenced_path": ref,
                        "resolved_path": str(resolved),
                        "exists": stats["exists"],
                        "size_bytes": stats["size_bytes"],
                        "width": stats["width"],
                        "height": stats["height"],
                        "format": stats["format"],
                        "decodable": stats["decodable"],
                        "channels": stats["channels"],
                        "finite": stats["finite"],
                        "variance": stats["variance"],
                        "tiny_or_strip": stats["tiny_or_strip"],
                        "referenced_before_generation": bool(
                            generated_cell is not None and displayed_cell is not None and displayed_cell < generated_cell
                        ),
                        "display_matches_saved_file": not filename or filename not in figure_generation or ref == f"images/{filename}",
                    }
                )
        output_path = OUTPUT_DIR / "image_audit.json"
        output_path.write_text(json.dumps(audit_rows, indent=2))
        return audit_rows, output_path

    def build_contact_sheet(paths, destination, thumb_size=(280, 180), columns=3):
        cards = []
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                canvas = Image.new("RGB", thumb_size, "white")
                preview = rgb.copy()
                preview.thumbnail((thumb_size[0] - 16, thumb_size[1] - 34))
                paste_x = (thumb_size[0] - preview.width) // 2
                paste_y = 8 + max(0, (thumb_size[1] - 42 - preview.height) // 2)
                canvas.paste(preview, (paste_x, paste_y))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, thumb_size[1] - 22), path.name, fill="black")
                cards.append(canvas)
        rows = int(np.ceil(len(cards) / columns))
        sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "#f3f4f6")
        for idx, card in enumerate(cards):
            x = (idx % columns) * thumb_size[0]
            y = (idx // columns) * thumb_size[1]
            sheet.paste(card, (x, y))
        sheet.save(destination)
        return destination

    def generated_markdown_reference_counts(notebook_cells):
        counts = {name: 0 for name in EXPECTED_IMAGE_NAMES}
        for cell in notebook_cells:
            source = normalize_cell_source(cell)
            refs = re.findall(r'images/([A-Za-z0-9._-]+\\.png)', source)
            for name in refs:
                if name in counts:
                    counts[name] += 1
        return counts

    def validate_notebook_rendering_contract():
        notebook = json.loads((NOTEBOOK_DIR / "index.ipynb").read_text())
        cells = notebook["cells"]
        ref_counts = generated_markdown_reference_counts(cells)
        duplicates = {name: count for name, count in ref_counts.items() if count != 1}
        assert not duplicates, f"Generated figures must appear exactly once in markdown/HTML: {duplicates}"
        joined_source = "\\n".join(normalize_cell_source(cell) for cell in cells if cell.get("cell_type") == "code")
        sanitized_source = (
            joined_source.replace('"display(fig)"', "").replace('"plt.show()"', "").replace('"Image("', "")
        )
        for forbidden in ["display(fig)", "plt.show()", "Image("]:
            assert forbidden not in sanitized_source, f"Notebook rendering contract violated by explicit code display: {forbidden}"
        all_source = "\\n".join(normalize_cell_source(cell) for cell in cells)
        sanitized_all_source = all_source.replace('"images/generated_figures_contact_sheet.png"', "")
        assert "images/generated_figures_contact_sheet.png" not in sanitized_all_source, "Contact sheet should not be referenced under images/."
        return ref_counts

    def benchmark_configuration(reference, target, patch_size, max_disparity, direction="left_to_right", warmups=2, measured=10):
        result = None
        for _ in range(warmups):
            result = run_match(reference, target, max_disparity=max_disparity, patch_size=patch_size, direction=direction)
        runtimes = []
        for _ in range(measured):
            start = time.perf_counter()
            result = run_match(reference, target, max_disparity=max_disparity, patch_size=patch_size, direction=direction)
            runtimes.append(time.perf_counter() - start)
        assert result is not None
        median_runtime = float(np.median(runtimes))
        return result, median_runtime, {
            "median_runtime_seconds": median_runtime,
            "min_runtime_seconds": float(np.min(runtimes)),
            "max_runtime_seconds": float(np.max(runtimes)),
            "warmup_runs": warmups,
            "measured_runs": measured,
        }

    def as_float_tensor(array):
        if isinstance(array, torch.Tensor):
            return array.to(device=DEVICE, dtype=torch.float32)
        return torch.as_tensor(array, dtype=torch.float32, device=DEVICE)

    def tensor_to_numpy(array):
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def masked_invalid_numpy(array):
        return np.ma.masked_invalid(tensor_to_numpy(array))

    def finite_quantile(array, q):
        tensor = as_float_tensor(array)
        values = tensor[torch.isfinite(tensor)]
        return float(torch.quantile(values, q).item())

    def finite_min(array):
        tensor = as_float_tensor(array)
        values = tensor[torch.isfinite(tensor)]
        return float(values.min().item())

    def finite_max(array):
        tensor = as_float_tensor(array)
        values = tensor[torch.isfinite(tensor)]
        return float(values.max().item())

    def box_filter(image, radius):
        image_t = as_float_tensor(image)
        if radius <= 0:
            return image_t.clone()
        kernel = 2 * radius + 1
        padded = F.pad(image_t[None, None], (radius, radius, radius, radius), mode="replicate")
        filtered = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
        return filtered[0, 0]

    def disparity_to_depth(disparity, focal_length=FOCAL_LENGTH_PX, baseline=BASELINE_UNITS):
        disparity_t = as_float_tensor(disparity)
        depth = torch.full_like(disparity_t, float("nan"))
        valid = torch.isfinite(disparity_t)
        depth[valid] = focal_length * baseline / torch.clamp(disparity_t[valid], min=1e-3)
        return depth if isinstance(disparity, torch.Tensor) else tensor_to_numpy(depth)

    def depth_to_disparity(depth, focal_length=FOCAL_LENGTH_PX, baseline=BASELINE_UNITS):
        depth_t = as_float_tensor(depth)
        disparity = torch.full_like(depth_t, float("nan"))
        valid = depth_t > 1e-6
        disparity[valid] = focal_length * baseline / depth_t[valid]
        return disparity if isinstance(depth, torch.Tensor) else tensor_to_numpy(disparity)

    def create_synthetic_scene(height=HEIGHT, width=WIDTH, focal_length=FOCAL_LENGTH_PX, baseline=BASELINE_UNITS):
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=DEVICE),
            torch.arange(width, dtype=torch.float32, device=DEVICE),
            indexing="ij",
        )
        left = 0.28 + 0.10 * torch.sin(x / 6.0) + 0.08 * torch.cos(y / 8.0) + 0.04 * torch.sin((x + 1.3 * y) / 11.0)
        disparity = torch.full((height, width), 2.2, dtype=torch.float32, device=DEVICE)
        region = np.full((height, width), "background", dtype=object)

        checker = (x > 16) & (x < 60) & (y > 18) & (y < 66)
        left[checker] = 0.38 + 0.24 * torch.remainder(torch.floor(x[checker] / 5.0) + torch.floor(y[checker] / 5.0), 2.0)
        disparity[checker] = 8.7
        region[tensor_to_numpy(checker)] = "checker"

        textureless = (x > 18) & (x < 64) & (y > 72) & (y < 91)
        left[textureless] = 0.63
        disparity[textureless] = 3.1
        region[tensor_to_numpy(textureless)] = "textureless"

        repeated = (x > 96) & (x < 138) & (y > 16) & (y < 52)
        left[repeated] = 0.34 + 0.22 * torch.remainder(torch.floor((x[repeated] - 96.0) / 4.0), 2.0)
        disparity[repeated] = 8.0
        region[tensor_to_numpy(repeated)] = "repeated"

        occluder = (x > 66) & (x < 76) & (y > 14) & (y < 82)
        left[occluder] = 0.86 - 0.10 * torch.remainder(torch.floor((y[occluder] - 14.0) / 6.0), 2.0)
        disparity[occluder] = 9.6
        region[tensor_to_numpy(occluder)] = "occluder"

        circle = (x - 88.0) ** 2 + (y - 38.0) ** 2 < 17.0 ** 2
        left[circle] = 0.22 + 0.55 * torch.exp(-((x[circle] - 88.0) ** 2 + (y[circle] - 38.0) ** 2) / 140.0)
        disparity[circle] = 4.6
        region[tensor_to_numpy(circle)] = "circle"

        ramp = (x > 78) & (x < 132) & (y > 60) & (y < 89)
        left[ramp] = 0.25 + 0.30 * ((x[ramp] - 78.0) / (132.0 - 78.0))
        disparity[ramp] = 3.5 + 2.8 * ((x[ramp] - 78.0) / (132.0 - 78.0))
        region[tensor_to_numpy(ramp)] = "ramp"

        left = torch.clamp(left, 0.0, 1.0)
        depth = disparity_to_depth(disparity, focal_length, baseline)

        right = torch.full_like(left, float("nan"))
        right_disp = torch.full_like(disparity, float("-inf"))
        source_visible = torch.zeros((height, width), dtype=torch.bool, device=DEVICE)
        invalid_projection = torch.zeros((height, width), dtype=torch.bool, device=DEVICE)
        source_xr = torch.full((height, width), -1, dtype=torch.long, device=DEVICE)

        for yy in range(height):
            for xx in range(width):
                d = float(disparity[yy, xx].item())
                xr = int(round(xx - d))
                source_xr[yy, xx] = xr
                if xr < 0 or xr >= width:
                    invalid_projection[yy, xx] = True
                    continue
                if d > float(right_disp[yy, xr].item()):
                    right_disp[yy, xr] = d
                    right[yy, xr] = left[yy, xx]

        row_x = torch.arange(width, dtype=torch.float32, device=DEVICE)
        for yy in range(height):
            valid_cols = torch.isfinite(right[yy])
            if int(valid_cols.sum().item()) >= 2:
                xp = row_x[valid_cols]
                fp = right[yy, valid_cols]
                right_idx = torch.searchsorted(xp, row_x)
                left_idx = torch.clamp(right_idx - 1, 0, xp.numel() - 1)
                right_idx = torch.clamp(right_idx, 0, xp.numel() - 1)
                x0 = xp[left_idx]
                x1 = xp[right_idx]
                y0 = fp[left_idx]
                y1 = fp[right_idx]
                denom = torch.where(torch.abs(x1 - x0) < 1e-6, torch.ones_like(x1), x1 - x0)
                alpha = torch.where(
                    torch.abs(x1 - x0) < 1e-6,
                    torch.zeros_like(row_x),
                    (row_x - x0) / denom,
                )
                fill = y0 + alpha * (y1 - y0)
                right[yy] = fill
            elif int(valid_cols.sum().item()) == 1:
                right[yy] = right[yy, valid_cols][0]
            else:
                right[yy] = 0.0

        for yy in range(height):
            for xx in range(width):
                xr = int(source_xr[yy, xx].item())
                if xr < 0 or xr >= width:
                    continue
                source_visible[yy, xx] = abs(float(right_disp[yy, xr].item()) - float(disparity[yy, xx].item())) < 1e-6

        occlusion_mask = ~source_visible
        return {
            "left": left,
            "right": right,
            "gt_disparity": disparity,
            "gt_depth": depth,
            "visible_mask": source_visible,
            "occlusion_mask": occlusion_mask,
            "invalid_projection": invalid_projection,
            "region": region,
        }

    def shift_for_disparity(image, disparity, direction):
        image_t = as_float_tensor(image)
        shifted = torch.full_like(image_t, float("nan"))
        valid = torch.zeros_like(image_t, dtype=torch.bool)
        if disparity == 0:
            shifted[:] = image_t
            valid[:] = True
            return shifted, valid
        if direction == "left_to_right":
            shifted[:, disparity:] = image_t[:, :-disparity]
            valid[:, disparity:] = True
        elif direction == "right_to_left":
            shifted[:, :-disparity] = image_t[:, disparity:]
            valid[:, :-disparity] = True
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        return shifted, valid

    def build_support_mask(height, width, radius, max_disparity, direction):
        mask = torch.zeros((height, width), dtype=torch.bool, device=DEVICE)
        y0, y1 = radius, height - radius
        if direction == "left_to_right":
            x0, x1 = radius + max_disparity, width - radius
        elif direction == "right_to_left":
            x0, x1 = radius, width - radius - max_disparity
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        if y1 > y0 and x1 > x0:
            mask[y0:y1, x0:x1] = True
        return mask

    def build_cost_volume(reference, target, patch_size, max_disparity, direction):
        reference_t = as_float_tensor(reference)
        target_t = as_float_tensor(target)
        radius = patch_size // 2
        h, w = reference_t.shape
        cost_volume = torch.full((max_disparity + 1, h, w), float("nan"), dtype=torch.float32, device=DEVICE)
        support_mask = build_support_mask(h, w, radius, max_disparity, direction)
        for d in range(max_disparity + 1):
            aligned, aligned_valid = shift_for_disparity(target_t, d, direction)
            diff = (reference_t - aligned) ** 2
            diff = torch.where(aligned_valid, diff, torch.full_like(diff, 1e3))
            cost = box_filter(diff, radius)
            valid = support_mask & aligned_valid
            layer = torch.full((h, w), float("nan"), dtype=torch.float32, device=DEVICE)
            layer[valid] = cost[valid]
            cost_volume[d] = layer
        return cost_volume, support_mask

    def disparity_from_cost_volume(cost_volume):
        cost_volume_t = as_float_tensor(cost_volume)
        filled = torch.where(torch.isfinite(cost_volume_t), cost_volume_t, torch.full_like(cost_volume_t, float("inf")))
        best = torch.argmin(filled, dim=0).to(torch.float32)
        valid = torch.isfinite(cost_volume_t).any(dim=0)
        best = torch.where(valid, best, torch.full_like(best, float("nan")))
        best_cost = torch.min(filled, dim=0).values
        best_cost = torch.where(valid, best_cost, torch.full_like(best_cost, float("nan")))
        return best, valid, best_cost

    def compute_error_maps(pred_disparity, gt_disparity, gt_depth):
        pred_disparity_t = as_float_tensor(pred_disparity)
        gt_disparity_t = as_float_tensor(gt_disparity)
        gt_depth_t = as_float_tensor(gt_depth)
        disparity_error = torch.abs(pred_disparity_t - gt_disparity_t)
        pred_depth = as_float_tensor(disparity_to_depth(pred_disparity_t))
        depth_error = torch.abs(pred_depth - gt_depth_t)
        return disparity_error, pred_depth, depth_error

    def summarize_metrics(pred_disparity, gt_disparity, gt_depth, eval_mask, runtime_seconds, consistency_mask=None):
        eval_mask_t = torch.as_tensor(eval_mask, dtype=torch.bool, device=DEVICE)
        disparity_error, pred_depth, depth_error = compute_error_maps(pred_disparity, gt_disparity, gt_depth)
        disp_values = disparity_error[eval_mask_t]
        useful_depth = eval_mask_t & torch.isfinite(depth_error) & torch.isfinite(as_float_tensor(pred_disparity)) & (as_float_tensor(pred_disparity) > 0.25)
        depth_values = depth_error[useful_depth]
        metrics = {
            "disparity_mae_px": float(disp_values.mean().item()),
            "bad_pixel_rate_gt_1px": float((disp_values > BAD_PIXEL_THRESHOLD).to(torch.float32).mean().item()),
            "valid_pixel_ratio": float(eval_mask_t.to(torch.float32).mean().item()),
            "depth_rmse_scene_units": float(torch.sqrt((depth_values ** 2).mean()).item()),
            "runtime_seconds": float(runtime_seconds),
        }
        if consistency_mask is not None:
            consistency_mask_t = torch.as_tensor(consistency_mask, dtype=torch.bool, device=DEVICE)
            metrics["left_right_consistency_rate"] = float(consistency_mask_t[eval_mask_t].to(torch.float32).mean().item())
        return metrics, disparity_error, pred_depth, depth_error

    def run_match(reference, target, max_disparity, patch_size, direction="left_to_right"):
        start = time.perf_counter()
        cost_volume, support_mask = build_cost_volume(reference, target, patch_size, max_disparity, direction)
        pred_disparity, valid_mask, min_cost = disparity_from_cost_volume(cost_volume)
        runtime = time.perf_counter() - start
        return {
            "cost_volume": cost_volume,
            "support_mask": support_mask,
            "pred_disparity": pred_disparity,
            "valid_mask": valid_mask,
            "min_cost": min_cost,
            "runtime_seconds": runtime,
            "direction": direction,
        }

    def sample_cost_curve(cost_volume, yy, xx):
        return as_float_tensor(cost_volume)[:, yy, xx]

    def bilateral_brightness_variant(left, right):
        left_t = as_float_tensor(left)
        right_t = as_float_tensor(right)
        right_bright = torch.clamp(0.12 + 0.82 * right_t, 0.0, 1.0)
        return left_t.clone(), right_bright

    def compute_right_to_left_consistency(left_disp, right_disp, left_valid_mask, right_valid_mask, eval_mask, tol=1.0):
        left_disp_t = as_float_tensor(left_disp)
        right_disp_t = as_float_tensor(right_disp)
        left_valid_t = torch.as_tensor(left_valid_mask, dtype=torch.bool, device=DEVICE)
        right_valid_t = torch.as_tensor(right_valid_mask, dtype=torch.bool, device=DEVICE)
        eval_mask_t = torch.as_tensor(eval_mask, dtype=torch.bool, device=DEVICE)
        h, w = left_disp_t.shape
        x_coords = torch.arange(w, device=DEVICE).view(1, w).expand(h, w)
        disp_indices = torch.round(torch.where(torch.isfinite(left_disp_t), left_disp_t, torch.zeros_like(left_disp_t))).to(torch.long)
        xr = x_coords - disp_indices
        in_bounds = (xr >= 0) & (xr < w)
        xr_safe = xr.clamp(0, w - 1)
        sampled_right_disp = torch.gather(right_disp_t, 1, xr_safe)
        sampled_right_valid = torch.gather(right_valid_t.to(torch.int64), 1, xr_safe).to(torch.bool)
        consistency = (
            eval_mask_t
            & left_valid_t
            & torch.isfinite(left_disp_t)
            & in_bounds
            & sampled_right_valid
            & torch.isfinite(sampled_right_disp)
            & (torch.abs(left_disp_t - sampled_right_disp) <= tol)
        )
        return consistency

    def fit_subpixel_quadratic(cost_curve, best_disp):
        curve_t = as_float_tensor(cost_curve)
        if best_disp <= 0 or best_disp >= len(curve_t) - 1:
            return float(best_disp), None
        c1 = float(curve_t[best_disp - 1].item())
        c2 = float(curve_t[best_disp].item())
        c3 = float(curve_t[best_disp + 1].item())
        if not np.all(np.isfinite([c1, c2, c3])):
            return float(best_disp), None
        denom = c1 - 2.0 * c2 + c3
        if abs(denom) < 1e-9:
            return float(best_disp), None
        offset = 0.5 * (c1 - c3) / denom
        refined = float(best_disp + offset)
        xs = np.linspace(best_disp - 1, best_disp + 1, 200)
        ys = c2 + 0.5 * denom * (xs - best_disp) ** 2 + 0.5 * (c3 - c1) * (xs - best_disp)
        return refined, (xs, ys)

    def create_constant_disparity_pair(height=72, width=112, disparity_px=4):
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=DEVICE),
            torch.arange(width, dtype=torch.float32, device=DEVICE),
            indexing="ij",
        )
        left = 0.28 + 0.22 * torch.sin(x / 3.7) + 0.18 * torch.cos(y / 5.1) + 0.11 * torch.sin((1.9 * x + 0.8 * y) / 6.3)
        left = torch.clamp(left, 0.0, 1.0)
        right = torch.zeros_like(left)
        right[:, :-disparity_px] = left[:, disparity_px:]
        left_visible_mask = torch.zeros((height, width), dtype=torch.bool, device=DEVICE)
        left_visible_mask[:, disparity_px:] = True
        right_visible_mask = torch.zeros((height, width), dtype=torch.bool, device=DEVICE)
        right_visible_mask[:, : width - disparity_px] = True
        return {
            "left": left,
            "right": right,
            "left_visible_mask": left_visible_mask,
            "right_visible_mask": right_visible_mask,
            "disparity_px": float(disparity_px),
        }

    def run_constant_disparity_sanity_test():
        sanity = create_constant_disparity_pair()
        left_match = run_match(
            sanity["left"],
            sanity["right"],
            max_disparity=8,
            patch_size=7,
            direction="left_to_right",
        )
        right_match = run_match(
            sanity["right"],
            sanity["left"],
            max_disparity=8,
            patch_size=7,
            direction="right_to_left",
        )
        left_eval_mask = sanity["left_visible_mask"] & left_match["valid_mask"]
        right_eval_mask = sanity["right_visible_mask"] & right_match["valid_mask"]
        consistency = compute_right_to_left_consistency(
            left_match["pred_disparity"],
            right_match["pred_disparity"],
            left_match["valid_mask"],
            right_match["valid_mask"],
            left_eval_mask,
            tol=0.5,
        )
        median_left = float(torch.median(left_match["pred_disparity"][left_eval_mask]).item())
        median_right = float(torch.median(right_match["pred_disparity"][right_eval_mask]).item())
        consistency_rate = float(consistency[left_eval_mask].to(torch.float32).mean().item())
        print(
            f"Controlled 4 px sanity test: median left-to-right disparity = {median_left:.3f} px, "
            f"median right-to-left disparity = {median_right:.3f} px, "
            f"left-right consistency rate = {consistency_rate:.3f}"
        )
        assert abs(median_left - 4.0) <= 0.25
        assert abs(median_right - 4.0) <= 0.25
        assert consistency_rate >= 0.95
        return {
            "median_left_disparity": median_left,
            "median_right_disparity": median_right,
            "left_right_consistency_rate": consistency_rate,
        }

    def warp_left_to_right_fractional(left_image, disparity_px):
        left_t = as_float_tensor(left_image)
        h, w = left_t.shape
        ys, xs = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=DEVICE),
            torch.arange(w, dtype=torch.float32, device=DEVICE),
            indexing="ij",
        )
        source_x = xs + disparity_px
        source_y = ys
        valid = (source_x >= 0.0) & (source_x <= (w - 1))
        grid_x = 2.0 * source_x / max(w - 1, 1) - 1.0
        grid_y = 2.0 * source_y / max(h - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)[None]
        warped = F.grid_sample(
            left_t[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]
        return warped, valid

    def run_fractional_disparity_experiment(ground_truth_disparity=8.7):
        height, width = 48, 96
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=DEVICE),
            torch.arange(width, dtype=torch.float32, device=DEVICE),
            indexing="ij",
        )
        left = 0.31 + 0.24 * torch.sin(x / 2.9) + 0.19 * torch.cos(y / 4.1) + 0.12 * torch.sin((1.4 * x + 0.7 * y) / 5.3)
        left = torch.clamp(left, 0.0, 1.0)
        right, valid = warp_left_to_right_fractional(left, ground_truth_disparity)
        match = run_match(left, right, max_disparity=12, patch_size=9, direction="left_to_right")
        yy, xx = 22, 48
        assert bool(valid[yy, xx].item()), "Fractional-disparity sample point fell outside the valid warp support."
        curve = sample_cost_curve(match["cost_volume"], yy, xx)
        best_disp = int(torch.argmin(torch.where(torch.isfinite(curve), curve, torch.full_like(curve, float("inf")))).item())
        refined_disp, fitted = fit_subpixel_quadratic(curve, best_disp)
        integer_error = abs(best_disp - ground_truth_disparity)
        refined_error = abs(refined_disp - ground_truth_disparity)
        print(
            f"Fractional disparity experiment: gt = {ground_truth_disparity:.3f} px, "
            f"integer estimate = {best_disp:.3f} px, refined estimate = {refined_disp:.3f} px, "
            f"integer error = {integer_error:.3f} px, refined error = {refined_error:.3f} px"
        )
        assert refined_error < integer_error
        assert refined_error < 0.15
        return {
            "left": left,
            "right": right,
            "valid_mask": valid,
            "curve": curve,
            "yy": yy,
            "xx": xx,
            "ground_truth_disparity": float(ground_truth_disparity),
            "integer_disparity": float(best_disp),
            "refined_disparity": float(refined_disp),
            "integer_error": float(integer_error),
            "refined_error": float(refined_error),
            "fitted_curve": fitted,
        }

    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    def rot_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    def skew(vec):
        tx, ty, tz = vec
        return np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]], dtype=np.float64)

    def project_points(K, R, t, points):
        pixels = []
        for point in points:
            cam = R @ point + t
            pix = K @ cam
            pixels.append(np.array([pix[0] / pix[2], pix[1] / pix[2], 1.0], dtype=np.float64))
        return np.stack(pixels)

    def line_segment_in_frame(line, width, height):
        a, b, c = line
        points = []
        for x in [0.0, width - 1.0]:
            if abs(b) > 1e-9:
                y = -(a * x + c) / b
                if -10.0 <= y <= height + 10.0:
                    points.append((x, y))
        for y in [0.0, height - 1.0]:
            if abs(a) > 1e-9:
                x = -(b * y + c) / a
                if -10.0 <= x <= width + 10.0:
                    points.append((x, y))
        unique = []
        for point in points:
            if all((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 > 1e-6 for other in unique):
                unique.append(point)
        return unique[:2] if len(unique) >= 2 else None

    scene = create_synthetic_scene()
    eval_mask_template = scene["visible_mask"]
    global_artifacts = {"sanity_test": run_constant_disparity_sanity_test()}
    """

    md1 = f"""
    ## 40.1 Introduction

    Stereo begins as a perceptual phenomenon: the left and right eyes see slightly different image
    positions, and those displacements can be interpreted as depth.

    {mit_40_01}

    The chapter quickly turns that perception into a two-part computational problem:

    1. **geometry**: where can a corresponding point lie?
    2. **matching**: which candidate point is the correct one?

    The rest of the notebook follows the MIT chapter order but adds executable experiments for the
    same ideas.
    """

    md2 = f"""
    ## 40.2 Stereo Cues

    ### 40.2.1 How Far Away Is a Boat?

    The boat example introduces two depth cues before stereo algorithms appear:

    - a single-view estimate using observer height `h` and angle `alpha`, with `d = h / tan(alpha)`;
    - a two-view triangulation estimate using baseline `t` and angles `alpha` and `beta`, with
      `d = t sin(alpha) sin(beta) / sin(alpha + beta)`.

    {mit_40_02}

    ### 40.2.2 Depth from Image Disparities

    The random-dot stereogram isolates disparity from recognizable object identity. It shows that
    a depth percept can arise purely from left-right displacement.

    {mit_40_03}

    ### 40.2.3 Building a Stereo Pinhole Camera

    The chapter also grounds the perceptual story in image formation by showing a homemade
    anaglyph pinhole camera.

    {mit_40_04}

    The next code cell turns those ideas into five generated figures: geometric stereo cues,
    rectified stereo geometry, the inverse disparity-depth curve, far-depth sensitivity, and
    baseline sensitivity.
    """

    code2 = """
    single_height = 30.0
    single_alpha_deg = 8.0
    single_distance = single_height / math.tan(math.radians(single_alpha_deg))

    baseline_t = 120.0
    alpha_deg = 38.0
    beta_deg = 29.0
    triangulated_distance = baseline_t * math.sin(math.radians(alpha_deg)) * math.sin(math.radians(beta_deg)) / math.sin(
        math.radians(alpha_deg + beta_deg)
    )

    def make_random_dot_stereogram(height=84, width=128, square_size=28, shift=7):
        base = (np.random.rand(height, width) > 0.5).astype(np.float32)
        left = base.copy()
        right = base.copy()
        y0 = height // 2 - square_size // 2
        x0 = width // 2 - square_size // 2
        patch = (np.random.rand(square_size, square_size) > 0.5).astype(np.float32)
        left[y0 : y0 + square_size, x0 : x0 + square_size] = patch
        right[y0 : y0 + square_size, x0 - shift : x0 - shift + square_size] = patch
        return left, right, shift

    stereogram_left, stereogram_right, stereogram_shift = make_random_dot_stereogram()

    fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 5.0], [0, 0], color="#6b7280", lw=2.0)
    ax.scatter([0.8], [single_height], color="#111827", s=65)
    ax.scatter([4.6], [0], color=ACCENT_BLUE, s=65)
    ax.plot([0.8, 0.8], [0, single_height], color=EDGE_COLOR, lw=2.2)
    ax.plot([0.8, 4.6], [single_height, 0], color=ACCENT_RED, lw=2.2)
    ax.annotate("", xy=(1.65, single_height), xytext=(0.8, single_height), arrowprops=dict(arrowstyle="->", color="#6b7280"))
    ax.add_patch(patches.Arc((0.8, single_height), 1.2, 0.9, angle=0, theta1=318, theta2=360, color=ACCENT_GOLD, lw=2.0))
    ax.text(0.55, single_height + 3.0, "observer")
    ax.text(4.3, 2.8, "boat", color=ACCENT_BLUE)
    ax.text(0.95, single_height / 2.0, r"$h$", color=EDGE_COLOR)
    ax.text(1.18, single_height - 3.0, r"$\\alpha$", color=ACCENT_GOLD)
    ax.text(2.15, 1.2, rf"$d = h / \\tan(\\alpha) = {single_distance:.1f}$", color=ACCENT_RED)
    ax.set_title("Generated study of MIT Fig. 40.2: single-view cue")
    ax.set_xlim(-0.1, 5.2)
    ax.set_ylim(-2.0, single_height + 8.0)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot([0, 5.0], [0, 0], color="#6b7280", lw=2.0)
    left_obs = np.array([0.8, 0.0])
    right_obs = np.array([4.2, 0.0])
    boat = np.array([2.75, 2.6])
    ax.scatter([left_obs[0], right_obs[0]], [left_obs[1], right_obs[1]], color="#111827", s=65)
    ax.scatter([boat[0]], [boat[1]], color=ACCENT_BLUE, s=65)
    ax.plot([left_obs[0], boat[0]], [left_obs[1], boat[1]], color=EDGE_COLOR, lw=2.2)
    ax.plot([right_obs[0], boat[0]], [right_obs[1], boat[1]], color=ACCENT_RED, lw=2.2)
    ax.text(left_obs[0] - 0.18, 0.22, "A")
    ax.text(right_obs[0] - 0.06, 0.22, "B")
    ax.text(boat[0] + 0.08, boat[1] + 0.1, "boat", color=ACCENT_BLUE)
    ax.text(1.15, 0.42, rf"$\\alpha = {alpha_deg:.0f}^\\circ$", color=EDGE_COLOR)
    ax.text(3.02, 0.42, rf"$\\beta = {beta_deg:.0f}^\\circ$", color=ACCENT_RED)
    ax.text(1.88, -0.35, rf"$t = {baseline_t:.0f}$")
    ax.text(0.9, 2.95, rf"$d = t\\sin\\alpha\\sin\\beta / \\sin(\\alpha+\\beta) = {triangulated_distance:.1f}$")
    ax.set_title("Generated study of MIT Fig. 40.2: triangulation")
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-0.8, 3.6)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    combined = np.concatenate([stereogram_left, np.full((stereogram_left.shape[0], 6), 0.6), stereogram_right], axis=1)
    ax.imshow(combined, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.text(8, 6, "left-eye image", color="#111827", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.text(stereogram_left.shape[1] + 14, 6, "right-eye image", color="#111827", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.text(8, stereogram_left.shape[0] + 8, rf"hidden square disparity = {stereogram_shift} px", color=EDGE_COLOR)
    ax.set_title("Generated study of MIT Fig. 40.3: random-dot stereogram")
    ax.set_xticks([])
    ax.set_yticks([])
    save_figure(fig, "01-stereo-cues.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axes[0]
    ax.scatter([0.0, BASELINE_UNITS], [0.0, 0.0], color="#111827", s=110)
    point_depth = 3.2
    point_disp = FOCAL_LENGTH_PX * BASELINE_UNITS / point_depth
    point = np.array([BASELINE_UNITS / 2.0, point_depth])
    ax.scatter([point[0]], [point[1]], color=ACCENT_RED, s=95)
    ax.plot([0.0, point[0]], [0.0, point[1]], color=EDGE_COLOR, lw=2.2, linestyle="--")
    ax.plot([BASELINE_UNITS, point[0]], [0.0, point[1]], color=ACCENT_BLUE, lw=2.2, linestyle="--")
    ax.plot([0.0, BASELINE_UNITS], [0.0, 0.0], color=ACCENT_GOLD, lw=2.2)
    ax.plot([0.0, 0.0], [0.0, 0.65], color=EDGE_COLOR, lw=2.2)
    ax.plot([BASELINE_UNITS, BASELINE_UNITS], [0.0, 0.65], color=ACCENT_BLUE, lw=2.2)
    ax.annotate("", xy=(0.0, 0.36), xytext=(BASELINE_UNITS, 0.36), arrowprops=dict(arrowstyle="<->", lw=1.8, color=ACCENT_GOLD))
    ax.text(BASELINE_UNITS / 2.0 - 0.01, 0.42, "baseline $B$")
    ax.text(-0.02, 0.67, "$f$", color=EDGE_COLOR)
    ax.text(BASELINE_UNITS + 0.01, 0.67, "$f$", color=ACCENT_BLUE)
    ax.text(0.03, point_depth * 0.52, "$Z$")
    ax.text(point[0] + 0.01, point[1] + 0.12, "3D point $P$", color=ACCENT_RED)
    ax.set_title("3D stereo rig")
    ax.set_xlim(-0.08, BASELINE_UNITS + 0.12)
    ax.set_ylim(-0.2, point_depth + 0.8)
    ax.axis("off")

    ax = axes[1]
    x_left = 0.72
    x_right = x_left - point_disp / FOCAL_LENGTH_PX
    ax.axhline(0.0, color="#6b7280", lw=1.6)
    ax.plot([0.0, 0.0], [-0.38, 0.38], color=EDGE_COLOR, lw=2.2)
    ax.plot([1.0, 1.0], [-0.38, 0.38], color=ACCENT_BLUE, lw=2.2)
    ax.scatter([x_left], [0.0], color=EDGE_COLOR, s=55)
    ax.scatter([x_right], [0.0], color=ACCENT_BLUE, s=55)
    ax.annotate("", xy=(x_right, -0.16), xytext=(x_left, -0.16), arrowprops=dict(arrowstyle="<->", lw=1.8, color=ACCENT_RED))
    ax.text((x_left + x_right) / 2.0 - 0.04, -0.29, "$d = x_L - x_R$")
    ax.text(x_left - 0.02, 0.09, "$x_L$", color=EDGE_COLOR)
    ax.text(x_right - 0.02, 0.09, "$x_R$", color=ACCENT_BLUE)
    ax.text(-0.03, 0.28, "left image plane", color=EDGE_COLOR)
    ax.text(0.85, 0.28, "right image plane", color=ACCENT_BLUE)
    ax.text(0.03, -0.43, rf"$Z = fB/d = {point_depth:.2f}$")
    ax.set_title("Rectified image coordinates")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.55, 0.55)
    ax.axis("off")
    save_figure(fig, "02-rectified-stereo-geometry.png")

    disparities = np.linspace(0.5, 18.0, 300)
    depths = FOCAL_LENGTH_PX * BASELINE_UNITS / disparities
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(disparities, depths, color=EDGE_COLOR, lw=2.4)
    for disp, label in [(12.0, "near"), (6.0, "medium"), (2.0, "far")]:
        z = FOCAL_LENGTH_PX * BASELINE_UNITS / disp
        ax.scatter([disp], [z], color=ACCENT_RED, s=45)
        ax.annotate(label, (disp, z), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("disparity $d$ (pixels)")
    ax.set_ylabel("depth $Z$ (scene units)")
    ax.set_title("Inverse relationship: $Z = fB / d$")
    save_figure(fig, "03-depth-vs-disparity.png")

    depth_grid = np.linspace(1.5, 11.0, 240)
    disparity_nominal = depth_to_disparity(depth_grid)
    disparity_errors = [0.25, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for delta, color in zip(disparity_errors, [EDGE_COLOR, ACCENT_BLUE, ACCENT_RED], strict=True):
        depth_plus = disparity_to_depth(np.maximum(disparity_nominal - delta, 0.1))
        depth_minus = disparity_to_depth(disparity_nominal + delta)
        abs_error = 0.5 * ((depth_plus - depth_grid) + (depth_grid - depth_minus))
        ax.plot(depth_grid, abs_error, lw=2.2, color=color, label=rf"$\\pm {delta}$ px disparity error")
    ax.set_xlabel("true depth $Z$")
    ax.set_ylabel("approximate depth error")
    ax.set_title("Far-depth sensitivity to disparity uncertainty")
    ax.legend()
    save_figure(fig, "04-depth-error-sensitivity.png")

    baselines = [0.12, 0.24, 0.36]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for baseline, color in zip(baselines, [EDGE_COLOR, ACCENT_BLUE, ACCENT_RED], strict=True):
        disp = FOCAL_LENGTH_PX * baseline / depth_grid
        axes[0].plot(depth_grid, disp, color=color, lw=2.2, label=rf"$B = {baseline:.2f}$")
        depth_error_from_one_px = np.abs((FOCAL_LENGTH_PX * baseline) / np.maximum(disp - 0.5, 0.5) - (FOCAL_LENGTH_PX * baseline) / (disp + 0.5))
        valid = disp >= 1.25
        axes[1].plot(depth_grid[valid], depth_error_from_one_px[valid], color=color, lw=2.2, label=rf"$B = {baseline:.2f}$")
    axes[0].set_xlabel("depth $Z$")
    axes[0].set_ylabel("disparity (px)")
    axes[0].set_title("Disparity increases with baseline")
    axes[1].set_xlabel("depth $Z$")
    axes[1].set_ylabel("depth error from a 1 px disparity mistake")
    axes[1].set_title("Larger baselines reduce depth sensitivity")
    for ax in axes:
        ax.legend()
    save_figure(fig, "05-baseline-sensitivity.png")

    global_artifacts["geometry_summary"] = {
        "single_view_distance": single_distance,
        "triangulated_distance": triangulated_distance,
        "stereogram_shift_px": int(stereogram_shift),
    }
    print(json.dumps(global_artifacts["geometry_summary"], indent=2))
    """

    md3 = (
        gen_block(
            "01-stereo-cues.png",
            "Boat triangulation and a random-dot stereogram make the chapter’s depth cue visible before any stereo algorithm appears.",
        )
        + "\n\n"
        + gen_block(
            "02-rectified-stereo-geometry.png",
            "Rectified stereo geometry with baseline, focal length, corresponding points, disparity, and a triangulated 3D point.",
        )
        + "\n\n"
        + gen_block(
            "03-depth-vs-disparity.png",
            "The inverse relationship between disparity and depth. Equal disparity steps do not correspond to equal depth steps.",
        )
        + "\n\n"
        + gen_block(
            "04-depth-error-sensitivity.png",
            "A fixed disparity error creates much larger depth error at long range, which is why far geometry is fragile.",
        )
        + "\n\n"
        + gen_block(
            "05-baseline-sensitivity.png",
            "Larger baselines improve disparity signal but also create stronger view changes, overlap loss, and occlusion risk.",
        )
        + """

        The generated geometry figures sharpen three MIT chapter points:

        - positive disparity means the corresponding point appears farther left in the right image;
        - depth is **nonlinear** in disparity;
        - far points are especially sensitive to small disparity errors, so geometry alone does not
          make stereo easy.
        """
    )

    md4 = f"""
    ## 40.3 Model-Based Methods

    ### 40.3.1 Triangulation

    The simple rectified geometry above is the special case used to derive the chapter’s core
    equations:

    - disparity: `d = x_L - x_R`
    - depth: `Z = fB / d`

    {mit_40_05}

    ### 40.3.2 Stereo Matching

    Once the geometry is fixed, the practical problem is correspondence. Which pixel in the right
    image matches a given pixel in the left?

    {mit_40_06}

    The next experiments use a synthetic rectified pair so the notebook can compute a cost volume,
    run winner-takes-all disparity estimation, and measure actual error.
    """

    code4 = """
    scene = create_synthetic_scene()
    left = scene["left"]
    right = scene["right"]
    gt_disparity = scene["gt_disparity"]
    gt_depth = scene["gt_depth"]
    visible_mask = scene["visible_mask"]
    region = scene["region"]

    sample_y, sample_x = 36, 44
    pixel_match = run_match(left, right, patch_size=1, max_disparity=DEFAULT_MAX_DISPARITY, direction="left_to_right")
    patch_match = run_match(left, right, patch_size=DEFAULT_PATCH_SIZE, max_disparity=DEFAULT_MAX_DISPARITY, direction="left_to_right")
    bright_left, bright_right = bilateral_brightness_variant(left, right)
    bright_match = run_match(bright_left, bright_right, patch_size=DEFAULT_PATCH_SIZE, max_disparity=DEFAULT_MAX_DISPARITY, direction="left_to_right")

    fig = plt.figure(figsize=(12.8, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 0.9])
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, :]), None],
        ],
        dtype=object,
    )
    axes[0, 0].imshow(tensor_to_numpy(left), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].scatter([sample_x], [sample_y], color=ACCENT_RED, s=45)
    axes[0, 0].set_title("Reference pixel in the left image")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    disparities = np.arange(DEFAULT_MAX_DISPARITY + 1)
    axes[0, 1].plot(disparities, tensor_to_numpy(sample_cost_curve(pixel_match["cost_volume"], sample_y, sample_x)), marker="o", lw=2.0, label="single pixel")
    for patch_size, color in zip([3, 7, 11], [EDGE_COLOR, ACCENT_BLUE, ACCENT_RED], strict=True):
        match = run_match(left, right, patch_size=patch_size, max_disparity=DEFAULT_MAX_DISPARITY, direction="left_to_right")
        axes[0, 1].plot(disparities, tensor_to_numpy(sample_cost_curve(match["cost_volume"], sample_y, sample_x)), marker="o", lw=1.7, color=color, label=rf"patch {patch_size}x{patch_size}")
    axes[0, 1].axvline(float(gt_disparity[sample_y, sample_x].item()), color=ACCENT_GOLD, linestyle="--", lw=1.8, label="ground truth")
    axes[0, 1].set_xlabel("candidate disparity")
    axes[0, 1].set_ylabel("matching cost")
    axes[0, 1].set_title("Pixel vs patch matching costs")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(disparities, tensor_to_numpy(sample_cost_curve(patch_match["cost_volume"], sample_y, sample_x)), marker="o", lw=2.0, color=EDGE_COLOR, label="nominal pair")
    axes[1, 0].plot(disparities, tensor_to_numpy(sample_cost_curve(bright_match["cost_volume"], sample_y, sample_x)), marker="s", lw=2.0, color=ACCENT_RED, label="brightness-shifted right image")
    axes[1, 0].axvline(float(gt_disparity[sample_y, sample_x].item()), color=ACCENT_GOLD, linestyle="--", lw=1.8, label="ground truth")
    axes[1, 0].set_xlabel("candidate disparity")
    axes[1, 0].set_ylabel("matching cost")
    axes[1, 0].set_title("Brightness change shifts the cost surface")
    axes[1, 0].legend(fontsize=8, loc="upper right")
    save_figure(fig, "06-pixel-vs-patch-matching.png")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), constrained_layout=True)
    disp_levels = [2, 5, 8, 11]
    vmax = finite_quantile(patch_match["cost_volume"], 0.95)
    for ax, disp in zip(axes.ravel(), disp_levels, strict=True):
        im = ax.imshow(tensor_to_numpy(patch_match["cost_volume"][disp]), cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(f"Cost slice at disparity {disp}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="matching cost")
    save_figure(fig, "07-cost-volume-slices.png")

    disparity_error, pred_depth, depth_error = compute_error_maps(
        patch_match["pred_disparity"], gt_disparity, gt_depth
    )
    eval_mask = visible_mask & patch_match["valid_mask"]
    metrics_base, disparity_error, pred_depth, depth_error = summarize_metrics(
        patch_match["pred_disparity"], gt_disparity, gt_depth, eval_mask, patch_match["runtime_seconds"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
    vmin = finite_min(gt_disparity[eval_mask])
    vmax = finite_max(gt_disparity[eval_mask])
    im0 = axes[0].imshow(tensor_to_numpy(gt_disparity), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground-truth disparity")
    im1 = axes[1].imshow(masked_invalid_numpy(patch_match["pred_disparity"]), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Winner-takes-all disparity")
    im2 = axes[2].imshow(masked_invalid_numpy(disparity_error), cmap="magma", vmin=0.0, vmax=finite_quantile(disparity_error[eval_mask], 0.98))
    axes[2].set_title("Absolute disparity error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im0, ax=axes[:2].tolist(), fraction=0.03, pad=0.02, label="disparity (px)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="error (px)")
    save_figure(fig, "08-winner-takes-all-disparity.png")

    patch_sizes = [3, 5, 7, 9]
    patch_results = []
    for patch_size in patch_sizes:
        result = run_match(left, right, patch_size=patch_size, max_disparity=DEFAULT_MAX_DISPARITY, direction="left_to_right")
        mask = visible_mask & result["valid_mask"]
        metrics, _, _, _ = summarize_metrics(result["pred_disparity"], gt_disparity, gt_depth, mask, result["runtime_seconds"])
        patch_results.append((patch_size, result, metrics))

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), constrained_layout=True)
    for ax, (patch_size, result, metrics) in zip(axes.ravel(), patch_results, strict=True):
        im = ax.imshow(masked_invalid_numpy(result["pred_disparity"]), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Patch {patch_size}x{patch_size}  |  MAE {metrics['disparity_mae_px']:.2f} px")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.07, pad=0.07, label="disparity (px)")
    save_figure(fig, "09-patch-size-sweep.png")

    max_ranges = [6, 10, 14, 18]
    range_results = []
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), constrained_layout=True)
    for ax, max_disp in zip(axes.ravel(), max_ranges, strict=True):
        result = run_match(left, right, patch_size=DEFAULT_PATCH_SIZE, max_disparity=max_disp, direction="left_to_right")
        mask = visible_mask & result["valid_mask"]
        metrics, _, _, _ = summarize_metrics(result["pred_disparity"], gt_disparity, gt_depth, mask, result["runtime_seconds"])
        range_results.append((max_disp, result, metrics))
        im = ax.imshow(masked_invalid_numpy(result["pred_disparity"]), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(
            f"Range {max_disp} px  |  MAE {metrics['disparity_mae_px']:.2f} px  |  {metrics['runtime_seconds']:.3f} s"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.07, pad=0.07, label="disparity (px)")
    save_figure(fig, "10-max-disparity-sweep.png")

    global_artifacts["base_match"] = patch_match
    global_artifacts["patch_results"] = patch_results
    global_artifacts["range_results"] = range_results
    global_artifacts["metrics_base"] = metrics_base
    global_artifacts["scene_summary"] = {
        "ground_truth_disparity_range_px": [finite_min(gt_disparity), finite_max(gt_disparity)],
        "visible_pixel_count": int(visible_mask.to(torch.int64).sum().item()),
        "base_metrics": metrics_base,
    }
    print(json.dumps(global_artifacts["scene_summary"], indent=2))
    """

    md5 = (
        gen_block(
            "06-pixel-vs-patch-matching.png",
            "Single-pixel matching is brittle; patch aggregation stabilizes the cost surface, while brightness shifts still move the optimum.",
        )
        + "\n\n"
        + gen_block(
            "07-cost-volume-slices.png",
            "Each disparity slice of the cost volume answers one question: how plausible is this disparity at every image location?",
        )
        + "\n\n"
        + gen_block(
            "08-winner-takes-all-disparity.png",
            "Winner-takes-all disparity estimation chooses `d*(x,y) = argmin_d C(x,y,d)` independently at each pixel.",
        )
        + "\n\n"
        + gen_block(
            "09-patch-size-sweep.png",
            "Patch size is a real tradeoff: too small is noisy, too large blurs across discontinuities and occlusions.",
        )
        + "\n\n"
        + gen_block(
            "10-max-disparity-sweep.png",
            "The disparity search range must be large enough to include valid solutions but small enough to avoid wasted runtime and extra ambiguity.",
        )
        + """

        Quantitatively, the notebook now exposes the actual matching objective:

        `d*(x, y) = argmin_d C(x, y, d)`

        where `C(x, y, d)` is the matching cost at pixel `(x, y)` for candidate disparity `d`.
        The generated sweeps make three failure modes visible:

        - **too-small patches** are unstable;
        - **too-large patches** bleed across depth discontinuities;
        - **incorrect disparity ranges** either truncate the solution or waste compute.
        """
    )

    md6 = f"""
    ### 40.3.2.1 Finding image features

    Feature-based stereo is the chapter’s answer to the fragility of raw intensity matching.

    {mit_40_08}

    A good feature is localizable under small translations. The chapter expresses that with the
    Harris patch energy

    `E(Delta x, Delta y) = sum_(x,y in P) (l(x,y) - l(x + Delta x, y + Delta y))^2`

    and then motivates richer local descriptors.

    ### 40.3.2.2 Local image descriptors

    {mit_40_09}

    Oriented local structure is often more stable than raw intensity values, especially under
    small view changes.

    ### 40.3.2.3 Interpolation between feature matches

    Even after sparse feature matching, a system still needs interpolation or regularization to
    obtain dense depth. The next figures analyze the failure cases that make that interpolation
    necessary.
    """

    code6 = """
    base_match = global_artifacts["base_match"]
    pred_left = base_match["pred_disparity"]
    texture_y, texture_x = 80, 30
    repeat_y, repeat_x = 30, 112
    subpixel_y, subpixel_x = 36, 44

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4))
    axes[0].imshow(tensor_to_numpy(left), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].scatter([texture_x], [texture_y], color=ACCENT_RED, s=45)
    axes[0].set_title("Textureless region")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].plot(np.arange(DEFAULT_MAX_DISPARITY + 1), tensor_to_numpy(sample_cost_curve(base_match["cost_volume"], texture_y, texture_x)), marker="o", color=EDGE_COLOR)
    axes[1].axvline(float(gt_disparity[texture_y, texture_x].item()), color=ACCENT_GOLD, linestyle="--", lw=1.8)
    axes[1].set_xlabel("candidate disparity")
    axes[1].set_ylabel("matching cost")
    axes[1].set_title("Flat cost curve in weak texture")
    texture_patch = left[texture_y - 6 : texture_y + 7, texture_x - 6 : texture_x + 7]
    axes[2].imshow(tensor_to_numpy(texture_patch), cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Almost constant local patch")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    save_figure(fig, "11-textureless-failure.png")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4))
    axes[0].imshow(tensor_to_numpy(left), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].scatter([repeat_x], [repeat_y], color=ACCENT_RED, s=45)
    axes[0].set_title("Repetitive stripe region")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    curve = sample_cost_curve(base_match["cost_volume"], repeat_y, repeat_x)
    axes[1].plot(np.arange(DEFAULT_MAX_DISPARITY + 1), tensor_to_numpy(curve), marker="o", color=ACCENT_BLUE)
    axes[1].axvline(float(gt_disparity[repeat_y, repeat_x].item()), color=ACCENT_GOLD, linestyle="--", lw=1.8)
    axes[1].set_xlabel("candidate disparity")
    axes[1].set_ylabel("matching cost")
    axes[1].set_title("Multiple plausible minima")
    repeated_patch = left[repeat_y - 8 : repeat_y + 9, repeat_x - 12 : repeat_x + 13]
    axes[2].imshow(tensor_to_numpy(repeated_patch), cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Repeated texture causes aliasing")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    save_figure(fig, "12-repetitive-pattern-failure.png")

    right_to_left = run_match(right, left, patch_size=DEFAULT_PATCH_SIZE, max_disparity=DEFAULT_MAX_DISPARITY, direction="right_to_left")
    consistency_eval_mask = visible_mask & base_match["valid_mask"]
    consistency = compute_right_to_left_consistency(
        pred_left,
        right_to_left["pred_disparity"],
        base_match["valid_mask"],
        right_to_left["valid_mask"],
        consistency_eval_mask,
        tol=1.0,
    )
    inconsistency = consistency_eval_mask & ~consistency
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8), constrained_layout=True)
    im0 = axes[0, 0].imshow(masked_invalid_numpy(pred_left), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Left-to-right disparity")
    im1 = axes[0, 1].imshow(masked_invalid_numpy(right_to_left["pred_disparity"]), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Right-to-left disparity")
    occlusion_cmap = mcolors.ListedColormap(["#f8fafc", "#111827"])
    inconsistency_cmap = mcolors.ListedColormap(["#f8fafc", ACCENT_RED])
    axes[1, 0].imshow(tensor_to_numpy(scene["occlusion_mask"]).astype(np.float32), cmap=occlusion_cmap, vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("True synthetic occlusion mask")
    axes[1, 1].imshow(tensor_to_numpy(inconsistency).astype(np.float32), cmap=inconsistency_cmap, vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Left-right inconsistency mask")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im0, ax=[axes[0, 0], axes[0, 1]], orientation="horizontal", fraction=0.07, pad=0.08, label="disparity (px)")
    save_figure(fig, "13-occlusion-and-left-right-check.png")

    fractional_demo = run_fractional_disparity_experiment(ground_truth_disparity=8.7)
    curve = fractional_demo["curve"]
    best_disp = fractional_demo["integer_disparity"]
    refined_disp = fractional_demo["refined_disparity"]
    fitted = fractional_demo["fitted_curve"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = np.arange(DEFAULT_MAX_DISPARITY + 1)
    ax.plot(xs, tensor_to_numpy(curve), marker="o", color=EDGE_COLOR, label="integer-disparity costs")
    ax.axvline(best_disp, color=ACCENT_RED, linestyle="--", lw=1.8, label=f"best integer = {best_disp:.2f}")
    if fitted is not None:
        fit_x, fit_y = fitted
        ax.plot(fit_x, fit_y, color=ACCENT_BLUE, lw=2.0, label=f"parabolic fit, refined = {refined_disp:.2f}")
        ax.scatter([refined_disp], [np.interp(refined_disp, fit_x, fit_y)], color=ACCENT_BLUE, s=55)
    ax.axvline(fractional_demo["ground_truth_disparity"], color=ACCENT_GOLD, linestyle=":", lw=2.0, label="ground truth")
    ax.text(
        0.02,
        0.96,
        "\\n".join(
            [
                f"integer error = {fractional_demo['integer_error']:.3f} px",
                f"refined error = {fractional_demo['refined_error']:.3f} px",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", edgecolor="#d1d5db", alpha=0.92),
    )
    ax.set_xlabel("candidate disparity")
    ax.set_ylabel("matching cost")
    ax.set_title("Subpixel refinement around the best integer disparity")
    ax.legend(fontsize=8)
    save_figure(fig, "14-subpixel-refinement.png")

    global_artifacts["consistency_mask"] = consistency
    global_artifacts["right_to_left"] = right_to_left
    global_artifacts["subpixel_demo"] = {
        "integer_disparity": float(best_disp),
        "refined_disparity": float(refined_disp),
        "ground_truth_disparity": fractional_demo["ground_truth_disparity"],
        "integer_error": fractional_demo["integer_error"],
        "refined_error": fractional_demo["refined_error"],
    }
    print(json.dumps({"subpixel_demo": global_artifacts["subpixel_demo"]}, indent=2))
    """

    md7 = (
        gen_block(
            "11-textureless-failure.png",
            "In a low-texture region the matching-cost surface becomes flat, so many disparities look almost equally plausible.",
        )
        + "\n\n"
        + gen_block(
            "12-repetitive-pattern-failure.png",
            "Repeated patterns create several near-identical alignments, which makes the cost surface multimodal.",
        )
        + "\n\n"
        + gen_block(
            "13-occlusion-and-left-right-check.png",
            "Left-right consistency reveals pixels whose correspondences are unstable or absent because of occlusion.",
        )
        + "\n\n"
        + gen_block(
            "14-subpixel-refinement.png",
            "A local parabola fit around the best integer disparity can recover a subpixel estimate when the cost curve is well behaved.",
        )
        + """

        Failure analysis is now explicit:

        - **textureless regions** fail because the cost surface is flat;
        - **repetitive textures** fail because several disparities have similar cost;
        - **occlusions** fail because a point may not exist in both views at all;
        - **brightness changes** shift the entire cost curve, even when geometry is correct;
        - **too-small patches** are noisy;
        - **too-large patches** cross depth boundaries and mix objects;
        - **far-depth sensitivity** amplifies small disparity mistakes into large depth errors.

        Practical mitigations exist, but each one leaves a tradeoff behind. Smoother costs reduce
        noise but blur discontinuities, larger baselines improve precision but increase occlusion,
        and subpixel fits help only when the local minimum is already reliable.
        """
    )

    md8 = f"""
    ### 40.3.3 Constraints for Arbitrary Cameras

    Once the chapter leaves the rectified special case, corresponding points are no longer found
    by searching the same row in the second image.

    {mit_40_10}

    {mit_40_11}

    {mit_40_12}

    The epipolar constraint is the algebraic form of that geometry:

    `x'^T F x = 0`

    Valid correspondences should make the residual close to zero.

    ### 40.3.4 The Essential and Fundamental Matrices

    The chapter explains that the essential matrix uses calibrated camera coordinates, while the
    fundamental matrix absorbs the intrinsic calibration and operates in image coordinates.

    ### 40.3.4.4 Epipolar lines: The game

    {mit_40_13}

    ### 40.3.5 Image Rectification

    Rectification is a practical warp that makes epipolar lines horizontal again, reducing a 2D
    search problem to a 1D search problem.
    """

    code8 = """
    width_ep, height_ep = 320, 240
    K = np.array([[220.0, 0.0, 160.0], [0.0, 220.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R1 = np.eye(3)
    t1 = np.zeros(3)
    R2 = rot_y(np.deg2rad(12.0)) @ rot_x(np.deg2rad(-5.0))
    t2 = np.array([0.55, 0.06, 0.02], dtype=np.float64)

    world_points = np.array(
        [[0.00, 0.00, 3.2], [0.20, -0.05, 3.8], [-0.28, 0.08, 4.1], [0.18, 0.14, 2.9]],
        dtype=np.float64,
    )
    p1 = project_points(K, R1, t1, world_points)
    p2 = project_points(K, R2, t2, world_points)
    E = skew(t2) @ R2
    Fmat = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    residuals = [float(p2[idx] @ Fmat @ p1[idx]) for idx in range(len(world_points))]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    ax = axes[0]
    ax.set_xlim(0, width_ep)
    ax.set_ylim(height_ep, 0)
    ax.set_facecolor("#f8fafc")
    line = Fmat @ p1[0]
    segment = line_segment_in_frame(line, width_ep, height_ep)
    if segment is not None:
        (x0, y0), (x1, y1) = segment
        ax.plot([x0, x1], [y0, y1], color=ACCENT_RED, lw=2.2, label="epipolar line in image 2")
    ax.scatter(p2[:, 0], p2[:, 1], color=EDGE_COLOR, s=42, label="valid correspondences")
    ax.scatter([p2[0, 0]], [p2[0, 1]], color=ACCENT_BLUE, s=60, marker="s", label="match for highlighted point")
    ax.text(8, 16, rf"$|x'^T F x|$ for highlighted pair = {abs(residuals[0]):.2e}")
    ax.set_title("Point in image 1 induces an epipolar line in image 2")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    residual_abs = np.abs(residuals)
    ax.bar(np.arange(len(residuals)), residual_abs, color=[EDGE_COLOR, ACCENT_BLUE, ACCENT_RED, ACCENT_GOLD], label="absolute residual")
    ax.axhline(1e-12, color=ACCENT_RED, linestyle="--", lw=1.6, label="1e-12 reference")
    ax.set_xlabel("correspondence index")
    ax.set_ylabel("absolute numerical residual")
    ax.set_title("Epipolar residuals near floating-point precision")
    ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, "15-epipolar-constraint.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    xs = np.linspace(0, width_ep - 1, 200)
    axes[0].set_xlim(0, width_ep)
    axes[0].set_ylim(height_ep, 0)
    axes[0].set_title("Conceptual rectification illustration: before")
    axes[0].set_facecolor("#f8fafc")
    for offset, slope, color in zip([30, 80, 130, 180], [0.22, 0.15, 0.08, -0.02], [EDGE_COLOR, ACCENT_BLUE, ACCENT_RED, ACCENT_GOLD], strict=True):
        ys = offset + slope * (xs - 30.0)
        axes[0].plot(xs, ys, color=color, lw=2.0)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_xlim(0, width_ep)
    axes[1].set_ylim(height_ep, 0)
    axes[1].set_title("Conceptual rectification illustration: after")
    axes[1].set_facecolor("#f8fafc")
    for yy, color in zip([40, 85, 130, 175], [EDGE_COLOR, ACCENT_BLUE, ACCENT_RED, ACCENT_GOLD], strict=True):
        axes[1].plot([0, width_ep - 1], [yy, yy], color=color, lw=2.0)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    save_figure(fig, "16-before-after-rectification.png")

    global_artifacts["epipolar"] = {"residuals": residuals}
    print(json.dumps(global_artifacts["epipolar"], indent=2))
    """

    md9 = (
        gen_block(
            "15-epipolar-constraint.png",
            "The point-to-line relation becomes algebraic through the epipolar constraint `x'^T F x = 0`.",
        )
        + "\n\n"
        + gen_block(
            "16-before-after-rectification.png",
            "Conceptual rectification illustration: it shows the geometric goal of horizontal scanlines, not the output of a full calibrated rectification pipeline.",
        )
        + """

        These figures reconnect the later epipolar geometry back to the earlier rectified stereo
        experiments. Rectification is not a different problem; it is a practical reparameterization
        of the same correspondence constraint.
        """
    )

    md10 = f"""
    ## 40.4 Learning-Based Methods

    The chapter does not present a full deep-learning survey. Instead, it emphasizes why many
    learned systems still predict **disparity** rather than depth directly, and why they often use
    a rectified cost-volume pipeline.

    {mit_40_14}

    Important chapter ideas preserved here:

    - disparity is easier to regularize than depth under a fixed stereo rig;
    - cost volumes remain central even in learned stereo;
    - the network predicts disparity, but geometry still interprets the result.

    ## 40.5 Evaluation

    The notebook keeps the evaluation executable by measuring classical block matching on the
    synthetic stereo pair.
    """

    code10 = """
    base_match = global_artifacts["base_match"]
    pred_disparity = base_match["pred_disparity"]
    valid_mask = base_match["valid_mask"]
    consistency_mask = global_artifacts["consistency_mask"]
    eval_mask = visible_mask & valid_mask
    metrics_final, disparity_error, pred_depth, depth_error = summarize_metrics(
        pred_disparity,
        gt_disparity,
        gt_depth,
        eval_mask,
        base_match["runtime_seconds"],
        consistency_mask=consistency_mask,
    )

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6), constrained_layout=True)
    im0 = axes[0, 0].imshow(tensor_to_numpy(gt_disparity), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground-truth disparity")
    im1 = axes[0, 1].imshow(masked_invalid_numpy(pred_disparity), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Predicted disparity")
    im2 = axes[0, 2].imshow(masked_invalid_numpy(disparity_error), cmap="magma", vmin=0.0, vmax=finite_quantile(disparity_error[eval_mask], 0.98))
    axes[0, 2].set_title("Disparity error")
    robust_depth_error = torch.where(
        eval_mask & torch.isfinite(pred_disparity) & (pred_disparity > 0.75),
        depth_error,
        torch.full_like(depth_error, float("nan")),
    )
    im3 = axes[1, 0].imshow(masked_invalid_numpy(robust_depth_error), cmap="magma", vmin=0.0, vmax=finite_quantile(robust_depth_error, 0.98))
    axes[1, 0].set_title("Depth error")
    axes[1, 1].imshow(tensor_to_numpy(~eval_mask).astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Invalid / occluded mask")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.98,
        "\\n".join(
            [
                "Metric summary",
                f"MAE: {metrics_final['disparity_mae_px']:.3f} px",
                f"Bad-pixel rate (>1 px): {metrics_final['bad_pixel_rate_gt_1px']:.3f}",
                f"Valid-pixel ratio: {metrics_final['valid_pixel_ratio']:.3f}",
                f"Depth RMSE: {metrics_final['depth_rmse_scene_units']:.3f}",
                f"Left-right consistency: {metrics_final['left_right_consistency_rate']:.3f}",
                f"Runtime: {metrics_final['runtime_seconds']:.3f} s",
            ]
        ),
        va="top",
        family="monospace",
    )
    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im0, ax=[axes[0, 0], axes[0, 1]], fraction=0.025, pad=0.02, label="disparity (px)")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, label="error (px)")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04, label="depth error")
    save_figure(fig, "17-disparity-depth-error-maps.png")

    operating_points = []
    for patch_size in [3, 5, 7, 9]:
        result, median_runtime, _ = benchmark_configuration(
            left,
            right,
            patch_size=patch_size,
            max_disparity=DEFAULT_MAX_DISPARITY,
            direction="left_to_right",
        )
        mask = visible_mask & result["valid_mask"]
        metrics, _, _, _ = summarize_metrics(result["pred_disparity"], gt_disparity, gt_depth, mask, median_runtime)
        operating_points.append(
            {
                "label": f"patch={patch_size}, range={DEFAULT_MAX_DISPARITY}",
                "runtime_seconds": float(metrics["runtime_seconds"]),
                "disparity_mae_px": float(metrics["disparity_mae_px"]),
                "family": "patch sweep",
            }
        )
    for max_disp in [6, 10, 14, 18]:
        result, median_runtime, _ = benchmark_configuration(
            left,
            right,
            patch_size=DEFAULT_PATCH_SIZE,
            max_disparity=max_disp,
            direction="left_to_right",
        )
        mask = visible_mask & result["valid_mask"]
        metrics, _, _, _ = summarize_metrics(result["pred_disparity"], gt_disparity, gt_depth, mask, median_runtime)
        operating_points.append(
            {
                "label": f"patch={DEFAULT_PATCH_SIZE}, range={max_disp}",
                "runtime_seconds": float(metrics["runtime_seconds"]),
                "disparity_mae_px": float(metrics["disparity_mae_px"]),
                "family": "range sweep",
            }
        )
    operating_points.sort(key=lambda row: (row["runtime_seconds"], row["disparity_mae_px"]))

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    runtimes = np.array([row["runtime_seconds"] for row in operating_points])
    maes = np.array([row["disparity_mae_px"] for row in operating_points])
    markers = {"patch sweep": "o", "range sweep": "s"}
    colors = {"patch sweep": EDGE_COLOR, "range sweep": ACCENT_BLUE}
    label_offsets = [(6, 8), (8, -12), (8, 8), (8, -14), (-88, 8), (-88, -14), (-88, 8), (-88, -14)]
    for idx, row in enumerate(operating_points):
        ax.scatter(
            [row["runtime_seconds"]],
            [row["disparity_mae_px"]],
            marker=markers[row["family"]],
            color=colors[row["family"]],
            s=62,
            zorder=3,
        )
        ax.annotate(row["label"], (row["runtime_seconds"], row["disparity_mae_px"]), textcoords="offset points", xytext=label_offsets[idx], fontsize=8)
    best_idx = int(np.argmin(maes + 0.15 * runtimes))
    ax.scatter(
        [runtimes[best_idx]],
        [maes[best_idx]],
        marker="*",
        color=ACCENT_RED,
        edgecolors="#111827",
        linewidths=0.8,
        s=180,
        zorder=4,
        label=f"selected operating point: {operating_points[best_idx]['label']}",
    )
    ax.set_xlabel("median runtime over 10 timed runs (s)")
    ax.set_ylabel("disparity MAE (px)")
    ax.set_title("Runtime vs accuracy across operating points (median of 10 timed runs)")
    ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, "18-runtime-accuracy-tradeoff.png")

    global_artifacts["metrics_final"] = metrics_final
    global_artifacts["operating_points"] = operating_points
    print("Results table")
    print("-" * 74)
    print(f"{'label':<28} {'runtime(s)':>12} {'MAE(px)':>10}")
    print("-" * 74)
    for row in operating_points:
        print(f"{row['label']:<28} {row['runtime_seconds']:>12.3f} {row['disparity_mae_px']:>10.3f}")
    print("-" * 74)
    print(json.dumps({"metrics_final": metrics_final}, indent=2))
    """

    md11b = """
    ## Reproducibility and Validation

    This final pass audits every notebook image reference, every generated PNG, and the inline
    figure outputs produced by `save_figure(...)`. It also writes a machine-readable inventory to
    `output/image_audit.json` and a visual contact sheet to
    `output/generated_figures_contact_sheet.png`.
    """

    code11b = """
    audit_rows, audit_path = collect_notebook_image_audit()
    ref_counts = validate_notebook_rendering_contract()
    generated_paths = [IMAGES_DIR / name for name in EXPECTED_IMAGE_NAMES]
    generated_stats = [image_stats(path) for path in generated_paths]
    flagged = [
        row
        for row in generated_stats
        if (not row["exists"])
        or (not row["decodable"])
        or row["tiny_or_strip"]
        or (row["variance"] is not None and row["variance"] < 1.0)
        or (row["near_white_ratio"] is not None and row["near_white_ratio"] > 0.985)
        or (row["near_black_ratio"] is not None and row["near_black_ratio"] > 0.985)
        or (row["alpha_zero_ratio"] is not None and row["alpha_zero_ratio"] > 0.985)
    ]
    contact_sheet_path = build_contact_sheet(generated_paths, OUTPUT_DIR / "generated_figures_contact_sheet.png")
    print(f"Image audit rows: {len(audit_rows)}")
    print(f"Audit report: {audit_path}")
    print(f"Generated markdown reference counts: {ref_counts}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Flagged generated figures: {len(flagged)}")
    assert not flagged, f"Generated figures failed QA checks: {[row['path'] for row in flagged]}"
    """

    md11 = (
        gen_block(
            "17-disparity-depth-error-maps.png",
            "Evaluation should expose both disparity-space and depth-space error, because small disparity errors can turn into large depth errors far from the cameras.",
        )
        + "\n\n"
        + gen_block(
            "18-runtime-accuracy-tradeoff.png",
            "Runtime and accuracy should be read together. The timings are machine-dependent and are intended only for relative comparison within this run.",
        )
        + """

        Metrics reported by code:

        - **disparity MAE**: mean absolute disparity error over valid visible pixels;
        - **bad-pixel rate**: fraction of valid pixels with absolute disparity error greater than 1 pixel;
        - **valid-pixel ratio**: fraction of all pixels that remain usable after support and visibility checks;
        - **depth RMSE**: root-mean-square error after converting disparity to depth;
        - **left-right consistency rate**: fraction of valid pixels that agree with a reverse-direction disparity check;
        - **runtime**: wall-clock runtime for the selected matching configuration.
        """
    )

    md12 = """
    ## 40.6 Concluding Remarks

    The MIT chapter ends by staying realistic about stereo:

    - correspondence is hard even in the rectified case;
    - far geometry is fragile because depth is nonlinear in disparity;
    - occlusions and repeated structure create genuine ambiguity;
    - practical systems mix geometry, matching, regularization, and evaluation rather than relying
      on a single elegant formula.

    The notebook now reflects that balance. It keeps the MIT figures for perceptual and geometric
    context, but it also runs the chapter’s ideas as code and measures the consequences directly.
    """

    code12 = """
    expected_paths = [IMAGES_DIR / name for name in EXPECTED_IMAGE_NAMES]
    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, f"Missing generated images: {missing}"

    generated_pngs = sorted(path.name for path in IMAGES_DIR.glob("*.png"))
    expected_pngs = sorted(EXPECTED_IMAGE_NAMES)
    assert generated_pngs == expected_pngs, f"Unexpected PNG set under images/: {generated_pngs}"
    contact_sheet_path = OUTPUT_DIR / "generated_figures_contact_sheet.png"
    assert contact_sheet_path.exists(), f"Missing QA contact sheet: {contact_sheet_path}"

    image_shapes = {}
    for path in expected_paths:
        assert path.stat().st_size > 0, f"Generated image is empty: {path}"
        img = mpimg.imread(path)
        image_shapes[path.name] = list(img.shape[:2])

    metrics_final = global_artifacts["metrics_final"]
    finite_metrics = {key: float(value) for key, value in metrics_final.items()}
    assert all(np.isfinite(list(finite_metrics.values()))), f"Non-finite metric detected: {finite_metrics}"

    pred_disparity = global_artifacts["base_match"]["pred_disparity"]
    pred_depth = disparity_to_depth(pred_disparity)
    assert pred_disparity.shape == gt_disparity.shape, "Disparity output shape mismatch."
    valid_eval = visible_mask & global_artifacts["base_match"]["valid_mask"]
    assert bool(torch.isfinite(as_float_tensor(pred_depth)[valid_eval]).all().item()), "Unexpected infinite depth values in valid regions."
    assert OUTPUT_DIR.exists(), f"Output directory missing: {OUTPUT_DIR}"

    summary = {
        "generated_image_count": len(EXPECTED_IMAGE_NAMES),
        "qa_artifact_count": 1,
        "image_shapes": image_shapes,
        "metrics_final": finite_metrics,
        "disparity_shape": list(pred_disparity.shape),
        "validation_passed": True,
    }
    print(json.dumps(summary, indent=2))
    """

    cells = [
        markdown_cell(md0),
        code_cell(code0),
        markdown_cell(md1),
        markdown_cell(md2),
        code_cell(code2),
        markdown_cell(md3),
        markdown_cell(md4),
        code_cell(code4),
        markdown_cell(md5),
        markdown_cell(md6),
        code_cell(code6),
        markdown_cell(md7),
        markdown_cell(md8),
        code_cell(code8),
        markdown_cell(md9),
        markdown_cell(md10),
        code_cell(code10),
        markdown_cell(md11),
        markdown_cell(md11b),
        code_cell(code11b),
        markdown_cell(md12),
        code_cell(code12),
    ]
    return notebook_dict(cells)


def build_readme() -> str:
    generated = "\n".join(f"- `images/{name}`" for name in [
        "01-stereo-cues.png",
        "02-rectified-stereo-geometry.png",
        "03-depth-vs-disparity.png",
        "04-depth-error-sensitivity.png",
        "05-baseline-sensitivity.png",
        "06-pixel-vs-patch-matching.png",
        "07-cost-volume-slices.png",
        "08-winner-takes-all-disparity.png",
        "09-patch-size-sweep.png",
        "10-max-disparity-sweep.png",
        "11-textureless-failure.png",
        "12-repetitive-pattern-failure.png",
        "13-occlusion-and-left-right-check.png",
        "14-subpixel-refinement.png",
        "15-epipolar-constraint.png",
        "16-before-after-rectification.png",
        "17-disparity-depth-error-maps.png",
        "18-runtime-accuracy-tradeoff.png",
    ])
    return textwrap.dedent(
        f"""
        # Chapter 40: Stereo Vision

        This chapter folder contains a chapter-aligned executable notebook for MIT
        *Foundations of Computer Vision*, Chapter 40.

        The notebook now keeps all MIT Figures `40.1` to `40.14` visible in the markdown and adds
        eighteen code-generated teaching figures covering stereo geometry, matching, failure modes,
        epipolar geometry, and evaluation.

        ## Generated figures

        {generated}

        ## How to run

        ```bash
        python3 notebooks/CV/2026/spring/final/chapter-40-stereo-vision/build_assets.py
        jupyter nbconvert --to notebook --execute notebooks/CV/2026/spring/final/chapter-40-stereo-vision/index.ipynb --output index.executed.ipynb --ExecutePreprocessor.timeout=600
        ```
        """
    ).strip() + "\n"


def build_assets_readme() -> str:
    return textwrap.dedent(
        """
        # MIT Vision Book Reference Assets

        These are the local Chapter 40 source images used by the notebook. Their filenames preserve
        the official textbook figure numbering and were verified visually against the MIT Vision
        Book Chapter 40 HTML source.

        | MIT figure | Local asset | Official source file |
        | --- | --- | --- |
        | 40.1 | `figure-40-01-titanic.png` | `https://visionbook.mit.edu/figures/3d_scene_understanding/titanic.png` |
        | 40.2 | `figure-40-02-boats.png` | `https://visionbook.mit.edu/figures/3d_scene_understanding/boats.png` |
        | 40.3 | `figure-40-03-random-dot-stereogram.png` | `https://visionbook.mit.edu/figures/3d_scene_understanding/random_dot_stereogram.png` |
        | 40.4 | `figure-40-04-anaglyph-camera.png` | `https://visionbook.mit.edu/figures/3d_scene_understanding/anaglyph_camera.png` |
        | 40.5 | `figure-40-05-triangulation-stereo.png` | `https://visionbook.mit.edu/figures/3d_scene_understanding/triangularization_stereo.png` |
        | 40.6a | `figure-40-06-office-left.jpg` | `https://visionbook.mit.edu/figures/stereo/officeLeft.jpg` |
        | 40.6b | `figure-40-06-office-right.jpg` | `https://visionbook.mit.edu/figures/stereo/officeRight.jpg` |
        | 40.7 | `figure-40-07-intensity-matching-failure.jpg` | `https://visionbook.mit.edu/figures/stereo/stereolamp.jpg` |
        | 40.8a | `figure-40-08-points-left.jpg` | `https://visionbook.mit.edu/figures/stereo/pointsLeft.jpg` |
        | 40.8b | `figure-40-08-points-right.jpg` | `https://visionbook.mit.edu/figures/stereo/pointsRight.jpg` |
        | 40.8c | `figure-40-08-depth.jpg` | `https://visionbook.mit.edu/figures/stereo/officeDepth.jpg` |
        | 40.9a | `figure-40-09-oriented-features.jpg` | `https://visionbook.mit.edu/figures/stereo/hands.jpg` |
        | 40.9b | `figure-40-09-sift-descriptor.jpg` | `https://visionbook.mit.edu/figures/stereo/SIFT.jpg` |
        | 40.10 | `figure-40-10-correspondence-ambiguity.png` | `https://visionbook.mit.edu/figures/3d_scene_understanding/epipolar_1.png` |
        | 40.11 | `figure-40-11-epipolar-ray.png` | `https://visionbook.mit.edu/figures/3d_scene_understanding/epipolar_3.png` |
        | 40.12 | `figure-40-12-epipolar-geometry.png` | `https://visionbook.mit.edu/figures/3d_scene_understanding/epipolar_geometry.png` |
        | 40.13 | `figure-40-13-epipolar-game.png` | `https://visionbook.mit.edu/figures/3d_scene_understanding/epipolar_game_play.png` |
        | 40.14 | `figure-40-14-stereo-cnn-block-diagram.jpg` | `https://visionbook.mit.edu/figures/stereo/stereocnn.jpg` |
        """
    ).strip() + "\n"


def main() -> None:
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=1))
    README_PATH.write_text(build_readme())
    ASSETS_README_PATH.write_text(build_assets_readme())
    print(f"Wrote notebook: {NOTEBOOK_PATH}")
    print(f"Wrote README: {README_PATH}")
    print(f"Wrote assets README: {ASSETS_README_PATH}")


if __name__ == "__main__":
    main()
