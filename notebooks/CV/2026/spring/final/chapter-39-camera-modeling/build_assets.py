from __future__ import annotations

import json
import textwrap
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = CHAPTER_DIR / "index.ipynb"
README_PATH = CHAPTER_DIR / "README.md"
ASSETS_README_PATH = CHAPTER_DIR / "assets" / "mit-book" / "README.md"
ASSETS_DIR = CHAPTER_DIR / "assets" / "mit-book"


FIGURES = [
    {
        "figure": "39.1",
        "section": "39.1",
        "files": ["figure-39-01-world-and-camera-coordinates.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/world_and_camera_coordinates.png"],
        "purpose": "Camera-centric and world-centric coordinate systems for describing image formation.",
        "caption": "This original textbook figure contrasts a camera-centered frame with a world-centered frame. It matters here because the whole chapter is about moving between those frames cleanly.",
    },
    {
        "figure": "39.2",
        "section": "39.1",
        "files": ["figure-39-02-the-picture.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/the_picture.png"],
        "purpose": "The scene photograph associated with the introductory sketch.",
        "caption": "This original textbook figure shows the image whose geometry the chapter later models. It matters because calibration always connects a real picture back to a 3D scene.",
    },
    {
        "figure": "39.3",
        "section": "39.2",
        "files": ["figure-39-03-perspective-projection.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/camera_centric.png"],
        "purpose": "Perspective projection from similar triangles.",
        "caption": "This original textbook figure shows how a 3D point projects through a pinhole onto the image plane. It matters because homogeneous coordinates re-express this geometry in matrix form.",
    },
    {
        "figure": "39.4",
        "section": "39.3.1",
        "files": ["figure-39-04-pinhole-and-sensor.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/pinhole_and_sensor.png"],
        "purpose": "Projection onto the sensor and conversion from metric units to pixels.",
        "caption": "This original textbook figure shows how focal length, sensor width, and pixel sampling connect physical geometry to image coordinates. It matters because intrinsic calibration lives in that conversion.",
    },
    {
        "figure": "39.5",
        "section": "39.3.1",
        "files": ["figure-39-05-coordinate-conventions.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/conventions_coordinates.png"],
        "purpose": "Different image-plane coordinate conventions.",
        "caption": "This original textbook figure compares common image-coordinate conventions. It matters because sign choices and origin placement change how we write the intrinsic matrix.",
    },
    {
        "figure": "39.6",
        "section": "39.3.2",
        "files": ["figure-39-06-light-ray.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/light_ray.png"],
        "purpose": "Backprojecting an image point into a 3D ray.",
        "caption": "This original textbook figure shows that a single pixel defines a whole 3D ray, not a unique 3D point. It matters because depth is what turns that ray back into a specific scene location.",
    },
    {
        "figure": "39.7",
        "section": "39.3.3",
        "files": [
            "figure-39-07-simple-calibration-setup.jpg",
            "figure-39-07-simple-calibration-photo.jpg",
        ],
        "urls": [
            "https://visionbook.mit.edu/figures/imaging_geometry/simple_calibration_1.jpg",
            "https://visionbook.mit.edu/figures/imaging_geometry/simple_calibration_2.jpg",
        ],
        "purpose": "A simple calibration setup and the resulting chessboard image.",
        "caption": "This original textbook figure groups the physical setup and the captured chessboard image. It matters because the chapter first introduces calibration as a measurement problem before presenting more general estimation methods.",
    },
    {
        "figure": "39.8",
        "section": "39.4",
        "files": ["figure-39-08-world-and-camera-coordinates-2.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/world_and_camera_coordinates_2.png"],
        "purpose": "Transforming world coordinates into camera coordinates with rotation and translation.",
        "caption": "This original textbook figure shows the extrinsic relationship between the world frame and the camera frame. It matters because `R` and `T` explain where the camera is and how it is oriented.",
    },
    {
        "figure": "39.9",
        "section": "39.5",
        "files": ["figure-39-09-summary-camera-model.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/summary_camera_model.png"],
        "purpose": "Summary of the full camera model from world point to image point.",
        "caption": "This original textbook figure summarizes the full pipeline from world coordinates to camera coordinates to pixels. It matters because the projection matrix combines all of those steps.",
    },
    {
        "figure": "39.10",
        "section": "39.6",
        "files": ["figure-39-10-camera-calibration-scenarios.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/camera_calibration_scenarios.png"],
        "purpose": "Four camera-pose examples of increasing geometric complexity.",
        "caption": "This original textbook figure collects the concrete camera-pose cases discussed in the chapter. It matters because the same matrix model can represent all of them.",
    },
    {
        "figure": "39.11",
        "section": "39.6",
        "files": ["figure-39-11-horizon-heads.jpg"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/horizon_heads.jpg"],
        "purpose": "People of similar height align near the horizon when the camera is level.",
        "caption": "This original textbook figure shows a practical horizon cue in a level camera. It matters because camera orientation leaves visible traces in ordinary photographs.",
    },
    {
        "figure": "39.12",
        "section": "39.6",
        "files": ["figure-39-12-eyes-location.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/eyes_location.png"],
        "purpose": "Equal-height people at different depths sharing a common image row.",
        "caption": "This original textbook figure sketches why equal-height points can line up in the image despite being at different depths. It matters because the horizon is a geometric consequence of camera pose.",
    },
    {
        "figure": "39.13",
        "section": "39.6",
        "files": [
            "figure-39-13-low-horizon.jpg",
            "figure-39-13-high-horizon.jpg",
        ],
        "urls": [
            "https://visionbook.mit.edu/figures/imaging_geometry/low_horizon_vp.jpg",
            "https://visionbook.mit.edu/figures/imaging_geometry/high_horizon_vp.jpg",
        ],
        "purpose": "Two real photos with different horizon-line locations caused by tilt changes.",
        "caption": "This original textbook figure groups two photos taken with different camera tilt angles. It matters because changing the camera pitch shifts the horizon line in a predictable way.",
    },
    {
        "figure": "39.14",
        "section": "39.7.4",
        "files": ["figure-39-14-reprojection-error.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/reprojection_error.png"],
        "purpose": "Reprojection error as a distance between observed and predicted image points.",
        "caption": "This original textbook figure shows reprojection error directly on the image plane. It matters because nonlinear refinement usually optimizes camera parameters by minimizing this quantity.",
    },
    {
        "figure": "39.15",
        "section": "39.7.5",
        "files": ["figure-39-15-office-measurements.jpg"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/office_with_measures_2.jpg"],
        "purpose": "Office photo with measured real-world distances.",
        "caption": "This original textbook figure shows the real office scene used in the toy calibration example. It matters because the later 3D annotations are grounded in these measurements.",
    },
    {
        "figure": "39.16",
        "section": "39.7.5",
        "files": ["figure-39-16-office-correspondences.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/correspondences_img.png"],
        "purpose": "Office image plus 3D world coordinates for sparse correspondences.",
        "caption": "This original textbook figure pairs image observations with measured 3D coordinates. It matters because calibration needs matched 3D scene points and 2D image points.",
    },
    {
        "figure": "39.17",
        "section": "39.7.5",
        "files": ["figure-39-17-estimated-camera.png"],
        "urls": ["https://visionbook.mit.edu/figures/imaging_geometry/result_toymodel_3dscene_and_estimated_camera_b.png"],
        "purpose": "Estimated camera pose for the office toy example.",
        "caption": "This original textbook figure visualizes the inferred camera location from several viewpoints. It matters because a good calibration should produce a physically plausible camera pose.",
    },
]


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
          <p><em>{label} {caption}</em></p>
        </div>
        """
    ).strip()


def mit_figure_block(figure: str, section: str, files: list[str], explanation: str, widths: list[str] | None = None) -> str:
    paths = [f"assets/mit-book/{name}" for name in files]
    label = f"MIT Vision Book — Section {section}, Figure {figure}."
    return image_block(paths, f"MIT Vision Book Figure {figure}", explanation, label, widths=widths)


def generated_block(filename: str, label: str, explanation: str) -> str:
    return image_block([f"images/{filename}"], filename, explanation, label)


def build_notebook() -> dict[str, object]:
    mit = {item["figure"]: item for item in FIGURES}

    cells: list[dict[str, object]] = []

    cells.append(
        markdown_cell(
            """
            # Chapter 39: Camera Modeling and Calibration

            This notebook is a compact executable companion to MIT Vision Book Chapter 39.
            It keeps the chapter's original structure, embeds every official MIT figure locally,
            and adds a small number of deterministic CPU-friendly visualizations to make the
            geometry concrete.
            """
        )
    )

    cells.append(
        code_cell(
            """
            import json
            import math
            import random
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch

            SEED = 39
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            NOTEBOOK_DIR = Path.cwd().resolve()
            if not (NOTEBOOK_DIR / "index.ipynb").exists():
                candidate = NOTEBOOK_DIR / "notebooks" / "CV" / "2026" / "spring" / "final" / "chapter-39-camera-modeling"
                if candidate.exists():
                    NOTEBOOK_DIR = candidate.resolve()
                else:
                    matches = list(NOTEBOOK_DIR.rglob("chapter-39-camera-modeling/index.ipynb"))
                    assert matches, f"Could not resolve the chapter directory from {Path.cwd()}"
                    NOTEBOOK_DIR = matches[0].parent.resolve()

            CHAPTER_DIR = NOTEBOOK_DIR
            ASSETS_DIR = CHAPTER_DIR / "assets" / "mit-book"
            IMAGES_DIR = CHAPTER_DIR / "images"
            OUTPUT_DIR = CHAPTER_DIR / "output"
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            plt.rcParams.update(
                {
                    "figure.facecolor": "white",
                    "axes.facecolor": "#fbfbf8",
                    "axes.edgecolor": "#374151",
                    "font.size": 11,
                }
            )

            def savefig(fig, filename: str) -> None:
                path = IMAGES_DIR / filename
                fig.savefig(path, dpi=180, bbox_inches="tight")
                plt.close(fig)

            def homogenize(points: torch.Tensor) -> torch.Tensor:
                ones = torch.ones((points.shape[0], 1), dtype=points.dtype)
                return torch.cat([points, ones], dim=1)

            def rotation_x(angle_deg: float) -> torch.Tensor:
                t = math.radians(angle_deg)
                c, s = math.cos(t), math.sin(t)
                return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)

            def rotation_y(angle_deg: float) -> torch.Tensor:
                t = math.radians(angle_deg)
                c, s = math.cos(t), math.sin(t)
                return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)

            def make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
                return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float64)

            def project_points(points_world: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                points_cam = (R @ (points_world - T).T).T
                pixels_h = (K @ points_cam.T).T
                pixels = pixels_h[:, :2] / pixels_h[:, 2:].clamp_min(1e-9)
                return pixels, points_cam

            def build_dlt_matrix(points_world: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
                rows = []
                for P, p in zip(points_world, pixels, strict=True):
                    X, Y, Z = [float(v) for v in P]
                    x, y = [float(v) for v in p]
                    rows.append([-X, -Y, -Z, -1.0, 0.0, 0.0, 0.0, 0.0, x * X, x * Y, x * Z, x])
                    rows.append([0.0, 0.0, 0.0, 0.0, -X, -Y, -Z, -1.0, y * X, y * Y, y * Z, y])
                return torch.tensor(rows, dtype=torch.float64)

            def estimate_projection_matrix_dlt(points_world: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
                A = build_dlt_matrix(points_world, pixels)
                _, _, vh = torch.linalg.svd(A)
                M = vh[-1].reshape(3, 4)
                return M / torch.linalg.norm(M)

            def project_with_matrix(points_world: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
                Xh = homogenize(points_world).T
                ph = (M @ Xh).T
                return ph[:, :2] / ph[:, 2:3].clamp_min(1e-9)

            EXPECTED_ASSETS = [
                "figure-39-01-world-and-camera-coordinates.png",
                "figure-39-02-the-picture.png",
                "figure-39-03-perspective-projection.png",
                "figure-39-04-pinhole-and-sensor.png",
                "figure-39-05-coordinate-conventions.png",
                "figure-39-06-light-ray.png",
                "figure-39-07-simple-calibration-setup.jpg",
                "figure-39-07-simple-calibration-photo.jpg",
                "figure-39-08-world-and-camera-coordinates-2.png",
                "figure-39-09-summary-camera-model.png",
                "figure-39-10-camera-calibration-scenarios.png",
                "figure-39-11-horizon-heads.jpg",
                "figure-39-12-eyes-location.png",
                "figure-39-13-low-horizon.jpg",
                "figure-39-13-high-horizon.jpg",
                "figure-39-14-reprojection-error.png",
                "figure-39-15-office-measurements.jpg",
                "figure-39-16-office-correspondences.png",
                "figure-39-17-estimated-camera.png",
            ]
            missing_assets = [name for name in EXPECTED_ASSETS if not (ASSETS_DIR / name).exists()]
            assert not missing_assets, f"Missing MIT Book assets: {missing_assets}"
            """
        )
    )

    cells.append(
        markdown_cell(
            f"""
            ## 39.1 Introduction

            Chapter 39 shifts from the simple pinhole stories of earlier chapters to a more useful
            camera model: one that can describe where the camera sits in the world, how it is
            oriented, and how 3D points become pixels.

            {mit_figure_block(mit["39.1"]["figure"], mit["39.1"]["section"], mit["39.1"]["files"], mit["39.1"]["caption"])}

            {mit_figure_block(mit["39.2"]["figure"], mit["39.2"]["section"], mit["39.2"]["files"], mit["39.2"]["caption"], widths=["38%"])}

            The chapter's core idea is that calibration is about estimating the transformation from
            world coordinates to image coordinates. Intrinsic parameters describe the camera itself.
            Extrinsic parameters describe the camera's position and pose relative to the scene.
            """
        )
    )

    cells.append(
        markdown_cell(
            f"""
            ## 39.2 3D Camera Projections in Homogeneous Coordinates

            Perspective projection contains a division by depth, which makes the equations awkward in
            ordinary Euclidean coordinates. Homogeneous coordinates let us express the same mapping as
            a matrix multiplication followed by a normalization step.

            {mit_figure_block(mit["39.3"]["figure"], mit["39.3"]["section"], mit["39.3"]["files"], mit["39.3"]["caption"], widths=["82%"])}

            The point to keep in mind is that a 3D location `P = [X, Y, Z]^T` projects to image
            coordinates roughly proportional to `X/Z` and `Y/Z`. Nearby points look larger because
            dividing by a smaller depth amplifies the image coordinates.

            ### 39.2.1 Parallel Projection

            Parallel projection removes that depth-dependent division. It is a simpler model that is
            sometimes useful for distant scenes or for deriving intuition, but it does not capture
            the foreshortening that real pinhole cameras produce.
            """
        )
    )

    cells.append(
        code_cell(
            """
            z_values = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)
            x_offsets = torch.tensor([1.2, 1.2, 1.2], dtype=torch.float64)
            y_offsets = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float64)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            for depth, x_val, y_val in zip(z_values, x_offsets, y_offsets, strict=True):
                perspective = torch.tensor([x_val / depth, y_val / depth], dtype=torch.float64)
                parallel = torch.tensor([x_val, y_val], dtype=torch.float64)
                axes[0].scatter(float(perspective[0]), float(perspective[1]), s=70, label=f"Z={depth:.0f}")
                axes[1].scatter(float(parallel[0]), float(parallel[1]), s=70, label=f"Z={depth:.0f}")

            for ax, title in zip(axes, ["Perspective projection", "Parallel projection"], strict=True):
                ax.axhline(0.0, color="#9ca3af", lw=1.0)
                ax.axvline(0.0, color="#9ca3af", lw=1.0)
                ax.set_xlabel("image x")
                ax.set_ylabel("image y")
                ax.set_title(title)
                ax.legend(frameon=False)
                ax.set_aspect("equal", adjustable="box")

            axes[0].text(0.15, 0.72, "Farther points move toward the origin", fontsize=10)
            axes[1].text(0.62, 0.88, "Depth does not change image location", fontsize=10)
            fig.suptitle("Projection model comparison", fontsize=14)
            savefig(fig, "01-projection-models.png")
            """
        )
    )

    cells.append(
        markdown_cell(
            generated_block(
                "01-projection-models.png",
                "**Supplemental visualization for MIT Vision Book Section 39.2.**",
                "We projected the same lateral 3D point at three depths using both perspective and parallel projection. The perspective panel shrinks image coordinates with increasing depth, while the parallel panel keeps them fixed. That is the key geometric difference between Sections 39.2 and 39.2.1.",
            )
        )
    )

    cells.append(
        markdown_cell(
            f"""
            ## 39.3 Camera-Intrinsic Parameters

            Intrinsic parameters describe how the camera turns rays into pixels. In practice, this
            includes focal scaling, the principal point, and sometimes unequal pixel scales, skew,
            or lens distortion.

            ### 39.3.1 From Meters to Pixels

            {mit_figure_block(mit["39.4"]["figure"], mit["39.4"]["section"], mit["39.4"]["files"], mit["39.4"]["caption"])}

            The intrinsic matrix tells us how to convert camera-plane coordinates into pixel
            coordinates. The chapter uses `a` and `b` for horizontal and vertical focal scaling, and
            `(c_x, c_y)` for the principal point.

            {mit_figure_block(mit["39.5"]["figure"], mit["39.5"]["section"], mit["39.5"]["files"], mit["39.5"]["caption"])}

            In code, it is worth being explicit about conventions: image origins, axis directions,
            and sign choices all affect the exact matrix form even when the geometry is the same.

            ### 39.3.2 From Pixels to Rays

            {mit_figure_block(mit["39.6"]["figure"], mit["39.6"]["section"], mit["39.6"]["files"], mit["39.6"]["caption"], widths=["76%"])}

            Backprojection reverses the forward camera map only up to a ray. If a pixel is known and
            the depth is unknown, there are infinitely many 3D points consistent with that pixel.

            ### 39.3.3 A Simple, although Unreliable, Calibration Method

            {mit_figure_block(mit["39.7"]["figure"], mit["39.7"]["section"], mit["39.7"]["files"], mit["39.7"]["caption"], widths=["60%", "32%"])}

            This sanity-check method uses a known target, a measured distance, and the apparent size
            of the target in the image to estimate focal scaling. It is useful for intuition, but it
            ignores many practical effects such as distortion and imperfect alignment.

            ### 39.3.4 Other Camera Parameters

            Real cameras may also need skew terms, separate horizontal and vertical pixel scales, and
            especially distortion correction. In practice, radial distortion is often the first extra
            effect that visibly breaks the simplest pinhole model.
            """
        )
    )

    cells.append(
        code_cell(
            """
            fx_values = [450.0, 750.0]
            principal_points = [(0.0, 0.0), (0.55, -0.35)]
            grid = torch.tensor(
                [
                    [-1.0, -0.8, 4.0],
                    [-0.4,  0.4, 4.0],
                    [ 0.3, -0.1, 4.0],
                    [ 0.9,  0.7, 4.0],
                ],
                dtype=torch.float64,
            )

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            for fx in fx_values:
                K = make_intrinsics(fx, fx, 0.0, 0.0)
                pixels, _ = project_points(grid, K, torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
                axes[0].scatter(pixels[:, 0], pixels[:, 1], s=70, label=f"f={fx:.0f}")

            for cx, cy in principal_points:
                K = make_intrinsics(600.0, 600.0, cx * 600.0, cy * 600.0)
                pixels, _ = project_points(grid, K, torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
                axes[1].scatter(pixels[:, 0], pixels[:, 1], s=70, label=f"(cx, cy)=({cx:.2f}, {cy:.2f})")

            axes[0].set_title("Focal scaling changes magnification")
            axes[1].set_title("Principal point shifts the whole image")
            for ax in axes:
                ax.axhline(0.0, color="#9ca3af", lw=1.0)
                ax.axvline(0.0, color="#9ca3af", lw=1.0)
                ax.set_xlabel("pixel x")
                ax.set_ylabel("pixel y")
                ax.legend(frameon=False)
                ax.set_aspect("equal", adjustable="box")

            fig.suptitle("Two key intrinsic-parameter effects", fontsize=14)
            savefig(fig, "02-intrinsics-effects.png")
            """
        )
    )

    cells.append(
        markdown_cell(
            generated_block(
                "02-intrinsics-effects.png",
                "**Computational reconstruction related to MIT Vision Book Section 39.3, Figure 39.4.**",
                "We projected the same four 3D points while changing either focal scaling or the principal point. Larger focal length magnifies the image, while changing `(c_x, c_y)` translates all projected points together. Those are two of the central effects encoded by the intrinsic matrix.",
            )
        )
    )

    cells.append(
        code_cell(
            """
            pixel = torch.tensor([220.0, -120.0], dtype=torch.float64)
            focal = 500.0
            depths = torch.tensor([1.5, 3.0, 5.0], dtype=torch.float64)
            ray_dir = torch.tensor([pixel[0] / focal, pixel[1] / focal, 1.0], dtype=torch.float64)
            ray_points = depths.unsqueeze(1) * ray_dir.unsqueeze(0)

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot([0.0, float(ray_points[-1, 0])], [0.0, float(ray_points[-1, 1])], [0.0, float(ray_points[-1, 2])], color="#0f766e", lw=2.5)
            ax.scatter(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], color="#dc2626", s=55)
            ax.scatter([0.0], [0.0], [0.0], color="#111827", s=65)

            for idx, depth in enumerate(depths):
                point = ray_points[idx]
                ax.text(float(point[0]), float(point[1]), float(point[2]) + 0.12, f"Z={depth:.1f}", fontsize=10)

            ax.set_title("Backprojecting one pixel into a 3D ray")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=25, azim=-60)
            savefig(fig, "03-pixel-to-ray-backprojection.png")
            """
        )
    )

    cells.append(
        markdown_cell(
            generated_block(
                "03-pixel-to-ray-backprojection.png",
                "**Computational reconstruction related to MIT Vision Book Section 39.3.2, Figure 39.6.**",
                "We chose one pixel direction and sampled three depths along its backprojected ray. The computed points all correspond to the same image location, which is exactly why a pixel alone is insufficient to recover a unique 3D point.",
            )
        )
    )

    cells.append(
        markdown_cell(
            f"""
            ## 39.4 Camera-Extrinsic Parameters

            Extrinsic parameters answer a different question from intrinsics: where is the camera in
            the world, and how is it rotated relative to the world frame?

            {mit_figure_block(mit["39.8"]["figure"], mit["39.8"]["section"], mit["39.8"]["files"], mit["39.8"]["caption"], widths=["82%"])}

            The chapter writes this mapping as a rotation followed by a translation in homogeneous
            coordinates. Once points are expressed in the camera frame, the intrinsic matrix can
            project them into the image.

            ## 39.5 Full Camera Model

            {mit_figure_block(mit["39.9"]["figure"], mit["39.9"]["section"], mit["39.9"]["files"], mit["39.9"]["caption"], widths=["82%"])}

            The full projection matrix composes intrinsics with extrinsics. In compact notation, the
            camera matrix is often written as `M = K [R | -RT]`.

            ## 39.6 A Few Concrete Examples

            {mit_figure_block(mit["39.10"]["figure"], mit["39.10"]["section"], mit["39.10"]["files"], mit["39.10"]["caption"])}

            The chapter then walks through level cameras, tilted cameras, and more structured ground
            scenes to show how pose affects the final image equations.

            {mit_figure_block(mit["39.11"]["figure"], mit["39.11"]["section"], mit["39.11"]["files"], mit["39.11"]["caption"], widths=["62%"])}

            {mit_figure_block(mit["39.12"]["figure"], mit["39.12"]["section"], mit["39.12"]["files"], mit["39.12"]["caption"], widths=["92%"])}

            {mit_figure_block(mit["39.13"]["figure"], mit["39.13"]["section"], mit["39.13"]["files"], mit["39.13"]["caption"], widths=["48%", "48%"])}

            A helpful way to read these examples is to ask which quantities stay fixed when the
            camera tilts or translates. Horizon-line motion is especially useful because it gives a
            visible signature of camera pitch.
            """
        )
    )

    cells.append(
        code_cell(
            """
            points_world = torch.tensor(
                [
                    [-2.0, 1.7, 6.0],
                    [ 0.0, 1.7, 10.0],
                    [ 2.0, 1.7, 14.0],
                    [-2.0, 0.0, 8.0],
                    [ 2.0, 0.0, 12.0],
                ],
                dtype=torch.float64,
            )
            K = make_intrinsics(500.0, 500.0, 0.0, 0.0)
            T = torch.tensor([0.0, 1.6, 0.0], dtype=torch.float64)
            pitches = [0.0, 15.0]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, pitch in zip(axes, pitches, strict=True):
                R = rotation_x(pitch)
                pixels, _ = project_points(points_world, K, R, T)
                ax.scatter(pixels[:, 0], pixels[:, 1], s=65, color="#0f766e")
                ax.axhline(500.0 * math.tan(math.radians(pitch)), color="#dc2626", linestyle="--", lw=2, label="predicted horizon")
                ax.axhline(0.0, color="#9ca3af", lw=1.0)
                ax.axvline(0.0, color="#9ca3af", lw=1.0)
                ax.set_title(f"Pitch = {pitch:.0f}°")
                ax.set_xlabel("pixel x")
                ax.set_ylabel("pixel y")
                ax.legend(frameon=False)
                ax.set_aspect("equal", adjustable="box")

            fig.suptitle("Extrinsic pose changes projected layout and horizon height", fontsize=14)
            savefig(fig, "04-extrinsics-and-horizon.png")
            """
        )
    )

    cells.append(
        markdown_cell(
            generated_block(
                "04-extrinsics-and-horizon.png",
                "**Supplemental visualization for MIT Vision Book Section 39.6.**",
                "We projected a small set of equal-height scene points with a level camera and with a 15 degree downward pitch. The dashed line marks the predicted horizon height `f tan(theta)`, showing how camera pitch shifts where equal-height points land in the image.",
            )
        )
    )

    cells.append(
        markdown_cell(
            f"""
            ## 39.7 Camera Calibration

            Calibration estimates the mapping from known 3D scene points to observed 2D image
            points. The chapter first introduces a linear estimate of the projection matrix and then
            discusses how to recover more interpretable intrinsic and extrinsic parameters.

            ### 39.7.1 Direct Linear Transform

            DLT solves for a projection matrix by stacking linear equations from several 3D-to-2D
            correspondences. The result is determined only up to an overall scale, which is fine for
            projective geometry.

            ### 39.7.2 Recovering Intrinsic and Extrinsic Camera Parameters

            Once a projection matrix has been estimated, we still need to factor it into camera
            intrinsics and pose. Conceptually, this means separating the left `3x3` calibration part
            from the world-to-camera pose terms.

            ### 39.7.3 Multiplane Calibration Method

            Multiplane calibration uses several views of a known planar target to stabilize the
            parameter estimates. This is closer to what practical camera-calibration toolkits do.

            ### 39.7.4 Nonlinear Optimization by Minimizing Reprojection Error

            {mit_figure_block(mit["39.14"]["figure"], mit["39.14"]["section"], mit["39.14"]["files"], mit["39.14"]["caption"], widths=["62%"])}

            Linear estimates are often only the starting point. A better model is usually obtained by
            refining parameters to minimize the distance between observed image points and predicted
            image points.

            ### 39.7.5 A Toy Example

            {mit_figure_block(mit["39.15"]["figure"], mit["39.15"]["section"], mit["39.15"]["files"], mit["39.15"]["caption"])}

            {mit_figure_block(mit["39.16"]["figure"], mit["39.16"]["section"], mit["39.16"]["files"], mit["39.16"]["caption"])}

            {mit_figure_block(mit["39.17"]["figure"], mit["39.17"]["section"], mit["39.17"]["files"], mit["39.17"]["caption"])}

            The office example shows the whole calibration story end to end: measure some 3D
            positions, annotate the corresponding image points, estimate the camera, and then inspect
            whether the result is plausible.
            """
        )
    )

    cells.append(
        code_cell(
            """
            world_points = torch.tensor(
                [
                    [-1.5, -0.5,  4.0],
                    [-0.5, -0.2,  5.2],
                    [ 0.6, -0.3,  6.4],
                    [ 1.4,  0.1,  7.0],
                    [-1.2,  0.7,  5.0],
                    [-0.2,  0.9,  6.0],
                    [ 0.8,  1.1,  6.8],
                    [ 1.6,  0.8,  8.0],
                ],
                dtype=torch.float64,
            )

            K_true = make_intrinsics(820.0, 790.0, 320.0, 240.0)
            R_true = rotation_y(8.0) @ rotation_x(-12.0)
            T_true = torch.tensor([0.3, 0.6, -0.4], dtype=torch.float64)

            observed_pixels, camera_points = project_points(world_points, K_true, R_true, T_true)
            noisy_pixels = observed_pixels + 0.8 * torch.tensor(
                [[0.5, -0.3], [-0.4, 0.2], [0.1, 0.4], [0.3, -0.5], [-0.2, 0.6], [0.4, -0.1], [-0.3, -0.2], [0.2, 0.3]],
                dtype=torch.float64,
            )

            M_est = estimate_projection_matrix_dlt(world_points, noisy_pixels)
            reproj_clean = project_with_matrix(world_points, M_est)
            reproj_error = torch.linalg.norm(reproj_clean - noisy_pixels, dim=1)

            A = build_dlt_matrix(world_points, noisy_pixels)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

            axes[0].scatter(noisy_pixels[:, 0], noisy_pixels[:, 1], color="#0f766e", s=55, label="observed")
            axes[0].scatter(reproj_clean[:, 0], reproj_clean[:, 1], facecolors="none", edgecolors="#dc2626", s=70, label="DLT reprojection")
            for obs, pred in zip(noisy_pixels, reproj_clean, strict=True):
                axes[0].plot([float(obs[0]), float(pred[0])], [float(obs[1]), float(pred[1])], color="#9ca3af", lw=1.2)
            axes[0].invert_yaxis()
            axes[0].set_title("Observed vs predicted image points")
            axes[0].set_xlabel("pixel x")
            axes[0].set_ylabel("pixel y")
            axes[0].legend(frameon=False)

            axes[1].imshow(torch.log10(A.abs() + 1e-6).numpy(), aspect="auto", cmap="magma")
            axes[1].set_title("DLT linear system matrix")
            axes[1].set_xlabel("unknown parameter index")
            axes[1].set_ylabel("equation row")

            axes[2].bar(range(len(reproj_error)), reproj_error.numpy(), color="#2563eb")
            axes[2].axhline(float(reproj_error.mean()), color="#dc2626", linestyle="--", lw=2, label=f"mean={reproj_error.mean():.2f}px")
            axes[2].set_title("Reprojection error per correspondence")
            axes[2].set_xlabel("point index")
            axes[2].set_ylabel("pixels")
            axes[2].legend(frameon=False)

            fig.suptitle("Synthetic DLT calibration demo", fontsize=14)
            savefig(fig, "05-dlt-reprojection-demo.png")

            print("Estimated projection matrix (normalized):")
            print(M_est)
            print(f"Mean reprojection error: {reproj_error.mean().item():.3f} px")
            """
        )
    )

    cells.append(
        markdown_cell(
            generated_block(
                "05-dlt-reprojection-demo.png",
                "**Computational reconstruction related to MIT Vision Book Section 39.7, Figure 39.14.**",
                "We generated synthetic 3D points, projected them with a known camera, added small image noise, built the DLT linear system, and estimated a projection matrix up to scale. The left panel compares observed and predicted image points, the center panel shows the stacked DLT system, and the right panel summarizes reprojection error.",
            )
        )
    )

    cells.append(
        markdown_cell(
            """
            ## 39.8 Concluding Remarks

            The chapter's practical message is simple: a camera model is only useful when it links
            geometry, coordinates, and measurable image data. Intrinsics explain how rays become
            pixels, extrinsics explain where the camera is, and calibration ties both pieces to real
            correspondences.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Section Coverage

            | Section | Topic | Original figures included | Generated visualization | Coverage status |
            | --- | --- | --- | --- | --- |
            | 39.1 | Introduction | 39.1, 39.2 | None | Complete |
            | 39.2 | Homogeneous perspective projection | 39.3 | `01-projection-models.png` | Complete |
            | 39.2.1 | Parallel projection | None | `01-projection-models.png` | Complete |
            | 39.3 | Camera-intrinsic parameters | 39.4, 39.5, 39.6, 39.7 | `02-intrinsics-effects.png`, `03-pixel-to-ray-backprojection.png` | Complete |
            | 39.3.1 | From meters to pixels | 39.4, 39.5 | `02-intrinsics-effects.png` | Complete |
            | 39.3.2 | From pixels to rays | 39.6 | `03-pixel-to-ray-backprojection.png` | Complete |
            | 39.3.3 | Simple calibration method | 39.7 | None | Complete |
            | 39.3.4 | Other camera parameters | None | None | Complete |
            | 39.4 | Camera-extrinsic parameters | 39.8 | `04-extrinsics-and-horizon.png` | Complete |
            | 39.5 | Full camera model | 39.9 | None | Complete |
            | 39.6 | Concrete examples | 39.10, 39.11, 39.12, 39.13 | `04-extrinsics-and-horizon.png` | Complete |
            | 39.7 | Camera calibration | 39.14, 39.15, 39.16, 39.17 | `05-dlt-reprojection-demo.png` | Complete |
            | 39.7.1 | Direct linear transform | None | `05-dlt-reprojection-demo.png` | Complete |
            | 39.7.2 | Recover intrinsics and extrinsics | None | Explanation only | Complete |
            | 39.7.3 | Multiplane calibration | None | Explanation only | Complete |
            | 39.7.4 | Reprojection error minimization | 39.14 | `05-dlt-reprojection-demo.png` | Complete |
            | 39.7.5 | Toy example | 39.15, 39.16, 39.17 | Explanation only | Complete |
            | 39.8 | Concluding remarks | None | None | Complete |
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Figure Coverage

            | Figure | Section | Original included | Generated counterpart | Treatment |
            | --- | --- | --- | --- | --- |
            | 39.1 | 39.1 | Yes | None | Original figure with explanation |
            | 39.2 | 39.1 | Yes | None | Original figure with explanation |
            | 39.3 | 39.2 | Yes | `01-projection-models.png` | Computational reconstruction |
            | 39.4 | 39.3.1 | Yes | `02-intrinsics-effects.png` | Supplemental visualization |
            | 39.5 | 39.3.1 | Yes | None | Original figure with explanation |
            | 39.6 | 39.3.2 | Yes | `03-pixel-to-ray-backprojection.png` | Computational reconstruction |
            | 39.7 | 39.3.3 | Yes | None | Original figure with explanation |
            | 39.8 | 39.4 | Yes | `04-extrinsics-and-horizon.png` | Computational reconstruction |
            | 39.9 | 39.5 | Yes | None | Original figure with explanation |
            | 39.10 | 39.6 | Yes | `04-extrinsics-and-horizon.png` | Supplemental visualization |
            | 39.11 | 39.6 | Yes | `04-extrinsics-and-horizon.png` | Supplemental visualization |
            | 39.12 | 39.6 | Yes | `04-extrinsics-and-horizon.png` | Supplemental visualization |
            | 39.13 | 39.6 | Yes | `04-extrinsics-and-horizon.png` | Supplemental visualization |
            | 39.14 | 39.7.4 | Yes | `05-dlt-reprojection-demo.png` | Computational reconstruction |
            | 39.15 | 39.7.5 | Yes | None | Original figure with explanation |
            | 39.16 | 39.7.5 | Yes | None | Original figure with explanation |
            | 39.17 | 39.7.5 | Yes | None | Original figure with explanation |
            """
        )
    )

    cells.append(
        code_cell(
            """
            expected_pngs = {
                "01-projection-models.png",
                "02-intrinsics-effects.png",
                "03-pixel-to-ray-backprojection.png",
                "04-extrinsics-and-horizon.png",
                "05-dlt-reprojection-demo.png",
            }
            generated_pngs = {path.name for path in IMAGES_DIR.glob("*.png")}
            assert expected_pngs.issubset(generated_pngs), (expected_pngs, generated_pngs)
            print("Generated figures verified:", sorted(expected_pngs))
            """
        )
    )

    return notebook_dict(cells)


def build_assets_readme() -> str:
    lines = [
        "# MIT Vision Book Reference Assets",
        "",
        "These files are local copies of the original MIT Vision Book Chapter 39 figures.",
        "They are included here for educational attribution-preserving notebook use.",
        "",
        "| Local filename | MIT figure | Section | Official source URL | Purpose | Attribution |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in FIGURES:
        for filename, url in zip(item["files"], item["urls"], strict=True):
            lines.append(
                f"| `{filename}` | {item['figure']} | {item['section']} | `{url}` | {item['purpose']} | MIT Vision Book, *Foundations of Computer Vision* |"
            )
    lines.append("")
    return "\n".join(lines)


def build_readme() -> str:
    return textwrap.dedent(
        """
        # Chapter 39: Camera Modeling and Calibration

        This chapter folder contains a compact executable notebook companion to MIT Vision Book
        Chapter 39. It preserves the official section order, keeps all original MIT Figures `39.1`
        through `39.17` visible in the notebook, and adds five small generated visualizations to
        make the core geometry and calibration ideas more concrete.

        ## Notebook contents

        - All official sections and subsections from Chapter 39.
        - All 19 original MIT figure assets stored locally under `assets/mit-book/`.
        - Five deterministic generated figures under `images/`.
        - Final section-coverage and figure-coverage tables.

        ## Original vs generated material

        - Original MIT Vision Book figures are shown from local files in `assets/mit-book/` and are
          labeled explicitly as original textbook figures.
        - Generated figures are saved under `images/` and are labeled either as computational
          reconstructions or as supplemental visualizations.

        ## Environment

        - Registered notebook environment: `torch.dev.cpu`
        - Designed to run on CPU from a fresh kernel

        ## How to build and execute

        ```bash
        python3 notebooks/CV/2026/spring/final/chapter-39-camera-modeling/build_assets.py
        jupyter nbconvert --to notebook --execute notebooks/CV/2026/spring/final/chapter-39-camera-modeling/index.ipynb --output index.executed.ipynb --ExecutePreprocessor.timeout=600
        ```

        ## Output locations

        - Original MIT assets: `assets/mit-book/`
        - Generated notebook figures: `images/`
        - Executed notebook: `index.executed.ipynb`

        ## Attribution

        Original textbook figures are from the MIT Vision Book, *Foundations of Computer Vision*,
        Chapter 39: Camera Modeling and Calibration.

        ## Known limitations

        - The generated visuals are intentionally selective and do not reproduce every textbook
          derivation.
        - The DLT example uses synthetic correspondences rather than the full office-scene recovery
          pipeline from the textbook.
        """
    ).strip() + "\n"


def validate_assets() -> None:
    expected = [filename for item in FIGURES for filename in item["files"]]
    missing = [name for name in expected if not (ASSETS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing MIT Book assets: {missing}")


def main() -> None:
    validate_assets()
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=2) + "\n")
    README_PATH.write_text(build_readme())
    ASSETS_README_PATH.write_text(build_assets_readme())
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Wrote {README_PATH}")
    print(f"Wrote {ASSETS_README_PATH}")


if __name__ == "__main__":
    main()
