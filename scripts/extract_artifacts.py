#!/usr/bin/env python3
"""Extract artifacts (images, Plotly charts, HTML tables) from executed notebooks."""

import base64
import json
import re
import sys
from pathlib import Path


def _extract_title_from_source(source: str) -> str | None:
    """Extract plot title from cell source code.

    Looks for plt.title(), plt.suptitle(), ax.set_title(), or fig.suptitle() calls.
    """
    patterns = [
        r"\.suptitle\s*\(\s*[\"']([^\"']+)[\"']",
        r"\.set_title\s*\(\s*[\"']([^\"']+)[\"']",
        r"plt\.title\s*\(\s*[\"']([^\"']+)[\"']",
    ]
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            title = match.group(1)
            # Sanitize for filename
            title = re.sub(r"[^\w\s-]", "", title).strip()
            title = re.sub(r"[-\s]+", "_", title).lower()
            return title
    return None


def _save_png(data: str, path: Path) -> None:
    """Decode base64 PNG and save to path."""
    image_bytes = base64.b64decode(data)
    path.write_bytes(image_bytes)


def _save_plotly_html(data: dict, path: Path) -> None:
    """Write a self-contained Plotly HTML file from plotly JSON data."""
    plotly_json = json.dumps(data)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="plot"></div>
    <script>
        var figure = {plotly_json};
        Plotly.newPlot('plot', figure.data, figure.layout || {{}});
    </script>
</body>
</html>"""
    path.write_text(html)


def extract_artifacts(executed_nb_path: Path, output_dir: Path) -> dict:
    """Extract all outputs from an executed notebook into output_dir.

    Args:
        executed_nb_path: Path to the executed notebook (.ipynb).
        output_dir: Base output directory (images/ and text/ subdirs will be created).

    Returns:
        Dict with counts of extracted artifacts by type.
    """
    images_dir = output_dir / "images"
    text_dir = output_dir / "text"
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    with open(executed_nb_path) as f:
        nb = json.load(f)

    counts = {"png": 0, "plotly": 0, "html_table": 0}

    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        title = _extract_title_from_source(source)
        plot_idx = 0

        for output in cell.get("outputs", []):
            out_data = output.get("data", {})

            # Extract PNG images
            if "image/png" in out_data:
                plot_idx += 1
                if title and plot_idx == 1:
                    name = f"cell_{cell_idx}_{title}"
                else:
                    name = f"cell_{cell_idx}_plot_{plot_idx}"
                png_path = images_dir / f"{name}.png"
                _save_png(out_data["image/png"], png_path)
                counts["png"] += 1

            # Extract Plotly interactive charts
            if "application/vnd.plotly.v1+json" in out_data:
                plot_idx += 1
                if title and plot_idx == 1:
                    name = f"cell_{cell_idx}_{title}"
                else:
                    name = f"cell_{cell_idx}_plot_{plot_idx}"
                html_path = images_dir / f"{name}.html"
                _save_plotly_html(out_data["application/vnd.plotly.v1+json"], html_path)
                counts["plotly"] += 1

            # Extract HTML tables (skip trivial repr HTML)
            if "text/html" in out_data:
                html_content = out_data["text/html"]
                if isinstance(html_content, list):
                    html_content = "".join(html_content)
                # Only save if it looks like a real table, not a trivial repr
                if "<table" in html_content.lower() and len(html_content) > 200:
                    table_path = text_dir / f"cell_{cell_idx}.html"
                    table_path.write_text(html_content)
                    counts["html_table"] += 1

    return counts


def main() -> None:
    """CLI entry point: extract artifacts from an executed notebook."""
    if len(sys.argv) < 2:
        print("Usage: extract_artifacts.py <executed_notebook.ipynb> [output_dir]", file=sys.stderr)
        sys.exit(1)

    nb_path = Path(sys.argv[1])
    if not nb_path.exists():
        print(f"Error: Notebook not found: {nb_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else nb_path.parent / "output"

    print(f"Extracting artifacts from: {nb_path}")
    print(f"Output directory: {output_dir}")

    counts = extract_artifacts(nb_path, output_dir)

    total = sum(counts.values())
    if total > 0:
        print(f"  Extracted {counts['png']} PNG image(s), {counts['plotly']} Plotly chart(s), {counts['html_table']} HTML table(s)")
    else:
        print("  No artifacts found in notebook outputs")


if __name__ == "__main__":
    main()
