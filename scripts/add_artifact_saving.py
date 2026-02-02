#!/usr/bin/env python3
"""
Auto-convert notebooks to save plot artifacts.

Adds a parameters cell and inserts plt.savefig() calls before plt.show().
"""

import json
import re
import sys
from pathlib import Path


def add_parameters_cell(notebook: dict) -> bool:
    """
    Add a parameters cell if it doesn't exist.

    Returns:
        True if cell was added, False if it already exists
    """
    cells = notebook.get("cells", [])

    # Check if parameters cell already exists
    for cell in cells:
        if cell.get("cell_type") == "code":
            metadata = cell.get("metadata", {})
            tags = metadata.get("tags", [])
            if "parameters" in tags:
                return False

    # Add parameters cell at the beginning
    parameters_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["parameters"]},
        "outputs": [],
        "source": [
            "# Parameters (injected by papermill)\n",
            'output_dir = "."\n',
            'images_dir = "./images"\n',
            'videos_dir = "./videos"\n',
            'audio_dir = "./audio"\n',
            'text_dir = "./text"\n',
            "\n",
            "# Import cv2 for saving supervision images\n",
            "try:\n",
            "    import cv2\n",
            "except ImportError:\n",
            "    pass\n",
        ],
    }

    # Insert at the very beginning so parameters are available to all cells
    cells.insert(0, parameters_cell)
    notebook["cells"] = cells

    return True


def generate_plot_filename(cell_source: str, plot_index: int) -> str:
    """Generate a descriptive filename for a plot based on cell content."""
    # Try to extract a meaningful name from the title or content
    title_match = re.search(r'title\s*[=\(]\s*["\']([^"\']+)["\']', cell_source, re.IGNORECASE)
    if title_match:
        # Clean the title to make it a valid filename
        title = title_match.group(1)
        filename = re.sub(r"[^\w\s-]", "", title).strip()
        filename = re.sub(r"[-\s]+", "_", filename).lower()
        return f"{filename}.png"

    # Fallback to generic name
    return f"plot_{plot_index}.png"


def add_savefig_calls(notebook: dict) -> int:
    """
    Add plt.savefig() and sv.plot_image saving calls in code cells.

    Returns:
        Number of modifications made
    """
    cells = notebook.get("cells", [])
    modifications = 0
    plot_counter = 0

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, str):
            source = source.split("\n")

        # Check if this cell has plotting calls
        has_plt_show = any("plt.show()" in line for line in source)
        has_sv_plot = any("sv.plot_image(" in line or "sv.plot_images_grid(" in line for line in source)

        if not (has_plt_show or has_sv_plot):
            continue

        # Check if savefig is already present
        has_savefig = any("savefig" in line or "cv2.imwrite" in line for line in source)
        if has_savefig:
            continue

        plot_counter += 1
        cell_source_str = "".join(source)
        filename = generate_plot_filename(cell_source_str, plot_counter)

        # Insert save calls
        new_source = []
        for i, line in enumerate(source):
            # Handle plt.show()
            if "plt.show()" in line:
                indent = len(line) - len(line.lstrip())
                savefig_line = " " * indent + f'plt.savefig(f"{{images_dir}}/{filename}", dpi=150, bbox_inches="tight")\n'
                new_source.append(savefig_line)
                modifications += 1

            # Handle sv.plot_image() - save the image before displaying
            elif "sv.plot_image(" in line:
                indent = len(line) - len(line.lstrip())
                # Extract the variable being plotted
                var_match = re.search(r'sv\.plot_image\((\w+)', line)
                if var_match:
                    var_name = var_match.group(1)
                    save_line = " " * indent + f'cv2.imwrite(f"{{images_dir}}/{filename}", cv2.cvtColor({var_name}, cv2.COLOR_RGB2BGR))\n'
                    new_source.append(save_line)
                    modifications += 1

            # Handle sv.plot_images_grid()
            elif "sv.plot_images_grid(" in line:
                # For grids, we'll save after the complete call
                indent = len(line) - len(line.lstrip())
                # This is more complex, skip for now or add comment
                pass

            new_source.append(line)

        cell["source"] = new_source

    return modifications


def process_notebook(notebook_path: Path, output_path: Path = None) -> dict:
    """
    Process a notebook to add artifact saving capabilities.

    Args:
        notebook_path: Path to input notebook
        output_path: Path to save modified notebook (default: overwrite input)

    Returns:
        Dict with processing statistics
    """
    if output_path is None:
        output_path = notebook_path

    # Read notebook
    with notebook_path.open() as f:
        notebook = json.load(f)

    # Add parameters cell
    params_added = add_parameters_cell(notebook)

    # Add savefig calls
    savefigs_added = add_savefig_calls(notebook)

    # Write modified notebook
    with output_path.open("w") as f:
        json.dump(notebook, f, indent=1)

    return {
        "parameters_cell_added": params_added,
        "savefig_calls_added": savefigs_added,
        "modified": params_added or savefigs_added > 0,
    }


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: add_artifact_saving.py <notebook.ipynb> [output.ipynb]", file=sys.stderr)
        print("\nAdds parameters cell and plt.savefig() calls to notebooks.", file=sys.stderr)
        print("If output path is not specified, overwrites the input file.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path

    if not input_path.exists():
        print(f"Error: Notebook not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing: {input_path}")

    stats = process_notebook(input_path, output_path)

    if stats["modified"]:
        print(f"✓ Modified notebook saved to: {output_path}")
        if stats["parameters_cell_added"]:
            print("  • Added parameters cell")
        if stats["savefig_calls_added"] > 0:
            print(f"  • Added {stats['savefig_calls_added']} plt.savefig() call(s)")
    else:
        print("• No modifications needed (already has parameters and savefig calls)")


if __name__ == "__main__":
    main()
