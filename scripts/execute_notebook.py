#!/usr/bin/env python3
"""Execute a notebook with papermill and save artifacts."""

import sys
from pathlib import Path

import papermill as pm


def execute_notebook(notebook_path: str, output_base: str = "notebooks") -> None:
    """
    Execute a notebook and save outputs.

    Args:
        notebook_path: Path to the notebook (relative to notebooks/)
        output_base: Base directory for notebooks (default: "notebooks")
    """
    # Normalize paths
    notebook_path = notebook_path.removeprefix("notebooks/")
    nb_file = Path(output_base) / notebook_path

    if not nb_file.exists():
        print(f"Error: Notebook not found: {nb_file}", file=sys.stderr)
        sys.exit(1)

    # Create output directory structure
    nb_dir = nb_file.parent
    output_dir = nb_dir / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)

    # Generate output notebook name
    output_nb = nb_dir / f"{nb_file.stem}-executed.ipynb"

    print(f"Executing notebook: {nb_file}")
    print(f"Output notebook: {output_nb}")
    print(f"Artifact directory: {output_dir}")

    # Execute with papermill
    try:
        pm.execute_notebook(
            str(nb_file),
            str(output_nb),
            parameters={
                "output_dir": str(output_dir),
                "images_dir": str(output_dir / "images"),
                "videos_dir": str(output_dir / "videos"),
                "audio_dir": str(output_dir / "audio"),
                "text_dir": str(output_dir / "text"),
            },
            log_output=True,
        )
        print(f"\n✓ Notebook executed successfully: {output_nb}")
    except Exception as e:
        print(f"\n✗ Notebook execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: execute_notebook.py <notebook_path>", file=sys.stderr)
        print("Example: execute_notebook.py transfer-learning/transfer_learning_tutorial.ipynb", file=sys.stderr)
        sys.exit(1)

    notebook_path = sys.argv[1]
    execute_notebook(notebook_path)
