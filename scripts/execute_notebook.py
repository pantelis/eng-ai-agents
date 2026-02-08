#!/usr/bin/env python3
"""Execute a notebook with papermill and save artifacts."""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import papermill as pm

from extract_artifacts import extract_artifacts
from update_registry import update_registry_entry
from wandb_utils import finish_wandb_run, init_wandb_run, log_notebook_result


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
    run = None
    try:
        # Init W&B immediately before execution so W&B's built-in
        # runtime matches the actual notebook execution duration.
        run = init_wandb_run(notebook_path)
        start = time.monotonic()
        pm.execute_notebook(
            str(nb_file),
            str(output_nb),
            kernel_name="python3",
            log_output=True,
        )
        duration = time.monotonic() - start
        executed_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        print(f"\n✓ Notebook executed successfully: {output_nb}")
        print(f"  Duration: {duration:.1f}s")

        # Update registry with execution metadata
        update_registry_entry(notebook_path, executed_date, duration)

        # Extract artifacts from the executed notebook
        counts = extract_artifacts(output_nb, output_dir)
        total = sum(counts.values())
        if total > 0:
            print(
                f"  Extracted {counts['png']} PNG(s), "
                f"{counts['plotly']} Plotly chart(s), "
                f"{counts['html_table']} HTML table(s)"
            )

        # Log results to W&B
        log_notebook_result(run, duration, executed_date, counts, output_dir)
    except Exception as e:
        if run is not None:
            run.summary["status"] = "failed"
        print(f"\n✗ Notebook execution failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        finish_wandb_run(run)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: execute_notebook.py <notebook_path>", file=sys.stderr)
        print("Example: execute_notebook.py transfer-learning/transfer_learning_tutorial.ipynb", file=sys.stderr)
        sys.exit(1)

    notebook_path = sys.argv[1]
    execute_notebook(notebook_path)
