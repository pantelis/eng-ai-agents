"""W&B experiment tracking helpers for notebook execution.

Provides init/log/finish functions with graceful no-op when wandb
is not installed or WANDB_API_KEY is not set.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

WANDB_PROJECT = "eng-ai-agents"
WANDB_ENTITY = "pantelis"


def _wandb_available() -> bool:
    return wandb is not None and bool(os.environ.get("WANDB_API_KEY"))


def _make_run_id(notebook_path: str) -> str:
    """Generate a unique W&B run ID from notebook path + timestamp.

    Returns an ID like "exec-ml-math-probability-gaussians-1738900000".
    """
    stem = Path(notebook_path).with_suffix("").as_posix().replace("/", "-")
    return f"exec-{stem}-{int(time.time())}"


def _delete_run(run_id: str) -> None:
    """Delete a W&B run by ID. Silently ignores errors."""
    try:
        api = wandb.Api()
        api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}").delete()
    except Exception:
        pass


def init_wandb_run(
    notebook_path: str,
    environment: str = "torch.dev.gpu",
) -> Any:
    """Create a fresh W&B run for a notebook execution.

    Each execution gets a unique run ID so W&B's Runtime reflects
    the actual execution time. The old run (if any) is deleted
    after the new run finishes (see finish_wandb_run).

    Returns the run object, or None if wandb is unavailable.
    """
    if not _wandb_available():
        return None

    nb = Path(notebook_path)
    parts = nb.parts
    category = parts[0] if len(parts) > 1 else "uncategorized"

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        id=_make_run_id(notebook_path),
        name=nb.stem,
        group=category,
        tags=[category, environment],
        job_type="notebook-execution",
        config={
            "notebook": str(nb),
            "environment": environment,
        },
        settings=wandb.Settings(init_timeout=120),
    )
    return run


def log_notebook_result(
    run: Any,
    duration: float,
    executed_date: str,
    counts: dict[str, int],
    output_dir: Path,
) -> None:
    """Log execution results and upload artifacts to an existing W&B run."""
    if run is None:
        return

    run.summary["status"] = "success"
    run.summary["duration_s"] = duration
    run.summary["executed_date"] = executed_date
    run.summary["png_count"] = counts.get("png", 0)
    run.summary["plotly_count"] = counts.get("plotly", 0)
    run.summary["html_table_count"] = counts.get("html_table", 0)

    # Upload PNG images
    images_dir = output_dir / "images"
    if images_dir.exists():
        for png in sorted(images_dir.glob("*.png")):
            run.log({png.stem: wandb.Image(str(png))})

    # Upload Plotly HTML files as artifacts
    html_files = list(images_dir.glob("*.html")) if images_dir.exists() else []
    if html_files:
        art = wandb.Artifact(
            name=f"{run.name}-plotly",
            type="plotly-charts",
        )
        for html in html_files:
            art.add_file(str(html))
        run.log_artifact(art)


def finish_wandb_run(run: Any) -> None:
    """Finish the run, then delete any prior runs for the same notebook.

    This keeps one row per notebook in the table with correct Runtime.
    """
    if run is None:
        return

    current_id = run.id
    notebook = run.config.get("notebook", "")

    try:
        run.finish()
    except Exception:
        pass

    # Delete stale runs for this notebook (different ID = older execution).
    if notebook:
        try:
            api = wandb.Api()
            old_runs = api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={"config.notebook": notebook},
            )
            for old_run in old_runs:
                if old_run.id != current_id:
                    try:
                        old_run.delete()
                    except Exception:
                        pass
        except Exception:
            pass
