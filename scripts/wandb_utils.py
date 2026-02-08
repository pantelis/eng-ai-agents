"""W&B experiment tracking helpers for notebook execution.

Provides init/log/finish functions with graceful no-op when wandb
is not installed or WANDB_API_KEY is not set.
"""

from __future__ import annotations

import os
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


def _make_run_id(prefix: str, notebook_path: str) -> str:
    """Generate a deterministic W&B run ID from a prefix and notebook path.

    Examples:
        _make_run_id("exec", "optimization/regularization/index.ipynb")
        → "exec-optimization-regularization-index"
    """
    stem = Path(notebook_path).with_suffix("").as_posix().replace("/", "-")
    return f"{prefix}-{stem}"


def init_wandb_run(
    notebook_path: str,
    environment: str = "torch.dev.gpu",
) -> Any:
    """Create a W&B run for a notebook execution.

    Uses a deterministic run ID so re-executions overwrite previous runs
    instead of creating duplicates.

    Returns the run object, or None if wandb is unavailable.
    """
    if not _wandb_available():
        return None

    nb = Path(notebook_path)
    # Extract category from path (e.g. "transfer-learning" from
    # "transfer-learning/transfer_learning_tutorial.ipynb")
    parts = nb.parts
    category = parts[0] if len(parts) > 1 else "uncategorized"

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        id=_make_run_id("exec", notebook_path),
        resume="allow",
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
    """Safely finish a W&B run."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
