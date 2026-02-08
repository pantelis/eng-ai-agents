#!/usr/bin/env python3
"""CLI tool to query W&B runs and display summary reports.

Uses wandb.Api() to query experiment data logged by the notebook
execution pipeline. Provides tabular and JSON views of runs without
requiring the W&B web UI.

Usage:
    python scripts/wandb_report.py summary [--group GROUP] [--tag TAG] [--since YYYY-MM-DD] [--json]
    python scripts/wandb_report.py training [--group GROUP] [--tag TAG] [--since YYYY-MM-DD] [--json]
    python scripts/wandb_report.py compare --group GROUP [--json]
    python scripts/wandb_report.py history --notebook PATH [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any


try:
    import wandb  # type: ignore[import-untyped]
except ImportError:
    wandb = None  # type: ignore[assignment]

from wandb_utils import WANDB_ENTITY, WANDB_PROJECT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wandb_api_available() -> bool:
    """Check if wandb is installed and an API key is configured."""
    return wandb is not None and bool(os.environ.get("WANDB_API_KEY"))


def _get_runs(
    api: Any,
    *,
    job_type: str | None = None,
    group: str | None = None,
    tag: str | None = None,
    since: str | None = None,
) -> list[Any]:
    """Query W&B runs with optional filters."""
    filters: dict[str, Any] = {}
    if job_type:
        filters["jobType"] = job_type
    if group:
        filters["group"] = group
    if tag:
        filters["tags"] = {"$in": [tag]}
    if since:
        filters["created_at"] = {"$gte": since}

    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters, order="-created_at")
    return list(runs)


def _format_duration(seconds: float | None) -> str:
    """Format seconds as 'Xm Ys' or 'Ys'."""
    if seconds is None:
        return "-"
    s = int(seconds)
    minutes, secs = divmod(s, 60)
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{s}s"


def _safe_summary(run: Any, key: str) -> str:
    """Safely get a summary value, returning '-' if missing."""
    val = run.summary.get(key)
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _compact_config(run: Any) -> str:
    """Produce a compact 'key=val, ...' string from run config."""
    skip = {"notebook", "environment", "_wandb"}
    pairs: list[str] = []
    for k, v in sorted(run.config.items()):
        if k.startswith("_") or k in skip:
            continue
        if isinstance(v, float):
            pairs.append(f"{k}={v:g}")
        else:
            pairs.append(f"{k}={v}")
    return ", ".join(pairs) if pairs else "-"


def _print_markdown_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print an aligned markdown table to stdout."""
    if not rows:
        print("(no runs found)")
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        parts = [cell.ljust(widths[i]) for i, cell in enumerate(cells)]
        return "| " + " | ".join(parts) + " |"

    print(fmt_row(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(fmt_row(row))


def _print_json(data: list[dict[str, Any]]) -> None:
    """Print data as formatted JSON."""
    json.dump(data, sys.stdout, indent=2, default=str)
    print()


def _run_date(run: Any) -> str:
    """Extract date string from run."""
    date = run.summary.get("executed_date")
    if date:
        return str(date)
    created = run.created_at
    if created:
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            return str(created)[:10]
    return "-"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_summary(api: Any, args: argparse.Namespace) -> None:
    """Show overview of notebook-execution runs."""
    runs = _get_runs(api, job_type="notebook-execution", group=args.group, tag=args.tag, since=args.since)

    if args.json:
        data = [
            {
                "group": run.group or "-",
                "notebook": run.name,
                "status": _safe_summary(run, "status"),
                "duration": run.summary.get("duration_s"),
                "date": _run_date(run),
                "png_count": run.summary.get("png_count", 0),
                "plotly_count": run.summary.get("plotly_count", 0),
            }
            for run in runs
        ]
        _print_json(data)
        return

    headers = ["Group", "Notebook", "Status", "Duration", "Date", "PNGs", "Plotly"]
    rows = [
        [
            run.group or "-",
            run.name,
            _safe_summary(run, "status"),
            _format_duration(run.summary.get("duration_s")),
            _run_date(run),
            str(run.summary.get("png_count", 0)),
            str(run.summary.get("plotly_count", 0)),
        ]
        for run in runs
    ]
    _print_markdown_table(headers, rows)


def cmd_training(api: Any, args: argparse.Namespace) -> None:
    """Show training runs with final metrics."""
    runs = _get_runs(api, job_type="training", group=args.group, tag=args.tag, since=args.since)

    if args.json:
        data = [
            {
                "group": run.group or "-",
                "run_name": run.name,
                "best_val_loss": run.summary.get("best_val_loss"),
                "best_val_acc": run.summary.get("best_val_accuracy"),
                "epochs": run.summary.get("epoch"),
            }
            for run in runs
        ]
        _print_json(data)
        return

    headers = ["Group", "Run Name", "Best Val Loss", "Best Val Acc", "Epochs"]
    rows = [
        [
            run.group or "-",
            run.name,
            _safe_summary(run, "best_val_loss"),
            _safe_summary(run, "best_val_accuracy"),
            _safe_summary(run, "epoch"),
        ]
        for run in runs
    ]
    _print_markdown_table(headers, rows)


def cmd_compare(api: Any, args: argparse.Namespace) -> None:
    """Side-by-side comparison of runs within a group."""
    if not args.group:
        print("Error: --group is required for the compare subcommand", file=sys.stderr)
        sys.exit(1)

    runs = _get_runs(api, group=args.group, tag=args.tag, since=args.since)

    if args.json:
        data = [
            {
                "run_name": run.name,
                "job_type": run.job_type or "-",
                "tags": run.tags,
                "best_val_loss": run.summary.get("best_val_loss"),
                "epochs": run.summary.get("epoch"),
                "config": {
                    k: v
                    for k, v in run.config.items()
                    if not k.startswith("_") and k not in {"notebook", "environment"}
                },
            }
            for run in runs
        ]
        _print_json(data)
        return

    print(f"Group: {args.group}")
    headers = ["Run Name", "Tags", "Val Loss (best)", "Epochs", "Config"]
    rows = [
        [
            run.name,
            ", ".join(run.tags) if run.tags else "-",
            _safe_summary(run, "best_val_loss"),
            _safe_summary(run, "epoch"),
            _compact_config(run),
        ]
        for run in runs
    ]
    _print_markdown_table(headers, rows)


def cmd_history(api: Any, args: argparse.Namespace) -> None:
    """Show execution history for a specific notebook over time."""
    if not args.notebook:
        print("Error: --notebook is required for the history subcommand", file=sys.stderr)
        sys.exit(1)

    notebook = args.notebook.removeprefix("notebooks/")
    # Query all runs whose config.notebook matches
    filters: dict[str, Any] = {"config.notebook": {"$regex": notebook}}
    if args.since:
        filters["created_at"] = {"$gte": args.since}

    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters, order="-created_at")
    runs = list(runs)

    if args.json:
        data = [
            {
                "job_type": run.job_type or "-",
                "date": _run_date(run),
                "duration": run.summary.get("duration_s"),
                "status": _safe_summary(run, "status"),
                "key_metric": _history_metric(run),
            }
            for run in runs
        ]
        _print_json(data)
        return

    print(f"Notebook: {notebook}")
    headers = ["Job Type", "Date", "Duration", "Status", "Key Metric"]
    rows = [
        [
            run.job_type or "-",
            _run_date(run),
            _format_duration(run.summary.get("duration_s")),
            _safe_summary(run, "status"),
            _history_metric(run),
        ]
        for run in runs
    ]
    _print_markdown_table(headers, rows)


def _history_metric(run: Any) -> str:
    """Pick the most relevant metric for a history row."""
    if run.job_type == "notebook-execution":
        pngs = run.summary.get("png_count", 0)
        plotly = run.summary.get("plotly_count", 0)
        parts = []
        if pngs:
            parts.append(f"{pngs} PNGs")
        if plotly:
            parts.append(f"{plotly} Plotly")
        return ", ".join(parts) if parts else "-"

    # For training runs, pick the first available metric
    for key in ("best_val_loss", "best_val_accuracy", "test_loss", "test_accuracy", "val_loss", "val_accuracy"):
        val = run.summary.get(key)
        if val is not None:
            label = key.replace("_", " ")
            return f"{label}={val:.4f}" if isinstance(val, float) else f"{label}={val}"
    return "-"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query W&B runs and display summary reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--group", help="Filter by W&B group")
        p.add_argument("--tag", help="Filter by tag")
        p.add_argument("--since", help="Only runs after date (YYYY-MM-DD)")
        p.add_argument("--json", action="store_true", help="Output as JSON")

    p_summary = sub.add_parser("summary", help="Overview of notebook execution runs")
    add_common(p_summary)

    p_training = sub.add_parser("training", help="Training runs with final metrics")
    add_common(p_training)

    p_compare = sub.add_parser("compare", help="Side-by-side comparison within a group")
    add_common(p_compare)

    p_history = sub.add_parser("history", help="Execution history for a specific notebook")
    add_common(p_history)
    p_history.add_argument("--notebook", help="Notebook path (e.g. regression/sgd_sinusoidal_dataset.ipynb)")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not _wandb_api_available():
        print("W&B not available (wandb not installed or WANDB_API_KEY not set).")
        print("Set WANDB_API_KEY to query experiment data.")
        sys.exit(0)

    api = wandb.Api()

    commands = {
        "summary": cmd_summary,
        "training": cmd_training,
        "compare": cmd_compare,
        "history": cmd_history,
    }
    commands[args.command](api, args)


if __name__ == "__main__":
    main()
