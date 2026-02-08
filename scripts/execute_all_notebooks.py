#!/usr/bin/env python3
"""Execute all notebooks from the registry in a single container session."""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import papermill as pm
import yaml

from extract_artifacts import extract_artifacts
from update_registry import update_registry_entry
from wandb_utils import finish_wandb_run, init_wandb_run, log_notebook_result


def main() -> int:
    registry_path = Path("notebooks/notebook-database.yml")
    if not registry_path.exists():
        print(f"Error: Registry not found: {registry_path}", file=sys.stderr)
        return 1

    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    notebooks = [e for e in registry["notebooks"] if e != "---"]

    passed = []
    failed = []
    skipped = []

    for entry in notebooks:
        nb_rel = entry["notebook"]
        env = entry.get("environment", "torch.dev.gpu")

        if env == "colab":
            print(f"\n[SKIP] {nb_rel} (colab-only)")
            skipped.append(nb_rel)
            continue

        nb_file = Path("notebooks") / nb_rel
        if not nb_file.exists():
            print(f"\n[SKIP] {nb_rel} (file not found)")
            skipped.append(nb_rel)
            continue

        # Output alongside the source notebook
        output_nb = nb_file.parent / f"{nb_file.stem}-executed.ipynb"
        output_dir = nb_file.parent / "output"
        output_dir.mkdir(exist_ok=True)
        for subdir in ("images", "videos", "audio", "text"):
            (output_dir / subdir).mkdir(exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Executing: {nb_rel}")
        print(f"{'=' * 60}")

        run = None
        try:
            # Init W&B immediately before execution so W&B's built-in
            # runtime matches the actual notebook execution duration.
            run = init_wandb_run(nb_rel, environment=env)
            start = time.monotonic()
            pm.execute_notebook(
                str(nb_file),
                str(output_nb),
                kernel_name="python3",
                log_output=True,
            )
            duration = time.monotonic() - start
            executed_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            print(f"  Duration: {duration:.1f}s")

            # Update registry with execution metadata
            update_registry_entry(nb_rel, executed_date, duration)

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

            print(f"[PASS] {nb_rel}")
            passed.append(nb_rel)
        except Exception as e:
            if run is not None:
                run.summary["status"] = "failed"
            print(f"[FAIL] {nb_rel}: {e}", file=sys.stderr)
            failed.append(nb_rel)
        finally:
            finish_wandb_run(run)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Passed:  {len(passed)}")
    print(f"  Failed:  {len(failed)}")
    print(f"  Skipped: {len(skipped)}")

    if failed:
        print("\nFailed notebooks:")
        for nb in failed:
            print(f"  - {nb}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
