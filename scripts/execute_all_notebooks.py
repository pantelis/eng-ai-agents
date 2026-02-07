#!/usr/bin/env python3
"""Execute all notebooks from the registry in a single container session."""

import sys
from pathlib import Path

import papermill as pm
import yaml


def main() -> int:
    registry_path = Path("notebooks/stripped-notebooks.yml")
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
        nb_rel = entry["stripped"]
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

        try:
            pm.execute_notebook(
                str(nb_file),
                str(output_nb),
                kernel_name="python3",
                parameters={
                    "output_dir": str(output_dir),
                    "images_dir": str(output_dir / "images"),
                    "videos_dir": str(output_dir / "videos"),
                    "audio_dir": str(output_dir / "audio"),
                    "text_dir": str(output_dir / "text"),
                },
                log_output=True,
            )
            print(f"[PASS] {nb_rel}")
            passed.append(nb_rel)
        except Exception as e:
            print(f"[FAIL] {nb_rel}: {e}", file=sys.stderr)
            failed.append(nb_rel)

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
