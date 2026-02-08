#!/usr/bin/env python3
"""Get the Docker environment for a notebook from the registry."""

import sys
from pathlib import Path

import yaml


def get_notebook_environment(notebook_path: str, registry_path: str = "notebooks/notebook-database.yml") -> str:
    """
    Get the Docker environment for a notebook from the registry.

    Args:
        notebook_path: Path to the notebook (relative to notebooks/)
        registry_path: Path to the registry YAML file

    Returns:
        Environment name (e.g., "torch.dev.gpu")

    Raises:
        SystemExit: If notebook not found in registry
    """
    registry_file = Path(registry_path)
    if not registry_file.exists():
        print(f"Error: Registry file not found: {registry_path}", file=sys.stderr)
        sys.exit(1)

    with registry_file.open() as f:
        registry = yaml.safe_load(f)

    notebooks = registry.get("notebooks", [])

    # Normalize notebook path (remove leading "notebooks/" if present)
    search_path = notebook_path.removeprefix("notebooks/")

    for entry in notebooks:
        # Skip separator entries
        if entry == "---":
            continue

        stripped = entry.get("notebook", "")
        if stripped == search_path:
            environment = entry.get("environment")
            if not environment:
                print(
                    f"Error: No environment specified for notebook: {search_path}",
                    file=sys.stderr,
                )
                sys.exit(1)
            return environment

    print(f"Error: Notebook not found in registry: {search_path}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: get_notebook_environment.py <notebook_path>", file=sys.stderr)
        print("Example: get_notebook_environment.py transfer-learning/transfer_learning_tutorial.ipynb", file=sys.stderr)
        sys.exit(1)

    notebook_path = sys.argv[1]
    environment = get_notebook_environment(notebook_path)
    print(environment)
