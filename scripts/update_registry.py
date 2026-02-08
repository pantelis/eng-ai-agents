#!/usr/bin/env python3
"""Update notebook registry entries with execution metadata."""

from pathlib import Path


def update_registry_entry(
    notebook_rel: str,
    last_executed: str,
    duration_seconds: float,
    registry_path: str = "notebooks/notebook-database.yml",
) -> bool:
    """Update a registry entry with execution date and duration.

    Preserves comments and formatting by operating on raw lines.

    Args:
        notebook_rel: Notebook path relative to notebooks/ (the 'notebook' value).
        last_executed: ISO date string (e.g. "2026-02-07").
        duration_seconds: Total execution time in seconds.
        registry_path: Path to the registry YAML file.

    Returns:
        True if the entry was found and updated, False otherwise.
    """
    path = Path(registry_path)
    lines = path.read_text().splitlines(keepends=True)

    # Find the line with matching stripped: value
    target_idx = None
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line == f"notebook: {notebook_rel}":
            target_idx = i
            break

    if target_idx is None:
        return False

    # Find the block boundary: next entry (line starting with "  -") or EOF
    block_end = len(lines)
    for i in range(target_idx + 1, len(lines)):
        stripped_line = lines[i].strip()
        if stripped_line.startswith("- ") or stripped_line == '- "---"':
            block_end = i
            break

    # Remove existing last_executed / duration_seconds lines in this block
    kept = []
    for i in range(len(lines)):
        if target_idx < i < block_end:
            s = lines[i].strip()
            if s.startswith("last_executed:") or s.startswith("duration_seconds:"):
                continue
        kept.append(lines[i])

    # Recalculate target_idx and block_end after removals
    target_idx = None
    for i, line in enumerate(kept):
        if line.strip() == f"notebook: {notebook_rel}":
            target_idx = i
            break

    block_end = len(kept)
    for i in range(target_idx + 1, len(kept)):
        stripped_line = kept[i].strip()
        if stripped_line.startswith("- ") or stripped_line == '- "---"':
            block_end = i
            break

    # Insert new metadata lines just before block_end
    indent = "    "
    new_lines = [
        f"{indent}last_executed: {last_executed}\n",
        f"{indent}duration_seconds: {duration_seconds:.1f}\n",
    ]
    kept[block_end:block_end] = new_lines

    path.write_text("".join(kept))
    return True
