#!/usr/bin/env python3
"""List all notebooks from the registry."""

import yaml

with open("notebooks/notebook-database.yml") as f:
    registry = yaml.safe_load(f)

for entry in registry["notebooks"]:
    if entry != "---":
        print(entry["stripped"])
