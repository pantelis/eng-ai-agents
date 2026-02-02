# Notebook Execution Plan - eng-ai-agents

## Overview

This repository stores **stripped notebooks** (code-only) from the private `aiml-common` repository and provides:
- Isolated execution environments per notebook
- GitHub Actions for automated execution
- Registry tracking of all notebooks
- Executed notebooks with outputs for documentation

## Directory Structure

```
eng-ai-agents/
├── pyproject.toml                  # Workspace root
├── uv.lock                         # Shared lockfile
├── notebooks/
│   ├── stripped-notebooks.yml      # Registry (existing)
│   ├── cv/
│   │   ├── basketball-detection/
│   │   │   ├── pyproject.toml     # Environment for this notebook
│   │   │   ├── notebook.ipynb     # Stripped (code-only)
│   │   │   └── notebook-executed.ipynb
│   │   └── object-tracking/
│   │       ├── pyproject.toml
│   │       └── notebook.ipynb
│   ├── nlp/
│   └── rl/
├── scripts/
│   ├── strip_and_copy.py          # Strip from aiml-common → here
│   ├── execute_notebook.py        # Execute single notebook
│   ├── execute_category.py        # Execute all in category
│   └── convert_to_mdx.py          # Convert executed → MDX
└── .github/
    └── workflows/
        ├── execute-notebooks.yml   # CI/CD execution
        └── publish-to-docs.yml     # Push MDX to main repo
```

## Workflow

### 1. Strip and Copy from aiml-common

```bash
# From main repo root
uv run python eng-ai-agents/scripts/strip_and_copy.py \
    src/aiml-common/projects/cv/sports-analytics/basketball/index.ipynb \
    eng-ai-agents/notebooks/cv/basketball-detection
```

This script:
- Strips markdown cells using `pipeline/tools/notebook/strip_markdown_cells.py`
- Copies to eng-ai-agents with proper structure
- Creates `pyproject.toml` if missing
- Updates `stripped-notebooks.yml` registry

### 2. Define Dependencies

Each notebook directory gets a `pyproject.toml`:

```toml
[project]
name = "basketball-detection"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "roboflow>=1.1.0",
    "opencv-python>=4.8.0",
]
```

### 3. Execute Locally

```bash
cd eng-ai-agents/notebooks/cv/basketball-detection

# Sync dependencies
uv sync

# Execute notebook
uv run papermill notebook.ipynb notebook-executed.ipynb
```

### 4. Execute via GitHub Actions

Push to `eng-ai-agents` triggers execution:

```yaml
on:
  push:
    paths:
      - 'notebooks/**/*.ipynb'
  workflow_dispatch:
```

### 5. Convert to MDX

```bash
# From executed notebook with outputs
uv run python scripts/convert_to_mdx.py \
    notebooks/cv/basketball-detection/notebook-executed.ipynb \
    ../eaia/src/products/sports-analytics-detection-tracking.mdx
```

### 6. Publish to Main Repo

GitHub Action pushes converted MDX back to main `eaia` repo.

## Implementation Phases

### Phase 1: Setup Workspace (This PR)
- [x] Create workspace `pyproject.toml`
- [ ] Create `scripts/strip_and_copy.py`
- [ ] Create `scripts/execute_notebook.py`
- [ ] Update `stripped-notebooks.yml` format

### Phase 2: Basketball Notebook (Example)
- [ ] Strip basketball notebooks to `notebooks/cv/basketball-detection/`
- [ ] Create `pyproject.toml` with dependencies
- [ ] Test local execution
- [ ] Verify outputs

### Phase 3: GitHub Actions
- [ ] Create `execute-notebooks.yml` workflow
- [ ] Test with basketball notebook
- [ ] Add caching for dependencies
- [ ] Add artifact upload

### Phase 4: Scale to All Notebooks
- [ ] Migrate existing notebooks from `notebooks/{topic}/` to new structure
- [ ] Add dependencies for each
- [ ] Execute and verify
- [ ] Update registry

### Phase 5: MDX Conversion Pipeline
- [ ] Create `scripts/convert_to_mdx.py`
- [ ] Create `publish-to-docs.yml` workflow
- [ ] Test round-trip: strip → execute → convert → publish

## Benefits

✅ **Isolation**: Each notebook has its own environment
✅ **Reproducible**: Lockfiles ensure consistent execution
✅ **Public**: Notebooks executable in Colab from public repo
✅ **Automated**: CI/CD executes and publishes automatically
✅ **Efficient**: uv's shared cache prevents duplicate downloads
✅ **Tracked**: Registry knows source → destination mapping

## Registry Format Enhancement

Update `stripped-notebooks.yml`:

```yaml
notebooks:
  - source: aiml-common/projects/cv/sports-analytics/basketball/index.ipynb
    destination: notebooks/cv/basketball-detection/notebook.ipynb
    category: cv
    dependencies: notebooks/cv/basketball-detection/pyproject.toml
    last_executed: 2026-02-01T19:00:00Z
    execution_time: 180s
    status: success
```
