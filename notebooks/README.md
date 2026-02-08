# Notebooks

This folder contains notebooks from the site https://aegean.ai/courses. You are free to edit those and issue Pull Requests with improvements. We are considering each PR for inclusion in the main repository.

## Notebook Execution System

This directory contains stripped notebooks from the private `aiml-common` repository with support for isolated execution environments and artifact storage.

### Registry

The `notebook-database.yml` file tracks all notebooks and their execution environments:

```yaml
notebooks:
  - source: aiml-common/lectures/transfer-learning/transfer_learning_tutorial.ipynb
    stripped: transfer-learning/transfer_learning_tutorial.ipynb
    code_cells: 13
    environment: torch.dev.gpu  # Maps to Docker service
    description: Transfer learning tutorial with PyTorch
```

### Execution

#### Execute a Single Notebook

```bash
make execute-notebook NOTEBOOK=transfer-learning/transfer_learning_tutorial.ipynb
```

This will:
1. Read the registry to determine the Docker environment
2. Execute the notebook in the appropriate container (e.g., `torch.dev.gpu`)
3. Create output directories: `output/{images,videos,audio,text}`
4. Save the executed notebook as `*-executed.ipynb`

#### Execute All Notebooks

```bash
make execute-all-notebooks
```

Executes all notebooks listed in the registry.

### Artifact Storage

Each notebook has its own artifact directory structure:

```
notebooks/
  transfer-learning/
    transfer_learning_tutorial.ipynb          # Original stripped notebook
    transfer_learning_tutorial-executed.ipynb # With outputs
    output/
      images/      # Generated plots, visualizations
      videos/      # Training animations, demos
      audio/       # Audio outputs
      text/        # Logs, metrics, reports
```

### Available Environments

- `torch.dev.gpu` - PyTorch development with CUDA support (local Docker execution)
- `ros.dev.gpu` - ROS 2 (Jazzy) with GPU support (local Docker execution)
- `colab` - Google Colab execution (for notebooks with Colab-specific dependencies)

### Google Colab Execution

Notebooks with `environment: colab` are designed to run in Google Colab and cannot be executed locally. When you run:

```bash
make execute-notebook NOTEBOOK=projects/cv/sports-analytics/basketball/basketball_ai_how_to_detect_track_and_identify_basketball_players.ipynb
```

You'll receive a Colab link instead of local execution:

```
═══════════════════════════════════════════════════════════════
  This notebook requires Google Colab
═══════════════════════════════════════════════════════════════

🔗 Open in Colab:
   https://colab.research.google.com/github/pantelis/eng-ai-agents/blob/main/notebooks/...

Manual steps:
  1. Click the link above to open in Colab
  2. Run all cells in Colab
  3. Download any generated artifacts manually
```

**When to use `colab` environment:**
- Notebooks using `google.colab` imports
- Notebooks with `!pip install` or `!apt-get` shell commands
- Notebooks downloading data with `!gdown` or similar
- Notebooks requiring dependencies not in Docker images

### Notebook Parameters

Executed notebooks receive these parameters automatically:

- `output_dir` - Base output directory path
- `images_dir` - Path for image outputs
- `videos_dir` - Path for video outputs
- `audio_dir` - Path for audio outputs
- `text_dir` - Path for text outputs

Use these parameters in your notebooks to save artifacts to the correct locations.

#### Example Notebook Cell

```python
# Parameters cell (will be injected by papermill)
output_dir = "."  # Default value
images_dir = "./images"  # Default value

# Save a plot
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3, 4])
plt.savefig(f"{images_dir}/my_plot.png")
plt.close()
```

### Enabling Artifact Saving in Notebooks

**IMPORTANT**: Notebooks are modified in-place to include artifact saving code. This is a one-time setup per notebook.

#### Process a Single Notebook

```bash
make add-artifact-saving NOTEBOOK=transfer-learning/transfer_learning_tutorial.ipynb
```

This modifies the original notebook to:
1. Add a parameters cell at the beginning (if not present)
2. Insert `plt.savefig()` calls before each `plt.show()`
3. Generate descriptive filenames from plot titles
4. Use the injected `images_dir` variable for paths

**The original notebook file is overwritten** with these additions. Commit the changes to git.

#### Process All Notebooks

```bash
make add-artifact-saving-all
```

Modifies all notebooks in the registry in-place.

#### What the Script Does

**Before:**
```python
plt.title("Monte Carlo vs TD(0) on Random Walk")
plt.show()
```

**After:**
```python
# Parameters (injected by papermill)
images_dir = "./images"  # Added at top of notebook

# ... later in code ...
plt.title("Monte Carlo vs TD(0) on Random Walk")
plt.savefig(f"{images_dir}/monte_carlo_vs_td0_on_random_walk.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Adding New Notebooks

When adding a new notebook to the registry:

1. Add the entry to `notebook-database.yml`:
   ```yaml
   - source: aiml-common/path/to/notebook.ipynb
     stripped: destination/path/notebook.ipynb
     code_cells: <count>
     environment: torch.dev.gpu  # or ros.dev.gpu
     description: Brief description
   ```

2. Add artifact saving to the notebook (one-time setup):
   ```bash
   make add-artifact-saving NOTEBOOK=destination/path/notebook.ipynb
   ```
   This modifies the original notebook in-place. Commit the changes.

3. Execute the notebook:
   ```bash
   make execute-notebook NOTEBOOK=destination/path/notebook.ipynb
   ```

4. Verify outputs in the `output/` directory

### Recommended Workflow

For the best workflow, modify all notebooks once, then commit them:

```bash
# 1. Add artifact saving to all notebooks (one-time setup)
make add-artifact-saving-all

# 2. Review and commit the modified notebooks
git add notebooks/
git commit -m "Add artifact saving to notebooks"

# 3. Now you can execute any notebook and get artifacts
make execute-notebook NOTEBOOK=<any-notebook>
```

### See Also

- [EXECUTION_PLAN.md](EXECUTION_PLAN.md) - Detailed implementation plan
- [notebook-database.yml](notebook-database.yml) - Notebook registry
