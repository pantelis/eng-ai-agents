# Notebooks

This folder contains notebooks from the site https://aegean.ai/courses. You are free to edit those and issue Pull Requests with improvements. We are considering each PR for inclusion in the main repository.

## Notebook Execution System

This directory contains stripped notebooks from the private `aiml-common` repository with support for isolated execution environments and artifact storage.

### Registry

The `stripped-notebooks.yml` file tracks all notebooks and their execution environments:

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

- `torch.dev.gpu` - PyTorch development with CUDA support
- `ros.dev.gpu` - ROS 2 (Jazzy) with GPU support

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

### Adding New Notebooks

When adding a new notebook to the registry:

1. Add the entry to `stripped-notebooks.yml`:
   ```yaml
   - source: aiml-common/path/to/notebook.ipynb
     stripped: destination/path/notebook.ipynb
     code_cells: <count>
     environment: torch.dev.gpu  # or ros.dev.gpu
     description: Brief description
   ```

2. Execute the notebook:
   ```bash
   make execute-notebook NOTEBOOK=destination/path/notebook.ipynb
   ```

3. Verify outputs in the `output/` directory

### See Also

- [EXECUTION_PLAN.md](EXECUTION_PLAN.md) - Detailed implementation plan
- [stripped-notebooks.yml](stripped-notebooks.yml) - Notebook registry
