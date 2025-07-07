# Introduction

## What is this repository?

This is a template docker-based dev environment. It currently supports NVIDIA GPUs but with slight modifications it can target for x86 CPUs and Apple silicon chips. 

It currently includes the following tools:

* a `assignments` directory with an empty notebook where you need to populate with your code. The notebook can optionally use the artagents library. 
* a `project` directory for your project source code. The documentation for the project is stored separately in the `docs` directory. 
* a `docs` directory that contains the source code of [quarto](https://quarto.org/) markdown (qmd) and `ipynb` notebooks content. You use the docs folder to publish your project work. 

## How to Launch the Development Container in VS Code

This repository includes a VS Code development container configuration that can be launched with either CPU or GPU support.

### Prerequisites

1. **Install VS Code** with the "Dev Containers" extension
2. **Install Docker** and ensure it's running
3. **For GPU support**: Install NVIDIA Container Toolkit (for Linux) or Docker Desktop with GPU support

### Environment Configuration

The repository includes a `.env` file that controls the container configuration. You can modify this file to customize your setup:

```env
# UV Extra for PyTorch (cpu or cu128 for CUDA)
UV_EXTRA=cpu

# Workspace configuration
WORKSPACE_DIR=/workspaces/eng-ai-agents
WORKSPACE_USER=vscode

# Container configuration
CONTAINER_NAME=eng-ai-agents-dev

# Port mappings
QUARTO_PORT=4199
JUPYTER_PORT=8890
DEV_PORT=8088
```

### Option 1: CPU-Only Setup (Default)

The default configuration uses CPU-only PyTorch:

1. Ensure the `.env` file has `UV_EXTRA=cpu` (this is the default)
2. Open the repository in VS Code
3. When prompted, click "Reopen in Container" or use the command palette (`Ctrl+Shift+P`) and select "Dev Containers: Reopen in Container"

### Option 2: GPU Setup (CUDA Support)

To enable GPU support with CUDA:

1. **Edit the `.env` file** and change `UV_EXTRA=cpu` to `UV_EXTRA=cu128`
2. Open the repository in VS Code
3. When prompted, click "Reopen in Container" or use the command palette and select "Dev Containers: Reopen in Container"

### Manual Setup Commands

The project uses a comprehensive Makefile for all environment management tasks:

```bash
# View all available commands
make help

# Setup virtual environment based on .env file settings
make setup-venv

# Or explicitly choose CPU/GPU setup
make setup-venv-cpu    # For CPU-only setup
make setup-venv-gpu    # For GPU setup (CUDA 12.8)

# Update dependencies after changing UV_EXTRA in .env
make sync-cpu          # Sync CPU dependencies
make sync-gpu          # Sync GPU dependencies

# Docker container management
make docker-build      # Build container
make docker-up         # Start container
make docker-down       # Stop container
make docker-rebuild    # Rebuild and restart container

# Development tools
make jupyter           # Start Jupyter Lab
make pre-commit        # Run pre-commit hooks
```

### Verification

After the container starts, you can verify your setup:

```bash
# Check Python environment
python --version

# Check UV_EXTRA setting
echo $UV_EXTRA

# Check if GPU is available (for GPU setup)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Port Customization

You can customize the exposed ports by modifying the `.env` file:

* `QUARTO_PORT`: Quarto preview server (default: 4199)
* `JUPYTER_PORT`: Jupyter notebook server (default: 8890)
* `DEV_PORT`: Additional development server (default: 8088)

Note: The actual ports exposed will be the values from your `.env` file.

## What should I do with it?

* Follow all instructions under [resources in the class website](https://pantelis.github.io/aiml-common/resources/environment/) as you will need it to submit your work.
* Familiarize yourself with the `uv` package manager as you will use it to build and manage all your dependencies.
* Follow the instructions in the course web site under resources to [submit your github repo to the course's LLM system](https://pantelis.github.io/aiml-common/resources/environment/assignment-submission.html) (Canvas/Brightspace).