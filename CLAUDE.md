# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational Docker-based development environment for AI/ML and robotics courses. Supports PyTorch with GPU acceleration and ROS 2 (Jazzy) for robotics development.

## Development Workflow

**CRITICAL**: This is a **container-first development environment**. All development workflows and make commands are designed to run **inside the devcontainer**, not on the host machine.

### Starting Development
1. Open the project in VS Code with the Dev Containers extension
2. Use "Dev Containers: Reopen in Container" command
3. To switch services, edit `.devcontainer/devcontainer.json` and change the `"service"` field:
   - `"torch.dev.gpu"` for PyTorch development (default)
   - `"ros.dev.gpu"` for ROS 2 development
   - Then rebuild the container using "Dev Containers: Rebuild Container"
4. Once inside the container, run `make start` to initialize the environment

### Key Constraints
- **Python environment setup** (`make start`, `make venv-recreate`) must run inside the container
- The Makefile uses `uv` package manager which is only available in the container
- Virtual environment at `.venv` has system-site-packages access to container-installed packages
- UV package manager respects container constraints (`/etc/pip/constraint.txt`)

### Host vs. Container Operations
- **Inside Container**: All make commands, Python development, package management
- **On Host**: Git operations, editing `.devcontainer/devcontainer.json` to switch services

## Development Commands

**Note**: All commands below must be executed from within the devcontainer.

### Initial Setup (Inside Container)
```bash
make start                    # Create venv, sync deps, install package (MUST run in container)
source .venv/bin/activate     # Activate environment
```

### Code Quality
```bash
make format                   # Format with ruff
make lint                     # Lint with auto-fix
make lint-check               # Lint without modifications
make type-check               # MyPy strict type checking
make quality                  # All quality checks (lint-check + type-check)
make style                    # Format + lint combined
make fixup                    # Fix only modified files
```

### Testing
```bash
make test                     # Run pytest
make test-cov                 # Pytest with coverage (HTML + terminal)
```

### Dependencies
```bash
make deps-sync                # Sync dependencies from lock file
make deps-update              # Update dependencies
make venv-recreate            # Clean and recreate venv (MUST run in container)
```

### Build
```bash
make build                    # Build wheel
make clean                    # Remove build artifacts
```

## Architecture

### Container Services (docker-compose.yml)
- **torch.dev.gpu**: PyTorch development with CUDA 12.8 (default)
- **ros.dev.gpu**: ROS 2 Jazzy with TurtleBot3, slam-toolbox, foxglove-bridge

Switch services by editing the `"service"` field in `.devcontainer/devcontainer.json` and rebuilding the container.

### Directory Structure
- `assignments/` - Jupyter notebooks for course assignments
- `project/` - Student project source code
- `docs/` - Quarto documentation (qmd + ipynb)
- `ros_ws/` - ROS 2 workspace with colcon build system
- `docker/` - Dockerfile variants (torch/ros, gpu/cpu, dev/prod)

### Package Management
Uses **UV** package manager with constraints from base Docker image (`/etc/pip/constraint.txt`). Virtual environment at `.venv` with system-site-packages access.

## Code Style

- **Formatter/Linter**: Ruff (line length: 119)
- **Type Checker**: MyPy strict mode
- **Python**: 3.11-3.12.3

Key ruff rules enabled: E, W, F, I, C, N, UP, B, A, S, T20, SIM, ARG, PTH, PL, RUF

## ROS 2 Development

```bash
# Build ROS workspace
cd ros_ws && colcon build

# Source workspace
source ros_ws/install/setup.bash

# Launch Foxglove bridge
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

Useful aliases in ROS container: `cbuild` (colcon build), `ssetup` (source setup), `rosdi` (rosdep install), `cyclone`/`fastdds` (RMW selection).

## Port Mappings (configurable via .env)

| Service | Jupyter | Quarto | Dev Server | ROS Master | Foxglove |
|---------|---------|--------|------------|------------|----------|
| torch.dev.gpu | 8888 | 4100 | 8000 | - | - |
| ros.dev.gpu | 8880 | 4180 | 8078 | 11311 | 8765 |
