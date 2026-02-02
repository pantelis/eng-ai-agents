# Introduction

## What is this repository?

This is a template docker-based dev environment. It supports NVIDIA GPUs on Linux, and provides CPU-based containers for macOS (Apple Silicon and Intel).

It currently includes the following tools:

- An `assignments` directory with notebooks where you populate your code
- A `project` directory for your project source code. The documentation for the project is stored separately in the `docs` directory
- A `docs` directory that contains the source code of [Quarto](https://quarto.org/) markdown (qmd) and ipynb notebooks content. You use the docs folder to publish your project work

## How to Launch the Development Container in VS Code

This repository includes a VS Code development container configuration that can be launched with either CPU or GPU support.

### Prerequisites

1. **Install VS Code** with the "Dev Containers" extension
2. **Install Docker** and ensure it's running
3. **For GPU support**: Install NVIDIA Container Toolkit (for Linux) or Docker Desktop with GPU support

**IMPORTANT:** After the container is launched, you must run the following commands **inside the container** to set up the environment:

```bash
make start                    # Must run inside container (uses uv package manager)
source .venv/bin/activate     # Activate the virtual environment
```

**Critical Note:** The `make start` command (and `make venv-recreate`) must be executed from within the devcontainer, not on the host machine. The Makefile uses the `uv` package manager which is only available inside the container and respects container-specific constraints.

## Running with Docker Compose (without VS Code)

You can also run the containers directly with Docker Compose:

```bash
# Copy environment file
cp .env.example .env

# Build and start the PyTorch GPU container
docker compose up -d torch.dev.gpu

# Exec into the running container
docker compose exec torch.dev.gpu bash

# Or for ROS development
docker compose up -d ros.dev.gpu
docker compose exec ros.dev.gpu bash
```

## macOS Development (Apple Silicon & Intel)

This repository includes dedicated support for macOS users, including both Apple Silicon (M1/M2/M3/M4) and Intel-based Macs.

### Important Limitations

**GPU Acceleration (MPS) is NOT available inside Docker on macOS.** Docker Desktop uses virtualization (HyperKit/Apple Virtualization Framework) which lacks access to Apple's Metal GPU APIs. For GPU-accelerated PyTorch training using MPS, you must run PyTorch natively on macOS, not inside Docker.

For more details, see:

- [Apple Silicon GPUs, Docker and Ollama: Pick Two](https://chariotsolutions.com/blog/post/apple-silicon-gpus-docker-and-ollama-pick-two/)
- [PyTorch MPS Backend Documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)

### macOS Services

The repository provides macOS-specific services via `docker-compose-mac.yml`:

- **`torch.dev.mac`**: PyTorch development environment (CPU-only)
- **`ros.dev.mac`**: ROS 2 Jazzy development environment

### Running on macOS

```bash
# Copy environment file
cp .env.example .env

# Build and start the PyTorch container for macOS
docker compose -f docker-compose-mac.yml up -d torch.dev.mac

# Exec into the running container
docker compose -f docker-compose-mac.yml exec torch.dev.mac bash

# Or for ROS 2 development
docker compose -f docker-compose-mac.yml up -d ros.dev.mac
docker compose -f docker-compose-mac.yml exec ros.dev.mac bash
```

### VS Code Dev Container for macOS

To use the macOS containers with VS Code, update `.devcontainer/devcontainer.json`:

```json
{
  "dockerComposeFile": ["../docker-compose-mac.yml"],
  "service": "torch.dev.mac",
  "runServices": ["torch.dev.mac"]
}
```

### GUI Applications (RViz2, etc.) on macOS

Running GUI applications like RViz2 requires XQuartz:

1. Install XQuartz:

   ```bash
   brew install --cask xquartz
   ```

2. Open XQuartz and go to **Preferences > Security**
3. Enable **"Allow connections from network clients"**
4. **Reboot your Mac** (required for changes to take effect)
5. After reboot, allow connections:

   ```bash
   xhost +localhost
   ```

**Troubleshooting:** If you see `Error: Can't open display: host.docker.internal:0`, ensure you have:

- Completed all XQuartz configuration steps above
- Rebooted your Mac after enabling network clients
- Run `xhost +localhost` in a terminal on the host

For more details, see:

- [Installing ROS 2 on macOS with Docker (Foxglove)](https://foxglove.dev/blog/installing-ros2-on-macos-with-docker)
- [Setup ROS 2 Dev Docker with Emacs in macOS](https://qurobotics.de/blog/2024-01-11-setup-ros2-dev-docker-with-emacs-in-macos/)
- [Development Container for ROS 2 on ARM64 Mac](https://github.com/tatsuyai713/Development-Container-for-ROS2-on-Arm64-Mac)

### Native PyTorch with MPS (Recommended for GPU Training)

If you need GPU acceleration on macOS, install PyTorch natively (outside Docker):

```bash
# Create a virtual environment
python3 -m venv .venv-native
source .venv-native/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

See [Apple's PyTorch Metal documentation](https://developer.apple.com/metal/pytorch/) for more details.

### Port Customization

You can customize the exposed ports by modifying the `.env` file. Each service has its own set of port variables:

**Dev Service Ports:**

- `DEV_QUARTO_PORT`: Quarto preview server (default: 4100)
- `DEV_JUPYTER_PORT`: Jupyter notebook server (default: 8888)
- `DEV_PORT`: Additional development server (default: 8000)

**ROS Service Ports:**

- `ROS_PORT`: ROS master port (default: 11311)
- `ROS_QUARTO_PORT`: Quarto preview server (default: 4180)
- `ROS_JUPYTER_PORT`: Jupyter notebook server (default: 8880)
- `ROS_DEV_PORT`: Additional development server (default: 8078)
- `FOXGLOVE_PORT`: Foxglove bridge WebSocket (default: 8765)

Note: The actual ports exposed will be the values from your `.env` file.

### Service Selection

The repository supports multiple container configurations:

- **`torch.dev.gpu`**: PyTorch development environment with GPU support (Linux)
- **`ros.dev.gpu`**: ROS 2 (Jazzy) development environment with GPU support (Linux)
- **`torch.dev.mac`**: PyTorch development environment for macOS (CPU-only)
- **`ros.dev.mac`**: ROS 2 (Jazzy) development environment for macOS

#### Switching Services

To switch between services, modify the `service` field in `.devcontainer/devcontainer.json`:

```json
{
  "service": "torch.dev.gpu"  // or "ros.dev.gpu", "torch.dev.mac", "ros.dev.mac"
}
```

After changing the service configuration, rebuild the container using VS Code's "Dev Containers: Rebuild Container" command.

#### Why Two Containers Launch by Default

When VS Code opens the Dev Container, it runs `docker-compose up` which starts **all services** defined in `docker-compose.yml` by default. The `"service"` field in `devcontainer.json` only specifies which container VS Code attaches to—it doesn't limit which containers are started.

Both containers share the same network (`ai-agents-network`), allowing inter-container communication if needed.

**To start only a single container**, add the `runServices` property to `.devcontainer/devcontainer.json`:

```json
{
  "service": "torch.dev.gpu",
  "runServices": ["torch.dev.gpu"]
}
```

This explicitly tells VS Code to only start the specified service(s) rather than all services in the compose file.

## What should I do with it?

- Follow all instructions under [resources in the class website](https://aegean.ai/aiml-common/resources/environment/) as you will need it to submit your work
- Familiarize yourself with the `uv` package manager as you will use it to build and manage all your dependencies
- Follow the instructions in the course web site under resources to [submit your GitHub repo to the course's LMS system](https://aegean.ai/aiml-common/resources/environment/assignment-submission.html) (Canvas/Brightspace)

### Additional Notes for ROS Development

**ROS 2 Discovery Settings (macOS):** The macOS containers are configured with `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and `ROS_DOMAIN_ID=42` to ensure proper ROS 2 node discovery within Docker's network isolation.

**Foxglove Bridge:** Connect to the Foxglove app using:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

**Building ROS Packages:** Use the provided aliases:

```bash
# Install dependencies
rosdi

# Build with symlink install
cbuild

# Source the workspace
ssetup
```
