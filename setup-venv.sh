#!/bin/bash

# Setup script for the virtual environment
# This script can be run if the virtual environment needs to be recreated

set -e

echo "Setting up Python virtual environment..."

# Check if UV_EXTRA is set, default to 'cpu'
UV_EXTRA=${UV_EXTRA:-cpu}
echo "Using UV_EXTRA: $UV_EXTRA"

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new virtual environment
echo "Creating virtual environment..."
uv venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv sync --extra $UV_EXTRA --dev

# Install additional development tools
echo "Installing additional development tools..."
uv pip install jupyter jupyterlab ipython notebook

# Verify installation
echo "Verifying installation..."
python -c "import sys; print(f'Python: {sys.executable}')"
echo "UV version: $(uv --version)"

if python -c "import torch" 2>/dev/null; then
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    if [ "$UV_EXTRA" = "cu128" ]; then
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    fi
else
    echo "Warning: PyTorch not installed or not importable"
fi

echo "Virtual environment setup complete!"
echo "To activate the virtual environment, run: source .venv/bin/activate"
