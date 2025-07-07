.PHONY: install install-dev format lint lint-check type-check test test-cov test-examples clean build deps-update deps-sync quality style fixup venv venv-recreate setup-dev docker-build-gpu docker-build-cpu docker-build docker-run-gpu docker-run-cpu ci-quality ci-test
# Use stage 0 container pip constraints
CONSTRAINTS := --constraint /etc/pip/constraint.txt

export PYTHONPATH = src
check_dirs := examples tests src utils
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/python
UV := /usr/bin/uv

# Create venv with access to system packages (from stage 0 container)
$(VENV_DIR)/bin/activate:
	rm -rf $(VENV_DIR)
	python3 -m venv $(VENV_DIR) --system-site-packages

venv: $(VENV_DIR)/bin/activate

# Installation targets
install: venv
	$(UV) pip install -e . 

#$(CONSTRAINTS)

install-dev: venv
	$(UV) pip install -e ".[dev]" 
#$(CONSTRAINTS)


# Development workflow targets
format: venv
	$(VENV_DIR)/bin/ruff format $(check_dirs)

lint: venv
	$(VENV_DIR)/bin/ruff check $(check_dirs) --fix

lint-check: venv
	$(VENV_DIR)/bin/ruff check $(check_dirs)

type-check: venv
	$(VENV_DIR)/bin/mypy src/eng-ai-agents

# Combined quality checks
quality: lint-check type-check
	@echo "All quality checks passed!"

# Quick style fix
style: format lint
	@echo "Code formatting and linting completed!"

# Quick fix for modified files only
fixup: venv
	$(eval modified_py_files := $(shell $(VENV_PY) utils/get_modified_files.py $(check_dirs) 2>/dev/null || echo ""))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		$(VENV_DIR)/bin/ruff check $(modified_py_files) --fix; \
		$(VENV_DIR)/bin/ruff format $(modified_py_files); \
		$(VENV_DIR)/bin/mypy $(modified_py_files) || true; \
	else \
		echo "No library .py files were modified"; \
	fi

# Testing targets
test:
	$(UV) run pytest

test-cov:
	$(UV) run pytest --cov=eng-ai-agents --cov-report=html --cov-report=term

test-examples:
	$(UV) run pytest examples/

# Dependency management
deps-update: venv
	$(UV) pip compile pyproject.toml --upgrade $(CONSTRAINTS)

deps-sync: venv
	$(UV) pip compile pyproject.toml $(CONSTRAINTS) | $(UV) pip sync -

deps-table-update: venv
	$(VENV_PY) utils/update_dependency_table.py

# Build and release targets
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	$(UV) build

build-install: build
	$(UV) pip install dist/*.whl $(CONSTRAINTS)

# Utility targets
venv-recreate:
	rm -rf $(VENV_DIR)
	$(MAKE) venv


# Docker targets
docker-build-gpu:
	docker build -f docker/Dockerfile.nvidia.dgpu -t eng-ai-agents:gpu .

docker-build-cpu:
	docker build -f docker/Dockerfile.cpu.amd64 -t eng-ai-agents:cpu .

docker-build: docker-build-gpu docker-build-cpu

docker-run-gpu:
	docker run --gpus all -it --rm \
		-v $(PWD):/workspace \
		-w /workspace \
		eng-ai-agents:gpu

docker-run-cpu:
	docker run -it --rm \
		-v $(PWD):/workspace \
		-w /workspace \
		eng-ai-agents:cpu

# Development setup
setup-dev: install-dev
	$(VENV_DIR)/bin/pre-commit install
	@echo "Development environment setup complete!"

# CI/CD helpers
ci-quality: lint-check type-check
ci-test: test-cov

start:
	$(MAKE) venv-recreate
	$(MAKE) deps-sync
	$(MAKE) install
	@echo "To activate the virtual environment, run: source .venv/bin/activate"
