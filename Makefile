.PHONY: install install-dev format lint lint-check type-check test test-cov test-examples clean build deps-update deps-sync quality style fixup venv venv-recreate setup-dev docker-build-gpu docker-build-cpu docker-build docker-run-gpu docker-run-cpu ci-quality ci-test execute-notebook execute-all-notebooks add-artifact-saving add-artifact-saving-all
# Use stage 0 container pip constraints (only if file exists)
CONSTRAINT_FILE := /etc/pip/constraint.txt
CONSTRAINTS := $(if $(wildcard $(CONSTRAINT_FILE)),--constraint $(CONSTRAINT_FILE),)

export PYTHONPATH = src
check_dirs := examples tests src utils
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/python
UV := $(shell which uv)
# Find Python 3.11+ (prefer /usr/local/bin/python for PyTorch container compatibility)
PYTHON := $(shell \
	for cmd in /usr/local/bin/python /usr/bin/python3; do \
		if [ -x "$$cmd" ] && $$cmd --version 2>/dev/null | grep -qE "3\.(1[1-9]|[2-9][0-9])"; then \
			echo $$cmd; \
			exit 0; \
		fi; \
	done; \
	command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || echo python3)

# Create venv with access to system packages (from stage 0 container)
$(VENV_DIR)/bin/activate:
	rm -rf $(VENV_DIR)
	$(UV) venv $(VENV_DIR) --python $(PYTHON) --system-site-packages

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

# Notebook execution targets
execute-notebook:
ifndef NOTEBOOK
	@echo "Error: NOTEBOOK parameter is required"
	@echo "Usage: make execute-notebook NOTEBOOK=<notebook-path>"
	@echo "Example: make execute-notebook NOTEBOOK=transfer-learning/transfer_learning_tutorial.ipynb"
	@exit 1
endif
	@echo "Getting environment for notebook: $(NOTEBOOK)"
	$(eval ENV := $(shell python3 scripts/get_notebook_environment.py $(NOTEBOOK)))
	@echo "Environment: $(ENV)"
	@if [ "$(ENV)" = "colab" ]; then \
		echo ""; \
		echo "═══════════════════════════════════════════════════════════════"; \
		echo "  This notebook requires Google Colab"; \
		echo "═══════════════════════════════════════════════════════════════"; \
		echo ""; \
		echo "This notebook uses Colab-specific features and cannot be"; \
		echo "executed locally. Please run it in Google Colab instead."; \
		echo ""; \
		echo "📂 Notebook: $(NOTEBOOK)"; \
		echo ""; \
		echo "🔗 Open in Colab:"; \
		echo "   https://colab.research.google.com/github/pantelis/eng-ai-agents/blob/main/notebooks/$(NOTEBOOK)"; \
		echo ""; \
		echo "Manual steps:"; \
		echo "  1. Click the link above to open in Colab"; \
		echo "  2. Run all cells in Colab"; \
		echo "  3. Download any generated artifacts manually"; \
		echo ""; \
		echo "═══════════════════════════════════════════════════════════════"; \
	else \
		echo "Executing notebook in Docker environment: $(ENV)"; \
		docker compose run --rm $(ENV) python scripts/execute_notebook.py $(NOTEBOOK); \
		echo "✓ Notebook execution complete"; \
	fi

execute-all-notebooks:
	@echo "Executing all notebooks from registry..."
	@python3 scripts/list_notebooks.py | while read nb; do \
		echo ""; \
		echo "========================================"; \
		echo "Executing: $$nb"; \
		echo "========================================"; \
		$(MAKE) execute-notebook NOTEBOOK=$$nb || true; \
	done
	@echo ""
	@echo "✓ All notebooks processed"

# Add artifact saving to notebooks
add-artifact-saving:
ifndef NOTEBOOK
	@echo "Error: NOTEBOOK parameter is required"
	@echo "Usage: make add-artifact-saving NOTEBOOK=<notebook-path>"
	@echo "Example: make add-artifact-saving NOTEBOOK=transfer-learning/transfer_learning_tutorial.ipynb"
	@exit 1
endif
	@echo "Adding artifact saving to: notebooks/$(NOTEBOOK)"
	python3 scripts/add_artifact_saving.py notebooks/$(NOTEBOOK)
	@echo "✓ Notebook modified"

add-artifact-saving-all: venv
	@echo "Adding artifact saving to all notebooks in registry..."
	@$(VENV_PY) -c "import yaml; \
		notebooks = yaml.safe_load(open('notebooks/stripped-notebooks.yml'))['notebooks']; \
		[print(n['stripped']) for n in notebooks if n != '---']" | while read nb; do \
			echo "Processing: $$nb"; \
			python3 scripts/add_artifact_saving.py notebooks/$$nb || true; \
		done
	@echo ""
	@echo "✓ All notebooks processed"
