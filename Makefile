# ============================================================
#  Makefile — CS 495 Real-Time Truck ETA
#  Supports macOS/Linux and Windows (PowerShell / cmd via GNU Make)
# ============================================================

PYTHON   := python3.13
VENV     := .venv

# ── Platform detection ──────────────────────────────────────
ifeq ($(OS),Windows_NT)
    VENV_PYTHON  := $(VENV)/Scripts/python.exe
    VENV_PIP     := $(VENV)/Scripts/pip.exe
    VENV_ACT     := $(VENV)/Scripts/activate
    RM_VENV      := if exist $(VENV) rmdir /s /q $(VENV)
    PYTHON_CHECK := where python3.13 >nul 2>&1 || where python >nul 2>&1
else
    VENV_PYTHON  := $(VENV)/bin/python
    VENV_PIP     := $(VENV)/bin/pip
    VENV_ACT     := $(VENV)/bin/activate
    RM_VENV      := rm -rf $(VENV)
    PYTHON_CHECK := which $(PYTHON)
endif

.DEFAULT_GOAL := help

# ── Targets ─────────────────────────────────────────────────

.PHONY: help
help:          ## Show this help message
	@echo Usage: make [target]
	@echo.
	@echo Targets:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

.PHONY: venv
venv: $(VENV_PYTHON)  ## Create the virtual environment

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip

.PHONY: install
install: venv  ## Install all dependencies into the venv
	$(VENV_PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: install  ## Install dependencies + dev/test extras
	$(VENV_PIP) install pytest pytest-cov ruff

.PHONY: audit
audit: venv  ## Run the data audit script
	$(VENV_PYTHON) data_audit.py

.PHONY: segment
segment: venv  ## Run trip segmentation
	$(VENV_PYTHON) trip_segmentation.py

.PHONY: train
train: venv  ## Train the ETA model
	$(VENV_PYTHON) eta_model.py

.PHONY: test
test: venv  ## Run tests with pytest
	$(VENV_PYTHON) -m pytest tests/ -v

.PHONY: lint
lint: venv  ## Lint with ruff
	$(VENV_PYTHON) -m ruff check .

.PHONY: freeze
freeze: venv  ## Freeze current venv packages to requirements.txt
	$(VENV_PIP) freeze > requirements.txt

.PHONY: clean
clean:         ## Remove the virtual environment and cached files
	$(RM_VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

.PHONY: rebuild
rebuild: clean install  ## Clean and reinstall everything
