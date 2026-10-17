# Deliberately minimal: a few targets, as an example of the pattern (a "get me set up"
# target and a "prove it works" target), not a wrapper around every command in the
# project. Everything else — training, evaluating, the demo, linting, tests — is run
# directly (see README.md's Quick start / Development sections for the exact
# commands); add more targets here only if your team actually wants the shortcut.

PYTHON ?= python3.13
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup data test

setup: ## Create .venv, install runtime+dev deps and this package, install pre-commit hooks
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt
	$(BIN)/pip install -e .
	$(BIN)/pre-commit install

data: ## Download/verify Z-FPV-v1 -> frames -> YOLO labels (needs HF access to the gated dataset)
	$(BIN)/python scripts/download_data.py
	$(BIN)/python scripts/prepare_data.py

test: ## Leakage guards and conversion tests (no gated data needed)
	$(BIN)/pytest

# TODO(team): add `reproduce` (data -> train -> evaluate) once the baseline exists.
