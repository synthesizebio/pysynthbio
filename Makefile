# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Target to run all checks (lint, type check, format check, tests) using uv
checks:
	@echo "Running ruff checks..."
	@uv run ruff check .
	@echo "\nRunning mypy type checks..."
	@uv run mypy . --namespace-packages --explicit-package-bases --ignore-missing-imports
	@echo "\nRunning black formatting checks..."
	@uv run black --check .
	@echo "\nRunning pytest tests..."
	@uv run pytest
	@echo "\nChecks completed."

.PHONY: help Makefile checks

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
