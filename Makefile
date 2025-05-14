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

# Documentation-specific targets
docs:
	@echo "Building documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Documentation built in $(BUILDDIR)/html."

docs-live:
	@echo "Starting live documentation server..."
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) --open-browser

docs-strict:
	@echo "Building documentation with strict warnings-as-errors mode..."
	@$(SPHINXBUILD) -b html -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Documentation built in strict mode in $(BUILDDIR)/html."

# This target builds documentation but filters out specific common warnings
# It's useful for CI where we want to treat serious warnings as errors 
docs-except-title-warnings:
	@echo "Building documentation with modified warnings-as-errors mode..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) 2>&1 | grep -v "Title underline too short" > build.log || true
	@cat build.log
	@! grep -q "WARNING:" build.log
	@echo "Documentation built with warnings checked in $(BUILDDIR)/html."

docs-linkcheck:
	@echo "Checking documentation links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "Link check results in $(BUILDDIR)/linkcheck."

docs-pdf:
	@echo "Building PDF documentation..."
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Converting to PDF..."
	@make -C "$(BUILDDIR)/latex" all-pdf
	@echo "PDF documentation built in $(BUILDDIR)/latex."

docs-clean:
	@echo "Cleaning documentation build directory..."
	@rm -rf "$(BUILDDIR)"
	@echo "Documentation build directory cleaned."

docs-all: docs-clean docs docs-linkcheck
	@echo "All documentation tasks completed."

.PHONY: help Makefile checks docs docs-live docs-strict docs-except-title-warnings docs-linkcheck docs-pdf docs-clean docs-all

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
