#!/bin/bash
# Used as a common callback for all of the `devcontainer.json` file variants after
#   the end of the Docker image creation step.

# Pull the newest available submodule sources.
git submodule update --init --recursive

# Install the Python package locally.
python3 -m pip install --user -e .

# Install the pre-commit hook package manager.
python3 -m pip install pre-commit

# Tell git to specifically treat the newly populated directory in the container, 
#   `/workspaces/ivy`, as a safe workspace.
git config --global --add safe.directory /workspaces/ivy

# Install the pre-commit hooks.
( cd /workspaces/ivy/ && pre-commit install)
