#!/bin/bash

git submodule update --init --recursive

python3 -m pip install --user -e .

python3 -m pip install pre-commit

git config --global --add safe.directory /workspaces/ivy

( cd /workspaces/ivy/ && pre-commit install)
