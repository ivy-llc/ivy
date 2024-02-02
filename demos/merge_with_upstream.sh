#!/bin/bash -e
git checkout "$1"
git remote add upstream https://github.com/unifyai/demos.git || true
git fetch upstream
git merge upstream/main --no-edit
git push