#!/bin/bash -e
git checkout "$1"
git remote add upstream https://github.com/unifyai/ivy.git || true
git fetch upstream
git merge upstream/master --no-edit
git push
