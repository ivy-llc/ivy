#!/bin/bash -e
git checkout "$1"
<<<<<<< HEAD
git remote add upstream https://github.com/unifyai/ivy.git || true
git fetch upstream
git merge upstream/main --no-edit
git push
=======
git remote add upstream https://github.com/unifyai/demos.git || true
git fetch upstream
git merge upstream/main --no-edit
git push
>>>>>>> 73a66de61fa0c78c47a3b19b29c47f34c96ef441
