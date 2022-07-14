#!/bin/bash -e

mkdir -p .hypothesis
# shellcheck disable=SC2046
core_tests=( $(ls -d ivy_tests/test_ivy/test_functional/test_core/test*) )
docker run --rm -v IVY_BACKEND="$1"  -v $(pwd):/ivy -v $(pwd)/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest ivy/${core_tests[@]}
