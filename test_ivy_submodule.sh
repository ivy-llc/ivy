#!/bin/bash -e

mkdir -p .hypothesis
# shellcheck disable=SC2046

echo $PWD
docker run --rm -v IVY_BACKEND="$1"  -v $(pwd):/ivy -v $(pwd)/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest ivy/ivy_tests/test_ivy/test_functional/test_core/"$2"
