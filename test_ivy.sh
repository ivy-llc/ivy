#!/bin/bash -e
docker run --rm --env IVY_BACKEND="$1" -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest ivy/ivy_tests/test_functional/test_core/"$2".py
