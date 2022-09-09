#!/bin/bash
cd ..
# shellcheck disable=SC2046
docker run --rm -v "$(pwd)":/ivy unifyai/ivy:latest python3 ivy/run_tests_CLI/test_dependencies.py \
-fp ivy/requirements/requirements.txt,ivy/requirements/optional.txt