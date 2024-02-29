#!/bin/bash -e
docker run --rm --env REDIS_URL="$3" --env REDIS_PASSWD="$4" -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --backend "$1" ivy_tests/test_ivy/test_functional/test_experimental/test_nn/"$2".py --tb=short
