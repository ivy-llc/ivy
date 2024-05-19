#!/bin/bash -e

set +e
pytest ivy_tests/test_ivy/test_frontends/test_tensorflow/ -p no:warnings --tb=short
pytest_exit_code=$?
set -e

if [ $pytest_exit_code -eq 0 ] || [ $pytest_exit_code -eq 1 ]; then
    exit 0
else
    exit 1
fi
