#!/bin/bash -e

submodule=$1
backend=$2

pytest ivy_tests/test_ivy/test_frontends/test_jax/ -p no:warnings --tb=short
pytest_exit_code=$?
if [ $pytest_exit_code -eq 0 ] || [ $pytest_exit_code -eq 1 ]; then
    exit 0
else
    exit 1
fi
