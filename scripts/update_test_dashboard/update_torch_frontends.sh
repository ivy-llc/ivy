#!/bin/bash -e

# pip install pytest-json-report 
# pytest ivy_tests/test_ivy/test_functional/test_core/test_elementwise.py -p no:warnings --json-report --json-report-file=report.json

submodule=$1
backend=$2

set +e
pytest ivy_tests/test_ivy/test_frontends/test_torch/test_$submodule.py --backend $backend -p no:warnings --tb=short --json-report --json-report-file=test_report.json
pytest_exit_code=$?
set -e

if [ $pytest_exit_code -eq 0 ] || [ $pytest_exit_code -eq 1 ]; then
    exit 0
else
    exit 1
fi