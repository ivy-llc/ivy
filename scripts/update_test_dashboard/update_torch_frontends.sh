#!/bin/bash -e

submodule=$1
backend=$2
workflow_link=$3
db_key=$4

set +e
pytest ivy_tests/test_ivy/test_frontends/test_torch/test_$submodule.py --backend $backend -p no:warnings --tb=short --json-report --json-report-file=test_report.json
pytest_exit_code=$?
set -e

if [ $pytest_exit_code -eq 0 ] || [ $pytest_exit_code -eq 1 ]; then
    python scripts/update_test_dashboard/update_db.py --workflow-link $workflow_link --db-key $db_key
    exit 0
else
    exit 1
fi
