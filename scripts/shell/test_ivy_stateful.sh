#!/bin/bash -e
python3 -m pytest --backend "$1" ivy_tests/test_ivy/test_stateful/"$2".py --tb=line
