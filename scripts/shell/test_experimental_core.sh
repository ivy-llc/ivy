#!/bin/bash -e
python3 -m pytest --backend "$1" ivy_tests/test_ivy/test_functional/test_experimental/test_core/"$2".py --tb=short
