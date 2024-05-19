#!/bin/bash -e

submodule=$1
backend=$2

pytest ivy_tests/test_ivy/test_frontends/test_tensorflow/test_compat/test_v1/test_$submodule.py --backend $backend -p no:warnings --tb=short
