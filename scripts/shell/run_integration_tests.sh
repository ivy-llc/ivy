#!/bin/bash

integration=$1
target=$2

pip3 install kornia
pytest ivy_tests/test_integrations/test_$integration.py -p no:warnings --target $target
