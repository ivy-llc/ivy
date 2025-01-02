#!/bin/bash

folder=$1
export DEBUG=0

python3 -m pip install timm kornia FrEIA

python3 -m pytest ivy_tests/test_transpiler/$folder -p no:warnings
