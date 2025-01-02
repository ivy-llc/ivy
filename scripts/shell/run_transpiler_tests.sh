#!/bin/bash

folder=$1
export DEBUG=0

python3 -m pip install datasets transformers onnx timm kornia FrEIA tf-keras

python3 -m pytest ivy_tests/test_transpiler/$folder -p no:warnings
