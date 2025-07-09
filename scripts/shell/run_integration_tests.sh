#!/bin/bash

integration=$1
target=$2

export IVY_KEY=$3
export DEBUG=0

pip install -r requirements/requirements.txt --upgrade
pip install jax
pip install jaxlib
pip install flax
pip install opencv-python
pip install pytest
pip install tensorflow
pip install torch
pip install torchvision
pip install kornia
pip install accelerate
pip install transformers
pip install hypothesis

pytest ivy_tests/test_integrations/test_$integration.py -p no:warnings --tb=short --target $target
