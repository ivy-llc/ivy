#!/bin/bash

integration=$1
target=$2

export IVY_KEY=$3
export VERSION=linux-nightly
export DEBUG=0

pip3 install -r requirements/requirements.txt --upgrade
pip3 install jax
pip3 install jaxlib
pip3 install flax
pip3 install opencv-python
pip3 install pytest
pip3 install tensorflow
pip3 install torch
pip3 install torchvision
pip3 install kornia
pip3 install accelerate
pip3 install transformers
pip3 install redis
pip3 install hypothesis

# get the nightly binaries
python << 'EOF'
import ivy
ivy.utils.cleanup_and_fetch_binaries()
EOF

pytest ivy_tests/test_integrations/test_$integration.py -p no:warnings --tb=short --target $target
