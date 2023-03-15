#!/bin/bash -e

# For some reason torch needed to be installed sequentially before installing from 
# requirements.txt
pip install torch || exit 1

# torch-scatter supporting torch 2.0 is only available on github
apt-get install -y git
pip install git+https://github.com/rusty1s/pytorch_scatter.git || exit 1