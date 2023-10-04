#!/bin/bash -e

# For some reason torch needed to be installed sequentially before installing from 
# requirements.txt
pip install torch || exit 1
pip install torch-scatter || exit 1
