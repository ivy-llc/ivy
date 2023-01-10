#!/bin/bash
#conda env update --file docker/multicondaenv.yml  --prune
source activate multienv
conda install jsonpickle
python multiversion_testing.py