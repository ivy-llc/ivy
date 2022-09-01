#!/bin/bash

pip3 install black
pip3 install flake8

git submodule update --init --recursive

python3 -m pip install --user -e .
