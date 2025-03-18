#!/bin/bash

version=$1
export DEBUG=0

# install deps
python -m pip install build kornia

# build the project
python -m build
cd dist/

# install the built wheel with pip
python -m pip install ivy-$version-py3-none-any.whl

# test that a simple transpilation works
python -c "import ivy; import kornia; ivy.transpile(kornia.color.rgb_to_grayscale, target='numpy')"
