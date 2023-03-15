#!/bin/bash
# file to setup documentation building pipeline without docker

# install libraries for the doc-builder
pip install -r ./requirements.txt || exit 1

# install libraries for ivy
pip install -r ../requirements/requirements.txt || exit 1
if [[ $(arch) == 'arm64' ]]; then
  pip install -r ../requirements/optional_m1_1.txt || exit 1
  pip install -r ../requirements/optional_m1_2.txt || exit 1
else
  pip install -r ../requirements/optional.txt || exit 1
fi

# Delete any previously generated content
rm -rf docs

# delete any previously generated pages
rm -rf build

sphinx-build -b html . build

# Disable Jekyll in GitHub pages
touch build/.nojekyll
