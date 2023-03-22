#!/bin/bash
# file to setup documentation building pipeline without docker
# $1 : Path to the library being documented

# install libraries for the doc-builder
pip install -r $1/requirements.txt || exit 1

# Run a prebuild script if exists
[ -x docs/prebuild.sh ] && docs/prebuild.sh

if [ -d requirements ]; then
  # install libraries for ivy
  pip install -r requirements/requirements.txt || exit 1
  if [[ $(arch) == 'arm64' ]]; then
    pip install -r requirements/optional_m1_1.txt || exit 1
    pip install -r requirements/optional_m1_2.txt || exit 1
  else
    pip install -r requirements/optional.txt || exit 1
  fi
else
    pip install -r requirements.txt || exit 1
    [ -r optional.txt ] && (pip install -r optional.txt || exit 1)
fi

# delete any previously generated pages
rm -rf docs/build

function cleanup {
  echo "Cleaning up"
  # Restore the original docs
  rm -rf docs/
  mv docs.old/ docs/
  # Give read and write permissions to the docs folder, as docker root take ownership of 
  # the files
  chmod -R a+rw docs
}

function error_exit {
  echo "Error in building docs"
  cleanup
  exit 1
}

# Backing up the docs folder
cp -r $1/docs/ $1/docs.old/ || exit 1

# syncing ivy folder with the doc-builder folder
rsync -rav $1/docs/ docs/ || error_exit
rm -rf docs/partial_source/conf.py
cp docs/conf.py docs/partial_source/conf.py

sphinx-build -v -b html docs docs/build || error_exit

# Disable Jekyll in GitHub pages
touch docs/build/.nojekyll

# Move the build to docs.old
# mv docs/build docs.old/build || error_exit

# cleanup
