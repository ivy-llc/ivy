#!/bin/bash
# file to setup documentation building pipeline without docker
# $1 : Path to the library being documented

# install libraries for the doc-builder
pip install -r ../requirements.txt || exit 1

# Run a prebuild script if exists
[ -x $1/docs/prebuild.sh ] && $1/docs/prebuild.sh

if [ -d $1/requirements ]; then
  # install libraries for ivy
  pip install -r $1/requirements/requirements.txt || exit 1
  if [[ $(arch) == 'arm64' ]]; then
    pip install -r $1/requirements/optional_m1_1.txt || exit 1
    pip install -r $1/requirements/optional_m1_2.txt || exit 1
  else
    pip install -r $1/requirements/optional.txt || exit 1
  fi
else
    pip install -r $1/requirements.txt || exit 1
    [ -r $1/optional.txt ] && (pip install -r $1/optional.txt || exit 1)
fi

# delete any previously generated pages
rm -rf $1/docs/build

function cleanup {
  echo "Cleaning up"
  # Restore the original docs
  rm -rf $1/docs/
  mv $1/docs.old/ $1/docs/
  # Give read and write permissions to the docs folder, as docker root take ownership of 
  # the files
  chmod -R a+rw $1/docs
}

function error_exit {
  echo "Error in building docs"
  cleanup $1
  exit 1
}

# Backing up the docs folder
cp -r $1/docs/ $1/docs.old/ || exit 1

# syncing ivy folder with the doc-builder folder
rsync -rav docs/ $1/docs/ || error_exit $1
rm -rf partial_source/conf.py
cp conf.py partial_source/conf.py

sphinx-build -v html $1/docs $1/docs/build || error_exit $1

# Disable Jekyll in GitHub pages
touch $1/docs/build/.nojekyll

# Move the build to docs.old
mv $1/docs/build $1/docs.old/build || error_exit $1

cleanup $1
