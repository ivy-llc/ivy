#!/bin/bash
if [ -z "$1" ]; then
    echo "You need to provide an old version number"
    exit
fi
if [ -z "$2" ]; then
    echo "You need to provide a release version number"
    exit
fi

if ! grep -Fq "$1" setup.py; then
    echo "The old version is not present in setup.py, exiting"
    exit
fi

if grep -Fq "$2" setup.py; then
    echo "The new version already exists in setup.py, exiting"
    exit
fi

# shellcheck disable=SC2005
PACKAGE_NAME=$(echo "$(grep -F "name=" setup.py)" | cut -f2 -d"'")
PIP_RET=$(python3 -m pip index versions "$PACKAGE_NAME")
PIP_HAS_OLD=$(echo "$PIP_RET" | grep -F "$1")

if [ -z "$PIP_HAS_OLD" ]; then
    echo "The old version not found in PyPI, exiting"
    exit
fi

PIP_HAS_NEW=$(echo "$PIP_RET" | grep -F "$2")

if [ -n "$PIP_HAS_NEW" ]; then
    echo "The new version already in PyPI, exiting"
    exit
fi

sed -i "s/$1/$2/g" setup.py
git add -A
git commit -m "version $2"
git push
git tag -a "v$2" -m "version $2"
git push origin "v$2"
