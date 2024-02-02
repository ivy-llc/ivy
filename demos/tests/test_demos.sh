#!/bin/bash

if [ $3 == true ]; then
export VERSION=nightly
fi

cd ivy
pip install -e .

cd ../demos
mkdir .ivy
touch .ivy/key.pem
echo -n "$1" > .ivy/key.pem

# install dependencies
pip install -r requirements.txt >/dev/null 2>&1

# run test
echo "PATH : $2"
python3 tests/main.py "$2"
