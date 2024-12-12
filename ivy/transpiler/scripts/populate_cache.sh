#!/bin/bash

# Install dependencies
python3 -m pip install cryptography transformers black
cd /ivy/tracer-transpiler/
python3 -m pip install -r tests/source2source/requirements.txt

# Install the local ivy_repo
cd /ivy/tracer-transpiler/ivy_repo
python3 -m pip install --user -e .

# Populate the cache based on the arguments
cd ..
cp source_to_source_translator/caching/populate_cache.py .

# Check the number of arguments
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <batch>"
  exit 1
fi

# Parse the batch JSON
batch=$1

# Pass the entire batch to the populate_cache.py script
python3 populate_cache.py "$batch"
