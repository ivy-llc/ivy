#!/bin/bash

# Install dependencies
python3 -m pip install cryptography transformers black
cd /ivy/tracer-transpiler/
python3 -m pip install -r tests/source2source/requirements.txt

# Install the local ivy_repo
cd /ivy/tracer-transpiler/ivy_repo
python3 -m pip install --user -e .

cd ..
cp source_to_source_translator/caching/merge_cache.py .

# Run the Python script to merge the cache files
python3 merge_cache.py
