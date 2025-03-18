#!/bin/bash

# install dependencies
python3 -m pip install cryptography
python3 -m pip install dill
python3 -m pip install black 
python3 -m pip install pytest 

# TODO: update repo path
# install the local ivy_repo
cd /ivy/tracer-transpiler/ivy_repo
python3 -m pip install --user -e .
python3 -m pip install -r tests/source2source/requirements.txt

# Python script to generate the matrix
python3 <<EOF
import json
import os
from tests.source2source.translations.test_translations import get_test_list, get_target_list

runners = ["ubuntu-latest"]
python_versions = ["3.10"]

matrix = []
for test in get_test_list():
    for target in get_target_list():
        for runner in runners:
            for python_version in python_versions:
                matrix.append({
                    "display": f"source2source(main)_{test}({target})",
                    "test": test,
                    "target": target,
                    "runner": runner,
                    "python_version": python_version
                })

output = json.dumps(matrix)
print(output)

# Save the matrix to GitHub Actions output
with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
    print(f"matrix={output}", file=f)
EOF