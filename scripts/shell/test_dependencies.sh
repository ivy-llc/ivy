#!/bin/bash

# shellcheck disable=SC2046
python3 ivy/test_dependencies.py -fp ivy/requirements.txt,ivy/optional.txt
