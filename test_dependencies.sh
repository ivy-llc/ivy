#!/bin/bash

# shellcheck disable=SC2046
docker run --rm -v $(pwd):/ivy ivydl/ivy:latest python3 test_dependencies.py -fp ivy/requirements.txt,ivy/optional.txt