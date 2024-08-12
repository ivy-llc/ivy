#!/bin/bash

# shellcheck disable=SC2046
docker run --rm -v "$(pwd)":/ivy ivyllc/ivy:latest python3 ivy/test_dependencies.py -fp ivy/requirements.txt,ivy/optional.txt
