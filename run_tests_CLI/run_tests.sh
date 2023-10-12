#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy unifyai/ivy:latest python3 -m pytest ivy_tests/
