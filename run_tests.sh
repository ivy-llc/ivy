#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy unifyai/ivy:latest python3 -m pytest ivy/ivy_tests/test_ivy/test_functional/test_core/test_statistical.py::test_einsum
