#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/models unifyai/models:latest python3 -m pytest ivy_models_tests/
