#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy_models unifyai/ivy-models:latest python3 -m pytest ivy_models_tests/
