#!/bin/bash

docker build -t unifyai/ivy:latest --no-cache -f docker/Dockerfile .
docker build -t unifyai/ivy:latest-gpu --no-cache docker/DockerfileGPU .
docker build -t unifyai/ivy:latest-copsim --no-cache docker/DockerfileCopsim .
