#!/bin/bash

docker build -t unifyai/ivy:latest --no-cache -f Dockerfile ..
docker build -t unifyai/ivy:latest-gpu --no-cache DockerfileGPU ..
docker build -t unifyai/ivy:latest-copsim --no-cache DockerfileCopsim ..
