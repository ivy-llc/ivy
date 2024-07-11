#!/bin/bash

docker build -t transpileai/ivy:latest --no-cache -f Dockerfile ..
docker build -t transpileai/ivy:latest-gpu --no-cache -f DockerfileGPU ..
