#!/bin/bash

docker build -t unifyai/ivy:latest --no-cache -f Dockerfile ..
docker build -t unifyai/ivy:latest-gpu --no-cache -f DockerfileGPU ..
