#!/bin/bash

docker build -t unifyai/ivy:latest --no-cache -f Dockerfile ..
docker build -t unifyai/multicuda:base_and_requirements --no-cache -f DockerfileGPUMultiCuda ..
