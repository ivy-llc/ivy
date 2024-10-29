#!/bin/bash

docker build -t ivyllc/ivy:latest --no-cache -f Dockerfile ..
docker build -t ivyllc/ivy:latest-gpu --no-cache -f DockerfileGPU ..
