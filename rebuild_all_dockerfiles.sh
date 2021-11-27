#!/bin/bash

docker build -t ivydl/ivy:latest --no-cache -f Dockerfile .
docker build -t ivydl/ivy:latest-gpu --no-cache DockerfileGPU .
docker build -t ivydl/ivy:latest-copsim --no-cache DockerfileCopsim .
