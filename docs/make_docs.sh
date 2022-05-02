#!/bin/bash -e
IFS='/' read -r -a working_directory <<< "$PWD"
user="${working_directory[2]}"
docker run --rm -v "$(pwd)"/..:/home/"$user"/project unifyai/doc-builder:latest "../ivy"