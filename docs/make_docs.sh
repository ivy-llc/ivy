#!/bin/bash -e
if [ -z "$1" ]
  then
    IFS='/' read -r -a working_directory <<< "$PWD"
    user="${working_directory[2]}"
  else
    user="$1"
fi
docker run --rm -v "$(pwd)"/..:/home/"$user"/project unifyai/doc-builder:latest "../ivy"