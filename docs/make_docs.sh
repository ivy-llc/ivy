#!/bin/bash -e
<<<<<<< HEAD
if [ -z "$1" ]
  then
    IFS='/' read -r -a working_directory <<< "$PWD"
    user="${working_directory[2]}"
  else
    user="$1"
fi
docker run --rm -v "$(pwd)"/..:/home/"$user"/project unifyai/doc-builder:latest "../ivy"
=======
docker run --rm -v "$(pwd)"/..:/project unifyai/doc-builder:latest
>>>>>>> a3fa5ae9c4567371f82de20b15479e535a867ead
