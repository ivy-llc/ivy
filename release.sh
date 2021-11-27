#!/bin/bash
if [ -z "$1" ]; then
    echo "You need to provide an old version number"
    exit
fi
if [ -z "$2" ]; then
    echo "You need to provide a release version number"
    exit
fi

sed -i "s/$1/$2/g" setup.py
git add -A
git commit -m "version $2"
git push
#git tag -a "v$2" -m "version $2"
#git push origin "v$2"
