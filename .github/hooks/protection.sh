#!/bin/bash

zero_commit="0000000000000000000000000000000000000000"

# Ensure we are in a transaction, rather than doing a clone or fetch.
if [ "$1" = "$zero_commit" ]; then
    exit 0
fi

# Choose the author
required_author="vedpatwardhan"

for commit in $(git rev-list $1..$2)
do
    author=$(git show -s --format='%an' $commit)
    if [ "$author" != "$required_author" ]; then
        echo "ERROR: All commits must be authored by $required_author. Commit $commit was authored by $author."
        exit 1
    fi
done

exit 0
