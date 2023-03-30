#!/bin/bash

# Set git pull.rebase mode to true
git config pull.rebase true

# Stash any changes
if git diff-index --quiet HEAD --; then
    echo "No changes to stash"
else
    git stash save
fi

# Pull the latest changes from the remote repository
git pull

# Apply any stashed changes
if git stash list | grep -q "stash"; then
    git stash pop
fi
