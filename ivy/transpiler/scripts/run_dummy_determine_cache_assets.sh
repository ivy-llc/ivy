#!/bin/bash

# Emulate GitHub Actions environment variables
export GITHUB_OUTPUT="/tmp/github_output.txt"  # Temp file to simulate GitHub Actions output file
export TORCH_FRONTEND_FUNCS_BATCH_SIZE=10
export TORCH_LAYERS_BATCH_SIZE=5
export IVY_FUNCS_BATCH_SIZE=15

# Ensure the GITHUB_OUTPUT file is empty at the start
echo -n "" > $GITHUB_OUTPUT

# Run the determine_cache_assets.sh script
./source_to_source_translator/scripts/determine_cache_assets.sh

# Print the contents of the GITHUB_OUTPUT file to the console
echo "GITHUB_OUTPUT content:"
cat $GITHUB_OUTPUT
