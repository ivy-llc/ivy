#!/bin/bash

# Check for macOS ARM architecture
if [[ $(uname -m) != 'arm64' ]]; then
    echo "This script is intended only for macOS ARM."
    exit 1
fi

# Install Homebrew if not already installed (macOS package manager)
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install pandoc using Homebrew
echo "Installing pandoc..."
brew install pandoc

# Check if ivy_dev exists and remove it if it does
if [ -d "ivy_dev" ]; then
    echo "Removing existing ivy_dev directory..."
    rm -rf ivy_dev
fi

# Create a Python virtual environment
echo "Creating a Python virtual environment..."
python3.10 -m venv ivy_dev
source ivy_dev/bin/activate.fish

# Install Python dependencies from requirements.txt
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements/requirements.txt

# Install Python dependencies for Apple Silicon
echo "Installing Python dependencies for Apple Silicon..."
pip install -r requirements/optional_apple_silicon_1.txt
pip install -r requirements/optional_apple_silicon_2.txt

# Handle requirement mappings similar to the Dockerfile approach
# Ensure jq is installed for processing JSON
if ! command -v jq &> /dev/null; then
    echo "Installing jq..."
    brew install jq
fi

# Set the target directory for framework-specific dependencies inside the virtual environment
FW_DIR="ivy_dev/fw"
mkdir -p $FW_DIR

# Copy the requirement_mappings_apple_silicon.json file from the Docker context
cp docker/requirement_mappings_apple_silicon.json .

# Install requirements based on mappings
echo "Installing requirements based on mappings..."
while IFS= read -r line; do
    fw_dir=$(echo "$line" | jq -r '.key')
    packages=$(echo "$line" | jq -r '.value[]')

    for package in $packages; do
        if [ -n "$package" ]; then
            echo "Installing $package in $fw_dir"
            pip install --ignore-installed --target "ivy_dev/fw/$fw_dir" "$package"
        fi
    done
done < <(jq -c 'to_entries[] | select(.value != [])' requirement_mappings_apple_silicon.json)

# Add the directories to PYTHONPATH
export PYTHONPATH="$FW_DIR/mxnet:$FW_DIR/numpy:$FW_DIR/tensorflow:$FW_DIR/jax:$FW_DIR/torch:$PYTHONPATH"

# Clean up
rm requirement_mappings_apple_silicon.json

echo "Dependency installation completed."
