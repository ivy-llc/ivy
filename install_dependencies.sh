# This shell script is required by the doc-builder. Moving it might break
# the doc-building pipeline
pip install -e .
pip install -r requirements/requirements.txt
if [[ $(arch) == 'arm64' ]]; then
      brew install pandoc
      pip install -r requirements/optional_apple_silicon_1.txt
      pip install -r requirements/optional_apple_silicon_2.txt
else
    sudo apt-get update
    sudo apt-get install pandoc -y
    pip install -r requirements/optional.txt
fi
