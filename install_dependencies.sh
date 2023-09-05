pip install -r requirements/requirements.txt
if [[ $(arch) == 'arm64' ]]; then
      pip install -r requirements/optional_apple_silicon_1.txt
      pip install -r requirements/optional_apple_silicon_2.txt
else
    pip install -r requirements/optional.txt
fi