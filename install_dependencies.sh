pip install -r requirements/requirements.txt
if [[ $(arch) == 'arm64' ]]; then
      pip install -r requirements/optional_m1_1.txt
      pip install -r requirements/optional_m1_2.txt
else
    pip install -r requirements/optional.txt
fi