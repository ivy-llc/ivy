pip install torch
ver=$(pip show torch | grep Version | cut -d ' ' -f2)
pip install torch-scatter -f https://data.pyg.org/whl/torch-$ver+cpu.html
pip install -r requirements/requirements.txt
if [[ $(arch) == 'arm64' ]]; then
      pip install -r requirements/optional_m1_1.txt
      pip install -r requirements/optional_m1_2.txt
else
    pip install -r requirements/optional.txt
fi