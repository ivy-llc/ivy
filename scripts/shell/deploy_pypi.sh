jq -c '.compiler[]' available_configs.json | while read config; do
    export TAG=${config:1:${#config}-2}
    export CLEAN=true
    python -m build
    python3 scripts/rename_wheels.py
done
python3 -m twine upload dist/* -u "__token__" -p "$PYPI_PASSWORD" --verbose
