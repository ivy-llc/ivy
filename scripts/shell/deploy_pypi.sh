python -m pip install .
python -c "import ivy; ivy.utils.cleanup_and_fetch_binaries()"
jq -c '.compiler[]' available_configs.json | while read config; do
    export TAG=cp311-cp311-win_amd64
    export CLEAN=true
    python -m build
    python3 scripts/rename_wheels.py
done
# python3 -m twine upload dist/* -u "__token__" -p "$PYPI_PASSWORD" --verbose
