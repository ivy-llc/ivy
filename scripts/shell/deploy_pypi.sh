python -m build
python3 scripts/rename_wheels.py
python3 -m twine upload dist/* -u "__token__" -p "$PYPI_PASSWORD" --verbose
