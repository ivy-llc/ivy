python3 -m build
python3 -m twine upload dist/* -u "__token__" -p "$PYPI_PASSWORD" --verbose
