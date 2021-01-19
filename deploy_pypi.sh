python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/* -u "__token__" -p "$PYPI_PASSWORD" --verbose
