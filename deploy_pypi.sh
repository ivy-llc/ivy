python3 -m build
python3 -m twine upload dist/* --repository testpypi -u "__token__" -p "$PYPI_PASSWORD_TEST" --verbose
