# NOQA
from importlib import import_module as builtin_import

import ivy


def import_module(name, package=None):
    if ivy.is_local():
        with ivy.utils._importlib.LocalIvyImporter():
            return ivy.utils._importlib._import_module(name=name, package=package)
    return builtin_import(name=name, package=package)
