# NOQA
import ivy
from importlib import import_module as builtin_import


def import_module(name, package=None):
    if ivy.is_local():
        return ivy.utils.backend._ivy_import_module(name=name, package=package)
    return builtin_import(name=name, package=package)
