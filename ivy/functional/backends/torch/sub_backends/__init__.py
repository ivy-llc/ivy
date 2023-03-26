import importlib

import ivy

_available_sub_backends = []
_current_sub_backends = []


def current_sub_backends():
    return _current_sub_backends


def available_sub_backends():
    return _available_sub_backends


if importlib.util.find_spec("xformers"):
    _available_sub_backends.append("xformers")
