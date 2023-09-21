import os
from ivy.utils.backend.sub_backend_handler import find_available_sub_backends


sub_backends_loc = __file__.rpartition(os.path.sep)[0]

_available_sub_backends = find_available_sub_backends(sub_backends_loc)
_current_sub_backends = []


def current_sub_backends():
    return _current_sub_backends


def available_sub_backends():
    return _available_sub_backends
