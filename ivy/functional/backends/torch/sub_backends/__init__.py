import importlib
import os


_available_sub_backends = []
_current_sub_backends = []


def current_sub_backends():
    return _current_sub_backends


def available_sub_backends():
    return _available_sub_backends


sub_backends_loc = __file__.rpartition(os.path.sep)[0]

for sub_backend in os.listdir(sub_backends_loc):
    if sub_backend.startswith("__") or not os.path.isdir(
        os.path.join(sub_backends_loc, sub_backend)
    ):
        continue

    elif importlib.util.find_spec(sub_backend):
        _available_sub_backends.append(sub_backend)
