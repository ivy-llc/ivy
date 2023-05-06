import os
from ivy.utils.backend.sub_backend_handler import find_available_sub_backends

sub_backends_loc = __file__.rpartition(os.path.sep)[0]


def current_sub_backends():
    raise NotImplementedError("mxnet.current_sub_backends Not Implemented")


def available_sub_backends():
    raise NotImplementedError("mxnet.available_sub_backends Not Implemented")
