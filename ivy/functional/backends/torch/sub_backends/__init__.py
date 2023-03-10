import importlib

import ivy
from ivy.utils.backend.sub_backend_handler import create_enable_function

sub_backends_attrs = []
available_sub_backends = []
original_dict = {}
backend = "torch"


if importlib.util.find_spec("xformers"):
    available_sub_backends.append('xformers')
  