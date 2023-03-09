import importlib

import ivy
from ivy.utils.backend.sub_backend_handler import create_enable_function

sub_backends_attrs = []
original_dict = {}
backend = "torch"



if importlib.util.find_spec("xformers"):
    print("found xformers installation")

    ivy.__dict__['enable_xformers'] = create_enable_function('xformers', backend)
    ivy.__dict__['is_xformers_enabled'] = False

    sub_backends_attrs.append('enable_xformers')
    sub_backends_attrs.append('is_xformers_enabled')