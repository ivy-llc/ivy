import importlib
from types import ModuleType, FunctionType

import ivy
from ivy.utils.verbosity import hide_prints

_sub_backend_dict = dict()
# torch sub_backends
_sub_backend_dict["xformers"] = "ivy.functional.backends.torch.sub_backends.xformers"




# TODO(Yasser): Wrap the functions after/before updating the dict
def set_sub_backend(sub_backend: ModuleType):
    for k, v in sub_backend.__dict__.items():
        if k in ivy.__dict__ and '__' not in k:
            if isinstance(ivy.__dict__[k], ModuleType):
                ivy.__dict__[k].__dict__.update(v.__dict__)
            elif isinstance(ivy.__dict__[k], FunctionType):
                ivy.__dict__[k] = v


def unset_sub_backend(sub_backend: ModuleType):
    original_dict = ivy.current_backend().sub_backends.original_dict
    for k, _ in sub_backend.__dict__.items():
        if k in ivy.__dict__ and '__' not in k:
            if isinstance(ivy.__dict__[k], ModuleType):
                ivy.__dict__[k].__dict__.update(original_dict[k].__dict__)
            elif isinstance(ivy.__dict__[k], FunctionType):
                ivy.__dict__[k] = original_dict[k]


def create_enable_function(sub_backend_name: str, backend_name: str):
    # TODO(Yasser): make sure to unset any sub_backend
    # that the current sub_backend is incompatible with (functions overlapping) 
    def enable_function(enable: bool=False):
        sub_backend = importlib.import_module(_sub_backend_dict[sub_backend_name])
        original_dict = ivy.current_backend().sub_backends.original_dict
        if not original_dict:
            with hide_prints():
                ivy.current_backend().sub_backends.original_dict = ivy.with_backend(backend_name).__dict__
        if enable:
            set_sub_backend(sub_backend)
            ivy.__dict__[f"is_{sub_backend_name}_enabled"] = True
        if not enable:
            if original_dict:
                unset_sub_backend(sub_backend)
                ivy.__dict__[f"is_{sub_backend_name}_enabled"] = False

    return enable_function