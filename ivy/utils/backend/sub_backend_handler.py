import importlib
from types import ModuleType, FunctionType

import ivy
from ivy.func_wrapper import _wrap_function

_sub_backend_dict = dict()
# torch sub_backends
_sub_backend_dict["xformers"] = "ivy.functional.backends.torch.sub_backends.xformers"


# Recursively updates the modules
def update_functions(sub_backend: ModuleType, target: ModuleType):
    for k, v in sub_backend.__dict__.items():
        if k in target.__dict__ and not k.startswith("__"):
            if isinstance(v, FunctionType):
                target.__dict__[k] = _wrap_function(
            key=k, to_wrap=sub_backend.__dict__[k], original=v, compositional=False
        )

            elif isinstance(v, ModuleType) and "ivy.functional." in v.__name__:
                update_functions(sub_backend.__dict__[k], target.__dict__[k])


def set_sub_backend(sub_backend: ModuleType, current_ivy):
    update_functions(sub_backend, current_ivy)


def unset_sub_backend(sub_backend: ModuleType, current_ivy):
    original_dict = current_ivy.current_backend().sub_backends.original_dict
    for k, v in sub_backend.__dict__.items():
        if k in current_ivy.__dict__ and not k.startswith('__'):
            if isinstance(current_ivy.__dict__[k], ModuleType) and "ivy.functional." in v.__name__:
                current_ivy.__dict__[k].__dict__.update(original_dict[k].__dict__)
            elif isinstance(current_ivy.__dict__[k], FunctionType):
                current_ivy.__dict__[k] = original_dict[k]


def create_enable_function(sub_backend_name: str, backend_name: str, current_ivy):
 
    def enable_function(enable: bool=False):
        # TODO(Yasser): make sure to unset any sub_backend
        # that the current sub_backend is incompatible with (functions overlapping)
        sub_backend = importlib.import_module(_sub_backend_dict[sub_backend_name])
        original_dict = current_ivy.current_backend().sub_backends.original_dict
        if not original_dict:
            # TODO: Find a better way to copy the original backend dict, calling with_backend is probably not so efficient
            current_ivy.current_backend().sub_backends.original_dict = current_ivy.with_backend(backend_name).__dict__
        if enable:
            set_sub_backend(sub_backend, current_ivy)
            current_ivy.__dict__[f"is_{sub_backend_name}_enabled"] = True
        if not enable:
            if original_dict:
                unset_sub_backend(sub_backend, current_ivy)
                current_ivy.__dict__[f"is_{sub_backend_name}_enabled"] = False

    enable_function.__doc__ = current_ivy.current_backend().sub_backends.docstrings[sub_backend_name]
    return enable_function


def add_sub_backend_attributes(current_ivy):
    backend = current_ivy.current_backend_str()
    if backend:
        for sub_backend in ivy.current_backend().available_sub_backends:
            current_ivy.__dict__[f'enable_{sub_backend}'] = create_enable_function(sub_backend, backend, current_ivy)
            current_ivy.__dict__[f'is_{sub_backend}_enabled'] = False

            current_ivy.current_backend().sub_backends_attrs.append(f'enable_{sub_backend}')
            current_ivy.current_backend().sub_backends_attrs.append(f'is_{sub_backend}_enabled')



def remove_sub_backend_attributes(current_ivy):
    # if backend agnostic ivy is set (i.e no backend is set), this will return an empty string
    backend = current_ivy.current_backend_str()
    if backend:
        for k in current_ivy.current_backend().sub_backends_attrs:
            del current_ivy.__dict__[k]
        current_ivy.current_backend().sub_backends_attrs = []
    
