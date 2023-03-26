import os
import importlib
from types import ModuleType, FunctionType

import ivy
from ivy.func_wrapper import _wrap_function
from ivy.utils.exceptions import IvyException

_sub_backend_dict = dict()
_backend_to_sub_backends_dict = dict()
# torch sub_backends

_sub_backend_dict["xformers"] = "ivy.functional.backends.torch.sub_backends.xformers"


_backend_to_sub_backends_dict["torch"] = ["xformers"]


_all_sub_backends = _backend_to_sub_backends_dict["torch"]


original_backend_dict = None


def set_sub_backend(sub_backend_str: str):
    if ivy.backend == "none":
        print("You must set a backend first:")
        available_sub_backends()
        return

    if ivy.current_backend_str() not in _backend_to_sub_backends_dict.keys():
        print(
            f"backend {ivy.current_backend_str()} does not have any supported sub_backends"
        )
        return

    if sub_backend_str not in _all_sub_backends:
        raise IvyException(
            f"sub_backend must be one from {_backend_to_sub_backends_dict[ivy.current_backend_str()]}"
        )

    if sub_backend_str not in _backend_to_sub_backends_dict[ivy.current_backend_str()]:
        print(
            f"{ivy.current_backend_str()} does not support {sub_backend_str} as a sub_backend"
        )
        return

    if sub_backend_str in ivy.current_sub_backends():
        return

    global original_backend_dict
    if original_backend_dict is None:
        original_backend_dict = ivy.__dict__.copy()
    sub_backend = importlib.import_module(_sub_backend_dict[sub_backend_str])
    _set_sub_backend_as_ivy(ivy.__dict__.copy(), ivy, sub_backend)
    ivy.current_backend().sub_backends._current_sub_backends.append(sub_backend_str)


def _set_sub_backend_as_ivy(
    original: dict, target: ModuleType, sub_backend: ModuleType
):
    backend_str = ivy.current_backend_str()
    for k, v in original.items():
        if k not in sub_backend.__dict__ and not k.startswith("__"):
            target.__dict__[k] = v
        if (
            k in sub_backend.__dict__
            and not k.startswith("__")
            and isinstance(v, FunctionType)
        ):
            target.__dict__[k] = _wrap_function(
                key=k, to_wrap=sub_backend.__dict__[k], original=v, compositional=False
            )
        elif (
            k in sub_backend.__dict__
            and not k.startswith("__")
            and isinstance(v, ModuleType)
        ):
            mod = ModuleType(k)
            mod.__name__ = v.__name__
            mod.__file__ = v.__file__
            target.__dict__[k] = mod
        if (
            isinstance(v, ModuleType)
            and "ivy.functional." in v.__name__
            and os.path.join("{}", "__init__.py").format(backend_str) not in v.__file__
            and k in sub_backend.__dict__
        ):
            _set_sub_backend_as_ivy(
                v.__dict__,
                target.__dict__[k],
                sub_backend.__dict__[k],
            )


def unset_sub_backend(sub_backend_str: str):
    if sub_backend_str not in ivy.current_sub_backends():
        return
    global original_backend_dict
    sub_backend = importlib.import_module(
        _sub_backend_dict[sub_backend_str]
    )  # The sub-backend is cached so this is fast
    _unset_sub_backend_from_ivy(
        original_backend_dict, ivy, sub_backend, sub_backend.name
    )
    ivy.current_backend().sub_backends._current_sub_backends.remove(sub_backend_str)


def _unset_sub_backend_from_ivy(
    original: dict, target: ModuleType, sub_backend: ModuleType, sub_backend_str: str
):
    backend_str = ivy.current_backend_str()
    for k, v in sub_backend.__dict__.items():
        if k in target.__dict__:
            if (
                isinstance(v, FunctionType)
                and sub_backend_str in f"sub_backends.{sub_backend_str}" in v.__module__
            ):
                target.__dict__[k] = original[k]
            if (
                isinstance(v, ModuleType)
                and "ivy.functional." in v.__name__
                and os.path.join("{}", "__init__.py").format(backend_str)
                not in v.__file__
            ):
                _unset_sub_backend_from_ivy(
                    original[k].__dict__,
                    target.__dict__[k],
                    sub_backend.__dict__[k],
                    sub_backend_str,
                )


def clear_sub_backends():
    if ivy.current_sub_backends():
        ivy.__dict__.update(original_backend_dict)
        ivy.current_backend().sub_backends._current_sub_backends = []


# This is used in set_backend only
def _clear_current_sub_backends():
    global original_backend_dict
    original_backend_dict = None
    if ivy.current_sub_backends():
        ivy.current_backend().sub_backends._current_sub_backends = []


def available_sub_backends():
    for k, v in _backend_to_sub_backends_dict.items():
        print(f"backend: {k} supports sub_backends: {v}")
