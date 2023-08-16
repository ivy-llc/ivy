import functools
from typing import Callable, Iterable
from importlib import import_module

import ivy


def _to_ivy_array(x):
    if hasattr(x, "ivy_array"):
        return x.ivy_array

    elif isinstance(x, (ivy.NativeArray, list)):
        return ivy.array(x)

    return x


def _infer_return_array(x: Iterable) -> Callable:
    module_path = x.__class__.__module__
    frontend_array_name = x.__class__.__name__

    if frontend_array_name in ["int", "float", "bool", "complex"]:
        frontend_path = "ivy.functional.frontends.builtins"
        frontend_array_name = "list"

    elif "ivy" not in module_path:
        module_str = module_path.split(".")[0]
        module_str = "jax" if module_str == "jaxlib" else module_str
        frontend_path = "ivy.functional.frontends." + module_str
        frontend_array_name = "Array" if module_str == "jax" else frontend_array_name

    else:
        frontend_path = module_path

    module = import_module(frontend_path)
    frontend_array = getattr(module, frontend_array_name)

    if "numpy" in frontend_path:
        return functools.partial(frontend_array, _init_overload=True)
    return frontend_array


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_arrays_builtins(*args, **kwargs):
        ivy_args = ivy.nested_map(args, _to_ivy_array, shallow=False, to_ignore=list)
        ivy_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, shallow=False, to_ignore=list
        )

        frontend_array = _infer_return_array(args[0])

        return fn(*ivy_args, **ivy_kwargs), frontend_array

    return _inputs_to_ivy_arrays_builtins


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_arrays(*args, **kwargs):
        ret, frontend_array = fn(*args, **kwargs)

        if isinstance(ret, ivy.Array) or ivy.is_native_array(ret):
            return frontend_array(ret)
        return ret

    return _outputs_to_frontend_arrays


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))


def from_zero_dim_arrays_to_scalar(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _from_zero_dim_arrays_to_scalar(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if len(ret) > 1:
            return ret
        return ivy.to_scalar(ret)

    return _from_zero_dim_arrays_to_scalar
