import functools
from typing import Callable, Iterable
from importlib import import_module

import ivy


def _to_ivy_array(x):
    # if x is any frontend array including frontend list
    # then extract the wrapped ivy array
    if hasattr(x, "ivy_array"):
        return x.ivy_array

    # convert native arrays and lists to ivy arrays
    elif isinstance(x, ivy.NativeArray):
        return ivy.array(x)

    return x


def _infer_return_array(x: Iterable) -> Callable:
    # get module path
    module_path = x.__class__.__module__.split(".")

    # if function's input is a scalar, list, or tuple
    # convert to current backend's frontend array
    if "builtins" in module_path:
        cur_backend = ivy.current_backend_str()
        module_str = cur_backend if len(cur_backend) != 0 else "numpy"

    # in this case we're dealing with frontend array
    # module's name is always at index 3 on the path
    elif "ivy" in module_path:
        module_str = module_path[3]

    # native array, e.g. np.ndarray, torch.Tensor etc.
    else:
        module_str = module_path[0]
        # replace jaxlib with jax to construct a valid path
        module_str = "jax" if module_str == "jaxlib" else module_str

    # import the module and get a corresponding frontend array
    frontend_path = "ivy.functional.frontends." + module_str
    module = import_module(frontend_path)
    frontend_array = getattr(module, "_frontend_array")

    return frontend_array


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_arrays_builtins(*args, **kwargs):
        ivy_args = ivy.nested_map(args, _to_ivy_array, shallow=False)
        ivy_kwargs = ivy.nested_map(kwargs, _to_ivy_array, shallow=False)

        # array is the first argument given to a function
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
