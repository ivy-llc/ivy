import functools
from typing import Callable

import ivy
import ivy.functional.frontends.numpy as np_frontend


def _from_ivy_array_to_scipy_frontend_array(x, nested=False, include_derived=None):
    if nested:
        return ivy.nested_map(
            x, _from_ivy_array_to_scipy_frontend_array, include_derived, shallow=False
        )
    elif isinstance(x, ivy.Array):
        return np_frontend.array(x)
    return x


def _from_scipy_frontend_array_to_ivy_array(x):
    if (
        # add other types of frontend arrays
        isinstance(x, np_frontend.ndarray)
    ):
        return ivy.to_scalar(x.ivy_array)
    if hasattr(x, "ivy_array"):
        return x.ivy_array
    return x


def _native_to_ivy_array(x):
    if isinstance(x, ivy.NativeArray):
        return ivy.array(x)
    return x


def _to_ivy_array(x):
    return _from_scipy_frontend_array_to_ivy_array(_native_to_ivy_array(x))


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_arrays_scipy(*args, **kwargs):
        """Converts all `ndarray.NDArray` instances in both the positional and keyword
        arguments into `ivy.Array` instances, and then calls the function with the
        updated arguments."""
        # check if kwargs contains an out argument, and if so, remove it
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True
        # convert all arrays in the inputs to ivy.Array instances
        new_args = ivy.nested_map(
            args, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        new_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        # add the original out argument back to the keyword arguments
        if has_out:
            new_kwargs["out"] = out
        return fn(*new_args, **new_kwargs)

    _inputs_to_ivy_arrays_scipy.inputs_to_ivy_arrays = True
    return _inputs_to_ivy_arrays_scipy


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_arrays_scipy(*args, **kwargs):
        """Calls the function, and then converts all `ivy.Array` instances in the
        function return into `ndarray.NDArray` instances."""
        # call unmodified function
        ret = fn(*args, **kwargs)

        # convert all arrays in the return to `frontend.Tensorflow.tensor` instances
        return _from_ivy_array_to_scipy_frontend_array(
            ret, nested=True, include_derived={tuple: True}
        )

    _outputs_to_frontend_arrays_scipy.outputs_to_frontend_arrays = True
    return _outputs_to_frontend_arrays_scipy


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """Wraps `fn` so that input arrays are all converted to `ivy.Array` instances and
    return arrays are all converted to `ndarray.NDArray` instances."""
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))


def outputs_to_native_arrays(fn: Callable):
    @functools.wraps(fn)
    def _outputs_to_native_arrays(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if isinstance(ret, np_frontend.ndarray):
            ret = ret.ivy_array.data
        return ret

    return _outputs_to_native_arrays
