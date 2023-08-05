import functools
from typing import Callable

import ivy
import ivy.functional.frontends.onnx as onnx_frontend


def _from_ivy_array_to_onnx_frontend_tensor(x, nested=False, include_derived=None):
    if nested:
        return ivy.nested_map(
            x, _from_ivy_array_to_onnx_frontend_tensor, include_derived, shallow=False
        )
    elif isinstance(x, ivy.Array) or ivy.is_native_array(x):
        a = onnx_frontend.Tensor(x)
        return a
    return x


def _ivy_array_to_onnx(x):
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
        return onnx_frontend.Tensor(x)
    return x


def _onnx_frontend_array_to_ivy(x):
    if hasattr(x, "ivy_array"):
        return x.ivy_array
    return x


def _native_to_ivy_array(x):
    if isinstance(x, ivy.NativeArray):
        return ivy.array(x)
    return x


def _to_ivy_array(x):
    return _onnx_frontend_array_to_ivy(_native_to_ivy_array(x))


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_arrays_onnx(*args, **kwargs):
        """
        Convert `Tensor` into `ivy.Array` instances.

        Convert all `Tensor` instances in both the positional and
        keyword arguments into `ivy.Array` instances, and then calls the
        function with the updated arguments.
        """
        # convert all arrays in the inputs to ivy.Array instances
        new_args = ivy.nested_map(
            args, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        new_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        return fn(*new_args, **new_kwargs)

    return _inputs_to_ivy_arrays_onnx


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_arrays_onnx(*args, **kwargs):
        """
        Convert `ivy.Array` into `Tensor` instances.

        Call the function, and then converts all `ivy.Array` instances
        returned by the function into `Tensor` instances.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)

        # convert all arrays in the return to `frontend.onnx.Tensor` instances
        return _from_ivy_array_to_onnx_frontend_tensor(
            ret, nested=True, include_derived={tuple: True}
        )

    return _outputs_to_frontend_arrays_onnx


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """
    Wrap `fn` so it receives and returns `ivy.Array` instances.

    Wrap `fn` so that input arrays are all converted to `ivy.Array`
    instances and return arrays are all converted to `ndarray.NDArray`
    instances.
    """
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))
