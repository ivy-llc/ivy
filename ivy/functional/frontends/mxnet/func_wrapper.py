import mxnet
import functools
from typing import Callable

import ivy
import ivy.functional.frontends.mxnet as mxnet_frontend


def _ivy_array_to_mxnet(x):

    # TODO: replace isinstance with ivy.is_native_array once
    # it's implemented in the mxnet backend
    if isinstance(x, ivy.Array) or isinstance(x, mxnet.ndarray.NDArray):
        return mxnet.ndarray.array(x)
    return x


def _mxnet_frontend_array_to_ivy(x):
    # TODO: Implement mxnet tensor class
    if hasattr(x, "ivy_array"):
        return x.ivy_array
    return x


def _native_to_ivy_array(x):
    if isinstance(x, ivy.NativeArray):
        return ivy.array(x)
    return x


def _to_ivy_array(x):
    return _mxnet_frontend_array_to_ivy(_native_to_ivy_array(x))


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Converts all `ndarray.NDArray` instances in both the positional and keyword
        arguments into `ivy.Array` instances, and then calls the function with the
        updated arguments.
        """
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True

        # convert all arrays in the inputs to ivy.Array instances
        ivy_args = ivy.nested_map(
            args, _to_ivy_array, include_derived=True, shallow=False
        )
        ivy_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, include_derived=True, shallow=False
        )
        if has_out:
            ivy_kwargs["out"] = out
        return fn(*ivy_args, **ivy_kwargs)

    new_fn.inputs_to_ivy_arrays = True
    return new_fn


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Calls the function, and then converts all `ivy.Array` instances in
        the function return into `ndarray.NDArray` instances.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)

        # convert all arrays in the return to `frontend.Tensorflow.tensor` instances
        return ivy.nested_map(
            ret, _ivy_array_to_mxnet, include_derived={tuple: True}
        )

    new_fn.outputs_to_frontend_arrays = True
    return new_fn


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """
    Wraps `fn` so that input arrays are all converted to `ivy.Array` instances
    and return arrays are all converted to `ndarray.NDArray` instances.
    """
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))
