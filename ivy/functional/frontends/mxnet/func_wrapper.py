import functools
import inspect
from typing import Callable

import ivy
from ivy.functional.frontends.mxnet.numpy.ndarray import ndarray


def _ivy_array_to_mxnet(x):
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
        return ndarray(x)
    return x


def _mxnet_frontend_array_to_ivy(x):
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
    def _inputs_to_ivy_arrays_mxnet(*args, **kwargs):
        """
        Convert `ndarray.NDArray` into `ivy.Array` instances.

        Convert all `ndarray.NDArray` instances in both the positional
        and keyword arguments into `ivy.Array` instances, and then calls
        the function with the updated arguments.
        """
        # convert all arrays in the inputs to ivy.Array instances
        new_args = ivy.nested_map(
            args, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        new_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        return fn(*new_args, **new_kwargs)

    _inputs_to_ivy_arrays_mxnet.inputs_to_ivy_arrays = True
    return _inputs_to_ivy_arrays_mxnet


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_arrays_mxnet(*args, **kwargs):
        """
        Convert `ivy.Array` into `ndarray.NDArray` instances.

        Call the function, and then converts all `ivy.Array` instances
        in the function return into `ndarray.NDArray` instances.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)

        # convert all arrays in the return to `frontend.Tensorflow.tensor` instances
        return ivy.nested_map(ret, _ivy_array_to_mxnet, include_derived={tuple: True})

    _outputs_to_frontend_arrays_mxnet.outputs_to_frontend_arrays = True
    return _outputs_to_frontend_arrays_mxnet


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """
    Wrap `fn` so it receives and returns `ivy.Array` instances.

    Wrap `fn` so that input arrays are all converted to `ivy.Array`
    instances and return arrays are all converted to `ndarray.NDArray`
    instances.
    """
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))


def handle_mxnet_out(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_mxnet_out(*args, out=None, **kwargs):
        if len(args) > (out_pos + 1):
            out = args[out_pos]
            kwargs = {
                **dict(
                    zip(
                        list(inspect.signature(fn).parameters.keys())[
                            out_pos + 1 : len(args)
                        ],
                        args[out_pos + 1 :],
                    )
                ),
                **kwargs,
            }
            args = args[:out_pos]
        elif len(args) == (out_pos + 1):
            out = args[out_pos]
            args = args[:-1]
        if ivy.exists(out):
            if not isinstance(out, ndarray):
                raise ivy.utils.exceptions.IvyException(
                    "Out argument must be an ivy.frontends.mxnet.numpy.ndarray object"
                )
            return fn(*args, out=out.ivy_array, **kwargs)
        return fn(*args, **kwargs)

    out_pos = list(inspect.signature(fn).parameters).index("out")
    _handle_mxnet_out.handle_numpy_out = True
    return _handle_mxnet_out
