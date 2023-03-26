import functools
<<<<<<< HEAD
from typing import Callable

import ivy
import ivy.functional.frontends.mxnet as mxnet_frontend
=======
import inspect
from typing import Callable

import ivy
from ivy.functional.frontends.mxnet.numpy.ndarray import ndarray
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69


def _ivy_array_to_mxnet(x):
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
<<<<<<< HEAD
        return mxnet_frontend.numpy.array(x)
=======
        return ndarray(x)
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    return x


def _mxnet_frontend_array_to_ivy(x):
    if hasattr(x, "ivy_array"):
        return x.ivy_array
    return x


def _native_to_ivy_array(x):
<<<<<<< HEAD
    # TODO: replace `ndarray.NDArray` with `ivy.NativeArray``
    # and `x.asnumpy()` with `x` once the mxnet tensor class
    # is implemented
=======
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
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
<<<<<<< HEAD
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
=======
        # convert all arrays in the inputs to ivy.Array instances
        new_args = ivy.nested_map(
            args, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        new_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        return fn(*new_args, **new_kwargs)
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69

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
        return ivy.nested_map(ret, _ivy_array_to_mxnet, include_derived={tuple: True})

    new_fn.outputs_to_frontend_arrays = True
    return new_fn


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """
    Wraps `fn` so that input arrays are all converted to `ivy.Array` instances
    and return arrays are all converted to `ndarray.NDArray` instances.
    """
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))
<<<<<<< HEAD
=======


def handle_mxnet_out(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, out=None, **kwargs):
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
    new_fn.handle_numpy_out = True
    return new_fn
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
