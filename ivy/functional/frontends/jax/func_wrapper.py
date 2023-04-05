# global
import functools
from typing import Callable
import inspect

# local
import ivy
import ivy.functional.frontends.jax as jax_frontend
import ivy.functional.frontends.numpy as np_frontend


def _from_jax_frontend_array_to_ivy_array(x):
    if (
        isinstance(x, jax_frontend.DeviceArray)
        and x.weak_type
        and x.ivy_array.shape == ()
    ):
        return ivy.to_scalar(x.ivy_array)
    if hasattr(x, "ivy_array"):
        return x.ivy_array
    return x


def _from_ivy_array_to_jax_frontend_array(x, nested=False, include_derived=None):
    if nested:
        return ivy.nested_map(
            x, _from_ivy_array_to_jax_frontend_array, include_derived, shallow=False
        )
    elif isinstance(x, ivy.Array):
        return jax_frontend.DeviceArray(x)
    return x


def _from_ivy_array_to_jax_frontend_array_weak_type(
    x, nested=False, include_derived=None
):
    if nested:
        return ivy.nested_map(
            x,
            _from_ivy_array_to_jax_frontend_array_weak_type,
            include_derived,
            shallow=False,
        )
    elif isinstance(x, ivy.Array):
        return jax_frontend.DeviceArray(x, weak_type=True)
    return x


def _native_to_ivy_array(x):
    if isinstance(x, ivy.NativeArray):
        return ivy.array(x)
    return x


def _to_ivy_array(x):
    return _from_jax_frontend_array_to_ivy_array(_native_to_ivy_array(x))


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
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

    return new_fn


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        weak_type = not any(
            (isinstance(arg, jax_frontend.DeviceArray) and arg.weak_type is False)
            or ivy.is_array(arg)
            or isinstance(arg, (tuple, list))
            for arg in args
        )
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            weak_type = False
        # call unmodified function
        # ToDo: Remove this default dtype setting
        #  once frontend specific backend setting is added
        if jax_frontend.config.jax_enable_x64:
            ivy.set_default_int_dtype("int64")
            ivy.set_default_float_dtype("float64")
            try:
                ret = fn(*args, **kwargs)
            finally:
                ivy.unset_default_int_dtype()
                ivy.unset_default_float_dtype()
        else:
            ret = fn(*args, **kwargs)
        # convert all arrays in the return to `jax_frontend.DeviceArray` instances
        if weak_type:
            return _from_ivy_array_to_jax_frontend_array_weak_type(
                ret,
                nested=True,
                include_derived={tuple: True},
            )
        return _from_ivy_array_to_jax_frontend_array(
            ret, nested=True, include_derived={tuple: True}
        )

    return new_fn


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))


def handle_jax_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, dtype=None, **kwargs):
        if len(args) > (dtype_pos + 1):
            dtype = args[dtype_pos]
            kwargs = {
                **dict(
                    zip(
                        list(inspect.signature(fn).parameters.keys())[
                            dtype_pos + 1 : len(args)
                        ],
                        args[dtype_pos + 1 :],
                    )
                ),
                **kwargs,
            }
            args = args[:dtype_pos]
        elif len(args) == (dtype_pos + 1):
            dtype = args[dtype_pos]
            args = args[:-1]

        if not dtype:
            return fn(*args, dtype=dtype, **kwargs)

        dtype = np_frontend.to_ivy_dtype(dtype)
        if not jax_frontend.config.jax_enable_x64:
            dtype = (
                jax_frontend.numpy.dtype_replacement_dict[dtype]
                if dtype in jax_frontend.numpy.dtype_replacement_dict
                else dtype
            )

        return fn(*args, dtype=dtype, **kwargs)

    dtype_pos = list(inspect.signature(fn).parameters).index("dtype")
    return new_fn


def outputs_to_native_arrays(fn: Callable):
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if isinstance(ret, jax_frontend.DeviceArray):
            ret = ret.ivy_array.data
        return ret

    return new_fn
