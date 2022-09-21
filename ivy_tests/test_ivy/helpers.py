"""Collection of helpers for ivy unit tests."""

# global
import importlib
import inspect
import pytest
import numpy as np
import math
import gc
from typing import Optional, Union, List
from hypothesis import given, settings
import hypothesis.extra.numpy as nph  # noqa
from hypothesis.internal.floats import float_of
from functools import reduce
from operator import mul

# local
from ivy.functional.backends.jax.general import is_native_array as is_jax_native_array
from ivy.functional.backends.numpy.general import (
    is_native_array as is_numpy_native_array,
)
from ivy.functional.backends.tensorflow.general import (
    is_native_array as is_tensorflow_native_array,
)
from ivy.functional.backends.torch.general import (
    is_native_array as is_torch_native_array,
)
from ivy_tests.test_ivy.test_frontends import NativeClass
from ivy_tests.test_ivy.test_frontends.test_torch import convtorch
from ivy_tests.test_ivy.test_frontends.test_numpy import convnumpy
from ivy_tests.test_ivy.test_frontends.test_tensorflow import convtensor
from ivy_tests.test_ivy.test_frontends.test_jax import convjax
import ivy.func_wrapper


TOLERANCE_DICT = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
cmd_line_args = (
    "as_variable",
    "native_array",
    "with_out",
    "container",
    "instance_method",
    "test_gradients",
)
frontend_fw = None

try:
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except ImportError:
    tf = None
from hypothesis import strategies as st

# local
import ivy
import ivy.functional.backends.numpy as ivy_np


def convtrue(argument):
    """Convert NativeClass in argument to true framework counter part"""
    if isinstance(argument, NativeClass):
        return argument._native_class
    return argument


def get_ivy_numpy():
    """Import Numpy module from ivy"""
    try:
        import ivy.functional.backends.numpy
    except ImportError:
        return None
    return ivy.functional.backends.numpy


def get_ivy_jax():
    """Import JAX module from ivy"""
    try:
        import ivy.functional.backends.jax
    except ImportError:
        return None
    return ivy.functional.backends.jax


def get_ivy_tensorflow():
    """Import Tensorflow module from ivy"""
    try:
        import ivy.functional.backends.tensorflow
    except ImportError:
        return None
    return ivy.functional.backends.tensorflow


def get_ivy_torch():
    """Import Torch module from ivy"""
    try:
        import ivy.functional.backends.torch
    except ImportError:
        return None
    return ivy.functional.backends.torch


def get_valid_numeric_dtypes():
    return ivy.valid_numeric_dtypes


_ivy_fws_dict = {
    "numpy": lambda: get_ivy_numpy(),
    "jax": lambda: get_ivy_jax(),
    "tensorflow": lambda: get_ivy_tensorflow(),
    "tensorflow_graph": lambda: get_ivy_tensorflow(),
    "torch": lambda: get_ivy_torch(),
}

_iterable_types = [list, tuple, dict]
_excluded = []


def _convert_vars(
    *, vars_in, from_type, to_type_callable=None, keep_other=True, to_type=None
):
    new_vars = list()
    for var in vars_in:
        if type(var) in _iterable_types:
            return_val = _convert_vars(
                vars_in=var, from_type=from_type, to_type_callable=to_type_callable
            )
            new_vars.append(return_val)
        elif isinstance(var, from_type):
            if isinstance(var, np.ndarray):
                if var.dtype == np.float64:
                    var = var.astype(np.float32)
                if bool(sum([stride < 0 for stride in var.strides])):
                    var = var.copy()
            if to_type_callable:
                new_vars.append(to_type_callable(var))
            else:
                raise Exception("Invalid. A conversion callable is required.")
        elif to_type is not None and isinstance(var, to_type):
            new_vars.append(var)
        elif keep_other:
            new_vars.append(var)

    return new_vars


def var_fn(x, *, dtype=None, device=None):
    """Returns x as an Ivy Variable wrapping an Ivy Array with given dtype and device"""
    return ivy.variable(ivy.array(x, dtype=dtype, device=device))


def get_current_frontend():
    """Returns the current frontend framework, returns None if no frontend is set."""
    return frontend_fw()


@st.composite
def get_dtypes(draw, kind, index=0, full=True, none=False, key=None):
    """
    Draws a valid dtypes for the test function. For frontend tests,
    it draws the data types from the intersection between backend
    framework data types and frontend framework dtypes, otherwise,
    draws it from backend framework data types.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    kind
        Supported types are integer, float, valid, numeric, and unsigned
    index
        list indexing incase a test needs to be skipped for a particular dtype(s)
    full
        returns the complete list of valid types
    none
        allow none in the list of valid types

    Returns
    -------
    ret
        dtype string
    """

    def _get_type_dict(framework):
        return {
            "valid": framework.valid_dtypes,
            "numeric": framework.valid_numeric_dtypes,
            "float": framework.valid_float_dtypes,
            "integer": framework.valid_int_dtypes,
            "unsigned": framework.valid_uint_dtypes,
            "signed_integer": tuple(
                set(framework.valid_int_dtypes).difference(framework.valid_uint_dtypes)
            ),
        }

    backend_dtypes = _get_type_dict(ivy)[kind]
    if frontend_fw:
        fw_dtypes = _get_type_dict(frontend_fw())[kind]
        valid_dtypes = tuple(set(fw_dtypes).intersection(backend_dtypes))
    else:
        valid_dtypes = backend_dtypes

    if none:
        valid_dtypes += (None,)
    if full:
        return valid_dtypes[index:]
    if key is None:
        return draw(st.sampled_from(valid_dtypes[index:]))
    ret = draw(st.shared(st.sampled_from(valid_dtypes[index:]), key=key))
    return [ret]


@st.composite
def get_castable_dtype(draw, available_dtypes, dtype: str, x: Optional[list] = None):
    """
    Draws castable dtypes for the given dtype based on the current backend.

    Parameters
    ----------
    draw
        Special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    available_dtypes
        Castable data types are drawn from this list randomly.
    dtype
        Data type from which to cast.
    x
        Optional list of values to cast.

    Returns
    -------
    ret
        A tuple of inputs and castable dtype.
    """

    def cast_filter(d):
        if ivy.is_int_dtype(d):
            max_val = ivy.iinfo(d).max
        elif ivy.is_float_dtype(d):
            max_val = ivy.finfo(d).max
        else:
            max_val = 1
        if x is None:
            max_x = -1
        else:
            max_x = np.max(np.abs(np.asarray(x)))
        return max_x <= max_val and ivy.dtype_bits(d) >= ivy.dtype_bits(dtype)

    cast_dtype = draw(st.sampled_from(available_dtypes).filter(cast_filter))
    if x is None:
        return dtype, cast_dtype
    if "uint" in cast_dtype:
        x = np.abs(np.asarray(x)).tolist()
    return dtype, x, cast_dtype


@st.composite
def floats(
    draw,
    *,
    min_value=None,
    max_value=None,
    allow_nan=False,
    allow_inf=False,
    allow_subnormal=False,
    width=None,
    exclude_min=True,
    exclude_max=True,
    safety_factor=0.99,
    small_value_safety_factor=1.1,
):
    """Draws an arbitrarily sized list of floats with a safety factor applied
        to avoid values being generated at the edge of a dtype limit.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_value
        minimum value of floats generated.
    max_value
        maximum value of floats generated.
    allow_nan
        if True, allow Nans in the list.
    allow_inf
        if True, allow inf in the list.
    allow_subnormal
        if True, allow subnormals in the list.
    width
        The width argument specifies the maximum number of bits of precision
        required to represent the generated float. Valid values are 16, 32, or 64.
    exclude_min
        if True, exclude the minimum limit.
    exclude_max
        if True, exclude the maximum limit.
    safety_factor
        default = 0.99. Only values which are 99% or less than the edge of
        the limit for a given dtype are generated.
    small_value_safety_factor
        default = 1.1.

    Returns
    -------
    ret
        list of floats.
    """
    lim_float16 = 65504
    lim_float32 = 3.4028235e38
    lim_float64 = 1.7976931348623157e308

    if min_value is not None and max_value is not None:
        if (
            min_value > -lim_float16 * safety_factor
            and max_value < lim_float16 * safety_factor
            and (width == 16 or not ivy.exists(width))
        ):
            # dtype float16
            width = 16
        elif (
            min_value > -lim_float32 * safety_factor
            and max_value < lim_float32 * safety_factor
            and (width == 32 or not ivy.exists(width))
        ):
            # dtype float32
            width = 32
        else:
            # dtype float64
            width = 64

        min_value = float_of(min_value, width)
        max_value = float_of(max_value, width)

        values = draw(
            st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=allow_nan,
                allow_subnormal=allow_subnormal,
                allow_infinity=allow_inf,
                width=width,
                exclude_min=exclude_min,
                exclude_max=exclude_max,
            )
        )

    else:
        if min_value is not None:
            if min_value > -lim_float16 * safety_factor and (
                width == 16 or not ivy.exists(width)
            ):
                dtype_min = "float16"
            elif min_value > -lim_float32 * safety_factor and (
                width == 32 or not ivy.exists(width)
            ):
                dtype_min = "float32"
            else:
                dtype_min = "float64"
        else:
            dtype_min = draw(st.sampled_from(ivy_np.valid_float_dtypes))

        if max_value is not None:
            if max_value < lim_float16 * safety_factor and (
                width == 16 or not ivy.exists(width)
            ):
                dtype_max = "float16"
            elif max_value < lim_float32 * safety_factor and (
                width == 32 or not ivy.exists(width)
            ):
                dtype_max = "float32"
            else:
                dtype_max = "float64"
        else:
            dtype_max = draw(st.sampled_from(ivy_np.valid_float_dtypes))

        dtype = ivy.promote_types(dtype_min, dtype_max)

        if dtype == "float16" or 16 == ivy.default(width, 0):
            width = 16
            min_value = float_of(-lim_float16 * safety_factor, width)
            max_value = float_of(lim_float16 * safety_factor, width)
        elif dtype in ["float32", "bfloat16"] or 32 == ivy.default(width, 0):
            width = 32
            min_value = float_of(-lim_float32 * safety_factor, width)
            max_value = float_of(lim_float32 * safety_factor, width)
        else:
            width = 64
            min_value = float_of(-lim_float64 * safety_factor, width)
            max_value = float_of(lim_float64 * safety_factor, width)

        values = draw(
            st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=allow_nan,
                allow_subnormal=allow_subnormal,
                allow_infinity=allow_inf,
                width=width,
                exclude_min=exclude_min,
                exclude_max=exclude_max,
            )
        )
    return values


@st.composite
def ints(draw, *, min_value=None, max_value=None, safety_factor=0.95):
    """Draws an arbitrarily sized list of integers with a safety factor
    applied to values.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_value
        minimum value of integers generated.
    max_value
        maximum value of integers generated.
    safety_factor
        default = 0.95. Only values which are 95% or less than the edge of
        the limit for a given dtype are generated.

    Returns
    -------
    ret
        list of integers.
    """
    dtype = draw(st.sampled_from(ivy_np.valid_int_dtypes))

    if dtype == "int8":
        min_value = ivy.default(min_value, round(-128 * safety_factor))
        max_value = ivy.default(max_value, round(127 * safety_factor))
    elif dtype == "int16":
        min_value = ivy.default(min_value, round(-32768 * safety_factor))
        max_value = ivy.default(max_value, round(32767 * safety_factor))
    elif dtype == "int32":
        min_value = ivy.default(min_value, round(-2147483648 * safety_factor))
        max_value = ivy.default(max_value, round(2147483647 * safety_factor))
    elif dtype == "int64":
        min_value = ivy.default(min_value, round(-9223372036854775808 * safety_factor))
        max_value = ivy.default(max_value, round(9223372036854775807 * safety_factor))
    elif dtype == "uint8":
        min_value = ivy.default(min_value, round(0 * safety_factor))
        max_value = ivy.default(max_value, round(255 * safety_factor))
    elif dtype == "uint16":
        min_value = ivy.default(min_value, round(0 * safety_factor))
        max_value = ivy.default(max_value, round(65535 * safety_factor))
    elif dtype == "uint32":
        min_value = ivy.default(min_value, round(0 * safety_factor))
        max_value = ivy.default(max_value, round(4294967295 * safety_factor))
    elif dtype == "uint64":
        min_value = ivy.default(min_value, round(0 * safety_factor))
        max_value = ivy.default(max_value, round(18446744073709551615 * safety_factor))

    return draw(st.integers(min_value, max_value))


@st.composite
def ints_or_floats(draw, *, min_value=None, max_value=None, safety_factor=0.95):
    """Draws integers or floats with a safety factor
    applied to values.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_value
        minimum value of integers generated.
    max_value
        maximum value of integers generated.
    safety_factor
        default = 0.95. Only values which are 95% or less than the edge of
        the limit for a given dtype are generated.

    Returns
    -------
    ret
        integer or float.
    """
    return draw(
        ints(
            min_value=int(min_value),
            max_value=int(max_value),
            safety_factor=safety_factor,
        )
        | floats(min_value=min_value, max_value=max_value, safety_factor=safety_factor)
    )


def assert_all_close(
    ret_np, ret_from_gt_np, rtol=1e-05, atol=1e-08, ground_truth_backend="TensorFlow"
):
    """Matches the ret_np and ret_from_np inputs element-by-element to ensure that
    they are the same.

    Parameters
    ----------
    ret_np
        Return from the framework to test. Ivy Container or Numpy Array.
    ret_from_gt_np
        Return from the ground truth framework. Ivy Container or Numpy Array.
    rtol
        Relative Tolerance Value.
    atol
        Absolute Tolerance Value.
    ground_truth_backend
        Ground Truth Backend Framework.

    Returns
    -------
    None if the test passes, else marks the test as failed.
    """
    ret_dtype = str(ret_np.dtype)
    ret_from_gt_dtype = str(ret_from_gt_np.dtype).replace("longlong", "int64")
    assert ret_dtype == ret_from_gt_dtype, (
        "the return with a {} backend produced data type of {}, while the return with"
        " a {} backend returned a data type of {}.".format(
            ground_truth_backend,
            ret_from_gt_dtype,
            ivy.current_backend_str(),
            ret_dtype,
        )
    )
    if ivy.is_ivy_container(ret_np) and ivy.is_ivy_container(ret_from_gt_np):
        ivy.Container.multi_map(assert_all_close, [ret_np, ret_from_gt_np])
    else:
        if ret_np.dtype == "bfloat16" or ret_from_gt_np.dtype == "bfloat16":
            ret_np = ret_np.astype("float64")
            ret_from_gt_np = ret_from_gt_np.astype("float64")
        assert np.allclose(
            np.nan_to_num(ret_np), np.nan_to_num(ret_from_gt_np), rtol=rtol, atol=atol
        ), "{} != {}".format(ret_np, ret_from_gt_np)


def assert_same_type_and_shape(values, this_key_chain=None):
    x, y = values
    assert type(x) is type(y), "type(x) = {}, type(y) = {}".format(type(x), type(y))
    if isinstance(x, np.ndarray):
        assert x.shape == y.shape, "x.shape = {}, y.shape = {}".format(x.shape, y.shape)
        assert x.dtype == y.dtype, "x.dtype = {}, y.dtype = {}".format(x.dtype, y.dtype)


def kwargs_to_args_n_kwargs(*, num_positional_args, kwargs):
    """Splits the kwargs into args and kwargs, with the first num_positional_args ported
    to args.
    """
    args = [v for v in list(kwargs.values())[:num_positional_args]]
    kwargs = {k: kwargs[k] for k in list(kwargs.keys())[num_positional_args:]}
    return args, kwargs


def list_of_length(*, x, length):
    """Returns a random list of the given length from elements in x."""
    return st.lists(x, min_size=length, max_size=length)


def as_cont(*, x):
    """Returns x as an Ivy Container, containing x at all its leaves."""
    return ivy.Container({"a": x, "b": {"c": x, "d": x}})


def as_lists(*args):
    """Changes the elements in args to be of type list."""
    return (a if isinstance(a, list) else [a] for a in args)


def flatten_fw(*, ret, fw):
    """Returns a flattened numpy version of the arrays in ret for a given framework."""
    if not isinstance(ret, tuple):
        ret = (ret,)
    if fw == "jax":
        ret_idxs = ivy.nested_argwhere(
            ret, lambda x: ivy.is_ivy_array(x) or is_jax_native_array(x)
        )
    elif fw == "numpy":
        ret_idxs = ivy.nested_argwhere(
            ret, lambda x: ivy.is_ivy_array(x) or is_numpy_native_array(x)
        )
    elif fw == "tensorflow":
        ret_idxs = ivy.nested_argwhere(
            ret, lambda x: ivy.is_ivy_array(x) or is_tensorflow_native_array(x)
        )
    else:
        ret_idxs = ivy.nested_argwhere(
            ret, lambda x: ivy.is_ivy_array(x) or is_torch_native_array(x)
        )
    ret_flat = ivy.multi_index_nest(ret, ret_idxs)

    # convert the return to NumPy
    ret_np_flat = [ivy.to_numpy(x) for x in ret_flat]
    return ret_np_flat


def flatten(*, ret):
    """Returns a flattened numpy version of the arrays in ret."""
    if not isinstance(ret, tuple):
        ret = (ret,)
    ret_idxs = ivy.nested_argwhere(ret, ivy.is_ivy_array)
    return ivy.multi_index_nest(ret, ret_idxs)


def flatten_and_to_np(*, ret):
    # flatten the return
    ret_flat = flatten(ret=ret)
    return [ivy.to_numpy(x) for x in ret_flat]


def get_ret_and_flattened_np_array(fn, *args, **kwargs):
    """
    Runs func with args and kwargs, and returns the result along with its flattened
    version.
    """
    ret = fn(*args, **kwargs)
    return ret, flatten_and_to_np(ret=ret)


def value_test(
    *,
    ret_np_flat,
    ret_np_from_gt_flat,
    rtol=None,
    atol=1e-6,
    ground_truth_backend="TensorFlow",
):
    """Performs a value test for matching the arrays in ret_np_flat and
    ret_from_np_flat.

    Parameters
    ----------
    ret_np_flat
        A list (flattened) containing Numpy arrays. Return from the
        framework to test.
    ret_np_from_gt_flat
        A list (flattened) containing Numpy arrays. Return from the ground
        truth framework.
    rtol
        Relative Tolerance Value.
    atol
        Absolute Tolerance Value.
    ground_truth_backend
        Ground Truth Backend Framework.

    Returns
    -------
    None if the value test passes, else marks the test as failed.
    """
    if type(ret_np_flat) != list:
        ret_np_flat = [ret_np_flat]
    if type(ret_np_from_gt_flat) != list:
        ret_np_from_gt_flat = [ret_np_from_gt_flat]
    assert len(ret_np_flat) == len(ret_np_from_gt_flat), (
        "len(ret_np_flat) != len(ret_np_from_gt_flat):\n\n"
        "ret_np_flat:\n\n{}\n\nret_np_from_gt_flat:\n\n{}".format(
            ret_np_flat, ret_np_from_gt_flat
        )
    )
    # value tests, iterating through each array in the flattened returns
    if not rtol:
        for ret_np, ret_np_from_gt in zip(ret_np_flat, ret_np_from_gt_flat):
            rtol = TOLERANCE_DICT.get(str(ret_np_from_gt.dtype), 1e-03)
            assert_all_close(
                ret_np,
                ret_np_from_gt,
                rtol=rtol,
                atol=atol,
                ground_truth_backend=ground_truth_backend,
            )
    else:
        for ret_np, ret_np_from_gt in zip(ret_np_flat, ret_np_from_gt_flat):
            assert_all_close(
                ret_np,
                ret_np_from_gt,
                rtol=rtol,
                atol=atol,
                ground_truth_backend=ground_truth_backend,
            )


def args_to_container(array_args):
    array_args_container = ivy.Container({str(k): v for k, v in enumerate(array_args)})
    return array_args_container


def gradient_test(
    *,
    fn_name,
    all_as_kwargs_np,
    args_np,
    kwargs_np,
    input_dtypes,
    as_variable_flags,
    native_array_flags,
    container_flags,
    rtol_: float = None,
    atol_: float = 1e-06,
    ground_truth_backend: str = "torch",
):
    def grad_fn(xs):
        array_vals = [v for k, v in xs.to_iterator()]
        arg_array_vals = array_vals[0 : len(args_idxs)]
        kwarg_array_vals = array_vals[len(args_idxs) :]
        args_writeable = ivy.copy_nest(args)
        kwargs_writeable = ivy.copy_nest(kwargs)
        ivy.set_nest_at_indices(args_writeable, args_idxs, arg_array_vals)
        ivy.set_nest_at_indices(kwargs_writeable, kwargs_idxs, kwarg_array_vals)
        return ivy.mean(ivy.__dict__[fn_name](*args_writeable, **kwargs_writeable))

    # extract all arrays from the arguments and keyword arguments
    arg_np_vals, args_idxs, c_arg_vals = _get_nested_np_arrays(args_np)
    kwarg_np_vals, kwargs_idxs, c_kwarg_vals = _get_nested_np_arrays(kwargs_np)

    args, kwargs, _, args_idxs, kwargs_idxs = create_args_kwargs(
        args_np=args_np,
        arg_np_vals=arg_np_vals,
        args_idxs=args_idxs,
        kwargs_np=kwargs_np,
        kwarg_np_vals=kwarg_np_vals,
        kwargs_idxs=kwargs_idxs,
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable_flags,
        native_array_flags=native_array_flags,
        container_flags=container_flags,
    )
    arg_array_vals = list(ivy.multi_index_nest(args, args_idxs))
    kwarg_array_vals = list(ivy.multi_index_nest(kwargs, kwargs_idxs))
    xs = args_to_container(arg_array_vals + kwarg_array_vals)
    _, ret_np_flat = get_ret_and_flattened_np_array(
        ivy.execute_with_gradients, grad_fn, xs
    )
    grads_np_flat = ret_np_flat[1]
    # compute the return with a Ground Truth backend
    ivy.set_backend(ground_truth_backend)
    test_unsupported = check_unsupported_dtype(
        fn=ivy.__dict__[fn_name],
        input_dtypes=input_dtypes,
        all_as_kwargs_np=all_as_kwargs_np,
    )
    if test_unsupported:
        return
    args, kwargs, _, args_idxs, kwargs_idxs = create_args_kwargs(
        args_np=args_np,
        kwargs_np=kwargs_np,
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable_flags,
        native_array_flags=native_array_flags,
        container_flags=container_flags,
    )
    arg_array_vals = list(ivy.multi_index_nest(args, args_idxs))
    kwarg_array_vals = list(ivy.multi_index_nest(kwargs, kwargs_idxs))
    xs = args_to_container(arg_array_vals + kwarg_array_vals)
    _, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
        ivy.execute_with_gradients, grad_fn, xs
    )
    grads_np_from_gt_flat = ret_np_from_gt_flat[1]
    ivy.unset_backend()
    # value test
    value_test(
        ret_np_flat=grads_np_flat,
        ret_np_from_gt_flat=grads_np_from_gt_flat,
        rtol=rtol_,
        atol=atol_,
    )


def check_unsupported_dtype(*, fn, input_dtypes, all_as_kwargs_np):
    """Checks whether a function does not support the input data types or the output
    data type.

    Parameters
    ----------
    fn
        The function to check.
    input_dtypes
        data-types of the input arguments and keyword-arguments.
    all_as_kwargs_np
        All arguments in Numpy Format, to check for the presence of dtype argument.

    Returns
    -------
    True if the function does not support the given input or output data types, False
    otherwise.
    """
    test_unsupported = False
    unsupported_dtypes_fn = ivy.function_unsupported_dtypes(fn)
    supported_dtypes_fn = ivy.function_supported_dtypes(fn)
    if unsupported_dtypes_fn:
        for d in input_dtypes:
            if d in unsupported_dtypes_fn:
                test_unsupported = True
                break
        if (
            "dtype" in all_as_kwargs_np
            and all_as_kwargs_np["dtype"] in unsupported_dtypes_fn
        ):
            test_unsupported = True
    if supported_dtypes_fn and not test_unsupported:
        for d in input_dtypes:
            if d not in supported_dtypes_fn:
                test_unsupported = True
                break
        if (
            "dtype" in all_as_kwargs_np
            and all_as_kwargs_np["dtype"] not in supported_dtypes_fn
        ):
            test_unsupported = True
    return test_unsupported


def check_unsupported_device(*, fn, input_device, all_as_kwargs_np):
    """Checks whether a function does not support a given device.

    Parameters
    ----------
    fn
        The function to check.
    input_device
        The backend device.
    all_as_kwargs_np
        All arguments in Numpy Format, to check for the presence of dtype argument.

    Returns
    -------
    True if the function does not support the given device, False otherwise.
    """
    test_unsupported = False
    unsupported_devices_fn = ivy.function_unsupported_devices(fn)
    supported_devices_fn = ivy.function_supported_devices(fn)
    if unsupported_devices_fn:
        if input_device in unsupported_devices_fn:
            test_unsupported = True
        if (
            "device" in all_as_kwargs_np
            and all_as_kwargs_np["device"] in unsupported_devices_fn
        ):
            test_unsupported = True
    if supported_devices_fn and not test_unsupported:
        if input_device not in supported_devices_fn:
            test_unsupported = True
        if (
            "device" in all_as_kwargs_np
            and all_as_kwargs_np["device"] not in supported_devices_fn
        ):
            test_unsupported = True
    return test_unsupported


def check_unsupported_device_and_dtype(*, fn, device, input_dtypes, all_as_kwargs_np):
    """Checks whether a function does not support a given device or data types.

    Parameters
    ----------
    fn
        The function to check.
    device
        The backend device to check.
    input_dtypes
        data-types of the input arguments and keyword-arguments.
    all_as_kwargs_np
        All arguments in Numpy Format, to check for the presence of dtype argument.

    Returns
    -------
    True if the function does not support both the device and any data type, False
    otherwise.
    """
    unsupported_devices_dtypes_fn = ivy.function_unsupported_devices_and_dtypes(fn)

    if device in unsupported_devices_dtypes_fn:
        for d in input_dtypes:
            if d in unsupported_devices_dtypes_fn[device]:
                return True

    if "device" in all_as_kwargs_np and "dtype" in all_as_kwargs_np:
        dev = all_as_kwargs_np["device"]
        dtype = all_as_kwargs_np["dtype"]
        if dtype in unsupported_devices_dtypes_fn.get(dev, []):
            return True

    return False


def _get_nested_np_arrays(nest):
    """
    A helper function to search for a NumPy arrays in a nest
    Parameters
    ----------
    nest
        nest to search in.

    Returns
    -------
         Items found, indices, and total number of arrays found
    """
    indices = ivy.nested_argwhere(nest, lambda x: isinstance(x, np.ndarray))
    ret = ivy.multi_index_nest(nest, indices)
    return ret, indices, len(ret)


def create_args_kwargs(
    *,
    args_np,
    arg_np_vals,
    args_idxs,
    kwargs_np,
    kwarg_np_vals,
    kwargs_idxs,
    input_dtypes,
    as_variable_flags,
    native_array_flags=None,
    container_flags=None,
):
    """Creates arguments and keyword-arguments for the function to test.

    Parameters
    ----------
    args_np
        A dictionary of arguments in Numpy.
    kwargs_np
        A dictionary of keyword-arguments in Numpy.
    input_dtypes
        data-types of the input arguments and keyword-arguments.
    as_variable_flags
        A list of booleans. if True for a corresponding input argument, it is called
        as an Ivy Variable.
    native_array_flags
        if not None, the corresponding argument is called as a Native Array.
    container_flags
        if not None, the corresponding argument is called as an Ivy Container.

    Returns
    -------
    Arguments, Keyword-arguments, number of arguments, and indexes on arguments and
    keyword-arguments.
    """
    # create args
    num_arg_vals = len(arg_np_vals)
    arg_array_vals = [
        ivy.array(x, dtype=d) for x, d in zip(arg_np_vals, input_dtypes[:num_arg_vals])
    ]
    arg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(arg_array_vals, as_variable_flags[:num_arg_vals])
    ]
    if native_array_flags:
        arg_array_vals = [
            ivy.to_native(x) if n else x
            for x, n in zip(arg_array_vals, native_array_flags[:num_arg_vals])
        ]
    if container_flags:
        arg_array_vals = [
            as_cont(x=x) if c else x
            for x, c in zip(arg_array_vals, container_flags[:num_arg_vals])
        ]
    args = ivy.copy_nest(args_np, to_mutable=True)
    ivy.set_nest_at_indices(args, args_idxs, arg_array_vals)

    # create kwargs
    kwarg_array_vals = [
        ivy.array(x, dtype=d)
        for x, d in zip(kwarg_np_vals, input_dtypes[num_arg_vals:])
    ]
    kwarg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(kwarg_array_vals, as_variable_flags[num_arg_vals:])
    ]
    if native_array_flags:
        kwarg_array_vals = [
            ivy.to_native(x) if n else x
            for x, n in zip(kwarg_array_vals, native_array_flags[num_arg_vals:])
        ]
    if container_flags:
        kwarg_array_vals = [
            as_cont(x=x) if c else x
            for x, c in zip(kwarg_array_vals, container_flags[num_arg_vals:])
        ]
    kwargs = ivy.copy_nest(kwargs_np, to_mutable=True)
    ivy.set_nest_at_indices(kwargs, kwargs_idxs, kwarg_array_vals)
    return args, kwargs, num_arg_vals, args_idxs, kwargs_idxs


def test_unsupported_function(*, fn, args, kwargs):
    """Tests a function with an unsupported datatype to raise an exception.

    Parameters
    ----------
    fn
        callable function to test.
    args
        arguments to the function.
    kwargs
        keyword-arguments to the function.
    """
    try:
        fn(*args, **kwargs)
        assert False
    except:  # noqa
        return


def test_method(
    *,
    input_dtypes_init: Union[ivy.Dtype, List[ivy.Dtype]] = None,
    as_variable_flags_init: Union[bool, List[bool]] = None,
    num_positional_args_init: int = 0,
    native_array_flags_init: Union[bool, List[bool]] = None,
    all_as_kwargs_np_init: dict = None,
    input_dtypes_method: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags_method: Union[bool, List[bool]],
    num_positional_args_method: int,
    native_array_flags_method: Union[bool, List[bool]],
    container_flags_method: Union[bool, List[bool]],
    all_as_kwargs_np_method: dict,
    fw: str,
    class_name: str,
    method_name: str = "__call__",
    init_with_v: bool = False,
    method_with_v: bool = False,
    rtol_: float = None,
    atol_: float = 1e-06,
    test_values: Union[bool, str] = True,
    test_gradients: bool = False,
    ground_truth_backend: str = "tensorflow",
    device_: str = "cpu",
):
    """Tests a class-method that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    input_dtypes_init
        data types of the input arguments to the constructor in order.
    as_variable_flags_init
        dictates whether the corresponding input argument passed to the constructor
        should be treated as an ivy.Variable.
    num_positional_args_init
        number of input arguments that must be passed as positional arguments to the
        constructor.
    native_array_flags_init
        dictates whether the corresponding input argument passed to the constructor
        should be treated as a native array.
    all_as_kwargs_np_init:
        input arguments to the constructor as keyword arguments.
    input_dtypes_method
        data types of the input arguments to the method in order.
    as_variable_flags_method
        dictates whether the corresponding input argument passed to the method should
        be treated as an ivy.Variable.
    num_positional_args_method
        number of input arguments that must be passed as positional arguments to the
        method.
    native_array_flags_method
        dictates whether the corresponding input argument passed to the method should
        be treated as a native array.
    container_flags_method
        dictates whether the corresponding input argument passed to the method should
        be treated as an ivy Container.
    all_as_kwargs_np_method:
        input arguments to the method as keyword arguments.
    fw
        current backend (framework).
    class_name
        name of the class to test.
    method_name
        name of tthe method to test.
    init_with_v
        if the class being tested is an ivy.Module, then setting this flag as True will
        call the constructor with the variables v passed explicitly.
    method_with_v
        if the class being tested is an ivy.Module, then setting this flag as True will
        call the method with the variables v passed explicitly.
    rtol_
        relative tolerance value.
    atol_
        absolute tolerance value.
    test_values
        can be a bool or a string to indicate whether correctness of values should be
        tested. If the value is `with_v`, shapes are tested but not values.
    ground_truth_backend
        Ground Truth Backend to compare the result-values.
    device_
        The device on which to create arrays.

    Returns
    -------
    ret
        optional, return value from the function
    ret_gt
        optional, return value from the Ground Truth function
    """
    # split the arguments into their positional and keyword components

    # Constructor arguments #
    # convert single values to length 1 lists
    (input_dtypes_init, as_variable_flags_init, native_array_flags_init,) = as_lists(
        ivy.default(input_dtypes_init, []),
        ivy.default(as_variable_flags_init, []),
        ivy.default(native_array_flags_init, []),
    )
    all_as_kwargs_np_init = ivy.default(all_as_kwargs_np_init, dict())
    (
        input_dtypes_method,
        as_variable_flags_method,
        native_array_flags_method,
        container_flags_method,
    ) = as_lists(
        input_dtypes_method,
        as_variable_flags_method,
        native_array_flags_method,
        container_flags_method,
    )

    args_np_constructor, kwargs_np_constructor = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args_init,
        kwargs=all_as_kwargs_np_init,
    )

    # extract all arrays from the arguments and keyword arguments
    con_arg_np_vals, con_args_idxs, con_c_arg_vals = _get_nested_np_arrays(
        args_np_constructor
    )
    con_kwarg_np_vals, con_kwargs_idxs, con_c_kwarg_vals = _get_nested_np_arrays(
        kwargs_np_constructor
    )

    # make all lists equal in length
    num_arrays_constructor = con_c_arg_vals + con_c_kwarg_vals
    if len(input_dtypes_init) < num_arrays_constructor:
        input_dtypes_init = [
            input_dtypes_init[0] for _ in range(num_arrays_constructor)
        ]
    if len(as_variable_flags_init) < num_arrays_constructor:
        as_variable_flags_init = [
            as_variable_flags_init[0] for _ in range(num_arrays_constructor)
        ]
    if len(native_array_flags_init) < num_arrays_constructor:
        native_array_flags_init = [
            native_array_flags_init[0] for _ in range(num_arrays_constructor)
        ]

    # update variable flags to be compatible with float dtype
    as_variable_flags_init = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(as_variable_flags_init, input_dtypes_init)
    ]

    # Create Args
    args_constructor, kwargs_constructor, _, _, _ = create_args_kwargs(
        args_np=args_np_constructor,
        arg_np_vals=con_arg_np_vals,
        args_idxs=con_args_idxs,
        kwargs_np=kwargs_np_constructor,
        kwarg_np_vals=con_kwarg_np_vals,
        kwargs_idxs=con_kwargs_idxs,
        input_dtypes=input_dtypes_init,
        as_variable_flags=as_variable_flags_init,
        native_array_flags=native_array_flags_init,
    )
    # End constructor #

    # Method arguments #
    args_np_method, kwargs_np_method = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args_method, kwargs=all_as_kwargs_np_method
    )

    # extract all arrays from the arguments and keyword arguments
    met_arg_np_vals, met_args_idxs, met_c_arg_vals = _get_nested_np_arrays(
        args_np_method
    )
    met_kwarg_np_vals, met_kwargs_idxs, met_c_kwarg_vals = _get_nested_np_arrays(
        kwargs_np_method
    )

    # make all lists equal in length
    num_arrays_method = met_c_arg_vals + met_c_kwarg_vals
    if len(input_dtypes_method) < num_arrays_method:
        input_dtypes_method = [input_dtypes_method[0] for _ in range(num_arrays_method)]
    if len(as_variable_flags_method) < num_arrays_method:
        as_variable_flags_method = [
            as_variable_flags_method[0] for _ in range(num_arrays_method)
        ]
    if len(native_array_flags_method) < num_arrays_method:
        native_array_flags_method = [
            native_array_flags_method[0] for _ in range(num_arrays_method)
        ]
    if len(container_flags_method) < num_arrays_method:
        container_flags_method = [
            container_flags_method[0] for _ in range(num_arrays_method)
        ]

    as_variable_flags_method = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(as_variable_flags_method, input_dtypes_method)
    ]

    # Create Args
    args_method, kwargs_method, _, _, _ = create_args_kwargs(
        args_np=args_np_method,
        arg_np_vals=met_arg_np_vals,
        args_idxs=met_args_idxs,
        kwargs_np=kwargs_np_method,
        kwarg_np_vals=met_kwarg_np_vals,
        kwargs_idxs=met_kwargs_idxs,
        input_dtypes=input_dtypes_method,
        as_variable_flags=as_variable_flags_method,
        native_array_flags=native_array_flags_method,
        container_flags=container_flags_method,
    )
    # End Method #

    # Run testing
    ins = ivy.__dict__[class_name](*args_constructor, **kwargs_constructor)
    v_np = None
    if isinstance(ins, ivy.Module):
        if init_with_v:
            v = ivy.Container(
                ins._create_variables(device=device_, dtype=input_dtypes_method[0])
            )
            ins = ivy.__dict__[class_name](*args_constructor, **kwargs_constructor, v=v)
        else:
            v = ins.__getattribute__("v")
        v_np = v.map(lambda x, kc: ivy.to_numpy(x) if ivy.is_array(x) else x)
        if method_with_v:
            kwargs_method = dict(**kwargs_method, v=v)
    ret, ret_np_flat = get_ret_and_flattened_np_array(
        ins.__getattribute__(method_name), *args_method, **kwargs_method
    )

    # Compute the return with a Ground Truth backend
    ivy.set_backend(ground_truth_backend)
    args_gt_constructor, kwargs_gt_constructor, _, _, _ = create_args_kwargs(
        args_np=args_np_constructor,
        arg_np_vals=con_arg_np_vals,
        args_idxs=con_args_idxs,
        kwargs_np=kwargs_np_constructor,
        kwarg_np_vals=con_kwarg_np_vals,
        kwargs_idxs=con_kwargs_idxs,
        input_dtypes=input_dtypes_init,
        as_variable_flags=as_variable_flags_init,
        native_array_flags=native_array_flags_init,
    )
    args_gt_method, kwargs_gt_method, _, _, _ = create_args_kwargs(
        args_np=args_np_method,
        arg_np_vals=met_arg_np_vals,
        args_idxs=met_args_idxs,
        kwargs_np=kwargs_np_method,
        kwarg_np_vals=met_kwarg_np_vals,
        kwargs_idxs=met_kwargs_idxs,
        input_dtypes=input_dtypes_method,
        as_variable_flags=as_variable_flags_method,
        native_array_flags=native_array_flags_method,
        container_flags=container_flags_method,
    )
    ins_gt = ivy.__dict__[class_name](*args_gt_constructor, **kwargs_gt_constructor)
    if isinstance(ins_gt, ivy.Module):
        v_gt = v_np.map(
            lambda x, kc: ivy.asarray(x) if isinstance(x, np.ndarray) else x
        )
        kwargs_gt_method = dict(**kwargs_gt_method, v=v_gt)
    ret_from_gt, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
        ins_gt.__getattribute__(method_name), *args_gt_method, **kwargs_gt_method
    )
    ivy.unset_backend()
    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, ret_from_gt
    # value test
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=ret_np_from_gt_flat,
        rtol=rtol_,
        atol=atol_,
    )


def test_function(
    *,
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: Union[bool, List[bool]],
    with_out: bool,
    num_positional_args: int,
    native_array_flags: Union[bool, List[bool]],
    container_flags: Union[bool, List[bool]],
    instance_method: bool,
    fw: str,
    fn_name: str,
    rtol_: float = None,
    atol_: float = 1e-06,
    test_values: bool = True,
    test_gradients: bool = False,
    ground_truth_backend: str = "tensorflow",
    device_: str = "cpu",
    return_flat_np_arrays: bool = False,
    **all_as_kwargs_np,
):
    """Tests a function that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated
        as an ivy Variable.
    with_out
        if True, the function is also tested with the optional out argument.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    native_array_flags
        dictates whether the corresponding input argument should be treated
        as a native array.
    container_flags
        dictates whether the corresponding input argument should be treated
         as an ivy Container.
    instance_method
        if True, the function is run as an instance method of the first
         argument (should be an ivy Array or Container).
    fw
        current backend (framework).
    fn_name
        name of the function to test.
    rtol_
        relative tolerance value.
    atol_
        absolute tolerance value.
    test_values
        if True, test for the correctness of the resulting values.
    test_gradients
        if True, test for the correctness of gradients.
    ground_truth_backend
        Ground Truth Backend to compare the result-values.
    device_
        The device on which to create arrays
    return_flat_np_arrays
        If test_values is False, this flag dictates whether the original returns are
        returned, or whether the flattened numpy arrays are returned.
    all_as_kwargs_np
        input arguments to the function as keyword arguments.

    Returns
    -------
    ret
        optional, return value from the function
    ret_gt
        optional, return value from the Ground Truth function

    Examples
    --------
    >>> input_dtypes = 'float64'
    >>> as_variable_flags = False
    >>> with_out = False
    >>> num_positional_args = 0
    >>> native_array_flags = False
    >>> container_flags = False
    >>> instance_method = False
    >>> fw = "torch"
    >>> fn_name = "abs"
    >>> x = np.array([-1])
    >>> test_function(input_dtypes, as_variable_flags, with_out,\
                            num_positional_args, native_array_flags,\
                            container_flags, instance_method, fw, fn_name, x=x)

    >>> input_dtypes = ['float64', 'float32']
    >>> as_variable_flags = [False, True]
    >>> with_out = False
    >>> num_positional_args = 1
    >>> native_array_flags = [True, False]
    >>> container_flags = [False, False]
    >>> instance_method = False
    >>> fw = "numpy"
    >>> fn_name = "add"
    >>> x1 = np.array([1, 3, 4])
    >>> x2 = np.array([-3, 15, 24])
    >>> test_function(input_dtypes, as_variable_flags, with_out,\
                            num_positional_args, native_array_flags,\
                             container_flags, instance_method,\
                              fw, fn_name, x1=x1, x2=x2)
    """
    # convert single values to length 1 lists
    input_dtypes, as_variable_flags, native_array_flags, container_flags = as_lists(
        input_dtypes, as_variable_flags, native_array_flags, container_flags
    )

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args, kwargs=all_as_kwargs_np
    )

    # extract all arrays from the arguments and keyword arguments
    arg_np_vals, args_idxs, c_arg_vals = _get_nested_np_arrays(args_np)
    kwarg_np_vals, kwargs_idxs, c_kwarg_vals = _get_nested_np_arrays(kwargs_np)

    # make all lists equal in length
    num_arrays = c_arg_vals + c_kwarg_vals
    if len(input_dtypes) < num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(num_arrays)]
    if len(as_variable_flags) < num_arrays:
        as_variable_flags = [as_variable_flags[0] for _ in range(num_arrays)]
    if len(native_array_flags) < num_arrays:
        native_array_flags = [native_array_flags[0] for _ in range(num_arrays)]
    if len(container_flags) < num_arrays:
        container_flags = [container_flags[0] for _ in range(num_arrays)]

    # update variable flags to be compatible with float dtype and with_out args
    as_variable_flags = [
        v if ivy.is_float_dtype(d) and not with_out else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]

    # update instance_method flag to only be considered if the
    # first term is either an ivy.Array or ivy.Container
    instance_method = instance_method and (
        not native_array_flags[0] or container_flags[0]
    )

    fn = getattr(ivy, fn_name)
    if gradient_incompatible_function(fn=fn):
        return
    test_unsupported = check_unsupported_dtype(
        fn=fn, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
    )
    if not test_unsupported:
        test_unsupported = check_unsupported_device(
            fn=fn, input_device=device_, all_as_kwargs_np=all_as_kwargs_np
        )
    if not test_unsupported:
        test_unsupported = check_unsupported_device_and_dtype(
            fn=fn,
            device=device_,
            input_dtypes=input_dtypes,
            all_as_kwargs_np=all_as_kwargs_np,
        )
    if test_unsupported:
        try:
            args, kwargs, num_arg_vals, args_idxs, kwargs_idxs = create_args_kwargs(
                args_np=args_np,
                arg_np_vals=arg_np_vals,
                args_idxs=args_idxs,
                kwargs_np=kwargs_np,
                kwarg_np_vals=kwarg_np_vals,
                kwargs_idxs=kwargs_idxs,
                input_dtypes=input_dtypes,
                as_variable_flags=as_variable_flags,
                native_array_flags=native_array_flags,
                container_flags=container_flags,
            )
        except Exception:
            return
    else:
        args, kwargs, num_arg_vals, args_idxs, kwargs_idxs = create_args_kwargs(
            args_np=args_np,
            arg_np_vals=arg_np_vals,
            args_idxs=args_idxs,
            kwargs_np=kwargs_np,
            kwarg_np_vals=kwarg_np_vals,
            kwargs_idxs=kwargs_idxs,
            input_dtypes=input_dtypes,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
            container_flags=container_flags,
        )

    # run either as an instance method or from the API directly
    instance = None
    if instance_method:
        is_instance = [
            (not native_flag) or container_flag
            for native_flag, container_flag in zip(native_array_flags, container_flags)
        ]
        arg_is_instance = is_instance[:num_arg_vals]
        kwarg_is_instance = is_instance[num_arg_vals:]
        if arg_is_instance and max(arg_is_instance):
            i = 0
            for i, a in enumerate(arg_is_instance):
                if a:
                    break
            instance_idx = args_idxs[i]
            instance = ivy.index_nest(args, instance_idx)
            args = ivy.copy_nest(args, to_mutable=True)
            ivy.prune_nest_at_index(args, instance_idx)
        else:
            i = 0
            for i, a in enumerate(kwarg_is_instance):
                if a:
                    break
            instance_idx = kwargs_idxs[i]
            instance = ivy.index_nest(kwargs, instance_idx)
            kwargs = ivy.copy_nest(kwargs, to_mutable=True)
            ivy.prune_nest_at_index(kwargs, instance_idx)
        if test_unsupported:
            test_unsupported_function(
                fn=instance.__getattribute__(fn_name), args=args, kwargs=kwargs
            )
            return

        ret, ret_np_flat = get_ret_and_flattened_np_array(
            instance.__getattribute__(fn_name), *args, **kwargs
        )
    else:
        if test_unsupported:
            test_unsupported_function(
                fn=ivy.__dict__[fn_name], args=args, kwargs=kwargs
            )
            return
        ret, ret_np_flat = get_ret_and_flattened_np_array(
            ivy.__dict__[fn_name], *args, **kwargs
        )
    # assert idx of return if the idx of the out array provided
    if with_out:
        test_ret = ret
        if isinstance(ret, tuple):
            assert hasattr(ivy.__dict__[fn_name], "out_index")
            test_ret = ret[getattr(ivy.__dict__[fn_name], "out_index")]
        out = ivy.zeros_like(test_ret)
        if max(container_flags):
            assert ivy.is_ivy_container(test_ret)
        else:
            assert ivy.is_array(test_ret)
        if instance_method:
            ret, ret_np_flat = get_ret_and_flattened_np_array(
                instance.__getattribute__(fn_name), *args, **kwargs, out=out
            )
        else:
            ret, ret_np_flat = get_ret_and_flattened_np_array(
                ivy.__dict__[fn_name], *args, **kwargs, out=out
            )
        test_ret = ret
        if isinstance(ret, tuple):
            test_ret = ret[getattr(ivy.__dict__[fn_name], "out_index")]
        assert test_ret is out
        if not max(container_flags) and ivy.native_inplace_support:
            # these backends do not always support native inplace updates
            assert test_ret.data is out.data
    # compute the return with a Ground Truth backend
    ivy.set_backend(ground_truth_backend)
    try:
        fn = getattr(ivy, fn_name)
        test_unsupported = check_unsupported_dtype(
            fn=fn, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
        )
        # create args
        if test_unsupported:
            try:
                args, kwargs, _, _, _ = create_args_kwargs(
                    args_np=args_np,
                    arg_np_vals=arg_np_vals,
                    args_idxs=args_idxs,
                    kwargs_np=kwargs_np,
                    kwargs_idxs=kwargs_idxs,
                    kwarg_np_vals=kwarg_np_vals,
                    input_dtypes=input_dtypes,
                    as_variable_flags=as_variable_flags,
                    native_array_flags=native_array_flags,
                    container_flags=container_flags,
                )
            except Exception:
                ivy.unset_backend()
                return
        else:
            args, kwargs, _, _, _ = create_args_kwargs(
                args_np=args_np,
                arg_np_vals=arg_np_vals,
                args_idxs=args_idxs,
                kwargs_np=kwargs_np,
                kwargs_idxs=kwargs_idxs,
                kwarg_np_vals=kwarg_np_vals,
                input_dtypes=input_dtypes,
                as_variable_flags=as_variable_flags,
                native_array_flags=native_array_flags,
                container_flags=container_flags,
            )
        if test_unsupported:
            test_unsupported_function(
                fn=ivy.__dict__[fn_name], args=args, kwargs=kwargs
            )
            ivy.unset_backend()
            return
        ret_from_gt, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
            ivy.__dict__[fn_name], *args, **kwargs
        )
    except Exception as e:
        ivy.unset_backend()
        raise e
    ivy.unset_backend()
    # gradient test
    if (
        test_gradients
        and not fw == "numpy"
        and all(as_variable_flags)
        and not any(container_flags)
        and not instance_method
    ):
        gradient_test(
            fn_name=fn_name,
            all_as_kwargs_np=all_as_kwargs_np,
            args_np=args_np,
            kwargs_np=kwargs_np,
            input_dtypes=input_dtypes,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
            container_flags=container_flags,
            rtol_=rtol_,
            atol_=atol_,
        )

    # assuming value test will be handled manually in the test function
    if not test_values:
        if return_flat_np_arrays:
            return ret_np_flat, ret_np_from_gt_flat
        return ret, ret_from_gt
    # value test
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=ret_np_from_gt_flat,
        rtol=rtol_,
        atol=atol_,
        ground_truth_backend=ground_truth_backend,
    )


def test_frontend_function(
    *,
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: Union[bool, List[bool]],
    with_out: bool,
    with_inplace: bool = False,
    num_positional_args: int,
    native_array_flags: Union[bool, List[bool]],
    fw: str,
    device="cpu",
    frontend: str,
    fn_tree: str,
    rtol: float = None,
    atol: float = 1e-06,
    test_values: bool = True,
    **all_as_kwargs_np,
):
    """Tests a frontend function for the current backend by comparing the result with
    the function in the associated framework.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated
        as an ivy Variable.
    with_out
        if True, the function is also tested for inplace update to an array 
        passed to the optional out argument, should not be True together 
        with with_inplace.
    with_inplace
        if True, the function is also tested with direct inplace update back to
        the inputted array, should not be True together with with_out.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    native_array_flags
        dictates whether the corresponding input argument should be treated
        as a native array.
    fw
        current backend (framework).
    frontend
        current frontend (framework).
    fn_tree
        Path to function in frontend framework namespace.
    rtol
        relative tolerance value.
    atol
        absolute tolerance value.
    test_values
        if True, test for the correctness of the resulting values.
    all_as_kwargs_np
        input arguments to the function as keyword arguments.

    Returns
    -------
    ret
        optional, return value from the function
    ret_np
        optional, return value from the Numpy function
    """
    # convert single values to length 1 lists
    input_dtypes, as_variable_flags, native_array_flags = as_lists(
        input_dtypes, as_variable_flags, native_array_flags
    )

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args, kwargs=all_as_kwargs_np
    )

    # extract all arrays from the arguments and keyword arguments
    arg_np_vals, args_idxs, c_arg_vals = _get_nested_np_arrays(args_np)
    kwarg_np_vals, kwargs_idxs, c_kwarg_vals = _get_nested_np_arrays(kwargs_np)

    # make all lists equal in length
    num_arrays = c_arg_vals + c_kwarg_vals
    if len(input_dtypes) < num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(num_arrays)]
    if len(as_variable_flags) < num_arrays:
        as_variable_flags = [as_variable_flags[0] for _ in range(num_arrays)]
    if len(native_array_flags) < num_arrays:
        native_array_flags = [native_array_flags[0] for _ in range(num_arrays)]

    # update variable flags to be compatible with float dtype and with_out args
    as_variable_flags = [
        v if ivy.is_float_dtype(d) and not with_out else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]

    # parse function name and frontend submodules (i.e. jax.lax, jax.numpy etc.)
    *frontend_submods, fn_tree = fn_tree.split(".")

    # check for unsupported dtypes in backend framework
    function = getattr(ivy.functional.frontends.__dict__[frontend], fn_tree)
    test_unsupported = check_unsupported_dtype(
        fn=function, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
    )

    if not test_unsupported:
        test_unsupported = check_unsupported_device_and_dtype(
            fn=function,
            device=device,
            input_dtypes=input_dtypes,
            all_as_kwargs_np=all_as_kwargs_np,
        )

    # create args
    if test_unsupported:
        try:
            args, kwargs, _, _, _ = create_args_kwargs(
                args_np=args_np,
                arg_np_vals=arg_np_vals,
                args_idxs=args_idxs,
                kwargs_np=kwargs_np,
                kwarg_np_vals=kwarg_np_vals,
                kwargs_idxs=kwargs_idxs,
                input_dtypes=input_dtypes,
                as_variable_flags=as_variable_flags,
                native_array_flags=native_array_flags,
            )
            args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)
        except Exception:
            return
    else:
        args, kwargs, _, _, _ = create_args_kwargs(
            args_np=args_np,
            arg_np_vals=arg_np_vals,
            args_idxs=args_idxs,
            kwargs_np=kwargs_np,
            kwarg_np_vals=kwarg_np_vals,
            kwargs_idxs=kwargs_idxs,
            input_dtypes=input_dtypes,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
        )
        args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)

    # frontend function
    frontend_fn = ivy.functional.frontends.__dict__[frontend].__dict__[fn_tree]

    # check and replace NativeClass object in arguments with ivy counterparts
    convs = {
        "jax": convjax,
        "numpy": convnumpy,
        "tensorflow": convtensor,
        "torch": convtorch,
    }
    if frontend in convs:
        conv = convs[frontend]
        args = ivy.nested_map(args, fn=conv, include_derived=True)
        kwargs = ivy.nested_map(kwargs, fn=conv, include_derived=True)

    # run from the Ivy API directly
    if test_unsupported:
        test_unsupported_function(fn=frontend_fn, args=args, kwargs=kwargs)
        return

    ret = frontend_fn(*args, **kwargs)
    ret = ivy.array(ret) if with_out and not ivy.is_array(ret) else ret
    out = ret
    assert not with_out or not with_inplace, "only one of with_out or with_inplace can be set as True"
    if with_out:
        assert not isinstance(ret, tuple)
        assert ivy.is_array(ret)
        # pass return value to out argument
        # check if passed reference is correctly updated
        kwargs["out"] = out
        ret = frontend_fn(*args, **kwargs)
        if ivy.native_inplace_support:
            assert ret.data is out.data
        assert ret is out
    elif with_inplace:
        assert not isinstance(ret, tuple)
        assert ivy.is_array(ret)
        if "inplace" in inspect.getfullargspec(frontend_fn).args:
            # the function provides optional inplace update
            # set inplace update to be True and check
            # if returned reference is inputted reference
            # and if inputted reference's content is correctly updated
            kwargs["inplace"] = True
            first_array = ivy.func_wrapper._get_first_array(args, kwargs)
            ret = frontend_fn(*args, **kwargs)
            if ivy.native_inplace_support:
                assert ret.data is first_array.data
            assert first_array is ret
        else:
            # the function provides inplace update by default
            # check if returned reference is inputted reference
            first_array = ivy.func_wrapper._get_first_array(args, kwargs)
            if ivy.native_inplace_support:
                assert ret.data is first_array.data
            assert first_array is ret

    # create NumPy args
    args_np = ivy.nested_map(
        args_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )
    kwargs_np = ivy.nested_map(
        kwargs_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )

    # temporarily set frontend framework as backend
    ivy.set_backend(frontend)
    backend_returned_scalar = False
    try:
        # check for unsupported dtypes in frontend framework
        function = getattr(ivy.functional.frontends.__dict__[frontend], fn_tree)
        test_unsupported = check_unsupported_dtype(
            fn=function, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
        )

        # create frontend framework args
        args_frontend = ivy.nested_map(
            args_np,
            lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
        )
        kwargs_frontend = ivy.nested_map(
            kwargs_np,
            lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
        )

        # change ivy dtypes to native dtypes
        if "dtype" in kwargs_frontend:
            kwargs_frontend["dtype"] = ivy.as_native_dtype(kwargs_frontend["dtype"])

        # change ivy device to native devices
        if "device" in kwargs_frontend:
            kwargs_frontend["device"] = ivy.as_native_dev(kwargs_frontend["device"])

        # check and replace the NativeClass objects in arguments with true counterparts
        args_frontend = ivy.nested_map(
            args_frontend, fn=convtrue, include_derived=True, max_depth=10
        )
        kwargs_frontend = ivy.nested_map(
            kwargs_frontend, fn=convtrue, include_derived=True, max_depth=10
        )

        # compute the return via the frontend framework
        frontend_fw = importlib.import_module(".".join([frontend] + frontend_submods))
        if test_unsupported:
            test_unsupported_function(
                fn=frontend_fw.__dict__[fn_tree],
                args=args_frontend,
                kwargs=kwargs_frontend,
            )
            return
        frontend_ret = frontend_fw.__dict__[fn_tree](*args_frontend, **kwargs_frontend)

        if frontend == "numpy" and not isinstance(frontend_ret, np.ndarray):
            backend_returned_scalar = True
            frontend_ret_np_flat = [np.asarray(frontend_ret)]
        else:
            # tuplify the frontend return
            if not isinstance(frontend_ret, tuple):
                frontend_ret = (frontend_ret,)
            frontend_ret_idxs = ivy.nested_argwhere(frontend_ret, ivy.is_native_array)
            frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
            frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]
    except Exception as e:
        ivy.unset_backend()
        raise e
    # unset frontend framework from backend
    ivy.unset_backend()

    if backend_returned_scalar:
        ret_np_flat = ivy.to_numpy([ret])
    else:
        ret_np_flat = flatten_and_to_np(ret=ret)

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, frontend_ret

    # value tests, iterating through each array in the flattened returns
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=rtol,
        atol=atol,
        ground_truth_backend=frontend,
    )


def test_frontend_array_instance_method(
    *,
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: Union[bool, List[bool]],
    with_out: bool,
    num_positional_args: int,
    native_array_flags: Union[bool, List[bool]],
    fw: str,
    frontend: str,
    frontend_class: object,
    fn_tree: str,
    rtol: float = None,
    atol: float = 1e-06,
    test_values: bool = True,
    **all_as_kwargs_np,
):
    """Tests a frontend instance method for the current backend by comparing the
    result with the function in the associated framework.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated
        as an ivy Variable.
    with_out
        if True, the function is also tested with the optional out argument.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    native_array_flags
        dictates whether the corresponding input argument should be treated
        as a native array.
    fw
        current backend (framework).
    frontend
        current frontend (framework).
    frontend_class
        class in the frontend framework.
    fn_tree
        Path to function in frontend framework namespace.
    rtol
        relative tolerance value.
    atol
        absolute tolerance value.
    test_values
        if True, test for the correctness of the resulting values.
    all_as_kwargs_np
        input arguments to the function as keyword arguments.

    Returns
    -------
    ret
        optional, return value from the function
    ret_np
        optional, return value from the Numpy function
    """
    # num_positional_args ignores self, which we need to compensate for
    num_positional_args += 1

    # convert single values to length 1 lists
    input_dtypes, as_variable_flags, native_array_flags = as_lists(
        input_dtypes, as_variable_flags, native_array_flags
    )

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args, kwargs=all_as_kwargs_np
    )

    # extract all arrays from the arguments and keyword arguments
    arg_np_vals, args_idxs, c_arg_vals = _get_nested_np_arrays(args_np)
    kwarg_np_vals, kwargs_idxs, c_kwarg_vals = _get_nested_np_arrays(kwargs_np)

    # make all lists equal in length
    num_arrays = c_arg_vals + c_kwarg_vals
    if len(input_dtypes) < num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(num_arrays)]
    if len(as_variable_flags) < num_arrays:
        as_variable_flags = [as_variable_flags[0] for _ in range(num_arrays)]
    if len(native_array_flags) < num_arrays:
        native_array_flags = [native_array_flags[0] for _ in range(num_arrays)]

    # update variable flags to be compatible with float dtype and with_out args
    as_variable_flags = [
        v if ivy.is_float_dtype(d) and not with_out else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]

    # parse function name and frontend submodules (i.e. jax.lax, jax.numpy etc.)
    *frontend_submods, fn_tree = fn_tree.split(".")

    # check for unsupported dtypes in backend framework
    function = getattr(ivy.functional.frontends.__dict__[frontend], fn_tree)
    test_unsupported = check_unsupported_dtype(
        fn=function, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
    )

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args, kwargs=all_as_kwargs_np
    )

    # create args
    if test_unsupported:
        try:
            args, kwargs, _, _, _ = create_args_kwargs(
                args_np=args_np,
                arg_np_vals=arg_np_vals,
                args_idxs=args_idxs,
                kwargs_np=kwargs_np,
                kwarg_np_vals=kwarg_np_vals,
                kwargs_idxs=kwargs_idxs,
                input_dtypes=input_dtypes,
                as_variable_flags=as_variable_flags,
                native_array_flags=native_array_flags,
            )
            args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)
        except Exception:
            return
    else:
        args, kwargs, _, _, _ = create_args_kwargs(
            args_np=args_np,
            arg_np_vals=arg_np_vals,
            args_idxs=args_idxs,
            kwargs_np=kwargs_np,
            kwarg_np_vals=kwarg_np_vals,
            kwargs_idxs=kwargs_idxs,
            input_dtypes=input_dtypes,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
        )
        args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)

    # get instance array
    if args == []:
        instance_array = list(kwargs.values())[0]
        del kwargs[(list(kwargs.keys())[0])]
    else:
        instance_array = args[0]
        args = args[1:]

    # create class instance
    class_instance = frontend_class(instance_array)

    # frontend function
    fn_name = fn_tree.split(".")[-1]
    frontend_fn = class_instance.__getattribute__(fn_name)

    # run from Ivy API directly
    if test_unsupported:
        test_unsupported_function(fn=frontend_fn, args=args, kwargs=kwargs)
        return

    ret = frontend_fn(*args, **kwargs)
    ret = ivy.array(ret) if with_out and not ivy.is_array(ret) else ret

    # assert idx of return if the idx of the out array provided
    out = ret
    if with_out:
        assert not isinstance(ret, tuple)
        assert ivy.is_array(ret)
        if "out" in kwargs:
            kwargs["out"] = out
            kwargs_ivy["out"] = ivy.asarray(out)  # case where ret is not ivy.array
        else:
            args[ivy.arg_info(frontend_fn, name="out")["idx"]] = out
            args_ivy = list(args_ivy)
            args_ivy[ivy.arg_info(frontend_fn, name="out")["idx"]] = ivy.asarray(
                out
            )  # case where ret is not ivy.array
            args_ivy = tuple(args_ivy)
        ret = frontend_fn(*args, **kwargs)

        if ivy.native_inplace_support:
            # these backends do not always support native inplace updates
            assert ret.data is out.data

    # create NumPy args
    args_np = ivy.nested_map(
        args_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )
    kwargs_np = ivy.nested_map(
        kwargs_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )

    # get instance array
    if args_np == [] or args_np == ():
        instance_np_array = list(kwargs_np.values())[0]
    else:
        instance_np_array = args_np[0]

    # create class instance
    class_instance_np = frontend_class(instance_np_array)

    # frontend function
    frontend_fn_np = class_instance_np.__getattribute__(fn_name)

    # remove self from all_as_kwargs_np
    del all_as_kwargs_np[(list(kwargs_np.keys())[0])]

    # temporarily set frontend framework as backend
    ivy.set_backend(frontend)
    backend_returned_scalar = False
    try:
        # run from Ivy API directly
        test_unsupported = check_unsupported_dtype(
            fn=frontend_fn_np,
            input_dtypes=input_dtypes,
            all_as_kwargs_np=all_as_kwargs_np,
        )

        # create frontend framework args
        args_frontend = ivy.nested_map(
            args_np,
            lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
        )
        kwargs_frontend = ivy.nested_map(
            kwargs_np,
            lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
        )

        # change ivy dtypes to native dtypes
        if "dtype" in kwargs_frontend:
            kwargs_frontend["dtype"] = ivy.as_native_dtype(kwargs_frontend["dtype"])

        # change ivy device to native devices
        if "device" in kwargs_frontend:
            kwargs_frontend["device"] = ivy.as_native_dev(kwargs_frontend["device"])

        # change out argument to ivy array
        if "out" in kwargs_frontend:
            kwargs_frontend["out"] = ivy.asarray(kwargs_frontend["out"])

        # get instance array
        if args_frontend == () or args_frontend == []:
            frontend_instance_array = list(kwargs_frontend.values())[0]
            del kwargs_frontend[(list(kwargs_frontend.keys())[0])]
        else:
            frontend_instance_array = args_frontend[0]
            args_frontend = args_frontend[1:]

        # create class instance
        frontend_class_instance = frontend_class(frontend_instance_array)

        # frontend function
        frontend_fn = frontend_class_instance.__getattribute__(fn_name)

        # return from frontend framework
        if test_unsupported:
            test_unsupported_function(
                fn=frontend_fn, args=args_frontend, kwargs=kwargs_frontend
            )
            return
        frontend_ret = frontend_fn(*args_frontend, **kwargs_frontend)

        if frontend == "numpy" and not isinstance(frontend_ret, np.ndarray):
            backend_returned_scalar = True
            frontend_ret_np_flat = [np.asarray(frontend_ret)]
        else:
            # tuplify the frontend return
            if not isinstance(frontend_ret, tuple):
                frontend_ret = (frontend_ret,)
            frontend_ret_idxs = ivy.nested_argwhere(frontend_ret, ivy.is_array)
            frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
            frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]
    except Exception as e:
        ivy.unset_backend()
        raise e
    # unset frontend framework from backend
    ivy.unset_backend()

    # handle scalar return
    if backend_returned_scalar:
        ret_np_flat = ivy.to_numpy([ret])
    else:
        ret_np_flat = flatten_and_to_np(ret=ret)

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, frontend_ret

    # value tests, iterating through each array in the flattened returns
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=rtol,
        atol=atol,
        ground_truth_backend=frontend,
    )


# Hypothesis #
# -----------#


@st.composite
def array_dtypes(
    draw,
    *,
    num_arrays=st.shared(ints(min_value=1, max_value=4), key="num_arrays"),
    available_dtypes=ivy_np.valid_float_dtypes,
    shared_dtype=False,
):
    """Draws a list of data types.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    num_arrays
        number of data types to be drawn.
    available_dtypes
        universe of available data types.
    shared_dtype
        if True, all data types in the list are same.

    Returns
    -------
    A strategy that draws a list.
    """
    if not isinstance(num_arrays, int):
        num_arrays = draw(num_arrays)
    if num_arrays == 1:
        dtypes = draw(list_of_length(x=st.sampled_from(available_dtypes), length=1))
    elif shared_dtype:
        dtypes = draw(list_of_length(x=st.sampled_from(available_dtypes), length=1))
        dtypes = [dtypes[0] for _ in range(num_arrays)]
    else:
        unwanted_types = set(ivy.all_dtypes).difference(set(available_dtypes))
        pairs = ivy.promotion_table.keys()
        available_dtypes = [
            pair for pair in pairs if not any([d in pair for d in unwanted_types])
        ]
        dtypes = list(draw(st.sampled_from(available_dtypes)))
        if num_arrays > 2:
            dtypes += [dtypes[i % 2] for i in range(num_arrays - 2)]
    return dtypes


@st.composite
def array_bools(
    draw,
    *,
    num_arrays=st.shared(ints(min_value=1, max_value=4), key="num_arrays"),
):
    """Draws a boolean list of a given size.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    num_arrays
        size of the list.

    Returns
    -------
    A strategy that draws a list.
    """
    size = num_arrays if isinstance(num_arrays, int) else draw(num_arrays)
    return draw(st.lists(st.booleans(), min_size=size, max_size=size))


@st.composite
def lists(draw, *, arg, min_size=None, max_size=None, size_bounds=None):
    """Draws a list from the dataset arg.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    arg
        dataset of elements.
    min_size
        least size of the list.
    max_size
        max size of the list.
    size_bounds
        if min_size or max_size is None, draw them randomly from the range
        [size_bounds[0], size_bounds[1]].

    Returns
    -------
    A strategy that draws a list.
    """
    integers = (
        ints(min_value=size_bounds[0], max_value=size_bounds[1])
        if size_bounds
        else ints()
    )
    if isinstance(min_size, str):
        min_size = draw(st.shared(integers, key=min_size))
    if isinstance(max_size, str):
        max_size = draw(st.shared(integers, key=max_size))
    return draw(st.lists(arg, min_size=min_size, max_size=max_size))


@st.composite
def dtype_and_values(
    draw,
    *,
    available_dtypes=ivy_np.valid_dtypes,
    num_arrays=1,
    min_value=None,
    max_value=None,
    large_abs_safety_factor=1.1,
    small_abs_safety_factor=1.1,
    safety_factor_scale="linear",
    allow_inf=False,
    allow_nan=False,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=0,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    shape=None,
    shared_dtype=False,
    ret_shape=False,
    dtype=None,
):
    """Draws a list of arrays with elements from the given corresponding data types.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    available_dtypes
        if dtype is None, data types are drawn from this list randomly.
    num_arrays
        Number of arrays to be drawn.
    min_value
        minimum value of elements in each array.
    max_value
        maximum value of elements in each array.
    large_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,

        when a "linear" safety factor scaler is used,  a safety factor of 2 means
        that only 50% of the range is included, a safety factor of 3 means that
        only 33% of the range is included etc.

        when a "log" safety factor scaler is used, a data type with maximum
        value of 2^32 and a safety factor of 2 transforms the maximum to 2^16.
    small_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,
        this has no effect on integer data types.

        when a "linear" safety factor scaler is used, a data type with minimum
        representable number of 0.0001 and a safety factor of 2 transforms the
        minimum to 0.0002, a safety factor of 3 transforms the minimum to 0.0003 etc.

        when a "log" safety factor scaler is used, a data type with minimum
        representable number of 0.5 * 2^-16 and a safety factor of 2 transforms the
        minimum to 0.5 * 2^-8, a safety factor of 3 transforms the minimum to 0.5 * 2^-4
    safety_factor_scale
        The operation to use when calculating the maximum value of the list. Can be
        "linear" or "log". Default value = "linear".
    allow_inf
        if True, allow inf in the arrays.
    allow_nan
        if True, allow Nans in the arrays.
    exclude_min
        if True, exclude the minimum limit.
    exclude_max
        if True, exclude the maximum limit.
    min_num_dims
        minimum size of the shape tuple.
    max_num_dims
        maximum size of the shape tuple.
    min_dim_size
        minimum value of each integer in the shape tuple.
    max_dim_size
        maximum value of each integer in the shape tuple.
    shape
        shape of the arrays in the list.
    shared_dtype
        if True, if dtype is None, a single shared dtype is drawn for all arrays.
    ret_shape
        if True, the shape of the arrays is also returned.
    dtype
        A list of data types for the given arrays.

    Returns
    -------
    A strategy that draws a list of dtype and arrays (as lists).
    """
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)
    if isinstance(available_dtypes, st._internal.SearchStrategy):
        available_dtypes = draw(available_dtypes)
    if not isinstance(num_arrays, int):
        num_arrays = draw(num_arrays)
    if dtype is None:
        dtype = draw(
            array_dtypes(
                num_arrays=num_arrays,
                available_dtypes=available_dtypes,
                shared_dtype=shared_dtype,
            )
        )
    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            shape = draw(shape)
    else:
        shape = draw(
            st.shared(
                get_shape(
                    min_num_dims=min_num_dims,
                    max_num_dims=max_num_dims,
                    min_dim_size=min_dim_size,
                    max_dim_size=max_dim_size,
                ),
                key="shape",
            )
        )
    values = []
    for i in range(num_arrays):
        values.append(
            draw(
                array_values(
                    dtype=dtype[i],
                    shape=shape,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    allow_nan=allow_nan,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                    large_abs_safety_factor=large_abs_safety_factor,
                    small_abs_safety_factor=small_abs_safety_factor,
                    safety_factor_scale=safety_factor_scale,
                )
            )
        )
    if num_arrays == 1:
        dtype = dtype[0] if isinstance(dtype, list) else dtype
        values = values[0]
    if ret_shape:
        return dtype, values, shape
    return dtype, values


@st.composite
def dtype_values_axis(
    draw,
    *,
    available_dtypes,
    min_value=None,
    max_value=None,
    large_abs_safety_factor=1.1,
    small_abs_safety_factor=1.1,
    safety_factor_scale="linear",
    allow_inf=False,
    allow_nan=False,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=0,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    shape=None,
    shared_dtype=False,
    min_axis=None,
    max_axis=None,
    valid_axis=False,
    allow_neg_axes=True,
    min_axes_size=1,
    max_axes_size=None,
    force_int_axis=False,
    ret_shape=False,
):
    """Draws an array with elements from the given data type, and a random axis of
    the array.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    available_dtypes
        if dtype is None, data type is drawn from this list randomly.
    min_value
        minimum value of elements in the array.
    max_value
        maximum value of elements in the array.
    large_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,

        when a "linear" safety factor scaler is used,  a safety factor of 2 means
        that only 50% of the range is included, a safety factor of 3 means that
        only 33% of the range is included etc.

        when a "log" safety factor scaler is used, a data type with maximum
        value of 2^32 and a safety factor of 2 transforms the maximum to 2^16.
    small_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,
        this has no effect on integer data types.

        when a "linear" safety factor scaler is used, a data type with minimum
        representable number of 0.0001 and a safety factor of 2 transforms the
        minimum to 0.0002, a safety factor of 3 transforms the minimum to 0.0003 etc.

        when a "log" safety factor scaler is used, a data type with minimum
        representable number of 0.5 * 2^-16 and a safety factor of 2 transforms the
        minimum to 0.5 * 2^-8, a safety factor of 3 transforms the minimum to 0.5 * 2^-4
    safety_factor_scale
        The operation to use when calculating the maximum value of the list. Can be
        "linear" or "log". Default value = "linear".
    allow_inf
        if True, allow inf in the array.
    allow_nan
        if True, allow Nans in the arrays.
    exclude_min
        if True, exclude the minimum limit.
    exclude_max
        if True, exclude the maximum limit.
    min_num_dims
        minimum size of the shape tuple.
    max_num_dims
        maximum size of the shape tuple.
    min_dim_size
        minimum value of each integer in the shape tuple.
    max_dim_size
        maximum value of each integer in the shape tuple.
    valid_axis
        if True, a valid axis will be drawn from the array dimensions.
    allow_neg_axes
        if True, returned axes may include negative axes.
    min_axes_size
        minimum size of the axis tuple.
    max_axes_size
        maximum size of the axis tuple.
    force_int_axis
        if True, and only one axis is drawn, the returned axis will be an integer.
    shape
        shape of the array. if None, a random shape is drawn.
    shared_dtype
        if True, if dtype is None, a single shared dtype is drawn for all arrays.
    min_axis
        if shape is None, axis is drawn from the range [min_axis, max_axis].
    max_axis
        if shape is None, axis is drawn from the range [min_axis, max_axis].
    ret_shape
        if True, the shape of the arrays is also returned.

    Returns
    -------
    A strategy that draws a dtype, an array (as list), and an axis.
    """
    results = draw(
        dtype_and_values(
            available_dtypes=available_dtypes,
            min_value=min_value,
            max_value=max_value,
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale=safety_factor_scale,
            allow_inf=allow_inf,
            allow_nan=allow_nan,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
            shape=shape,
            shared_dtype=shared_dtype,
            ret_shape=True,
        )
    )
    dtype, values, arr_shape = results
    if valid_axis or shape:
        if not isinstance(values, list):
            axis = None
        else:
            axis = draw(
                get_axis(
                    shape=arr_shape,
                    min_size=min_axes_size,
                    max_size=max_axes_size,
                    allow_neg=allow_neg_axes,
                    force_int=force_int_axis,
                )
            )
    else:
        axis = draw(ints(min_value=min_axis, max_value=max_axis))
    if ret_shape:
        return dtype, values, axis, shape
    return dtype, values, axis


# taken from
# https://github.com/data-apis/array-api-tests/array_api_tests/test_manipulation_functions.py
@st.composite
def reshape_shapes(draw, *, shape):
    """Draws a random shape with the same number of elements as the given shape.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    shape
        list/strategy/tuple of integers representing an array shape.

    Returns
    -------
    A strategy that draws a tuple.
    """
    if isinstance(shape, st._internal.SearchStrategy):
        shape = draw(shape)
    size = 1 if len(shape) == 0 else math.prod(shape)
    rshape = draw(st.lists(ints(min_value=0)).filter(lambda s: math.prod(s) == size))
    # assume(all(side <= MAX_SIDE for side in rshape))
    if len(rshape) != 0 and size > 0 and draw(st.booleans()):
        index = draw(ints(min_value=0, max_value=len(rshape) - 1))
        rshape[index] = -1
    return tuple(rshape)


# taken from https://github.com/HypothesisWorks/hypothesis/issues/1115
@st.composite
def subsets(draw, *, elements):
    """Draws a subset of elements from the given elements.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    elements
        set of elements to be drawn from.

    Returns
    -------
    A strategy that draws a subset of elements.
    """
    return tuple(e for e in elements if draw(st.booleans()))


@st.composite
def array_n_indices_n_axis(
    draw,
    *,
    array_dtypes,
    indices_dtypes=ivy_np.valid_int_dtypes,
    disable_random_axis=False,
    boolean_mask=False,
    allow_inf=False,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    first_dimension_only=False,
):
    """Generates two arrays x & indices, the values in the indices array are indices
    of the array x. Draws an integers randomly from the minimum and maximum number of
    positional arguments a given function can take.

    Parameters
    ----------
    array_dtypes
        list of data type to draw the array dtype from.
    indices_dtypes
        list of data type to draw the indices dtype from.
    disable_random_axis
        axis is set to -1 when True. Randomly generated with hypothesis if False.
    allow_inf
        inf values are allowed to be generated in the values array when True.
    min_num_dims
        The minimum number of dimensions the arrays can have.
    max_num_dims
        The maximum number of dimensions the arrays can have.
    min_dim_size
        The minimum size of the dimensions of the arrays.
    max_dim_size
        The maximum size of the dimensions of the arrays.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator
    which generates arrays of values and indices.

    Examples
    --------
    @given(
        array_n_indices_n_axis=array_n_indices_n_axis(
            array_dtypes=helpers.get_dtypes("valid"),
            indices_dtypes=helpers.get_dtypes("integer"),
            boolean_mask=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10
            )
    )
    """
    x_dtype, x, x_shape = draw(
        dtype_and_values(
            available_dtypes=array_dtypes,
            allow_inf=allow_inf,
            ret_shape=True,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    if disable_random_axis:
        axis = -1
    else:
        axis = draw(
            ints(
                min_value=-1 * len(x_shape),
                max_value=len(x_shape) - 1,
            )
        )
    if boolean_mask:
        indices_dtype, indices = draw(
            dtype_and_values(
                dtype=["bool"],
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            )
        )
    else:
        max_axis = max(x_shape[axis] - 1, 0)
        if first_dimension_only:
            max_axis = max(x_shape[0] - 1, 0)
        indices_dtype, indices = draw(
            dtype_and_values(
                available_dtypes=indices_dtypes,
                allow_inf=False,
                min_value=0,
                max_value=max_axis,
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            )
        )
    return [x_dtype, indices_dtype], x, indices, axis


def _zeroing_and_casting(x, cast_type):
    # covnert -0.0 to 0.0
    if x == 0.0:
        return 0.0
    x = float(np.array(x).astype(cast_type)) if x else None
    return x


def _clamp_value(x, dtype_info):
    if x > dtype_info.max:
        return dtype_info.max
    if x < dtype_info.min:
        return dtype_info.min
    return x


@st.composite
def array_values(
    draw,
    *,
    dtype,
    shape,
    min_value=None,
    max_value=None,
    allow_nan=False,
    allow_subnormal=False,
    allow_inf=False,
    exclude_min=True,
    exclude_max=True,
    large_abs_safety_factor=1.1,
    small_abs_safety_factor=1.1,
    safety_factor_scale="linear",
):
    """Draws a list (of lists) of a given shape containing values of a given data type.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    dtype
        data type of the elements of the list.
    shape
        shape of the required list.
    min_value
        minimum value of elements in the list.
    max_value
        maximum value of elements in the list.
    allow_nan
        if True, allow Nans in the list.
    allow_subnormal
        if True, allow subnormals in the list.
    allow_inf
        if True, allow inf in the list.
    exclude_min
        if True, exclude the minimum limit.
    exclude_max
        if True, exclude the maximum limit.
    large_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,

        when a "linear" safety factor scaler is used,  a safety factor of 2 means
        that only 50% of the range is included, a safety factor of 3 means that
        only 33% of the range is included etc.

        when a "log" safety factor scaler is used, a data type with maximum
        value of 2^32 and a safety factor of 2 transforms the maximum to 2^16.
    small_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,
        this has no effect on integer data types.

        when a "linear" safety factor scaler is used, a data type with minimum
        representable number of 0.0001 and a safety factor of 2 transforms the
        minimum to 0.0002, a safety factor of 3 transforms the minimum to 0.0003 etc.

        when a "log" safety factor scaler is used, a data type with minimum
        representable number of 0.5 * 2^-16 and a safety factor of 2 transforms the
        minimum to 0.5 * 2^-8, a safety factor of 3 transforms the minimum to 0.5 * 2^-4
    safety_factor_scale
        The operation to use when calculating the maximum value of the list. Can be
        "linear" or "log". Default value = "linear".

    In the case of min_value or max_value is not in the valid range
    the invalid value will be replaced by data type limit, the range
    of the numbers in that case is not preserved.

    Returns
    -------
        A strategy that draws a list.
    """
    assert small_abs_safety_factor >= 1, "small_abs_safety_factor must be >= 1"
    assert large_abs_safety_factor >= 1, "large_value_safety_factor must be >= 1"

    size = 1
    if isinstance(shape, int):
        size = shape
    else:
        for dim in shape:
            size *= dim

    if isinstance(dtype, st._internal.SearchStrategy):
        dtype = draw(dtype)
        dtype = dtype[0] if isinstance(dtype, list) else draw(dtype)

    if "float" in dtype:
        kind_dtype = "float"
        dtype_info = ivy.finfo(dtype)
    elif "int" in dtype:
        kind_dtype = "int"
        dtype_info = ivy.iinfo(dtype)
    elif "bool" in dtype:
        kind_dtype = "bool"
    else:
        raise TypeError(
            f"{dtype} is not a valid data type that can be generated,"
            f" only integers, floats and booleans are allowed."
        )

    if kind_dtype != "bool":
        if min_value is None:
            min_value = dtype_info.min
            b_scale_min = True
        else:
            min_value = _clamp_value(min_value, dtype_info)
            b_scale_min = False

        if max_value is None:
            max_value = dtype_info.max
            b_scale_max = True
        else:
            max_value = _clamp_value(max_value, dtype_info)
            b_scale_max = False

        assert max_value >= min_value

        # Scale the values
        if safety_factor_scale == "linear":
            if b_scale_min:
                min_value = min_value / large_abs_safety_factor
            if b_scale_max:
                max_value = max_value / large_abs_safety_factor
            if kind_dtype == "float":
                abs_smallest_val = dtype_info.smallest_normal * small_abs_safety_factor
        elif safety_factor_scale == "log":
            if b_scale_min:
                min_sign = math.copysign(1, min_value)
                min_value = abs(min_value) ** (1 / large_abs_safety_factor) * min_sign
            if b_scale_max:
                max_sign = math.copysign(1, max_value)
                max_value = abs(max_value) ** (1 / large_abs_safety_factor) * max_sign
            if kind_dtype == "float":
                m, e = math.frexp(dtype_info.smallest_normal)
                abs_smallest_val = m * (2 ** (e / small_abs_safety_factor))
        else:
            raise ValueError(
                f"{safety_factor_scale} is not a valid safety factor scale."
                f" use 'log' or 'linear'."
            )

        if kind_dtype == "int":
            if exclude_min:
                min_value += 1
            if exclude_max:
                max_value -= 1
            values = draw(
                list_of_length(
                    x=st.integers(int(min_value), int(max_value)), length=size
                )
            )
        elif kind_dtype == "float":
            floats_info = {
                "float16": {"cast_type": "float16", "width": 16},
                "bfloat16": {"cast_type": "float32", "width": 32},
                "float32": {"cast_type": "float32", "width": 32},
                "float64": {"cast_type": "float64", "width": 64},
            }
            # The smallest possible value is determined by one of the arguments
            if min_value > -abs_smallest_val or max_value < abs_smallest_val:
                float_strategy = st.floats(
                    # Using np.array to assert that value
                    # can be represented of compatible width.
                    min_value=np.array(
                        min_value, dtype=floats_info[dtype]["cast_type"]
                    ).tolist(),
                    max_value=np.array(
                        max_value, dtype=floats_info[dtype]["cast_type"]
                    ).tolist(),
                    allow_nan=allow_nan,
                    allow_subnormal=allow_subnormal,
                    allow_infinity=allow_inf,
                    width=floats_info[dtype]["width"],
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                )
            else:
                float_strategy = st.one_of(
                    st.floats(
                        min_value=np.array(
                            min_value, dtype=floats_info[dtype]["cast_type"]
                        ).tolist(),
                        max_value=np.array(
                            -abs_smallest_val, dtype=floats_info[dtype]["cast_type"]
                        ).tolist(),
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=floats_info[dtype]["width"],
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                    st.floats(
                        min_value=np.array(
                            abs_smallest_val, dtype=floats_info[dtype]["cast_type"]
                        ).tolist(),
                        max_value=np.array(
                            max_value, dtype=floats_info[dtype]["cast_type"]
                        ).tolist(),
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=floats_info[dtype]["width"],
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                )
            values = draw(
                list_of_length(
                    x=float_strategy,
                    length=size,
                )
            )
    else:
        values = draw(list_of_length(x=st.booleans(), length=size))

    array = np.array(values)
    if isinstance(shape, (tuple, list)):
        array = array.reshape(shape)
    return array.tolist()


@st.composite
def get_shape(
    draw,
    *,
    allow_none=False,
    min_num_dims=0,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
):
    """Draws a tuple of integers drawn randomly from [min_dim_size, max_dim_size]
     of size drawn from min_num_dims to max_num_dims. Useful for randomly
     drawing the shape of an array.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    allow_none
        if True, allow for the result to be None.
    min_num_dims
        minimum size of the tuple.
    max_num_dims
        maximum size of the tuple.
    min_dim_size
        minimum value of each integer in the tuple.
    max_dim_size
        maximum value of each integer in the tuple.

    Returns
    -------
    A strategy that draws a tuple.
    """
    if allow_none:
        shape = draw(
            st.none()
            | st.lists(
                ints(min_value=min_dim_size, max_value=max_dim_size),
                min_size=min_num_dims,
                max_size=max_num_dims,
            )
        )
    else:
        shape = draw(
            st.lists(
                ints(min_value=min_dim_size, max_value=max_dim_size),
                min_size=min_num_dims,
                max_size=max_num_dims,
            )
        )
    if shape is None:
        return shape
    return tuple(shape)


def none_or_list_of_floats(
    *,
    dtype,
    size,
    min_value=None,
    max_value=None,
    exclude_min=False,
    exclude_max=False,
    no_none=False,
):
    """Draws a List containing Nones or Floats.

    Parameters
    ----------
    dtype
        float data type ('float16', 'float32', or 'float64').
    size
        size of the list required.
    min_value
        lower bound for values in the list
    max_value
        upper bound for values in the list
    exclude_min
        if True, exclude the min_value
    exclude_max
        if True, exclude the max_value
    no_none
        if True, List does not contains None

    Returns
    -------
    A strategy that draws a List containing Nones or Floats.
    """
    if no_none:
        if dtype == "float16":
            values = list_of_length(
                x=floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=16,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
        elif dtype == "float32":
            values = list_of_length(
                x=st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=32,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
        elif dtype == "float64":
            values = list_of_length(
                x=st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=64,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
    else:
        if dtype == "float16":
            values = list_of_length(
                x=st.none()
                | st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=16,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
        elif dtype == "float32":
            values = list_of_length(
                x=st.none()
                | st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=32,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
        elif dtype == "float64":
            values = list_of_length(
                x=st.none()
                | st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=64,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
    return values


@st.composite
def get_mean_std(draw, *, dtype):
    """Draws two integers representing the mean and standard deviation for a given data
    type.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    dtype
        data type.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.
    """
    values = draw(none_or_list_of_floats(dtype=dtype, size=2))
    values[1] = abs(values[1]) if values[1] else None
    return values[0], values[1]


@st.composite
def get_bounds(draw, *, dtype):
    """Draws two integers low, high for a given data type such that low < high.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    dtype
        data type.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.
    """
    if "int" in dtype:
        values = draw(array_values(dtype=dtype, shape=2))
        values[0], values[1] = abs(values[0]), abs(values[1])
        low, high = min(values), max(values)
        if low == high:
            return draw(get_bounds(dtype=dtype))
    else:
        values = draw(none_or_list_of_floats(dtype=dtype, size=2))
        if values[0] is not None and values[1] is not None:
            low, high = min(values), max(values)
        else:
            low, high = values[0], values[1]
        if ivy.default(low, 0.0) >= ivy.default(high, 1.0):
            return draw(get_bounds(dtype=dtype))
    return low, high


@st.composite
def get_axis(
    draw,
    *,
    shape,
    allow_neg=True,
    allow_none=False,
    sorted=True,
    unique=True,
    min_size=1,
    max_size=None,
    force_tuple=False,
    force_int=False,
):
    """Draws one or more axis for the given shape.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    shape
        shape of the array as a tuple, or a hypothesis strategy from which the shape
        will be drawn
    allow_neg
        boolean; if True, allow negative axes to be drawn
    allow_none
        boolean; if True, allow None to be drawn
    sorted
        boolean; if True, and a tuple of axes is drawn, tuple is sorted in increasing
        fashion
    unique
        boolean; if True, and a tuple of axes is drawn, all axes drawn will be unique
    min_size
        int or hypothesis strategy; if a tuple of axes is drawn, the minimum number of
        axes drawn
    max_size
        int or hypothesis strategy; if a tuple of axes is drawn, the maximum number of
        axes drawn.
        If None and unique is True, then it is set to the number of axes in the shape
    force_tuple
        boolean, if true, all axis will be returned as a tuple. If force_tuple and
        force_int are true, then an AssertionError is raised
    force_int
        boolean, if true, all axis will be returned as an int. If force_tuple and
        force_int are true, then an AssertionError is raised

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.
    """
    assert not (force_int and force_tuple), (
        "Cannot return an int and a tuple. If "
        "both are valid then set 'force_int' "
        "and 'force_tuple' to False."
    )

    # Draw values from any strategies given
    if isinstance(shape, st._internal.SearchStrategy):
        shape = draw(shape)
    if isinstance(min_size, st._internal.SearchStrategy):
        min_size = draw(min_size)
    if isinstance(max_size, st._internal.SearchStrategy):
        max_size = draw(max_size)

    axes = len(shape)
    lower_axes_bound = axes if allow_neg else 0
    unique_by = (lambda x: shape[x]) if unique else None

    if max_size is None and unique:
        max_size = max(axes, min_size)

    valid_strategies = []

    if allow_none:
        valid_strategies.append(st.none())

    if not force_tuple:
        if axes == 0:
            valid_strategies.append(st.just(0))
        else:
            valid_strategies.append(st.integers(-lower_axes_bound, axes - 1))
    if not force_int:
        if axes == 0:
            valid_strategies.append(
                st.lists(st.just(0), min_size=min_size, max_size=max_size)
            )
        else:
            valid_strategies.append(
                st.lists(
                    st.integers(-lower_axes_bound, axes - 1),
                    min_size=min_size,
                    max_size=max_size,
                    unique_by=unique_by,
                )
            )

    axis = draw(st.one_of(*valid_strategies))

    if type(axis) == list:
        if sorted:

            def sort_key(ele, max_len):
                if ele < 0:
                    return ele + max_len - 1
                return ele

            axis.sort(key=(lambda ele: sort_key(ele, axes)))
        axis = tuple(axis)
    return axis


@st.composite
def num_positional_args(draw, *, fn_name: str = None):
    """Draws an integers randomly from the minimum and maximum number of positional
    arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    fn_name
        name of the function.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.

    Examples
    --------
    @given(
        num_positional_args=num_positional_args(fn_name="floor_divide")
    )
    @given(
        num_positional_args=num_positional_args(fn_name="add")
    )
    """
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    fn = None
    for i, fn_name_key in enumerate(fn_name.split(".")):
        if i == 0:
            fn = ivy.__dict__[fn_name_key]
        else:
            fn = fn.__dict__[fn_name_key]
    for param in inspect.signature(fn).parameters.values():
        if param.name == "self":
            continue
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind == param.KEYWORD_ONLY:
            num_keyword_only += 1
        elif param.kind == param.VAR_KEYWORD:
            num_keyword_only += 1
    return draw(
        ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )


@st.composite
def num_positional_args_from_fn(draw, *, fn: str = None):
    """Draws an integers randomly from the minimum and maximum number of positional
    arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    fn
        name of the function.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.

    Examples
    --------
    @given(
        num_positional_args=num_positional_args_from_fn(fn="floor_divide")
    )
    @given(
        num_positional_args=num_positional_args_from_fn(fn="add")
    )
    """
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    for param in inspect.signature(fn).parameters.values():
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind == param.KEYWORD_ONLY:
            num_keyword_only += 1
        elif param.kind == param.VAR_KEYWORD:
            num_keyword_only += 1
    return draw(
        ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )


@st.composite
def bool_val_flags(draw, cl_arg: Union[bool, None]):
    if cl_arg is not None:
        return draw(st.booleans().filter(lambda x: x == cl_arg))
    return draw(st.booleans())


def handle_cmd_line_args(test_fn):
    from ivy_tests.test_ivy.conftest import (
        FW_STRS,
        TEST_BACKENDS,
    )

    # first[1:-2] 5 arguments are all fixtures
    @given(data=st.data())
    @settings(max_examples=1)
    def new_fn(data, get_command_line_flags, device, f, fw, *args, **kwargs):
        gc.collect()
        flag, backend_string = (False, "")
        # skip test if device is gpu and backend is numpy
        if "gpu" in device and ivy.current_backend_str() == "numpy":
            # Numpy does not support GPU
            pytest.skip()
        if not f:
            # randomly draw a backend if not set
            backend_string = data.draw(st.sampled_from(FW_STRS))
            f = TEST_BACKENDS[backend_string]()
        else:
            # use the one which is parametrized
            flag = True

        global frontend_fw
        # Reset the global variable,
        # only set if frontend fw is inferred
        frontend_fw = None
        full_fn_test_path = test_fn.__module__.split(".")
        if len(full_fn_test_path) > 2:
            if full_fn_test_path[2] == "test_frontends":
                frontend_fw = TEST_BACKENDS[full_fn_test_path[3][5:]]

        # set backend using the context manager
        with f.use:
            # inspecting for keyword arguments in test function
            for param in inspect.signature(test_fn).parameters.values():
                if param.name in cmd_line_args:
                    kwargs[param.name] = data.draw(
                        bool_val_flags(get_command_line_flags[param.name])
                    )
                elif param.name == "fw":
                    kwargs["fw"] = fw if flag else backend_string
                elif param.name == "device":
                    kwargs["device"] = device
            return test_fn(*args, **kwargs)

    return new_fn


def gradient_incompatible_function(*, fn):
    return (
        not ivy.supports_gradients
        and hasattr(fn, "computes_gradients")
        and fn.computes_gradients
    )


@st.composite
def seed(draw):
    return draw(st.integers(min_value=0, max_value=2**8 - 1))


@st.composite
def arrays_and_axes(
    draw,
    allow_none=False,
    min_num_dims=0,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    num=2,
):
    shapes = list()
    for _ in range(num):
        shape = draw(
            get_shape(
                allow_none=False,
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            )
        )
        shapes.append(shape)
    arrays = list()
    for shape in shapes:
        arrays.append(
            draw(
                array_values(dtype="int32", shape=shape, min_value=-200, max_value=200)
            )
        )
    all_axes_ranges = list()
    for shape in shapes:
        if None in all_axes_ranges:
            all_axes_ranges.append(st.integers(0, len(shape) - 1))
        else:
            all_axes_ranges.append(st.one_of(st.none(), st.integers(0, len(shape) - 1)))
    axes = draw(st.tuples(*all_axes_ranges))
    return arrays, axes


@st.composite
def x_and_filters(draw, dim: int = 2, transpose: bool = False, depthwise=False):
    strides = draw(st.integers(min_value=1, max_value=2))
    padding = draw(st.sampled_from(["SAME", "VALID"]))
    batch_size = draw(st.integers(1, 5))
    filter_shape = draw(
        get_shape(min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5)
    )
    input_channels = draw(st.integers(1, 5))
    output_channels = draw(st.integers(1, 5))
    dilations = draw(st.integers(1, 2))
    dtype = draw(get_dtypes("float", full=False))
    if dim == 2:
        data_format = draw(st.sampled_from(["NCHW"]))
    elif dim == 1:
        data_format = draw(st.sampled_from(["NWC", "NCW"]))
    else:
        data_format = draw(st.sampled_from(["NDHWC", "NCDHW"]))

    x_dim = []
    if transpose:
        output_shape = []
        x_dim = draw(
            get_shape(
                min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=20
            )
        )
        for i in range(dim):
            output_shape.append(
                ivy.deconv_length(
                    x_dim[i], strides, filter_shape[i], padding, dilations
                )
            )
    else:
        for i in range(dim):
            min_x = filter_shape[i] + (filter_shape[i] - 1) * (dilations - 1)
            x_dim.append(draw(st.integers(min_x, 100)))
        x_dim = tuple(x_dim)
    if not depthwise:
        filter_shape = filter_shape + (input_channels, output_channels)
    else:
        filter_shape = filter_shape + (input_channels,)
    if data_format == "NHWC" or data_format == "NWC" or data_format == "NDHWC":
        x_shape = (batch_size,) + x_dim + (input_channels,)
    else:
        x_shape = (batch_size, input_channels) + x_dim
    vals = draw(
        array_values(
            dtype=dtype,
            shape=x_shape,
            large_abs_safety_factor=3,
            small_abs_safety_factor=4,
            safety_factor_scale="log",
        )
    )
    filters = draw(
        array_values(
            dtype=dtype,
            shape=filter_shape,
            large_abs_safety_factor=3,
            small_abs_safety_factor=4,
            safety_factor_scale="log",
        )
    )
    if transpose:
        return (
            dtype,
            vals,
            filters,
            dilations,
            data_format,
            strides,
            padding,
            output_shape,
        )
    return dtype, vals, filters, dilations, data_format, strides, padding


#      From array-api repo     #
# ---------------------------- #


def _broadcast_shapes(shape1, shape2):
    """Broadcasts `shape1` and `shape2`"""
    N1 = len(shape1)
    N2 = len(shape2)
    N = max(N1, N2)
    shape = [None for _ in range(N)]
    i = N - 1
    while i >= 0:
        n1 = N1 - N + i
        if N1 - N + i >= 0:
            d1 = shape1[n1]
        else:
            d1 = 1
        n2 = N2 - N + i
        if N2 - N + i >= 0:
            d2 = shape2[n2]
        else:
            d2 = 1

        if d1 == 1:
            shape[i] = d2
        elif d2 == 1:
            shape[i] = d1
        elif d1 == d2:
            shape[i] = d1
        else:
            raise Exception("Broadcast error")

        i = i - 1

    return tuple(shape)


# from array-api repo
def broadcast_shapes(*shapes):
    if len(shapes) == 0:
        raise ValueError("shapes=[] must be non-empty")
    elif len(shapes) == 1:
        return shapes[0]
    result = _broadcast_shapes(shapes[0], shapes[1])
    for i in range(2, len(shapes)):
        result = _broadcast_shapes(result, shapes[i])
    return result


# np.prod and others have overflow and math.prod is Python 3.8+ only
def prod(seq):
    return reduce(mul, seq, 1)


# from array-api repo
def mutually_broadcastable_shapes(
    num_shapes: int,
    *,
    base_shape=(),
    min_dims: int = 1,
    max_dims: int = 4,
    min_side: int = 1,
    max_side: int = 4,
):
    if max_dims is None:
        max_dims = min(max(len(base_shape), min_dims) + 5, 32)
    if max_side is None:
        max_side = max(base_shape[-max_dims:] + (min_side,)) + 5
    return (
        nph.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape,
            min_dims=min_dims,
            max_dims=max_dims,
            min_side=min_side,
            max_side=max_side,
        )
        .map(lambda BS: BS.input_shapes)
        .filter(lambda shapes: all(prod(i for i in s if i > 0) < 1000 for s in shapes))
    )


@st.composite
def array_and_broadcastable_shape(draw, dtype):
    """Returns an array and a shape that the array can be broadcast to"""
    in_shape = draw(nph.array_shapes(min_dims=1, max_dims=4))
    x = draw(nph.arrays(shape=in_shape, dtype=dtype))
    to_shape = draw(
        mutually_broadcastable_shapes(1, base_shape=in_shape)
        .map(lambda S: S[0])
        .filter(lambda s: broadcast_shapes(in_shape, s) == s),
        label="shape",
    )
    return x, to_shape
