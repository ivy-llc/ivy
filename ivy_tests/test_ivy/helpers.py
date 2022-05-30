"""Collection of helpers for ivy unit tests."""

# global
from contextlib import redirect_stdout
from io import StringIO
import sys
import re

import numpy as np
import math
from numpy import array_api as xp
from hypothesis.extra.array_api import make_strategies_namespace

xps = make_strategies_namespace(xp)


try:
    import jax.numpy as _jnp
except ImportError:
    _jnp = None
try:
    import tensorflow as _tf

    _tf_version = float(".".join(_tf.__version__.split(".")[0:2]))
    if _tf_version >= 2.3:
        # noinspection PyPep8Naming,PyUnresolvedReferences
        from tensorflow.python.types.core import Tensor as tensor_type
    else:
        # noinspection PyPep8Naming
        # noinspection PyProtectedMember,PyUnresolvedReferences
        from tensorflow.python.framework.tensor_like import _TensorLike as tensor_type
    physical_devices = _tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        _tf.config.experimental.set_memory_growth(device, True)
except ImportError:
    _tf = None
try:
    import torch as _torch
except ImportError:
    _torch = None
try:
    import mxnet as _mx
    import mxnet.ndarray as _mx_nd
except ImportError:
    _mx = None
    _mx_nd = None
from hypothesis import strategies as st
import hypothesis.extra.numpy as nph

# local
import ivy
import ivy.functional.backends.numpy as ivy_np


def get_ivy_numpy():
    try:
        import ivy.functional.backends.numpy
    except ImportError:
        return None
    return ivy.functional.backends.numpy


def get_ivy_jax():
    try:
        import ivy.functional.backends.jax
    except ImportError:
        return None
    return ivy.functional.backends.jax


def get_ivy_tensorflow():
    try:
        import ivy.functional.backends.tensorflow
    except ImportError:
        return None
    return ivy.functional.backends.tensorflow


def get_ivy_torch():
    try:
        import ivy.functional.backends.torch
    except ImportError:
        return None
    return ivy.functional.backends.torch


def get_ivy_mxnet():
    try:
        import ivy.functional.backends.mxnet
    except ImportError:
        return None
    return ivy.functional.backends.mxnet


_ivy_fws_dict = {
    "numpy": lambda: get_ivy_numpy(),
    "jax": lambda: get_ivy_jax(),
    "tensorflow": lambda: get_ivy_tensorflow(),
    "tensorflow_graph": lambda: get_ivy_tensorflow(),
    "torch": lambda: get_ivy_torch(),
    "mxnet": lambda: get_ivy_mxnet(),
}

_iterable_types = [list, tuple, dict]
_excluded = []


def _convert_vars(
    vars_in, from_type, to_type_callable=None, keep_other=True, to_type=None
):
    new_vars = list()
    for var in vars_in:
        if type(var) in _iterable_types:
            return_val = _convert_vars(var, from_type, to_type_callable)
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


def np_call(func, *args, **kwargs):
    ret = func(*args, **kwargs)
    if isinstance(ret, (list, tuple)):
        return ivy.to_native(ret, nested=True)
    return ivy.to_numpy(ret)


def jnp_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _jnp.asarray)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, _jnp.asarray)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (_jnp.ndarray, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (_jnp.ndarray, ivy.Array), ivy.to_numpy)[0]


def tf_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _tf.convert_to_tensor)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, _tf.convert_to_tensor)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (tensor_type, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (tensor_type, ivy.Array), ivy.to_numpy)[0]


def tf_graph_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _tf.convert_to_tensor)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, _tf.convert_to_tensor)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))

    @_tf.function
    def tf_func(*local_args, **local_kwargs):
        return func(*local_args, **local_kwargs)

    output = tf_func(*new_args, **new_kwargs)

    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (tensor_type, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (tensor_type, ivy.Array), ivy.to_numpy)[0]


def torch_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _torch.from_numpy)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, _torch.from_numpy)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (_torch.Tensor, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (_torch.Tensor, ivy.Array), ivy.to_numpy)[0]


def mx_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _mx_nd.array)
    new_kw_items = _convert_vars(kwargs.values(), np.ndarray, _mx_nd.array)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_items))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(
            _convert_vars(output, (_mx_nd.ndarray.NDArray, ivy.Array), ivy.to_numpy)
        )
    else:
        return _convert_vars(
            [output], (_mx_nd.ndarray.NDArray, ivy.Array), ivy.to_numpy
        )[0]


_calls = [np_call, jnp_call, tf_call, tf_graph_call, torch_call, mx_call]


def assert_compilable(fn):
    try:
        ivy.compile(fn)
    except Exception as e:
        raise e


# function that trims white spaces from docstrings
def trim(docstring):
    """Trim function from PEP-257"""
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    # Current code/unittests expects a line return at
    # end of multiline docstrings
    # workaround expected behavior from unittests
    if "\n" in docstring:
        trimmed.append("")

    # Return a single string:
    return "\n".join(trimmed)


def docstring_examples_run(fn):
    if not hasattr(fn, "__name__"):
        return True
    fn_name = fn.__name__
    if fn_name not in ivy.backend_handler.ivy_original_dict:
        return True

    docstring = ivy.backend_handler.ivy_original_dict[fn_name].__doc__

    if docstring is None:
        return True

    # removing extra new lines and trailing white spaces from the docstrings
    trimmed_docstring = trim(docstring)
    trimmed_docstring = trimmed_docstring.split("\n")

    # end_index: -1, if print statement is not found in the docstring
    end_index = -1

    # parsed_output is set as an empty string to manage functions with multiple inputs
    parsed_output = ""

    # parsing through the docstrings to find lines with print statement
    # following which is our parsed output
    sub = ">>> print("
    for index, line in enumerate(trimmed_docstring):
        if sub in line:
            end_index = trimmed_docstring.index("", index)
            p_output = trimmed_docstring[index + 1 : end_index]
            p_output = ("").join(p_output).replace(" ", "")
            parsed_output += p_output

    if end_index == -1:
        return True

    executable_lines = [
        line.split(">>>")[1][1:] for line in docstring.split("\n") if ">>>" in line
    ]

    # noinspection PyBroadException
    f = StringIO()
    with redirect_stdout(f):
        for line in executable_lines:
            try:
                exec(line)
            except RuntimeError:
                raise Exception("ERROR EXECUTING FUNCTION IN DOCSTRING")

    output = f.getvalue()
    output = output.rstrip()
    output = output.replace(" ", "").replace("\n", "")

    # handling cases when the stdout contains ANSI colour codes
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(
        r"""
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
    """,
        re.VERBOSE,
    )

    output = ansi_escape.sub("", output)

    print("Output: ", output)
    print("Putput: ", parsed_output)

    assert output == parsed_output, "Output is unequal to the docstrings output."
    return True


def var_fn(a, b=None, c=None, dtype=None):
    return ivy.variable(ivy.array(a, b, c))


def exclude(exclusion_list):
    global _excluded
    _excluded += list(set(exclusion_list) - set(_excluded))


def frameworks():
    return list(
        set(
            [
                ivy_fw()
                for fw_str, ivy_fw in _ivy_fws_dict.items()
                if ivy_fw() is not None and fw_str not in _excluded
            ]
        )
    )


def calls():
    return [
        call
        for (fw_str, ivy_fw), call in zip(_ivy_fws_dict.items(), _calls)
        if ivy_fw() is not None and fw_str not in _excluded
    ]


def f_n_calls():
    return [
        (ivy_fw(), call)
        for (fw_str, ivy_fw), call in zip(_ivy_fws_dict.items(), _calls)
        if ivy_fw() is not None and fw_str not in _excluded
    ]


def assert_all_close(x, y, rtol=1e-05, atol=1e-08):
    if ivy.is_ivy_container(x) and ivy.is_ivy_container(y):
        ivy.Container.multi_map(assert_all_close, [x, y])
    else:
        assert np.allclose(
            np.nan_to_num(x), np.nan_to_num(y), rtol=rtol, atol=atol
        ), "{} != {}".format(x, y)


def kwargs_to_args_n_kwargs(num_positional_args, kwargs):
    args = [v for v in list(kwargs.values())[:num_positional_args]]
    kwargs = {k: kwargs[k] for k in list(kwargs.keys())[num_positional_args:]}
    return args, kwargs


def list_of_length(x, length):
    return st.lists(x, min_size=length, max_size=length)


def as_cont(x):
    return ivy.Container({"a": x, "b": {"c": x, "d": x}})


def as_lists(dtype, as_variable, native_array, container):
    if not isinstance(dtype, list):
        dtype = [dtype]
    if not isinstance(as_variable, list):
        as_variable = [as_variable]
    if not isinstance(native_array, list):
        native_array = [native_array]
    if not isinstance(container, list):
        container = [container]
    return dtype, as_variable, native_array, container


def test_array_function(
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    fn_name,
    rtol=1e-03,
    atol=1e-06,
    **all_as_kwargs_np
):

    # convert single values to length 1 lists
    dtype, as_variable, native_array, container = as_lists(
        dtype, as_variable, native_array, container
    )

    # update variable flags to be compatible with float dtype and with_out args
    as_variable = [
        v if ivy.is_float_dtype(d) and not with_out else False
        for v, d in zip(as_variable, dtype)
    ]

    # update instance_method flag to only be considered if the
    # first term is either an ivy.Array or ivy.Container
    instance_method = instance_method and (not native_array[0] or container[0])

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(num_positional_args, all_as_kwargs_np)

    # change all data types so that they are supported by this framework
    dtype = ["float32" if d in ivy.invalid_dtypes else d for d in dtype]

    # create args
    args_idxs = ivy.nested_indices_where(args_np, lambda x: isinstance(x, np.ndarray))
    arg_np_vals = ivy.multi_index_nest(args_np, args_idxs)
    num_arg_vals = len(arg_np_vals)
    arg_array_vals = [
        ivy.array(x, dtype=d) for x, d in zip(arg_np_vals, dtype[:num_arg_vals])
    ]
    arg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(arg_array_vals, as_variable[:num_arg_vals])
    ]
    arg_array_vals = [
        ivy.to_native(x) if n else x
        for x, n in zip(arg_array_vals, native_array[:num_arg_vals])
    ]
    arg_array_vals = [
        as_cont(x) if c else x for x, c in zip(arg_array_vals, container[:num_arg_vals])
    ]
    args = ivy.copy_nest(args_np, to_mutable=True)
    ivy.set_nest_at_indices(args, args_idxs, arg_array_vals)

    # create kwargs
    kwargs_idxs = ivy.nested_indices_where(
        kwargs_np, lambda x: isinstance(x, np.ndarray)
    )
    kwarg_np_vals = ivy.multi_index_nest(kwargs_np, kwargs_idxs)
    kwarg_array_vals = [
        ivy.array(x, dtype=d) for x, d in zip(kwarg_np_vals, dtype[num_arg_vals:])
    ]
    kwarg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(kwarg_array_vals, as_variable[num_arg_vals:])
    ]
    kwarg_array_vals = [
        ivy.to_native(x) if n else x
        for x, n in zip(kwarg_array_vals, native_array[num_arg_vals:])
    ]
    kwarg_array_vals = [
        as_cont(x) if c else x
        for x, c in zip(kwarg_array_vals, container[num_arg_vals:])
    ]
    kwargs = ivy.copy_nest(kwargs_np, to_mutable=True)
    ivy.set_nest_at_indices(kwargs, kwargs_idxs, kwarg_array_vals)

    # create numpy args
    args_np = ivy.nested_map(
        args,
        lambda x: ivy.to_numpy(x) if ivy.is_ivy_container(x) or ivy.is_array(x) else x,
    )
    kwargs_np = ivy.nested_map(
        kwargs,
        lambda x: ivy.to_numpy(x) if ivy.is_ivy_container(x) or ivy.is_array(x) else x,
    )

    # run either as an instance method or from the API directly
    instance = None
    if instance_method:
        is_instance = [(not n) or c for n, c in zip(native_array, container)]
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
        ret = instance.__getattribute__(fn_name)(*args, **kwargs)
    else:
        ret = ivy.__dict__[fn_name](*args, **kwargs)

    # assert idx of return if the idx of the out array provided
    out = ret
    if with_out:
        assert not isinstance(ret, tuple)
        if max(container):
            assert ivy.is_ivy_container(ret)
        else:
            assert ivy.is_array(ret)
        if instance_method:
            ret = instance.__getattribute__(fn_name)(*args, **kwargs, out=out)
        else:
            ret = ivy.__dict__[fn_name](*args, **kwargs, out=out)

        if max(container):
            assert ret is out

        if max(container) or fw in ["tensorflow", "jax", "numpy"]:
            # these backends do not always support native inplace updates
            pass
        else:
            assert ret.data is out.data

    # value test
    if not isinstance(ret, tuple):
        ret = (ret,)
    if dtype == "bfloat16":
        return  # bfloat16 is not supported by numpy
    ret_idxs = ivy.nested_indices_where(ret, ivy.is_ivy_array)
    ret_flat = ivy.multi_index_nest(ret, ret_idxs)
    ret_np_flat = [ivy.to_numpy(x) for x in ret_flat]
    ivy.set_backend("numpy")
    ret_from_np = ivy.to_native(
        ivy.__dict__[fn_name](*args_np, **kwargs_np), nested=True
    )
    ivy.unset_backend()
    if not isinstance(ret_from_np, tuple):
        ret_from_np = (ret_from_np,)
    ret_from_np_flat = ivy.multi_index_nest(ret_from_np, ret_idxs)
    for ret_np, ret_from_np in zip(ret_np_flat, ret_from_np_flat):
        assert_all_close(ret_np, ret_from_np, rtol=rtol, atol=atol)


# Hypothesis #
# -----------#


@st.composite
def array_dtypes(draw, na=st.shared(st.integers(), key="num_arrays")):
    size = na if isinstance(na, int) else draw(na)
    return draw(
        st.lists(
            st.sampled_from(ivy_np.valid_float_dtypes), min_size=size, max_size=size
        )
    )


@st.composite
def array_bools(draw, na=st.shared(st.integers(), key="num_arrays")):
    size = na if isinstance(na, int) else draw(na)
    return draw(st.lists(st.booleans(), min_size=size, max_size=size))


@st.composite
def lists(draw, arg, min_size=None, max_size=None, size_bounds=None):
    ints = st.integers(size_bounds[0], size_bounds[1]) if size_bounds else st.integers()
    if isinstance(min_size, str):
        min_size = draw(st.shared(ints, key=min_size))
    if isinstance(max_size, str):
        max_size = draw(st.shared(ints, key=max_size))
    return draw(st.lists(arg, min_size=min_size, max_size=max_size))


@st.composite
def valid_axes(draw, ndim=None, size_bounds=None):
    ints = st.integers(size_bounds[0], size_bounds[1]) if size_bounds else st.integers()
    dims = draw(st.shared(ints, key=ndim))
    any_axis_strategy = (
        st.none() | st.integers(-dims, dims - 1) | nph.valid_tuple_axes(dims)
    )
    return draw(any_axis_strategy)


@st.composite
def integers(draw, min_value=None, max_value=None):
    if isinstance(min_value, str):
        min_value = draw(st.shared(st.integers(), key=min_value))
    if isinstance(max_value, str):
        max_value = draw(st.shared(st.integers(), key=max_value))
    return draw(st.integers(min_value=min_value, max_value=max_value))


@st.composite
def dtype_and_values(draw, available_dtypes, n_arrays=1, allow_inf=True):
    if n_arrays == 1:
        types = set(available_dtypes).difference(set(ivy.invalid_dtypes))
        dtype = draw(list_of_length(st.sampled_from(tuple(types)), 1))
    else:
        unwanted_types = set(ivy.invalid_dtypes).union(
            set(ivy.all_dtypes).difference(set(available_dtypes))
        )
        pairs = ivy.promotion_table.keys()
        types = [pair for pair in pairs if not any([d in pair for d in unwanted_types])]
        dtype = list(draw(st.sampled_from(types)))
    if n_arrays == 3:
        dtype.append(dtype[0])

    size = draw(st.integers(0, 10))
    values = []
    for i in range(n_arrays):
        values.append(draw(array_values(dtype[i], size, allow_inf)))
    if n_arrays == 1:
        dtype = dtype[0]
        values = values[0]
    return dtype, values


# taken from
# https://github.com/data-apis/array-api-tests/array_api_tests/test_manipulation_functions.py
@st.composite
def reshape_shapes(draw, shape):
    size = 1 if len(shape) == 0 else math.prod(shape)
    rshape = draw(st.lists(st.integers(0)).filter(lambda s: math.prod(s) == size))
    # assume(all(side <= MAX_SIDE for side in rshape))
    if len(rshape) != 0 and size > 0 and draw(st.booleans()):
        index = draw(st.integers(0, len(rshape) - 1))
        rshape[index] = -1
    return tuple(rshape)


# taken from https://github.com/HypothesisWorks/hypothesis/issues/1115
@st.composite
def subsets(draw, elements):
    return tuple(e for e in elements if draw(st.booleans()))


@st.composite
def array_values(draw, dtype, size, allow_inf=None):
    if dtype == "int8":
        values = draw(list_of_length(st.integers(-128, 127), size))
    elif dtype == "int16":
        values = draw(list_of_length(st.integers(-32768, 32767), size))
    elif dtype == "int32":
        values = draw(list_of_length(st.integers(-2147483648, 2147483647), size))
    elif dtype == "int64":
        values = draw(
            list_of_length(st.integers(-9223372036854775808, 9223372036854775807), size)
        )
    elif dtype == "uint8":
        values = draw(list_of_length(st.integers(0, 255), size))
    elif dtype == "uint16":
        values = draw(list_of_length(st.integers(0, 65535), size))
    elif dtype == "uint32":
        values = draw(list_of_length(st.integers(0, 4294967295), size))
    elif dtype == "uint64":
        values = draw(list_of_length(st.integers(0, 18446744073709551615), size))
    elif dtype == "float16":
        values = draw(
            list_of_length(
                st.floats(width=16, allow_subnormal=False, allow_infinity=allow_inf),
                size,
            )
        )
    elif dtype == "float32":
        values = draw(
            list_of_length(
                st.floats(width=32, allow_subnormal=False, allow_infinity=allow_inf),
                size,
            )
        )
    elif dtype == "float64":
        values = draw(
            list_of_length(
                st.floats(width=64, allow_subnormal=False, allow_infinity=allow_inf),
                size,
            )
        )
    elif dtype == "bool":
        values = draw(list_of_length(st.booleans(), size))
    return values


@st.composite
def get_shape(draw, allow_none=True, min_size=0):
    if allow_none:
        shape = draw(st.none() | st.lists(st.integers(min_value=1, max_value=8),
                                          min_size=min_size,
                                          max_size=8))
    else:
        shape = draw(st.lists(st.integers(min_value=1, max_value=8),
                              min_size=min_size,
                              max_size=8))
    if shape is None:
        return shape
    return tuple(shape)


def none_or_list_of_floats(
        dtype,
        size,
        min_value=None,
        max_value=None,
        exclude_min=False,
        exclude_max=False,
        no_none=False
):
    if no_none:
        if dtype == "float16":
            values = list_of_length(st.floats(min_value=min_value,
                                              max_value=max_value,
                                              width=16,
                                              allow_subnormal=False,
                                              allow_infinity=False,
                                              allow_nan=False,
                                              exclude_min=exclude_min,
                                              exclude_max=exclude_max), size)
        elif dtype == "float32":
            values = list_of_length(st.floats(min_value=min_value,
                                              max_value=max_value,
                                              width=32,
                                              allow_subnormal=False,
                                              allow_infinity=False,
                                              allow_nan=False,
                                              exclude_min=exclude_min,
                                              exclude_max=exclude_max), size)
        elif dtype == "float64":
            values = list_of_length(st.floats(min_value=min_value,
                                              max_value=max_value,
                                              width=64,
                                              allow_subnormal=False,
                                              allow_infinity=False,
                                              allow_nan=False,
                                              exclude_min=exclude_min,
                                              exclude_max=exclude_max), size)
    else:
        if dtype == "float16":
            values = list_of_length(st.none() | st.floats(min_value=min_value,
                                                          max_value=max_value,
                                                          width=16,
                                                          allow_subnormal=False,
                                                          allow_infinity=False,
                                                          allow_nan=False,
                                                          exclude_min=exclude_min,
                                                          exclude_max=exclude_max),
                                    size)
        elif dtype == "float32":
            values = list_of_length(st.none() | st.floats(min_value=min_value,
                                                          max_value=max_value,
                                                          width=32,
                                                          allow_subnormal=False,
                                                          allow_infinity=False,
                                                          allow_nan=False,
                                                          exclude_min=exclude_min,
                                                          exclude_max=exclude_max),
                                    size)
        elif dtype == "float64":
            values = list_of_length(st.none() | st.floats(min_value=min_value,
                                                          max_value=max_value,
                                                          width=64,
                                                          allow_subnormal=False,
                                                          allow_infinity=False,
                                                          allow_nan=False,
                                                          exclude_min=exclude_min,
                                                          exclude_max=exclude_max),
                                    size)
    return values


@st.composite
def get_mean_std(draw, dtype):
    values = draw(none_or_list_of_floats(dtype, 2))
    values[1] = abs(values[1]) if values[1] else None
    return values[0], values[1]


@st.composite
def get_bounds(draw, dtype):
    if 'int' in dtype:
        values = draw(array_values(dtype, 2))
        values[0], values[1] = abs(values[0]), abs(values[1])
        low, high = min(values), max(values)
        if low == high:
            return draw(get_bounds(dtype))
    else:
        values = draw(none_or_list_of_floats(dtype, 2))
        if values[0] is not None and values[1] is not None:
            low, high = min(values), max(values)
        else:
            low, high = values[0], values[1]
        if ivy.default(low, 0.0) >= ivy.default(high, 1.0):
            return draw(get_bounds(dtype))
    return low, high


@st.composite
def get_probs(draw, dtype):
    shape = draw(st.lists(st.integers(min_value=2, max_value=8),
                          min_size=2,
                          max_size=2))
    probs = []
    for i in range(shape[0]):
        probs.append(draw(none_or_list_of_floats(
            dtype,
            shape[1],
            min_value=0,
            exclude_min=True,
            no_none=True
        )))
    return probs, shape[1]


@st.composite
def get_float_array(draw, dtype, allow_nan=False):
    shape = draw(get_shape(allow_none=False, min_size=1))
    res = np.asarray(draw(xps.arrays(dtype, shape)))
    if not allow_nan:
        res[np.isnan(res)] = 0
    return res.tolist()
