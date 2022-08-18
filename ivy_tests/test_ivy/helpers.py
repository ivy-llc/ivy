"""Collection of helpers for ivy unit tests."""

# global
import importlib
from contextlib import redirect_stdout
from io import StringIO
import sys
import re
import inspect
import numpy as np
import math
from typing import Union, List
from hypothesis import assume
import hypothesis.extra.numpy as nph  # noqa

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


TOLERANCE_DICT = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
cmd_line_args = (
    "as_variable",
    "native_array",
    "with_out",
    "container",
    "instance_method",
    "test_gradients",
)

try:
    import jax.numpy as jnp
except (ImportError, RuntimeError, AttributeError):
    jnp = None
try:
    import tensorflow as tf

    _tf_version = float(".".join(tf.__version__.split(".")[0:2]))
    if _tf_version >= 2.3:
        # noinspection PyPep8Naming,PyUnresolvedReferences
        from tensorflow.python.types.core import Tensor as tensor_type
    else:
        # noinspection PyPep8Naming
        # noinspection PyProtectedMember,PyUnresolvedReferences
        from tensorflow.python.framework.tensor_like import _TensorLike as tensor_type
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except ImportError:
    tf = None
try:
    import torch
except ImportError:
    torch = None
try:
    import mxnet as mx
    import mxnet.ndarray as mx_nd
except ImportError:
    mx = None
    mx_nd = None
from hypothesis import strategies as st

# local
import ivy
import ivy.functional.backends.numpy as ivy_np


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


def get_ivy_mxnet():
    """Import MXNET module from ivy"""
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


def np_call(func, *args, **kwargs):
    """Call a given function and return the result as a Numpy Array.

    Parameters
    ----------
    func
        The given function (callable).
    args
        The arguments to be given.
    kwargs
        The keywords args to be given.

    Returns
    -------
    ret
        The result of the function call as a Numpy Array
    """
    ret = func(*args, **kwargs)
    if isinstance(ret, (list, tuple)):
        return ivy.to_native(ret, nested=True)
    return ivy.to_numpy(ret)


def jnp_call(func, *args, **kwargs):
    new_args = _convert_vars(
        vars_in=args, from_type=np.ndarray, to_type_callable=jnp.asarray
    )
    new_kw_vals = _convert_vars(
        vars_in=kwargs.values(), from_type=np.ndarray, to_type_callable=jnp.asarray
    )
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(
            _convert_vars(
                vars_in=output,
                from_type=(jnp.ndarray, ivy.Array),
                to_type_callable=ivy.to_numpy,
            )
        )
    else:
        return _convert_vars(
            vars_in=[output],
            from_type=(jnp.ndarray, ivy.Array),
            to_type_callable=ivy.to_numpy,
        )[0]


def tf_call(func, *args, **kwargs):
    new_args = _convert_vars(
        vars_in=args, from_type=np.ndarray, to_type_callable=tf.convert_to_tensor
    )
    new_kw_vals = _convert_vars(
        vars_in=kwargs.values(),
        from_type=np.ndarray,
        to_type_callable=tf.convert_to_tensor,
    )
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(
            _convert_vars(
                vars_in=output,
                from_type=(tensor_type, ivy.Array),
                to_type_callable=ivy.to_numpy,
            )
        )
    else:
        return _convert_vars(
            vars_in=[output],
            from_type=(tensor_type, ivy.Array),
            to_type_callable=ivy.to_numpy,
        )[0]


def tf_graph_call(func, *args, **kwargs):
    new_args = _convert_vars(
        vars_in=args, from_type=np.ndarray, to_type_callable=tf.convert_to_tensor
    )
    new_kw_vals = _convert_vars(
        vars_in=kwargs.values(),
        from_type=np.ndarray,
        to_type_callable=tf.convert_to_tensor,
    )
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))

    @tf.function
    def tf_func(*local_args, **local_kwargs):
        return func(*local_args, **local_kwargs)

    output = tf_func(*new_args, **new_kwargs)

    if isinstance(output, tuple):
        return tuple(
            _convert_vars(
                vars_in=output,
                from_type=(tensor_type, ivy.Array),
                to_type_callable=ivy.to_numpy,
            )
        )
    else:
        return _convert_vars(
            vars_in=[output],
            from_type=(tensor_type, ivy.Array),
            to_type_callable=ivy.to_numpy,
        )[0]


def torch_call(func, *args, **kwargs):
    new_args = _convert_vars(
        vars_in=args, from_type=np.ndarray, to_type_callable=torch.from_numpy
    )
    new_kw_vals = _convert_vars(
        vars_in=kwargs.values(), from_type=np.ndarray, to_type_callable=torch.from_numpy
    )
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(
            _convert_vars(
                vars_in=output,
                from_type=(torch.Tensor, ivy.Array),
                to_type_callable=ivy.to_numpy,
            )
        )
    else:
        return _convert_vars(
            vars_in=[output],
            from_type=(torch.Tensor, ivy.Array),
            to_type_callable=ivy.to_numpy,
        )[0]


def mx_call(func, *args, **kwargs):
    new_args = _convert_vars(
        vars_in=args, from_type=np.ndarray, to_type_callable=mx_nd.array
    )
    new_kw_items = _convert_vars(
        vars_in=kwargs.values(), from_type=np.ndarray, to_type_callable=mx_nd.array
    )
    new_kwargs = dict(zip(kwargs.keys(), new_kw_items))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(
            _convert_vars(
                vars_in=output,
                from_type=(mx_nd.ndarray.NDArray, ivy.Array),
                to_type_callable=ivy.to_numpy,
            )
        )
    else:
        return _convert_vars(
            vars_in=[output],
            from_type=(mx_nd.ndarray.NDArray, ivy.Array),
            to_type_callable=ivy.to_numpy,
        )[0]


_calls = [np_call, jnp_call, tf_call, tf_graph_call, torch_call, mx_call]


# function that trims white spaces from docstrings
def trim(*, docstring):
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


def docstring_examples_run(
    *, fn, from_container=False, from_array=False, num_sig_fig=3
):
    """Performs docstring tests for a given function.

    Parameters
    ----------
    fn
        Callable function to be tested.
    from_container
        if True, check docstring of the function as a method of an Ivy Container.
    from_array
        if True, check docstring of the function as a method of an Ivy Array.
    num_sig_fig
        Number of significant figures to check in the example.

    Returns
    -------
    None if the test passes, else marks the test as failed.
    """
    if not hasattr(fn, "__name__"):
        return True
    fn_name = fn.__name__
    if fn_name not in ivy.backend_handler.ivy_original_dict:
        return True

    if from_container:
        docstring = getattr(
            ivy.backend_handler.ivy_original_dict["Container"], fn_name
        ).__doc__
    elif from_array:
        docstring = getattr(
            ivy.backend_handler.ivy_original_dict["Array"], fn_name
        ).__doc__
    else:
        docstring = ivy.backend_handler.ivy_original_dict[fn_name].__doc__

    if docstring is None:
        return True

    # removing extra new lines and trailing white spaces from the docstrings
    trimmed_docstring = trim(docstring=docstring)
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
            if parsed_output != "":
                parsed_output += ","
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
            # noinspection PyBroadException
            try:
                if f.getvalue() != "" and f.getvalue()[-2] != ",":
                    print(",")
                exec(line)
            except Exception as e:
                print(e, " ", ivy.current_backend_str(), " ", line)

    output = f.getvalue()
    output = output.rstrip()
    output = output.replace(" ", "").replace("\n", "")
    output = output.rstrip(",")

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

    # print("Output: ", output)
    # print("Putput: ", parsed_output)

    # assert output == parsed_output, "Output is unequal to the docstrings output."
    sig_fig = float("1e-" + str(num_sig_fig))
    numeric_pattern = re.compile(
        r"""
            [\{\}\(\)\[\]]|\w+:
        """,
        re.VERBOSE,
    )
    num_output = output.replace("ivy.array", "")
    num_output = numeric_pattern.sub("", num_output)
    num_parsed_output = parsed_output.replace("ivy.array", "")
    num_parsed_output = numeric_pattern.sub("", num_parsed_output)
    num_output = num_output.split(",")
    num_parsed_output = num_parsed_output.split(",")
    docstr_result = True
    for (doc_u, doc_v) in zip(num_output, num_parsed_output):
        try:
            docstr_result = np.allclose(
                np.nan_to_num(complex(doc_u)),
                np.nan_to_num(complex(doc_v)),
                rtol=sig_fig,
            )
        except Exception:
            if str(doc_u) != str(doc_v):
                docstr_result = False
        if not docstr_result:
            print(
                "output for ",
                fn_name,
                " on run: ",
                output,
                "\noutput in docs :",
                parsed_output,
                "\n",
                doc_u,
                " != ",
                doc_v,
                "\n",
            )
            ivy.warn(
                "Output is unequal to the docstrings output: %s" % fn_name, stacklevel=0
            )
            break
    return docstr_result


def var_fn(x, *, dtype=None, device=None):
    """Returns x as an Ivy Variable wrapping an Ivy Array with given dtype and device"""
    return ivy.variable(ivy.array(x, dtype=dtype, device=device))


@st.composite
def ints(draw, *, min_value=None, max_value=None, safety_factor=0.95):
    """Draws an arbitrarily sized list of integers with a safety factor
    applied to values.

    Parameters
    ----------
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


def assert_all_close(
    ret_np, ret_from_np, rtol=1e-05, atol=1e-08, ground_truth_backend="TensorFlow"
):
    """Matches the ret_np and ret_from_np inputs element-by-element to ensure that they
    are the same.

    Parameters
    ----------
    ret_np
        Return from the framework to test. Ivy Container or Numpy Array.
    ret_from_np
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
    assert ret_np.dtype is ret_from_np.dtype, (
        "the return with a {} backend produced data type of {}, while the return with"
        " a {} backend returned a data type of {}.".format(
            ground_truth_backend,
            ret_from_np.dtype,
            ivy.current_backend_str(),
            ret_np.dtype,
        )
    )
    if ivy.is_ivy_container(ret_np) and ivy.is_ivy_container(ret_from_np):
        ivy.Container.multi_map(assert_all_close, [ret_np, ret_from_np])
    else:
        assert np.allclose(
            np.nan_to_num(ret_np), np.nan_to_num(ret_from_np), rtol=rtol, atol=atol
        ), "{} != {}".format(ret_np, ret_from_np)


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
        ret_idxs = ivy.nested_indices_where(
            ret, lambda x: ivy.is_ivy_array(x) or is_jax_native_array(x)
        )
    elif fw == "numpy":
        ret_idxs = ivy.nested_indices_where(
            ret, lambda x: ivy.is_ivy_array(x) or is_numpy_native_array(x)
        )
    elif fw == "tensorflow":
        ret_idxs = ivy.nested_indices_where(
            ret, lambda x: ivy.is_ivy_array(x) or is_tensorflow_native_array(x)
        )
    else:
        ret_idxs = ivy.nested_indices_where(
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
    ret_idxs = ivy.nested_indices_where(ret, ivy.is_ivy_array)
    return ivy.multi_index_nest(ret, ret_idxs)


def flatten_and_to_np(*, ret):
    # flatten the return
    ret_flat = flatten(ret=ret)
    # convert the return to NumPy
    return [ivy.to_numpy(x) for x in ret_flat]


def get_ret_and_flattened_np_array(func, *args, **kwargs):
    """
    Runs func with args and kwargs, and returns the result along with its flattened
    version.
    """
    ret = func(*args, **kwargs)
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
        A list (flattened) containing Numpy arrays. Return from the framework to test.
    ret_from_np_flat
        A list (flattened) containing Numpy arrays. Return from the ground truth
        framework.
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
        for ret_np, ret_from_np in zip(ret_np_flat, ret_np_from_gt_flat):
            rtol = TOLERANCE_DICT.get(str(ret_from_np.dtype), 1e-03)
            assert_all_close(
                ret_np,
                ret_from_np,
                rtol=rtol,
                atol=atol,
                ground_truth_backend=ground_truth_backend,
            )
    else:
        for ret_np, ret_from_np in zip(ret_np_flat, ret_np_from_gt_flat):
            assert_all_close(
                ret_np,
                ret_from_np,
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
    True if the function does not support either the device or any data type, False
    otherwise.
    """
    test_unsupported = False
    unsupported_devices_dtypes_fn = ivy.function_unsupported_devices_and_dtypes(fn)
    supported_devices_dtypes_fn = ivy.function_supported_devices_and_dtypes(fn)
    for i in range(len(unsupported_devices_dtypes_fn["devices"])):
        if device in unsupported_devices_dtypes_fn["devices"][i]:
            for d in input_dtypes:
                if d in unsupported_devices_dtypes_fn["dtypes"][i]:
                    test_unsupported = True
                    break
    if (
        "device" in all_as_kwargs_np
        and "dtype" in all_as_kwargs_np
        and all_as_kwargs_np["device"] in unsupported_devices_dtypes_fn["devices"]
    ):
        index = unsupported_devices_dtypes_fn["devices"].index(
            all_as_kwargs_np["device"]
        )
        if all_as_kwargs_np["dtype"] in unsupported_devices_dtypes_fn["dtypes"][index]:
            test_unsupported = True
    if test_unsupported:
        return test_unsupported

    for i in range(len(supported_devices_dtypes_fn["devices"])):
        if device in supported_devices_dtypes_fn["devices"][i]:
            for d in input_dtypes:
                if d not in supported_devices_dtypes_fn["dtypes"][i]:
                    test_unsupported = True
                    break
        else:
            test_unsupported = True
        if (
            "device" in all_as_kwargs_np
            and "dtype" in all_as_kwargs_np
            and all_as_kwargs_np["device"] in supported_devices_dtypes_fn["devices"]
        ):
            if all_as_kwargs_np["device"] not in supported_devices_dtypes_fn["devices"]:
                test_unsupported = True
            else:
                index = supported_devices_dtypes_fn["devices"].index(
                    all_as_kwargs_np["device"]
                )
                if (
                    all_as_kwargs_np["dtype"]
                    not in supported_devices_dtypes_fn["dtypes"][index]
                ):
                    test_unsupported = True
    return test_unsupported


def create_args_kwargs(
    *,
    args_np,
    kwargs_np,
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
        A list of booleans. if True for a corresponding input argument, it is called as
        an Ivy Variable.
    native_array_flags
        if not None, the corresponding argument is called as a Native Array.
    container_flags
        if not None, the corresponding argument is called as an Ivy Container.

    Returns
    -------
    Arguments, Keyword-arguments, number of arguments, and indexes on arguments and
    keyword-arguments.
    """
    # extract all arrays from the arguments and keyword arguments
    args_idxs = ivy.nested_indices_where(args_np, lambda x: isinstance(x, np.ndarray))
    arg_np_vals = ivy.multi_index_nest(args_np, args_idxs)
    kwargs_idxs = ivy.nested_indices_where(
        kwargs_np, lambda x: isinstance(x, np.ndarray)
    )
    kwarg_np_vals = ivy.multi_index_nest(kwargs_np, kwargs_idxs)

    # assert that the number of arrays aligns with the dtypes and as_variable_flags
    num_arrays = len(arg_np_vals) + len(kwarg_np_vals)
    if num_arrays > 0:
        assert num_arrays == len(input_dtypes), (
            "Found {} arrays in the input arguments, but {} dtypes and "
            "as_variable_flags. Make sure to pass in a sequence of bools for all "
            "associated boolean flag inputs to test_function, with the sequence length "
            "being equal to the number of arrays in the arguments.".format(
                num_arrays, len(input_dtypes)
            )
        )

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
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: Union[bool, List[bool]],
    all_as_kwargs_np,
    num_positional_args: int,
    input_dtypes_constructor: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags_constructor: Union[bool, List[bool]],
    constructor_kwargs,
    num_positional_args_constructor: int,
    fw: str,
    class_name: str,
    rtol: float = None,
    atol: float = 1e-06,
    test_values: bool = True,
    ground_truth_backend: str = "tensorflow",
):
    """Tests a class-method that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated as an
        ivy Variable.
    all_as_kwargs_np:
        input arguments to the function as keyword arguments.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    input_dtypes_constructor
        data types of the input arguments for the constructor in order.
    as_variable_flags_constructor
        dictates whether the corresponding input argument should be treated as an
        ivy Variable for the constructor
    constructor_kwargs:
        input arguments to the constructor as keyword arguments.
    num_positional_args_constructor
        number of input arguments that must be passed as positional
        arguments to the constructor.
    fw
        current backend (framework).
    class_name
        name of the class to test.
    rtol
        relative tolerance value.
    atol
        absolute tolerance value.
    test_values
        if True, test for the correctness of the resulting values.
    ground_truth_backend
        Ground Truth Backend to compare the result-values.

    Returns
    -------
    ret
        optional, return value from the function
    ret_gt
        optional, return value from the Ground Truth function
    """
    # convert single values to length 1 lists
    input_dtypes, as_variable_flags = as_lists(input_dtypes, as_variable_flags)
    # update variable flags to be compatible with float dtype
    as_variable_flags = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]

    # change all data types so that they are supported by this framework
    input_dtypes = ["float32" if d in ivy.invalid_dtypes else d for d in input_dtypes]

    # create args
    calling_args_np, calling_kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args, kwargs=all_as_kwargs_np
    )
    calling_args, calling_kwargs, _, _, _ = create_args_kwargs(
        args_np=calling_args_np,
        kwargs_np=calling_kwargs_np,
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable_flags,
    )

    constructor_args_np, constructor_kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args_constructor, kwargs=constructor_kwargs
    )
    constructor_args, constructor_kwargs, _, _, _ = create_args_kwargs(
        args_np=constructor_args_np,
        kwargs_np=constructor_kwargs_np,
        input_dtypes=input_dtypes_constructor,
        as_variable_flags=as_variable_flags_constructor,
    )
    # run
    ins = ivy.__dict__[class_name](*constructor_args, **constructor_kwargs)
    ret, ret_np_flat = get_ret_and_flattened_np_array(
        ins, *calling_args, **calling_kwargs
    )
    # compute the return with a Ground Truth backend
    ivy.set_backend(ground_truth_backend)
    calling_args_gt, calling_kwargs_gt, _, _, _ = create_args_kwargs(
        args_np=calling_args_np,
        kwargs_np=calling_kwargs_np,
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable_flags,
    )
    constructor_args_gt, constructor_kwargs_gt, _, _, _ = create_args_kwargs(
        args_np=constructor_args_np,
        kwargs_np=constructor_kwargs_np,
        input_dtypes=input_dtypes_constructor,
        as_variable_flags=as_variable_flags_constructor,
    )
    ins_gt = ivy.__dict__[class_name](*constructor_args_gt, **constructor_kwargs_gt)
    ret_from_gt, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
        ins_gt, *calling_args_gt, **calling_kwargs_gt
    )
    ivy.unset_backend()
    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, ret_from_gt
    # value test
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=ret_np_from_gt_flat,
        rtol=rtol,
        atol=atol,
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

    # make all lists equal in length
    num_arrays = max(
        len(input_dtypes),
        len(as_variable_flags),
        len(native_array_flags),
        len(container_flags),
    )
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

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args, kwargs=all_as_kwargs_np
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
                kwargs_np=kwargs_np,
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
            kwargs_np=kwargs_np,
            input_dtypes=input_dtypes,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
            container_flags=container_flags,
        )

    # run either as an instance method or from the API directly
    instance = None
    if instance_method:
        is_instance = [
            (not n) or c for n, c in zip(native_array_flags, container_flags)
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
                    kwargs_np=kwargs_np,
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
                kwargs_np=kwargs_np,
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
    num_positional_args: int,
    native_array_flags: Union[bool, List[bool]],
    fw: str,
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
    # make all lists equal in length
    num_arrays = max(
        len(input_dtypes),
        len(as_variable_flags),
        len(native_array_flags),
    )
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

    # change all data types so that they are supported by this framework
    input_dtypes = ["float32" if d in ivy.invalid_dtypes else d for d in input_dtypes]

    # create args
    if test_unsupported:
        try:
            args, kwargs, num_arg_vals, args_idxs, kwargs_idxs = create_args_kwargs(
                args_np=args_np,
                kwargs_np=kwargs_np,
                input_dtypes=input_dtypes,
                as_variable_flags=as_variable_flags,
                native_array_flags=native_array_flags,
            )
            args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)
        except Exception:
            return
    else:
        args, kwargs, num_arg_vals, args_idxs, kwargs_idxs = create_args_kwargs(
            args_np=args_np,
            kwargs_np=kwargs_np,
            input_dtypes=input_dtypes,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
        )
        args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)

    # frontend function
    frontend_fn = ivy.functional.frontends.__dict__[frontend].__dict__[fn_tree]

    # run from the Ivy API directly
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
            kwargs_ivy["out"] = out
        else:
            args[ivy.arg_info(frontend_fn, name="out")["idx"]] = out
            args_ivy = list(args_ivy)
            args_ivy[ivy.arg_info(frontend_fn, name="out")["idx"]] = out
            args_ivy = tuple(args_ivy)
        ret = frontend_fn(*args, **kwargs)

        if ivy.native_inplace_support:
            # these backends do not always support native inplace updates
            assert ret.data is out.data

    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in input_dtypes))

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

        # tuplify the frontend return
        if not isinstance(frontend_ret, tuple):
            frontend_ret = (frontend_ret,)

        # flatten the frontend return and convert to NumPy arrays
        frontend_ret_idxs = ivy.nested_indices_where(frontend_ret, ivy.is_native_array)
        frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
        frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]
    except Exception as e:
        ivy.unset_backend()
        raise e
    # unset frontend framework from backend
    ivy.unset_backend()

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, frontend_ret

    # flatten the return
    ret_np_flat = flatten_and_to_np(ret=ret)

    # value tests, iterating through each array in the flattened returns
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=rtol,
        atol=atol,
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
    large_value_safety_factor=1.1,
    small_value_safety_factor=1.1,
    allow_inf=False,
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
    safety_factor
        Ratio of max_value to maximum allowed number in the data type.
    allow_inf
        if True, allow inf in the arrays.
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
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                    large_value_safety_factor=large_value_safety_factor,
                    small_value_safety_factor=small_value_safety_factor,
                )
            )
        )
    if num_arrays == 1:
        dtype = dtype[0]
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
    allow_inf=True,
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
    allow_inf
        if True, allow inf in the array.
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
            allow_inf=allow_inf,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
            shape=shape,
            shared_dtype=shared_dtype,
            ret_shape=ret_shape,
        )
    )
    if ret_shape:
        dtype, values, shape = results
    else:
        dtype, values = results
    if not isinstance(values, list):
        return dtype, values, None
    if shape is not None:
        return dtype, values, draw(get_axis(shape=shape))
    axis = draw(ints(min_value=min_axis, max_value=max_axis))
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
def array_and_indices(
    draw,
    last_dim_same_size=True,
    allow_inf=False,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
):
    """Generates two arrays x & indices, the values in the indices array are indices
    of the array x. Draws an integers randomly from the minimum and maximum number of
    positional arguments a given function can take.

    Parameters
    ----------
    last_dim_same_size
        True:
            The shape of the indices array is the exact same as the shape of the values
            array.
        False:
            The last dimension of the second array is generated from a range of
            (0 -> dimension size of first array). This results in output shapes such as
            x = (5,5,5,5,5) & indices = (5,5,5,5,3) or x = (7,7) & indices = (7,2)
    allow_inf
        True: inf values are allowed to be generated in the values array
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
        array_and_indices=array_and_indices(
            last_dim_same_size= False
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10
            )
    )
    @given(
        array_and_indices=array_and_indices( last_dim_same_size= True)
    )
    """
    x_num_dims = draw(ints(min_value=min_num_dims, max_value=max_num_dims))
    x_dim_size = draw(ints(min_value=min_dim_size, max_value=max_dim_size))
    x = draw(
        dtype_and_values(
            available_dtypes=ivy_np.valid_numeric_dtypes,
            allow_inf=allow_inf,
            ret_shape=True,
            min_num_dims=x_num_dims,
            max_num_dims=x_num_dims,
            min_dim_size=x_dim_size,
            max_dim_size=x_dim_size,
        )
    )
    indices_shape = list(x[2])
    if not last_dim_same_size:
        indices_dim_size = draw(ints(min_value=1, max_value=x_dim_size))
        indices_shape[-1] = indices_dim_size
    indices = draw(
        dtype_and_values(
            available_dtypes=["int32", "int64"],
            allow_inf=False,
            min_value=0,
            max_value=max(x[2][-1] - 1, 0),
            shape=indices_shape,
        )
    )
    x = x[0:2]
    return (x, indices)


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
    allow_negative=True,
    large_value_safety_factor=1.1,
    small_value_safety_factor=1.1,
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
    allow_negative
        if True, allow negative numbers.
    safety_factor
        Ratio of max_value to maximum allowed number in the data type
    Returns
    -------
    A strategy that draws a list.
    """
    exclude_min = exclude_min if ivy.exists(min_value) else False
    exclude_max = exclude_max if ivy.exists(max_value) else False
    size = 1
    if isinstance(shape, int):
        size = shape
    else:
        for dim in shape:
            size *= dim
    values = None
    if "uint" in dtype:
        if dtype == "uint8":
            min_value = ivy.default(
                min_value, 1 if small_value_safety_factor < 1 else 0
            )
            max_value = ivy.default(max_value, round(255 * large_value_safety_factor))
        elif dtype == "uint16":
            min_value = ivy.default(
                min_value, 1 if small_value_safety_factor < 1 else 0
            )
            max_value = ivy.default(max_value, round(65535 * large_value_safety_factor))
        elif dtype == "uint32":
            min_value = ivy.default(
                min_value, 1 if small_value_safety_factor < 1 else 0
            )
            max_value = ivy.default(
                max_value, round(4294967295 * large_value_safety_factor)
            )
        elif dtype == "uint64":
            min_value = ivy.default(
                min_value, 1 if small_value_safety_factor < 1 else 0
            )
            max_value = ivy.default(
                max_value,
                min(
                    18446744073709551615,
                    round(18446744073709551615 * large_value_safety_factor),
                ),
            )
        values = draw(list_of_length(x=st.integers(min_value, max_value), length=size))
    elif "int" in dtype:

        if min_value is not None and max_value is not None:
            values = draw(
                list_of_length(
                    x=st.integers(min_value, max_value),
                    length=size,
                )
            )
        else:
            if dtype == "int8":
                min_value = ivy.default(
                    min_value, round(-128 * large_value_safety_factor)
                )
                max_value = ivy.default(
                    max_value, round(127 * large_value_safety_factor)
                )

            elif dtype == "int16":
                min_value = ivy.default(
                    min_value, round(-32768 * large_value_safety_factor)
                )
                max_value = ivy.default(
                    max_value, round(32767 * large_value_safety_factor)
                )

            elif dtype == "int32":
                min_value = ivy.default(
                    min_value, round(-2147483648 * large_value_safety_factor)
                )
                max_value = ivy.default(
                    max_value, round(2147483647 * large_value_safety_factor)
                )

            elif dtype == "int64":
                min_value = ivy.default(
                    min_value,
                    max(
                        -9223372036854775808,
                        round(-9223372036854775808 * large_value_safety_factor),
                    ),
                )
                max_value = ivy.default(
                    max_value,
                    min(
                        9223372036854775807,
                        round(9223372036854775807 * large_value_safety_factor),
                    ),
                )
            max_neg_value = -1 if small_value_safety_factor > 1 else 0
            min_pos_value = 1 if small_value_safety_factor > 1 else 0

            if min_value >= max_neg_value:
                min_value = min_pos_value
                max_neg_value = max_value
            elif max_value <= min_pos_value:
                min_pos_value = min_value
                max_value = max_neg_value

            values = draw(
                list_of_length(
                    x=st.integers(min_value, max_neg_value)
                    | st.integers(min_pos_value, max_value),
                    length=size,
                )
            )
    elif dtype == "float16":

        if min_value is not None and max_value is not None:
            values = draw(
                list_of_length(
                    x=st.floats(
                        min_value=min_value,
                        max_value=max_value,
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=16,
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                    length=size,
                )
            )

        else:
            limit = math.log(small_value_safety_factor)
            min_value_neg = min_value
            max_value_neg = round(-1 * limit, -3)
            min_value_pos = round(limit, -3)
            max_value_pos = max_value

            values = draw(
                list_of_length(
                    x=st.floats(
                        min_value=min_value_neg,
                        max_value=max_value_neg,
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=16,
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    )
                    | st.floats(
                        min_value=min_value_pos,
                        max_value=max_value_pos,
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=16,
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                    length=size,
                )
            )
        values = [v * large_value_safety_factor for v in values]
    elif dtype in ["float32", "bfloat16"]:
        if min_value is not None and max_value is not None:
            values = draw(
                list_of_length(
                    x=st.floats(
                        min_value=min_value,
                        max_value=max_value,
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=32,
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                    length=size,
                )
            )
        else:
            limit = math.log(small_value_safety_factor)
            min_value_neg = min_value
            max_value_neg = round(-1 * limit, -6)
            min_value_pos = round(limit, -6)
            max_value_pos = max_value

            values = draw(
                list_of_length(
                    x=st.floats(
                        min_value=min_value_neg,
                        max_value=max_value_neg,
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=32,
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    )
                    | st.floats(
                        min_value=min_value_pos,
                        max_value=max_value_pos,
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=32,
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                    length=size,
                )
            )
        values = [v * large_value_safety_factor for v in values]
    elif dtype == "float64":

        if min_value is not None and max_value is not None:
            values = draw(
                list_of_length(
                    x=st.floats(
                        min_value=min_value,
                        max_value=max_value,
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=64,
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                    length=size,
                )
            )
        else:
            limit = math.log(small_value_safety_factor)

            min_value_neg = min_value
            max_value_neg = round(-1 * limit, -15)
            min_value_pos = round(limit, -15)
            max_value_pos = max_value

            values = draw(
                list_of_length(
                    x=st.floats(
                        min_value=min_value_neg,
                        max_value=max_value_neg,
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=64,
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    )
                    | st.floats(
                        min_value=min_value_pos,
                        max_value=max_value_pos,
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=64,
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                    length=size,
                )
            )
        values = [v * large_value_safety_factor for v in values]
    elif dtype == "bool":
        values = draw(list_of_length(x=st.booleans(), length=size))
    array = np.array(values)
    if dtype != "bool" and not allow_negative:
        array = np.abs(array)
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
                x=st.floats(
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
            valid_strategies.append(st.integers(-axes, axes - 1))
    if not force_int:
        if axes == 0:
            valid_strategies.append(
                st.lists(st.just(0), min_size=min_size, max_size=max_size)
            )
        else:
            valid_strategies.append(
                st.lists(
                    st.integers(-axes, axes - 1),
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
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind == param.KEYWORD_ONLY:
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
    # first four arguments are all fixtures
    def new_fn(data, get_command_line_flags, fw, device, call, *args, **kwargs):
        # inspecting for keyword arguments in test function
        for param in inspect.signature(test_fn).parameters.values():
            if param.name in cmd_line_args:
                kwargs[param.name] = data.draw(
                    bool_val_flags(get_command_line_flags[param.name])
                )
            elif param.name == "data":
                kwargs["data"] = data
            elif param.name == "fw":
                kwargs["fw"] = fw
            elif param.name == "device":
                kwargs["device"] = device
            elif param.name == "call":
                kwargs["call"] = call
        return test_fn(*args, **kwargs)

    return new_fn


def gradient_incompatible_function(*, fn):
    return (
        not ivy.supports_gradients
        and hasattr(fn, "computes_gradients")
        and fn.computes_gradients
    )
