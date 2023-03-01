# global
import numpy as np
from hypothesis import strategies as st
from typing import Optional

try:
    import jsonpickle
except ImportError:
    pass
# local
import ivy
from . import number_helpers as nh
from . import array_helpers as ah
from .. import globals as test_globals
from ivy_tests import conftest

_dtype_kind_keys = {
    "valid",
    "numeric",
    "float",
    "unsigned",
    "integer",
    "signed_integer",
    "complex",
    "real_and_complex",
    "float_and_complex",
    "bool",
}


def _get_fn_dtypes(framework, kind="valid"):
    return test_globals.CURRENT_RUNNING_TEST.supported_device_dtypes[framework.backend][
        test_globals.CURRENT_DEVICE
    ][kind]


def _get_type_dict(framework, kind):
    if kind == "valid":
        return framework.valid_dtypes
    elif kind == "numeric":
        return framework.valid_numeric_dtypes
    elif kind == "integer":
        return framework.valid_int_dtypes
    elif kind == "float":
        return framework.valid_float_dtypes
    elif kind == "unsigned":
        return framework.valid_int_dtypes
    elif kind == "signed_integer":
        return tuple(
            set(framework.valid_int_dtypes).difference(framework.valid_uint_dtypes)
        )
    elif kind == "complex":
        return framework.valid_complex_dtypes
    elif kind == "real_and_complex":
        return tuple(
            set(framework.valid_numeric_dtypes).union(framework.valid_complex_dtypes)
        )
    elif kind == "float_and_complex":
        return tuple(
            set(framework.valid_float_dtypes).union(framework.valid_complex_dtypes)
        )
    elif kind == "bool":
        return tuple(
            set(framework.valid_dtypes).difference(framework.valid_numeric_dtypes)
        )
    else:
        raise RuntimeError("{} is an unknown kind!".format(kind))


def make_json_pickable(s):
    s = s.replace("builtins.bfloat16", "ivy.bfloat16")
    s = s.replace("jax._src.device_array.reconstruct_device_array", "jax.numpy.array")
    return s


@st.composite
def get_dtypes(
    draw, kind="valid", index=0, full=True, none=False, key=None, prune_function=True
):
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
        Supported types are integer, float, valid, numeric, signed_integer, complex,
        real_and_complex, float_and_complex, bool, and unsigned
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

    if prune_function:
        retrieval_fn = _get_fn_dtypes
        if test_globals.CURRENT_RUNNING_TEST is not test_globals._Notsetval:
            valid_dtypes = set(retrieval_fn(test_globals.CURRENT_BACKEND(), kind))
        else:
            raise RuntimeError(
                "No function is set to prune, calling "
                "prune_function=True without a function is redundant."
            )
    else:
        retrieval_fn = _get_type_dict
        valid_dtypes = set(retrieval_fn(ivy, kind))

    # The function may be called from a frontend test or an IVY api test
    # In the case of a IVY api test, the function should make sure it returns a valid
    # dtypes for the backend and also for the ground truth backend, if it is called from
    # a frontend test, we should also count for the frontend support data types
    # In conclusion, the following operations will get the intersection of
    # FN_DTYPES & BACKEND_DTYPES & FRONTEND_DTYPES & GROUND_TRUTH_DTYPES

    # If being called from a frontend test
    import time

    if test_globals.CURRENT_FRONTEND is not test_globals._Notsetval or isinstance(
        test_globals.CURRENT_FRONTEND_STR, list
    ):  # NOQA
        if isinstance(test_globals.CURRENT_FRONTEND_STR, list):
            process = test_globals.CURRENT_FRONTEND_STR[1]
            try:
                if test_globals.CURRENT_RUNNING_TEST.is_method:
                    process.stdin.write("1a" + "\n")
                    process.stdin.write(
                        jsonpickle.dumps(test_globals.CURRENT_RUNNING_TEST.is_method)
                        + "\n"
                    )
                else:
                    process.stdin.write("1" + "\n")
                process.stdin.write(f"{str(retrieval_fn.__name__)}" + "\n")
                process.stdin.write(f"{str(kind)}" + "\n")
                process.stdin.write(f"{test_globals.CURRENT_DEVICE}" + "\n")
                process.stdin.write(
                    f"{test_globals.CURRENT_RUNNING_TEST.fn_tree}" + "\n"
                )

                process.stdin.flush()
            except Exception as e:
                print(
                    "Something bad happened to the subprocess, here are the logs:\n\n"
                )
                print(process.stdout.readlines())
                raise e

            frontend_ret = process.stdout.readline()
            if frontend_ret:
                try:
                    frontend_ret = jsonpickle.loads(make_json_pickable(frontend_ret))
                except:
                    raise Exception(f"source of all bugsss   {frontend_ret}")
            else:
                print(process.stderr.readlines())
                raise Exception
            frontend_dtypes = frontend_ret
            valid_dtypes = valid_dtypes.intersection(frontend_dtypes)

        else:
            frontend_dtypes = retrieval_fn(test_globals.CURRENT_FRONTEND(), kind)
            valid_dtypes = valid_dtypes.intersection(frontend_dtypes)

    # Make sure we return dtypes that are compatiable with ground truth backend
    ground_truth_is_set = (
        test_globals.CURRENT_GROUND_TRUTH_BACKEND is not test_globals._Notsetval  # NOQA
    )
    if ground_truth_is_set:
        if isinstance(test_globals.CURRENT_GROUND_TRUTH_BACKEND, list):
            process = test_globals.CURRENT_GROUND_TRUTH_BACKEND[1]
            try:
                if test_globals.CURRENT_RUNNING_TEST.is_method:
                    process.stdin.write("1a" + "\n")
                else:
                    process.stdin.write("1" + "\n")
                process.stdin.write(f"{str(retrieval_fn.__name__)}" + "\n")
                process.stdin.write(f"{str(kind)}" + "\n")
                process.stdin.write(f"{test_globals.CURRENT_DEVICE}" + "\n")
                process.stdin.write(
                    f"{test_globals.CURRENT_RUNNING_TEST.fn_tree}" + "\n"
                )
                process.stdin.flush()
            except Exception as e:
                print(
                    "Something bad happened to the subprocess, here are the logs:\n\n"
                )
                print(process.stdout.readlines())
                raise e
            backend_ret = process.stdout.readline()
            if backend_ret:
                backend_ret = jsonpickle.loads(make_json_pickable(backend_ret))
            else:
                print(process.stderr.readlines())
                raise Exception

            valid_dtypes = valid_dtypes.intersection(backend_ret)
        else:
            valid_dtypes = valid_dtypes.intersection(
                retrieval_fn(test_globals.CURRENT_GROUND_TRUTH_BACKEND(), kind)
            )

    valid_dtypes = list(valid_dtypes)
    if none:
        valid_dtypes.append(None)
    if full:
        return valid_dtypes[index:]
    if key is None:
        return [draw(st.sampled_from(valid_dtypes[index:]))]
    return [draw(st.shared(st.sampled_from(valid_dtypes[index:]), key=key))]


@st.composite
def array_dtypes(
    draw,
    *,
    num_arrays=st.shared(nh.ints(min_value=1, max_value=4), key="num_arrays"),
    available_dtypes=get_dtypes("valid"),
    shared_dtype=False,
    array_api_dtypes=False,
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
    array_api_dtypes
        if True, use data types that can be promoted with the array_api_promotion
        table.

    Returns
    -------
    A strategy that draws a list.
    """
    if isinstance(available_dtypes, st._internal.SearchStrategy):
        available_dtypes = draw(available_dtypes)
    if not isinstance(num_arrays, int):
        num_arrays = draw(num_arrays)
    if num_arrays == 1:
        dtypes = draw(
            ah.list_of_size(
                x=st.sampled_from(available_dtypes),
                size=1,
            )
        )
    elif shared_dtype:
        dtypes = draw(
            ah.list_of_size(
                x=st.sampled_from(available_dtypes),
                size=1,
            )
        )
        dtypes = [dtypes[0] for _ in range(num_arrays)]
    else:
        unwanted_types = set(ivy.all_dtypes).difference(set(available_dtypes))
        if array_api_dtypes:
            pairs = ivy.array_api_promotion_table.keys()
        else:
            pairs = ivy.promotion_table.keys()
        # added to avoid complex dtypes from being sampled if they are not available.
        pairs = [pair for pair in pairs if all([d in available_dtypes for d in pair])]
        available_dtypes = [
            pair for pair in pairs if not any([d in pair for d in unwanted_types])
        ]
        dtypes = list(draw(st.sampled_from(available_dtypes)))
        if num_arrays > 2:
            dtypes += [dtypes[i % 2] for i in range(num_arrays - 2)]
    return dtypes


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
    bound_dtype_bits = (
        lambda d: ivy.dtype_bits(d) / 2
        if ivy.is_complex_dtype(d)
        else ivy.dtype_bits(d)
    )

    def cast_filter(d):
        if ivy.is_int_dtype(d):
            max_val = ivy.iinfo(d).max
        elif ivy.is_float_dtype(d) or ivy.is_complex_dtype(d):
            max_val = ivy.finfo(d).max
        else:
            max_val = 1
        if x is None:
            if ivy.is_int_dtype(dtype):
                max_x = ivy.iinfo(dtype).max
            elif ivy.is_float_dtype(dtype) or ivy.is_complex_dtype(dtype):
                max_x = ivy.finfo(dtype).max
            else:
                max_x = 1
        else:
            max_x = np.max(np.abs(np.asarray(x)))
        return max_x <= max_val and bound_dtype_bits(d) >= bound_dtype_bits(dtype)

    cast_dtype = draw(st.sampled_from(available_dtypes).filter(cast_filter))
    if x is None:
        return dtype, cast_dtype
    if "uint" in cast_dtype:
        x = np.abs(np.asarray(x))
    return dtype, x, cast_dtype
