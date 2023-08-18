# global


import numpy as np
from hypothesis import strategies as st
from typing import Optional

# local
import ivy
from ..pipeline_helper import BackendHandler, get_frontend_config
from . import number_helpers as nh
from . import array_helpers as ah
from .. import globals as test_globals


_dtype_kind_keys = {
    "valid",
    "numeric",
    "float",
    "unsigned",
    "integer",
    "signed_integer",
    "complex",
    "real_and_complex",
    "float_and_integer",
    "float_and_complex",
    "bool",
}


def _get_fn_dtypes(framework: str, kind="valid", mixed_fn_dtypes="compositional"):
    all_devices_dtypes = test_globals.CURRENT_RUNNING_TEST.supported_device_dtypes[
        framework
    ]
    if mixed_fn_dtypes in all_devices_dtypes:
        all_devices_dtypes = all_devices_dtypes[mixed_fn_dtypes]
    return all_devices_dtypes[test_globals.CURRENT_DEVICE_STRIPPED][kind]


def _get_type_dict(framework: str, kind: str, is_frontend_test=False):
    if is_frontend_test:
        framework_module = get_frontend_config(framework).supported_dtypes
    else:
        framework_module = ivy.with_backend(framework)

    if kind == "valid":
        return framework_module.valid_dtypes
    if kind == "numeric":
        return framework_module.valid_numeric_dtypes
    if kind == "integer":
        return framework_module.valid_int_dtypes
    if kind == "float":
        return framework_module.valid_float_dtypes
    if kind == "unsigned":
        return framework_module.valid_uint_dtypes
    if kind == "signed_integer":
        return tuple(
            set(framework_module.valid_int_dtypes).difference(
                framework_module.valid_uint_dtypes
            )
        )
    if kind == "complex":
        return framework_module.valid_complex_dtypes
    if kind == "real_and_complex":
        return tuple(
            set(framework_module.valid_numeric_dtypes).union(
                framework_module.valid_complex_dtypes
            )
        )
    if kind == "float_and_complex":
        return tuple(
            set(framework_module.valid_float_dtypes).union(
                framework_module.valid_complex_dtypes
            )
        )
    if kind == "float_and_integer":
        return tuple(
            set(framework_module.valid_float_dtypes).union(
                framework_module.valid_int_dtypes
            )
        )
    if kind == "bool":
        return tuple(
            set(framework_module.valid_dtypes).difference(
                framework_module.valid_numeric_dtypes
            )
        )

    raise RuntimeError("{} is an unknown kind!".format(kind))


@st.composite
def get_dtypes(
    draw,
    kind="valid",
    index=0,
    mixed_fn_compos=True,
    full=True,
    none=False,
    key=None,
    prune_function=True,
):
    """
    Draws a valid dtypes for the test function. For frontend tests, it draws the data
    types from the intersection between backend framework data types and frontend
    framework dtypes, otherwise, draws it from backend framework data types.

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
    mixed_fn_compos
        boolean if True, the function will return the dtypes of the compositional
        implementation for mixed partial functions and if False, it will return
        the dtypes of the primary implementation.
    full
        returns the complete list of valid types
    none
        allow none in the list of valid types
    key
        if provided, a shared value will be drawn from the strategy and passed to the
        function as the keyword argument with the given name.
    prune_function
        if True, the function will prune the data types to only include the ones that
        are supported by the current backend. If False, the function will return all
        the data types supported by the current backend.

    Returns
    -------
    ret
        A strategy that draws dtype strings

    Examples
    --------
    >>> get_dtypes()
    ['float16',
        'uint8',
        'complex128',
        'bool',
        'uint32',
        'float64',
        'int8',
        'int16',
        'complex64',
        'float32',
        'int32',
        'uint16',
        'int64',
        'uint64']

    >>> get_dtypes(kind='valid', full=False)
    ['int16']

    >>> get_dtypes(kind='valid', full=False)
    ['uint16']

    >>> get_dtypes(kind='numeric', full=False)
    ['complex64']

    >>> get_dtypes(kind='float', full=False, key="leaky_relu")
    ['float16']

    >>> get_dtypes(kind='float', full=False, key="searchsorted")
    ['bfloat16']

    >>> get_dtypes(kind='float', full=False, key="dtype")
    ['float32']

    >>> get_dtypes("numeric", prune_function=False)
    ['int16']

    >>> get_dtypes("valid", prune_function=False)
    ['uint32']

    >>> get_dtypes("valid", prune_function=False)
    ['complex128']

    >>> get_dtypes("valid", prune_function=False)
    ['bool']

    >>> get_dtypes("valid", prune_function=False)
    ['float16']
    """
    mixed_fn_dtypes = "compositional" if mixed_fn_compos else "primary"
    if prune_function:
        retrieval_fn = _get_fn_dtypes
        if test_globals.CURRENT_RUNNING_TEST is not test_globals._Notsetval:
            valid_dtypes = set(
                retrieval_fn(
                    test_globals.CURRENT_BACKEND,
                    mixed_fn_dtypes=mixed_fn_dtypes,
                    kind=kind,
                )
            )
        else:
            raise RuntimeError(
                "No function is set to prune, calling "
                "prune_function=True without a function is redundant."
            )
    else:
        retrieval_fn = _get_type_dict
        valid_dtypes = set(retrieval_fn(test_globals.CURRENT_BACKEND, kind))

    # The function may be called from a frontend test or an Ivy API test
    # In the case of an Ivy API test, the function should make sure it returns a valid
    # dtypes for the backend and also for the ground truth backend, if it is called from
    # a frontend test, we should also count for the frontend support data types
    # In conclusion, the following operations will get the intersection of
    # FN_DTYPES & BACKEND_DTYPES & FRONTEND_DTYPES & GROUND_TRUTH_DTYPES

    # If being called from a frontend test

    if test_globals.CURRENT_FRONTEND is not test_globals._Notsetval:
        frontend_dtypes = retrieval_fn(test_globals.CURRENT_FRONTEND, kind, True)
        valid_dtypes = valid_dtypes.intersection(frontend_dtypes)

    # Make sure we return dtypes that are compatible with ground truth backend
    ground_truth_is_set = (
        test_globals.CURRENT_GROUND_TRUTH_BACKEND is not test_globals._Notsetval  # NOQA
    )
    if ground_truth_is_set:
        valid_dtypes = valid_dtypes.intersection(
            retrieval_fn(test_globals.CURRENT_GROUND_TRUTH_BACKEND, kind=kind)
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
    """
    Draws a list of data types.

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
        A strategy that draws a list of data types.

    Examples
    --------
    >>> array_dtypes(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     shared_dtype=True,
    ... )
    ['float64']

    >>> array_dtypes(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     shared_dtype=True,
    ... )
    ['int8', 'int8']

    >>> array_dtypes(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     shared_dtype=True,
    ... )
    ['int32', 'int32', 'int32', 'int32']

    >>> array_dtypes(
    ...     num_arrays=5,
    ...     available_dtypes=get_dtypes("valid"),
    ...     shared_dtype=False,
    ... )
    ['int8', 'float64', 'complex64', 'int8', 'bool']

    >>> array_dtypes(
    ...     num_arrays=5,
    ...     available_dtypes=get_dtypes("valid"),
    ...     shared_dtype=False,
    ... )
    ['bool', 'complex64', 'bool', 'complex64', 'bool']

    >>> array_dtypes(
    ...     num_arrays=5,
    ...     available_dtypes=get_dtypes("valid"),
    ...     shared_dtype=False,
    ... )
    ['float64', 'int8', 'float64', 'int8', 'float64']
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
    with BackendHandler.update_backend(test_globals.CURRENT_BACKEND) as ivy_backend:
        bound_dtype_bits = lambda d: (
            ivy_backend.dtype_bits(d) / 2
            if ivy_backend.is_complex_dtype(d)
            else ivy_backend.dtype_bits(d)
        )

        def cast_filter(d):
            if ivy_backend.is_int_dtype(d):
                max_val = ivy_backend.iinfo(d).max
                min_val = ivy_backend.iinfo(d).min
            elif ivy_backend.is_float_dtype(d) or ivy_backend.is_complex_dtype(d):
                max_val = ivy_backend.finfo(d).max
                min_val = ivy_backend.finfo(d).min
            else:
                max_val = 1
                min_val = -1
            if x is None:
                if ivy_backend.is_int_dtype(dtype):
                    max_x = ivy_backend.iinfo(dtype).max
                    min_x = ivy_backend.iinfo(dtype).min
                elif ivy_backend.is_float_dtype(dtype) or ivy_backend.is_complex_dtype(
                    dtype
                ):
                    max_x = ivy_backend.finfo(dtype).max
                    min_x = ivy_backend.finfo(dtype).min
                else:
                    max_x = 1
                    min_x = -1
            else:
                max_x = np.max(np.asarray(x))
                min_x = np.min(np.asarray(x))
            return (
                max_x <= max_val
                and min_x >= min_val
                and bound_dtype_bits(d) >= bound_dtype_bits(dtype)
            )

        cast_dtype = draw(st.sampled_from(available_dtypes).filter(cast_filter))
        if x is None:
            return dtype, cast_dtype
        return dtype, x, cast_dtype
