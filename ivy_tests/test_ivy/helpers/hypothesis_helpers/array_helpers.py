# global
import numpy as np
import hypothesis.extra.numpy as nph
from hypothesis import strategies as st, assume
from hypothesis.internal.floats import float_of
from functools import reduce as _reduce
from operator import mul
import sys
import string

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.hypothesis_helpers.dtype_helpers import get_dtypes
from . import general_helpers as gh
from . import dtype_helpers, number_helpers


@st.composite
def array_bools(
    draw, *, size=st.shared(number_helpers.ints(min_value=1, max_value=4), key="size")
):
    """
    Draws a list of booleans with a given size.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    size
        size of the list.

    Returns
    -------
    ret
        A strategy that draws a list.

    Examples
    --------
    >>> array_bools(size=5)
    [False, True, False, False, False]

    >>> array_bools(size=5)
    [False, False, False, False, False]

    >>> array_bools(size=5)
    [True, False, False, False, False]

    >>> array_bools(size=1)
    [True]

    >>> array_bools(size=1)
    [False]

    >>> array_bools(size=1)
    [True]

    >>> array_bools()
    [False, False, False, False]

    >>> array_bools()
    [True, True, True, False]

    >>> array_bools()
    [True]
    """
    if not isinstance(size, int):
        size = draw(size)
    return draw(st.lists(st.booleans(), min_size=size, max_size=size))


def list_of_size(*, x, size):
    """
    Return a list of the given length with elements drawn randomly from x.

    Parameters
    ----------
    x
        a list to draw elements from.
    size
        length of the list.

    Returns
    -------
    ret
        A strategy that draws a list.

    Examples
    --------
    >>> list_of_size(
    ...     x=st.sampled_from([-1, 5, 9]),
    ...     size=4,
    ... )
    [-1, 5, -1, -1]

    >>> list_of_size(
    ...     x=st.sampled_from([-1, 5, 9]),
    ...     size=4,
    ... )
    [9, -1, -1, -1]

    >>> list_of_size(
    ...     x=st.sampled_from([-1, 5, 9]),
    ...     size=4,
    ... )
    [9, 9, -1, 9]

    >>> list_of_size(
    ...     x=st.integers(min_value=0, max_value=4),
    ...     size=10,
    ... )
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    >>> list_of_size(
    ...     x=st.integers(min_value=0, max_value=4),
    ...     size=10,
    ... )
    [3, 3, 2, 4, 1, 0, 4, 2, 1, 2]

    >>> list_of_size(
    ...     x=st.integers(min_value=0, max_value=4),
    ...     size=10,
    ... )
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    >>> list_of_size(
    ...     x=st.booleans(),
    ...     size=3,
    ... )
    [False, False, False]

    >>> list_of_size(
    ...     x=st.booleans(),
    ...     size=3,
    ... )
    [True, True, False]

    >>> list_of_size(
    ...     x=st.booleans(),
    ...     size=3,
    ... )
    [False, True, False]
    """
    return lists(x=x, min_size=size, max_size=size)


@st.composite
def lists(
    draw,
    *,
    x,
    min_size=None,
    max_size=None,
    size_bounds=None,
):
    """
    Draws a list with a random bounded size from the data-set x.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    x
        data-set of elements.
    min_size
        minimum size of the list.
    max_size
        max size of the list.
    size_bounds
        if min_size or max_size is None, draw them randomly from the range
        [size_bounds[0], size_bounds[1]].

    Returns
    -------
    ret
        A strategy that draws a list.

    Examples
    --------
    >>> lists(
    ...     x=st.sampled_from([-1, 5, 9]),
    ...     min_size=4,
    ...     max_size=5,
    ... )
    [5, 5, 5, 9, 9]

    >>> lists(
    ...     x=st.sampled_from([-1, 5, 9]),
    ...     min_size=4,
    ...     max_size=5,
    ... )
    [5, 9, -1, -1]

    >>> lists(
    ...     x=st.sampled_from([-1, 5, 9]),
    ...     min_size=4,
    ...     max_size=5,
    ... )
    [5, 9, 5, 9]

    >>> lists(
    ...     x=st.integers(min_value=0, max_value=4),
    ...     size_bounds=(9, 10),
    ... )
    [0, 2, 4, 3, 3, 3, 3, 2, 1, 4]

    >>> lists(
    ...     x=st.integers(min_value=0, max_value=4),
    ...     size_bounds=(9, 10),
    ... )
    [1, 0, 1, 2, 1, 4, 1, 3, 1]

    >>> lists(
    ...     x=st.integers(min_value=0, max_value=4),
    ...     size_bounds=(9, 10),
    ... )
    [1, 0, 1, 2, 1, 4, 1, 3, 1]

    >>> lists(
    ...     x=st.integers(min_value=0, max_value=4),
    ...     size_bounds=[9, 10],
    ... )
    [1, 3, 0, 2, 0, 0, 1, 4, 2, 3]

    >>> lists(
    ...     x=st.integers(min_value=0, max_value=4),
    ...     size_bounds=[9, 10],
    ... )
    [0, 0, 0, 0, 0, 0, 0, 0, 0]

    >>> lists(
    ...     x=st.integers(min_value=0, max_value=4),
    ...     size_bounds=[9, 10],
    ... )
    [1, 2, 4, 1, 1, 1, 4, 3, 2]

    >>> lists(
    ...     x=st.floats(
    ...         min_value=1,
    ...         max_value=3,
    ...         exclude_max=True,
    ...     ),
    ...     min_size=5,
    ...     max_size=5,
    ... )
    [1.1, 1.0, 1.0, 1.0, 1.0]

    >>> lists(
    ...     x=st.floats(
    ...         min_value=1,
    ...         max_value=3,
    ...         exclude_max=True,
    ...     ),
    ...     min_size=5,
    ...     max_size=5,
    ... )
    [2.00001, 2.00001, 1.0, 2.999999999999999, 1.9394938006792373]

    >>> lists(
    ...     x=st.floats(
    ...         min_value=1,
    ...         max_value=3,
    ...         exclude_max=True,
    ...     ),
    ...     min_size=5,
    ...     max_size=5,
    ... )
    [1.0, 2.00001, 1.0, 2.999999999999999, 1.9394938006792373]
    """
    if not isinstance(min_size, int) or not isinstance(max_size, int):
        integers = (
            number_helpers.ints(min_value=size_bounds[0], max_value=size_bounds[1])
            if size_bounds
            else number_helpers.ints()
        )
        if not isinstance(min_size, int):
            min_size = draw(st.shared(integers, key=min_size))
        if not isinstance(max_size, int):
            max_size = draw(st.shared(integers, key=max_size))

    return draw(st.lists(x, min_size=min_size, max_size=max_size))


@st.composite
def dtype_and_values(
    draw,
    *,
    available_dtypes=get_dtypes("valid"),
    num_arrays=1,
    abs_smallest_val=None,
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
    array_api_dtypes=False,
    shape_key="shape",
):
    """
    Draws a list of arrays with elements from the given corresponding data types.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    available_dtypes
        if dtype is None, data types are drawn from this list randomly.
    num_arrays
        Number of arrays to be drawn.
    abs_smallest_val
        sets the absolute smallest value to be generated for float data types,
        this has no effect on integer data types. If none, the default data type
        absolute smallest value is used.
    min_value
        minimum value of elements in each array.
    max_value
        maximum value of elements in each array.
    large_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,

        when a "linear" safety factor scaler is used, a safety factor of 2 means
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
        The operation to use for the safety factor scaling. Can be "linear" or "log".
        Default value = "linear".
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
    array_api_dtypes
        if True, use data types that can be promoted with the array_api_promotion
        table.

    Returns
    -------
    ret
        A strategy that draws a tuple of a list of dtypes and a list
        of their respective arrays.

    Examples
    --------
    >>> dtype_and_values(
    ...     num_arrays=3,
    ... )
    (['uint16', 'float16', 'uint16'], [array([37915, 6322, 26765, 12413,
        26986, 34665], dtype=uint16), array([-5.000e-01, -5.000e-01,
        -2.000e+00, -6.711e-05, -1.100e+00, -5.955e+04], dtype=float16),
        array([40817, 56193, 29200, 0, 5851, 9746], dtype=uint16)])

    >>> dtype_and_values(
    ...     num_arrays=3,
    ... )
    (['bool', 'uint32', 'bool'], [array(False), array(0, dtype=uint32),
        array(False)])

    >>> dtype_and_values(
    ...     num_arrays=3,
    ... )
    (['int8', 'int8', 'int8'], [array(0, dtype=int8), array(0, dtype=int8),
        array(0, dtype=int8)])

    >>> dtype_and_values(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     min_value=-10,
    ...     max_value=10,
    ...     num_arrays=2,
    ...     shared_dtype=True,
    ... ),
    (['float32', 'float32'], [array([1.1, 1.5], dtype=float32),
        array([-5.9604645e-08, 5.9604645e-08], dtype=float32)])

    >>> dtype_and_values(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     min_value=-10,
    ...     max_value=10,
    ...     num_arrays=2,
    ...     shared_dtype=True,
    ... ),
    (['int32', 'int32'], [array(-5, dtype=int32), array(-1, dtype=int32)])

    >>> dtype_and_values(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     min_value=-10,
    ...     max_value=10,
    ...     num_arrays=2,
    ...     shared_dtype=True,
    ... ),
    (['uint64', 'uint64'], [array([0], dtype=uint64), array([0],
        dtype=uint64)])

    >>> dtype_and_values(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=2,
    ...     ret_shape=True
    ... )
    (['int8', 'int32'], [array([27], dtype=int8), array([192],
        dtype=int32)], (1,))

    >>> dtype_and_values(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=2,
    ...     ret_shape=True
    ... )
    (['int32', 'int16'], [array(0, dtype=int32), array(0,
        dtype=int16)], ())

    >>> dtype_and_values(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=2,
    ...     ret_shape=True
    ... )
    (['int32', 'int16'], [array([[-103, 12, -41795, 1170789994,
        44251, 44209, 433075925]], dtype=int32), array([[24791,
        -24691, 24892, 16711, 7696, 972, 15357]], dtype=int16)],
        (1, 7))

    >>> dtype_and_values(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=1,
    ...     ret_shape=True,
    ... )
    (['uint8'], [array([0], dtype=uint8)], (1,))

    >>> dtype_and_values(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=1,
    ...     ret_shape=True,
    ... )
    (['float32'], [array(-1., dtype=float32)], ())

    >>> dtype_and_values(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=1,
    ...     ret_shape=True,
    ... )
    (['int64'], [array(72057594037927936)], ())
    """
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)
    if isinstance(available_dtypes, st._internal.SearchStrategy) and dtype is None:
        available_dtypes = draw(available_dtypes)
    if not isinstance(num_arrays, int):
        num_arrays = draw(num_arrays)
    if dtype is None:
        dtype = draw(
            dtype_helpers.array_dtypes(
                num_arrays=num_arrays,
                available_dtypes=available_dtypes,
                shared_dtype=shared_dtype,
                array_api_dtypes=array_api_dtypes,
            )
        )
    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            shape = draw(shape)
    else:
        shape = draw(
            st.shared(
                gh.get_shape(
                    min_num_dims=min_num_dims,
                    max_num_dims=max_num_dims,
                    min_dim_size=min_dim_size,
                    max_dim_size=max_dim_size,
                ),
                key=shape_key,
            )
        )
    values = []
    for i in range(num_arrays):
        values.append(
            draw(
                array_values(
                    dtype=dtype[i],
                    shape=shape,
                    abs_smallest_val=abs_smallest_val,
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
    if ret_shape:
        return dtype, values, shape
    return dtype, values


@st.composite
def dtype_values_axis(
    draw,
    *,
    available_dtypes,
    num_arrays=1,
    abs_smallest_val=None,
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
    force_tuple_axis=False,
    ret_shape=False,
):
    """
    Draws a list of arrays with elements from the given data type, and a random axis of
    the arrays.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    available_dtypes
        if dtype is None, data type is drawn from this list randomly.
    num_arrays
        Number of arrays to be drawn.
    abs_smallest_val
        sets the absolute smallest value to be generated for float data types,
        this has no effect on integer data types. If none, the default data type
        absolute smallest value is used.
    min_value
        minimum value of elements in the array.
    max_value
        maximum value of elements in the array.
    large_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,

        when a "linear" safety factor scaler is used, a safety factor of 2 means
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
    force_tuple_axis
        if true, all axis will be returned as a tuple.
    force_int_axis
        if true, the returned axis will be an int.
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
    ret
        A strategy that draws a tuple of a list of dtypes,
        a list of arrays, and an axis.

    Examples
    --------
    >>> dtype_values_axis()
    (['int16'], [array(29788, dtype=int16)])

    >>> dtype_values_axis()
    (['complex128'], [array(1.62222885e+156-2.68281172e-257j)])

    >>> dtype_values_axis()
    (['float64'], [array(-1.40129846e-45)])

    >>> dtype_values_axis(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=2,
    ... )
    (['int8', 'int16'], [array([[0]], dtype=int8), array([[1]], dtype=int16)], 0)

    >>> dtype_values_axis(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=2,
    ... )
    (['uint16', 'uint16'], [array(0, dtype=uint16), array(0, dtype=uint16)], 0)

    >>> dtype_values_axis(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=2,
    ... )
    (['float64', 'int16'], [array(-2.44758124e-308), array(0, dtype=int16)], 0)

    >>> dtype_values_axis(
    ...     available_dtypes=get_dtypes("float"),
    ...     min_num_dims=2,
    ...     max_num_dims=3,
    ...     min_dim_size=2,
    ...     max_dim_size=5,
    ...     min_axis=-2,
    ...     max_axis=1,
    ... )
    (['float64'], [array([[1.90000000e+000, 1.63426649e+308],
        [-1.50000000e+000, -1.91931887e+234]])], -1)

    >>> dtype_values_axis(
    ...     available_dtypes=get_dtypes("float"),
    ...     min_num_dims=2,
    ...     max_num_dims=3,
    ...     min_dim_size=2,
    ...     max_dim_size=5,
    ...     min_axis=-2,
    ...     max_axis=1,
    ... )
    (['bfloat16'], [array([[-1.29488e-38, -1.29488e-38],
        [-1.29488e-38, -1.29488e-38]], dtype=bfloat16)], 0)

    >>> dtype_values_axis(
    ...     available_dtypes=get_dtypes("float"),
    ...     min_num_dims=2,
    ...     max_num_dims=3,
    ...     min_dim_size=2,
    ...     max_dim_size=5,
    ...     min_axis=-2,
    ...     max_axis=1,
    ... )
    (['float64'], [array([[-2.44758124e-308, -2.44758124e-308],
        [-2.44758124e-308, -2.44758124e-308]])], 0)

    >>> dtype_values_axis(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=1,
    ...     allow_inf=True,
    ...     allow_nan=True,
    ... )
    (['float64'], [array([inf, -5.14361019e+16, 5.96046448e-08, 1.50000000e+00])], -51)

    >>> dtype_values_axis(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=1,
    ...     allow_inf=True,
    ...     allow_nan=True,
    ... )
    (['int16'], [array(12445, dtype=int16)], 171)

    >>> dtype_values_axis(
    ...     available_dtypes=get_dtypes("numeric"),
    ...     num_arrays=1,
    ...     allow_inf=True,
    ...     allow_nan=True,
    ... )
    (['uint32'], [array([0], dtype=uint32)], 0)
    """
    results = draw(
        dtype_and_values(
            available_dtypes=available_dtypes,
            num_arrays=num_arrays,
            abs_smallest_val=abs_smallest_val,
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
        if values[0].ndim == 0:
            axis = None
        else:
            axis = draw(
                gh.get_axis(
                    shape=arr_shape,
                    min_size=min_axes_size,
                    max_size=max_axes_size,
                    allow_neg=allow_neg_axes,
                    force_int=force_int_axis,
                    force_tuple=force_tuple_axis,
                )
            )
    else:
        axis = draw(number_helpers.ints(min_value=min_axis, max_value=max_axis))
    if ret_shape:
        return dtype, values, axis, arr_shape
    return dtype, values, axis


@st.composite
def array_indices_axis(
    draw,
    *,
    array_dtypes,
    indices_dtypes=get_dtypes("valid"),
    disable_random_axis=False,
    axis_zero=False,
    allow_inf=False,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    first_dimension_only=False,
    indices_same_dims=False,
    valid_bounds=True,
):
    """
    Generate two arrays x & indices, the values in the indices array are indices of the
    array x. Draws an integers randomly from the minimum and maximum number of
    positional arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    array_dtypes
        list of data type to draw the array dtype from.
    indices_dtypes
        list of data type to draw the indices dtype from.
    disable_random_axis
        axis is randomly generated with hypothesis if False. If True, axis is set
        to 0 if axis_zero is True, -1 otherwise.
    axis_zero
        If True, axis is set to zero if disable_random_axis is True.
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
    indices_same_dims
        Set x and indices dimensions to be the same
    valid_bounds
        If False, the strategy may produce out-of-bounds indices.

    Returns
    -------
    ret
        A strategy that can be used in the @given hypothesis
        decorator which generates arrays of values and indices.

    Examples
    --------
    @given(
        array_indices_axis=array_indices_axis(
            array_dtypes=helpers.get_dtypes("valid"),
            indices_dtypes=helpers.get_dtypes("integer"),
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10
            )
    )

    >>> array_indices_axis(
    ...    array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=["int64"],
    ...     max_num_dims=1,
    ...     indices_same_dims=True,
    ...     disable_random_axis=True,
    ...     axis_zero=True,
    ... )
    (['int64', 'int64'], array([-65536]), array([0]))

    >>> array_indices_axis(
    ...    array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=["int64"],
    ...     max_num_dims=1,
    ...     indices_same_dims=True,
    ...     disable_random_axis=True,
    ...     axis_zero=True,
    ... )
    (['bool', 'int64'], array([False, False, False, True,
        False, False, False, False]), array([0, 0, 2, 4,
        0, 0, 0, 1]))

    >>> array_indices_axis(
    ...    array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=["int64"],
    ...     max_num_dims=1,
    ...     indices_same_dims=True,
    ...     disable_random_axis=True,
    ...     axis_zero=True,
    ... )
    (['int64', 'int64'], array([0]), array([0]))

    >>> array_indices_axis(
    ...     array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=get_dtypes("integer"),
    ...     disable_random_axis=True,
    ...     first_dimension_only=True,
    ... )
    (['float64', 'uint64'], array([-2.44758124e-308]),
        array([0], dtype=uint64))

    >>> array_indices_axis(
    ...     array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=get_dtypes("integer"),
    ...     disable_random_axis=True,
    ...     first_dimension_only=True,
    ... )
    (['bool', 'uint64'], array([False]), array([0], dtype=uint64))

    >>> array_indices_axis(
    ...     array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=get_dtypes("integer"),
    ...     disable_random_axis=True,
    ...     first_dimension_only=True,
    ... )
    (['bool', 'int8'], array([False]), array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8))

    >>> array_indices_axis(
    ...     array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=["int64"],
    ...     max_num_dims=1,
    ...     indices_same_dims=True,
    ... )
    (['float16', 'int64'], array([-256.], dtype=float16),
        array([0]), 0, 0)

    >>> array_indices_axis(
    ...     array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=["int64"],
    ...     max_num_dims=1,
    ...     indices_same_dims=True,
    ... )
    (['uint8', 'int64'], array([1], dtype=uint8),
        array([0]), -1, 0)

    >>> array_indices_axis(
    ...     array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=["int64"],
    ...     max_num_dims=1,
    ...     indices_same_dims=True,
    ... )
    (['uint64', 'int64'], array([0], dtype=uint64),
        array([0]), 0, 0)
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
    x_dtype = x_dtype[0]
    x = x[0]
    if disable_random_axis:
        if axis_zero:
            axis = 0
        else:
            axis = -1
        batch_dims = 0
        batch_shape = x_shape[0:0]
    else:
        axis = draw(
            number_helpers.ints(
                min_value=-1 * len(x_shape),
                max_value=len(x_shape) - 1,
            )
        )
        batch_dims = draw(
            number_helpers.ints(
                min_value=0,
                max_value=max(0, axis),
            )
        )
        batch_shape = x_shape[0:batch_dims]
    if indices_same_dims:
        indices_shape = x_shape
    else:
        shape_var = draw(
            gh.get_shape(
                allow_none=False,
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims - batch_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            )
        )
        indices_shape = batch_shape + shape_var
    if first_dimension_only:
        max_axis = max(x_shape[0] - 1, 0)
    else:
        max_axis = max(x_shape[axis] - 1, 0)
    if not valid_bounds:
        max_axis = max_axis + 10
    indices_dtype, indices = draw(
        dtype_and_values(
            available_dtypes=indices_dtypes,
            allow_inf=False,
            min_value=0,
            max_value=max_axis,
            shape=indices_shape,
        )
    )
    indices_dtype = indices_dtype[0]
    indices = indices[0]
    if disable_random_axis:
        return [x_dtype, indices_dtype], x, indices

    return [x_dtype, indices_dtype], x, indices, axis, batch_dims


@st.composite
def array_indices_put_along_axis(
    draw,
    *,
    array_dtypes,
    indices_dtypes=get_dtypes("valid"),
    disable_random_axis=False,
    axis_zero=False,
    allow_inf=False,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    first_dimension_only=False,
    indices_same_dims=False,
    valid_bounds=True,
    values=None,
    values_dtypes=get_dtypes("valid"),
):
    """
    Generate two arrays x & indices, the values in the indices array are indices of the
    array x. Draws an integers randomly from the minimum and maximum number of
    positional arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    array_dtypes
        list of data type to draw the array dtype from.
    indices_dtypes
        list of data type to draw the indices dtype from.
    disable_random_axis
        axis is randomly generated with hypothesis if False. If True, axis is set
        to 0 if axis_zero is True, -1 otherwise.
    axis_zero
        If True, axis is set to zero if disable_random_axis is True.
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
    indices_same_dims
        Set x and indices dimensions to be the same
    valid_bounds
        If False, the strategy may produce out-of-bounds indices.
    values
        Custom values array to use instead of randomly generated values.
    values_dtypes : Union[None, List[str]]
        A list of dtypes for the values parameter. The function will use the dtypes
        returned by 'get_dtypes("valid")'.

    Returns
    -------
    ret
        A strategy that can be used in the @given hypothesis
        decorator which generates arrays of values and indices.

    Examples
    --------
    @given(
        array_indices_axis=array_indices_axis(
            array_dtypes=helpers.get_dtypes("valid"),
            indices_dtypes=helpers.get_dtypes("integer"),
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10
            )
    )

    >>> array_indices_axis(
    ...    array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=["int64"],
    ...     max_num_dims=1,
    ...     indices_same_dims=True,
    ...     disable_random_axis=True,
    ...     axis_zero=True,
    ... )
    (['int64', 'int64'], array([-65536]), array([0]))
    (['bool', 'int64'], array([False, False, False, True,
        False, False, False, False]), array([0, 0, 2, 4,
        0, 0, 0, 1]))
    (['int64', 'int64'], array([0]), array([0]))

    >>> array_indices_axis(
    ...     array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=get_dtypes("integer"),
    ...     disable_random_axis=True,
    ...     first_dimension_only=True,
    ... )
    (['float64', 'uint64'], array([-2.44758124e-308]),
        array([0], dtype=uint64))
    (['bool', 'uint64'], array([False]), array([0], dtype=uint64))
    (['bool', 'int8'], array([False]), array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8))

    >>> array_indices_axis(
    ...     array_dtypes=get_dtypes("valid"),
    ...     indices_dtypes=["int64"],
    ...     max_num_dims=1,
    ...     indices_same_dims=True,
    ... )
    (['float16', 'int64'], array([-256.], dtype=float16),
        array([0]), 0, 0)
    (['uint8', 'int64'], array([1], dtype=uint8),
        array([0]), -1, 0)
    (['uint64', 'int64'], array([0], dtype=uint64),
        array([0]), 0, 0)
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
    x_dtype = x_dtype[0]
    x = x[0]
    if disable_random_axis:
        if axis_zero:
            axis = 0
        else:
            axis = -1
        batch_dims = 0
        batch_shape = x_shape[0:0]
    else:
        axis = draw(
            number_helpers.ints(
                min_value=-1 * len(x_shape),
                max_value=len(x_shape) - 1,
            )
        )
        batch_dims = draw(
            number_helpers.ints(
                min_value=0,
                max_value=max(0, axis),
            )
        )
        batch_shape = x_shape[0:batch_dims]
    if indices_same_dims:
        indices_shape = x_shape
    else:
        shape_var = draw(
            gh.get_shape(
                allow_none=False,
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims - batch_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            )
        )
        indices_shape = batch_shape + shape_var
    if first_dimension_only:
        max_axis = max(x_shape[0] - 1, 0)
    else:
        max_axis = max(x_shape[axis] - 1, 0)
    if not valid_bounds:
        max_axis = max_axis + 10
    indices_dtype, indices = draw(
        dtype_and_values(
            available_dtypes=indices_dtypes,
            allow_inf=False,
            min_value=0,
            max_value=max_axis,
            shape=indices_shape,
        )
    )
    indices_dtype = indices_dtype[0]
    indices = indices[0]
    if disable_random_axis:
        return [x_dtype, indices_dtype], x, indices

    values_shape = indices_shape
    values_dtype, values = draw(
        dtype_and_values(
            available_dtypes=values_dtypes,
            allow_inf=False,
            shape=values_shape,
        )
    )
    values_dtype = values_dtype[0]
    values = values[0]
    return [x_dtype, indices_dtype, values_dtype], x, indices, axis, values, batch_dims


@st.composite
def arrays_and_axes(
    draw,
    available_dtypes=get_dtypes("float"),
    allow_none=False,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    num=2,
    return_dtype=False,
    force_int_axis=False,
):
    """
    Generate a list of arrays and axes.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    available_dtypes
        if dtype is None, data type is drawn from this list randomly.
    allow_none
        if True, one of the dimensions can be None
    min_num_dims
        The minimum number of dimensions the arrays can have.
    max_num_dims
        The maximum number of dimensions the arrays can have.
    min_dim_size
        The minimum size of the dimensions of the arrays.
    max_dim_size
        The maximum size of the dimensions of the arrays.
    num
        The number of arrays to be generated
    return_dtype
        If `True`, return a tuple of the form `(dtype, arrays, axes)` instead of
        `(arrays, axes)`
    force_int_axis
        If `True` and only one axis is drawn for each array, the returned axis will be
        an integer instead of a tuple containing one integer or `None`


    Returns
    -------
        A strategy that draws arrays and their axes

    Examples
    --------
    >>> arrays_and_axes(
    ...     allow_none=False,
    ...     min_num_dims=1,
    ...     max_num_dims=2,
    ...     min_dim_size=2,
    ...     max_dim_size=4,
    ...     num=2,
    ...     return_dtype=True,
    ... )
    (['float16', 'float16'], [array([[-1., -1.],
        [-1., -1.]], dtype=float16), array([[-1., -1.],
        [-1., -1.]], dtype=float16)], (None, None))

    >>> arrays_and_axes(
    ...     allow_none=False,
    ...     min_num_dims=1,
    ...     max_num_dims=2,
    ...     min_dim_size=2,
    ...     max_dim_size=4,
    ...     num=2,
    ...     return_dtype=True,
    ... )
    (['float16', 'float32'],
        [array([ 1.5 , -8.33], dtype=float16),
        array([8.26e+00, 9.10e+00, 6.72e-05], dtype=float16)],
        (0, None))

    >>> arrays_and_axes(
    ...     allow_none=False,
    ...     min_num_dims=1,
    ...     max_num_dims=2,
    ...     min_dim_size=2,
    ...     max_dim_size=4,
    ...     num=2,
    ...     return_dtype=True,
    ... )
    (['float64', 'float32'],
        [array([-1.1, -12.24322108]),
        array([[-2.44758124e-308, 8.26446279e+000, 5.96046448e-008],
        [1.17549435e-038, 1.06541027e-001, 1.13725760e+001]])],
        (None, None))

    >>> arrays_and_axes(
    ...     num=1,
    ...     force_int_axis=True,
    ... )
    ([array([0.07143888])], 0)

    >>> arrays_and_axes(
    ...     num=1,
    ...     force_int_axis=True,
    ... )
    ([array([-2.44758124e-308])], None)

    >>> arrays_and_axes(
    ...     num=1,
    ...     force_int_axis=True,
    ... )
    ([array([-6.72e-05, -6.72e-05, -6.72e-05, -6.72e-05, -6.72e-05],
        dtype=float16)], 0)
    """
    shapes = list()
    for _ in range(num):
        shape = draw(
            gh.get_shape(
                allow_none=False,
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            )
        )
        shapes.append(shape)
    if isinstance(available_dtypes, st._internal.SearchStrategy):
        available_dtypes = draw(available_dtypes)

    dtype = draw(
        dtype_helpers.array_dtypes(num_arrays=num, available_dtypes=available_dtypes)
    )
    arrays = list()
    for shape in shapes:
        arrays.append(
            draw(array_values(dtype=dtype[0], shape=shape, min_value=-20, max_value=20))
        )
    if force_int_axis:
        # ToDo: the following code references shape as the last element in shapes
        # while this is not the intended behavior of the code. This is a bug
        # that should be fixed in the future.
        if len(shape) <= 2:
            axes = draw(st.one_of(st.integers(0, len(shape) - 1), st.none()))
        else:
            axes = draw(st.integers(0, len(shape) - 1))
    else:
        all_axes_ranges = list()
        for shape in shapes:
            if None in all_axes_ranges:
                all_axes_ranges.append(st.integers(0, len(shape) - 1))
            else:
                all_axes_ranges.append(
                    st.one_of(st.none(), st.integers(0, len(shape) - 1))
                )
        axes = draw(st.tuples(*all_axes_ranges))
    if return_dtype:
        return dtype, arrays, axes
    return arrays, axes


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
    abs_smallest_val=None,
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
    """
    Draws a list (of lists) of a given shape containing values of a given data type.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    dtype
        data type of the elements of the list.
    shape
        shape of the required list.
    abs_smallest_val
        sets the absolute smallest value to be generated for float data types,
        this has no effect on integer data types. If none, the default data type
        absolute smallest value is used.
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

        when a "linear" safety factor scaler is used, a safety factor of 2 means
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

    Examples
    --------
    >>> array_values(
    ...     dtype=get_dtypes("valid"),
    ...     shape=get_shape(),
    ... )
    [1806 87 36912 6955 59576]

    >>> array_values(
    ...     dtype=get_dtypes("valid"),
    ...     shape=get_shape(),
    ... )
    1025
    """
    assert small_abs_safety_factor >= 1, "small_abs_safety_factor must be >= 1"
    assert large_abs_safety_factor >= 1, "large_value_safety_factor must be >= 1"

    if isinstance(shape, st._internal.SearchStrategy):
        shape = draw(shape)

    size = 1
    if isinstance(shape, int):
        size = shape
    else:
        for dim in shape:
            size *= dim

    if isinstance(dtype, st._internal.SearchStrategy):
        dtype = draw(dtype)
        dtype = dtype[0] if isinstance(dtype, list) else draw(dtype)

    if "float" in dtype or "complex" in dtype:
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
            " only integers, floats and booleans are allowed."
        )

    if kind_dtype != "bool":
        if min_value is not None:
            min_value = _clamp_value(min_value, dtype_info)

        if max_value is not None:
            max_value = _clamp_value(max_value, dtype_info)

        min_value, max_value, abs_smallest_val = gh.apply_safety_factor(
            dtype,
            min_value=min_value,
            max_value=max_value,
            abs_smallest_val=abs_smallest_val,
            small_abs_safety_factor=small_abs_safety_factor,
            large_abs_safety_factor=large_abs_safety_factor,
            safety_factor_scale=safety_factor_scale,
        )
        assert max_value >= min_value

        if kind_dtype == "int":
            if exclude_min:
                min_value += 1
            if exclude_max:
                max_value -= 1
            values = draw(list_of_size(x=st.integers(min_value, max_value), size=size))
        elif kind_dtype == "float":
            floats_info = {
                "float16": {"cast_type": "float16", "width": 16},
                "bfloat16": {"cast_type": "float32", "width": 32},
                "float32": {"cast_type": "float32", "width": 32},
                "float64": {"cast_type": "float64", "width": 64},
                "complex64": {"cast_type": "complex64", "width": 32},
                "complex128": {"cast_type": "complex128", "width": 64},
            }
            # The smallest possible value is determined by one of the arguments
            if min_value > -abs_smallest_val or max_value < abs_smallest_val:
                float_strategy = st.floats(
                    min_value=float_of(min_value, floats_info[dtype]["width"]),
                    max_value=float_of(max_value, floats_info[dtype]["width"]),
                    allow_nan=allow_nan,
                    allow_subnormal=allow_subnormal,
                    allow_infinity=allow_inf,
                    width=floats_info[dtype]["width"],
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                )
            # kind of a hack to not use the calculated max and min values
            elif allow_inf or allow_nan:
                float_strategy = st.floats(
                    allow_nan=allow_nan,
                    allow_subnormal=allow_subnormal,
                    allow_infinity=allow_inf,
                    width=floats_info[dtype]["width"],
                )
            else:
                float_strategy = st.one_of(
                    st.floats(
                        min_value=float_of(min_value, floats_info[dtype]["width"]),
                        max_value=float_of(
                            -abs_smallest_val, floats_info[dtype]["width"]
                        ),
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=floats_info[dtype]["width"],
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                    st.floats(
                        min_value=float_of(
                            abs_smallest_val, floats_info[dtype]["width"]
                        ),
                        max_value=float_of(max_value, floats_info[dtype]["width"]),
                        allow_nan=allow_nan,
                        allow_subnormal=allow_subnormal,
                        allow_infinity=allow_inf,
                        width=floats_info[dtype]["width"],
                        exclude_min=exclude_min,
                        exclude_max=exclude_max,
                    ),
                )
            if "complex" in dtype:
                float_strategy = st.tuples(float_strategy, float_strategy)
            values = draw(list_of_size(x=float_strategy, size=size))
            if "complex" in dtype:
                values = [complex(*v) for v in values]
    else:
        values = draw(list_of_size(x=st.booleans(), size=size))
    if dtype == "bfloat16":
        # check bfloat16 behavior enabled or not
        try:
            np.dtype("bfloat16")
        except Exception:
            # enables bfloat16 behavior with possibly no side effects

            import paddle_bfloat  # noqa

    array = np.asarray(values, dtype=dtype)

    if isinstance(shape, (tuple, list)):
        return array.reshape(shape)
    return np.asarray(array)


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
    return _reduce(mul, seq, 1)


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
    """Return an array and a shape that the array can be broadcast to."""
    if isinstance(dtype, st._internal.SearchStrategy):
        dtype = draw(dtype)
        dtype = dtype[0] if isinstance(dtype, list) else draw(dtype)
    in_shape = draw(nph.array_shapes(min_dims=1, max_dims=4))
    x = draw(array_values(shape=in_shape, dtype=dtype))
    to_shape = draw(
        mutually_broadcastable_shapes(1, base_shape=in_shape)
        .map(lambda S: S[0])
        .filter(lambda s: broadcast_shapes(in_shape, s) == s),
        label="shape",
    )
    return x, to_shape


@st.composite
def arrays_for_pooling(
    draw,
    min_dims,
    max_dims,
    min_side,
    max_side,
    explicit_or_str_padding=False,
    only_explicit_padding=False,
    return_dilation=False,
    mixed_fn_compos=True,
    data_format="channel_last",
):
    in_shape = draw(
        nph.array_shapes(
            min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=max_side
        )
    )
    dtype, x = draw(
        dtype_and_values(
            available_dtypes=get_dtypes("float", mixed_fn_compos=mixed_fn_compos),
            shape=in_shape,
            num_arrays=1,
            max_value=100,
            min_value=-100,
        )
    )

    if not isinstance(data_format, str):
        data_format = draw(data_format)
    array_dim = x[0].ndim
    if array_dim == 5:
        kernel = draw(
            st.tuples(
                st.integers(1, in_shape[1]),
                st.integers(1, in_shape[2]),
                st.integers(1, in_shape[3]),
            )
        )
    if array_dim == 4:
        kernel = draw(
            st.tuples(st.integers(1, in_shape[1]), st.integers(1, in_shape[2]))
        )
    if array_dim == 3:
        kernel = draw(st.tuples(st.integers(1, in_shape[1])))
    new_kernel = kernel
    if return_dilation:
        new_kernel = []
        dilations = []
        for i in range(len(kernel)):
            if kernel[i] > 1:
                max_dilation = (in_shape[i + 1] - kernel[i]) // (kernel[i] - 1) + 1
                dilations.append(draw(st.integers(1, max_dilation)))
                new_kernel.append(kernel[i] + (kernel[i] - 1) * (dilations[i] - 1))
            else:
                dilations.append(1)
                new_kernel.append(kernel[i])
    if explicit_or_str_padding or only_explicit_padding:
        padding = []
        for i in range(array_dim - 2):
            max_pad = new_kernel[i] // 2
            padding.append(
                draw(
                    st.tuples(
                        st.integers(0, max_pad),
                        st.integers(0, max_pad),
                    )
                )
            )
        if explicit_or_str_padding:
            padding = draw(
                st.one_of(st.just(padding), st.sampled_from(["VALID", "SAME"]))
            )
    else:
        padding = draw(st.sampled_from(["VALID", "SAME"]))

    strides = draw(st.tuples(st.integers(1, min(kernel))))

    if data_format == "channel_first":
        dim = len(in_shape)
        x[0] = np.transpose(x[0], (0, dim - 1, *range(1, dim - 1)))

    if return_dilation:
        return dtype, x, kernel, strides, padding, dilations
    return dtype, x, kernel, strides, padding


@st.composite
def dtype_array_query(
    draw,
    *,
    available_dtypes,
    min_num_dims=1,
    max_num_dims=3,
    min_dim_size=0,
    max_dim_size=10,
    allow_mask=True,
    allow_neg_step=True,
):
    dtype = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=available_dtypes,
        )
    )
    shape = draw(
        helpers.get_shape(
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    array = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape,
            large_abs_safety_factor=2,
            small_abs_safety_factor=2,
        )
    )
    if allow_mask and draw(st.booleans()):
        mask_shape = shape[: draw(st.integers(0, len(shape)))]
        index = draw(
            helpers.array_values(
                dtype="bool",
                shape=mask_shape,
            ).filter(lambda x: np.sum(x) > 0)
        )
        return dtype + ["bool"], array, index
    supported_index_types = ["int", "slice", "list", "array"]
    index_types = draw(
        st.lists(
            st.sampled_from(supported_index_types),
            min_size=0,
            max_size=len(shape),
        )
    )
    index_types = [v if shape[i] > 0 else "slice" for i, v in enumerate(index_types)]
    index = []
    for s, index_type in zip(shape, index_types):
        if index_type == "int":
            new_index = draw(st.integers(min_value=-s + 1, max_value=s - 1))
        elif index_type == "seq":
            new_index = draw(
                st.lists(
                    st.integers(min_value=-s + 1, max_value=s - 1),
                    min_size=1,
                    max_size=20,
                )
            )
        elif index_type == "array":
            _, new_index = draw(
                helpers.dtype_and_values(
                    min_value=-s + 1,
                    max_value=s - 1,
                    dtype=["int64"],
                )
            )
            new_index = new_index[0]
        else:
            new_index = slice(
                start := draw(
                    st.one_of(
                        st.integers(min_value=-s - 5, max_value=s + 5),
                        st.just(None),
                    )
                ),
                end := draw(
                    st.one_of(
                        st.integers(min_value=-s - 5, max_value=s + 5),
                        st.just(None),
                    )
                ),
                (
                    draw(st.integers(min_value=1, max_value=s + 5))
                    if (0 if start is None else s + start if start < 0 else start)
                    < (s - 1 if end is None else s + end if end < 0 else end)
                    else (
                        draw(st.integers(max_value=-1, min_value=-s - 5))
                        if allow_neg_step
                        else assume(False)
                    )
                ),
            )
        index += [new_index]
    if len(index_types) and draw(st.booleans()):
        start = draw(st.integers(min_value=0, max_value=len(index) - 1))
        min_ = len(index) if len(index_types) < len(shape) else start
        max_ = len(index) if len(index_types) < len(shape) else len(index) - 1
        end = draw(st.integers(min_value=min_, max_value=max_))
        if start != end:
            index = index[:start] + [Ellipsis] + index[end:]
    for _ in range(draw(st.integers(min_value=0, max_value=3))):
        index.insert(draw(st.integers(0, len(index))), None)
    index = tuple(index)
    if len(index) == 1 and draw(st.booleans()):
        index = index[0]
    return dtype + ["int64"] * index_types.count("array"), array, index


@st.composite
def dtype_array_query_val(
    draw,
    *,
    available_dtypes,
    min_num_dims=1,
    max_num_dims=3,
    min_dim_size=0,
    max_dim_size=10,
    allow_mask=True,
    allow_neg_step=True,
):
    input_dtype, x, query = draw(
        helpers.dtype_array_query(
            available_dtypes=available_dtypes,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
            allow_mask=allow_mask,
            allow_neg_step=allow_neg_step,
        )
    )
    real_shape = x[query].shape
    if len(real_shape):
        val_shape = real_shape[draw(st.integers(0, len(real_shape))) :]
    else:
        val_shape = real_shape
    val_dtype, val = draw(
        helpers.dtype_and_values(
            dtype=[input_dtype[0]],
            shape=val_shape,
            large_abs_safety_factor=2,
            small_abs_safety_factor=2,
        )
    )
    val_dtype = draw(
        helpers.get_castable_dtype(
            draw(available_dtypes),
            input_dtype[0],
            x if 0 not in x.shape else None
        )
    )[-1]
    val = val[0].astype(val_dtype)
    return input_dtype + [val_dtype], x, query, val


@st.composite
def create_nested_input(draw, dimensions, leaf_values):
    if len(dimensions) != 1:
        return [
            draw(create_nested_input(dimensions[1:], leaf_values))
            for _ in range(dimensions[0])
        ]
    value = draw(st.sampled_from(leaf_values))
    return [value for _ in range(dimensions[0])]


@st.composite
def cond_data_gen_helper(draw):
    dtype_x = helpers.dtype_and_values(
        available_dtypes=(ivy.float32, ivy.float64),
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
        max_value=10,
        min_value=-10,
        allow_nan=False,
        shared_dtype=True,
    ).filter(lambda x: np.linalg.cond(x[1][0].tolist()) < 1 / sys.float_info.epsilon)
    p = draw(
        st.sampled_from([None, 2, -2, 1, -1, "fro", "nuc", float("inf"), -float("inf")])
    )
    dtype, x = draw(dtype_x)
    return dtype, (x[0], p)


# helpers for tests (core and frontend) related to solve function
@st.composite
def get_first_solve_matrix(draw, adjoint=True):
    # batch_shape, random_size, shared

    # float16 causes a crash when filtering out matrices
    # for which `np.linalg.cond` is large.
    input_dtype_strategy = st.shared(
        st.sampled_from(draw(helpers.get_dtypes("float"))).filter(
            lambda x: "float16" not in x
        ),
        key="shared_dtype",
    )
    input_dtype = draw(input_dtype_strategy)

    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    matrix = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple([shared_size, shared_size]),
            min_value=2,
            max_value=5,
        ).filter(lambda x: np.linalg.cond(x) < 1 / sys.float_info.epsilon)
    )
    if adjoint:
        adjoint = draw(st.booleans())
        if adjoint:
            matrix = np.transpose(np.conjugate(matrix))
    return input_dtype, matrix, adjoint


@st.composite
def get_second_solve_matrix(draw):
    # batch_shape, shared, random_size
    # float16 causes a crash when filtering out matrices
    # for which `np.linalg.cond` is large.
    input_dtype_strategy = st.shared(
        st.sampled_from(draw(helpers.get_dtypes("float"))).filter(
            lambda x: "float16" not in x
        ),
        key="shared_dtype",
    )
    input_dtype = draw(input_dtype_strategy)

    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype, shape=tuple([shared_size, 1]), min_value=2, max_value=5
        )
    )


@st.composite
def einsum_helper(draw):
    # Todo: generalize to n equations and arrays

    # generate shapes as lists initially to allow updates in the loop ahead
    shape_1 = draw(
        st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5)
    )
    shape_2 = draw(
        st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5)
    )
    dims_1 = len(shape_1)
    dims_2 = len(shape_2)

    # generate equations as lists to allow updates and use unique=True
    eq_1 = draw(
        st.lists(
            st.sampled_from(string.ascii_lowercase),
            min_size=dims_1,
            max_size=dims_1,
            unique=True,
        )
    )
    eq_2 = draw(
        st.lists(
            st.sampled_from(string.ascii_lowercase),
            min_size=dims_2,
            max_size=dims_2,
            unique=True,
        )
    )

    # randomly change some of the dimensions and equations to match
    for i in range(min(dims_1, dims_2)):
        change = draw(st.booleans())
        if change:
            shape_2[i] = shape_1[i]
            eq_2[i] = eq_1[i]

    shape_1 = tuple(shape_1)
    shape_2 = tuple(shape_2)

    # join the lists to strings
    eq_1 = "".join(eq_1)
    eq_2 = "".join(eq_2)

    # generate arrays and dtypes
    dtype_1, value_1 = draw(
        dtype_and_values(
            available_dtypes=["float32"],
            shape=shape_1,
        )
    )

    dtype_2, value_2 = draw(
        dtype_and_values(
            available_dtypes=["float32"],
            shape=shape_2,
        )
    )

    output_length = min(dims_1, dims_2)
    output_eq = draw(
        st.lists(
            st.sampled_from(eq_1 + eq_2),
            min_size=output_length,
            max_size=output_length,
            unique=True,
        )
    )
    output_eq = "".join(output_eq)

    # explict einsum equation
    eq = eq_1 + "," + eq_2 + "->" + output_eq

    return eq, (value_1[0], value_2[0]), [dtype_1[0], dtype_2[0]]
