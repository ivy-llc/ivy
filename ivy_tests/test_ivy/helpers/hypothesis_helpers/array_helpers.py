# global
import numpy as np
import hypothesis.extra.numpy as nph
from hypothesis import strategies as st
from hypothesis.internal.floats import float_of
from functools import reduce
from operator import mul

# local
import ivy
from ivy_tests.test_ivy.helpers.hypothesis_helpers.dtype_helpers import get_dtypes
from . import general_helpers as gh
from . import dtype_helpers, number_helpers


@st.composite
def array_bools(
    draw,
    *,
    num_arrays=st.shared(
        number_helpers.ints(min_value=1, max_value=4), key="num_arrays"
    ),
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


def list_of_length(*, x, length):
    """Returns a random list of the given length from elements in x."""
    return st.lists(x, min_size=length, max_size=length)


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
        number_helpers.ints(min_value=size_bounds[0], max_value=size_bounds[1])
        if size_bounds
        else number_helpers.ints()
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
    """Draws a list of arrays with elements from the given data type,
    and a random axis of the arrays.

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
    force_tuple_axis
        if true, all axis will be returned as a tuple.
    force_int_axis
        if true and only one axis is drawn, the returned axis will be an int.
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
    indices_same_dims
        Set x and indices dimensions to be the same

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator
    which generates arrays of values and indices.

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
    max_axis = max(x_shape[axis] - 1, 0)
    if first_dimension_only:
        max_axis = max(x_shape[0] - 1, 0)
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
def arrays_and_axes(
    draw,
    available_dtypes=get_dtypes("float"),
    allow_none=False,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    num=2,
    returndtype=False,
    force_int_axis=False,
):
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
    if returndtype:
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
            f" only integers, floats and booleans are allowed."
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
            values = draw(
                list_of_length(x=st.integers(min_value, max_value), length=size)
            )
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
            values = draw(
                list_of_length(
                    x=float_strategy,
                    length=size,
                )
            )
            if "complex" in dtype:
                values = [complex(*v) for v in values]
    else:
        values = draw(list_of_length(x=st.booleans(), length=size))

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
    draw, min_dims, max_dims, min_side, max_side, allow_explicit_padding=False
):
    in_shape = draw(
        nph.array_shapes(
            min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=max_side
        )
    )
    dtype, x = draw(
        dtype_and_values(
            available_dtypes=get_dtypes("float"),
            shape=in_shape,
            num_arrays=1,
            max_value=100,
            min_value=-100,
        )
    )
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
    if allow_explicit_padding:
        padding = []
        for i in range(array_dim - 2):
            max_pad = kernel[i] // 2
            possible_pad_combos = [
                (i, max_pad - i)
                for i in range(0, max_pad)
                if i + (max_pad - i) == max_pad
            ]
            if len(possible_pad_combos) == 0:
                pad_selected_combo = (0, 0)
            else:
                pad_selected_combo = draw(st.sampled_from(possible_pad_combos))
            padding.append(
                draw(
                    st.tuples(
                        st.integers(0, pad_selected_combo[0]),
                        st.integers(0, pad_selected_combo[1]),
                    )
                )
            )
        padding = draw(st.one_of(st.just(padding), st.sampled_from(["VALID", "SAME"])))
    else:
        padding = draw(st.sampled_from(["VALID", "SAME"]))
    strides = draw(st.tuples(st.integers(1, in_shape[1])))
    return dtype, x, kernel, strides, padding
