# global
from hypothesis import strategies as st
from functools import lru_cache
import math
import numpy as np

# local
import ivy
from . import array_helpers, number_helpers, dtype_helpers
from ..pipeline_helper import WithBackendContext
from ivy.functional.ivy.layers import _deconv_length


def matrix_is_stable(x, cond_limit=30):
    """
    Check if a matrix is numerically stable or not.

    Used to avoid numerical instabilities in further computationally heavy calculations.

    Parameters
    ----------
    x
        The original matrix whose condition number is to be determined.
    cond_limit
        The greater the condition number, the more ill-conditioned the matrix
        will be, the more it will be prone to numerical instabilities.

        There is no rule of thumb for what the exact condition number
        should be to consider a matrix ill-conditioned(prone to numerical errors).
        But, if the condition number is "1", the matrix is perfectly said to be a
        well-conditioned matrix which will not be prone to any type of numerical
        instabilities in further calculations, but that would probably be a
        very simple matrix.

        The cond_limit should start with "30", gradually decreasing it according
        to our use, lower cond_limit would result in more numerically stable
        matrices but more simple matrices.

        The limit should always be in the range "1-30", greater the number greater
        the computational instability. Should not increase 30, it leads to strong
        multi-collinearity which leads to singularity.

    Returns
    -------
    ret
        If True, the matrix is suitable for further numerical computations.
    """
    return np.all(np.linalg.cond(x.astype("float64")) <= cond_limit)


@lru_cache(None)
def apply_safety_factor(
    dtype,
    *,
    backend: str,
    min_value=None,
    max_value=None,
    abs_smallest_val=None,
    small_abs_safety_factor=1.1,
    large_abs_safety_factor=1.1,
    safety_factor_scale="linear",
):
    """
    Apply safety factor scaling to numeric data type.

    Parameters
    ----------
    dtype
        the data type to apply safety factor scaling to.
    min_value
        the minimum value of the data type.
    max_value
        the maximum value of the data type.
    abs_smallest_val
        the absolute smallest representable value of the data type.
    large_abs_safety_factor
        the safety factor to apply to the maximum value.
    small_abs_safety_factor
        the safety factor to apply to the minimum value.
    safety_factor_scale
        the scale to apply the safety factor to, either 'linear' or 'log'.

    Returns
    -------
        A tuple of the minimum value, maximum value and absolute smallest representable
    """
    assert small_abs_safety_factor >= 1, "small_abs_safety_factor must be >= 1"
    assert large_abs_safety_factor >= 1, "large_value_safety_factor must be >= 1"

    if "float" in dtype or "complex" in dtype:
        kind_dtype = "float"
        with WithBackendContext(backend) as ivy_backend:
            dtype_info = ivy_backend.finfo(dtype)
    elif "int" in dtype:
        kind_dtype = "int"
        with WithBackendContext(backend) as ivy_backend:
            dtype_info = ivy_backend.iinfo(dtype)
    else:
        raise TypeError(
            f"{dtype} is not a valid numeric data type only integers and floats"
        )

    if min_value is None:
        min_value = dtype_info.min
    if max_value is None:
        max_value = dtype_info.max

    if safety_factor_scale == "linear":
        min_value = min_value / large_abs_safety_factor
        max_value = max_value / large_abs_safety_factor
        if kind_dtype == "float" and not abs_smallest_val:
            abs_smallest_val = dtype_info.smallest_normal * small_abs_safety_factor
    elif safety_factor_scale == "log":
        min_sign = math.copysign(1, min_value)
        min_value = abs(min_value) ** (1 / large_abs_safety_factor) * min_sign
        max_sign = math.copysign(1, max_value)
        max_value = abs(max_value) ** (1 / large_abs_safety_factor) * max_sign
        if kind_dtype == "float" and not abs_smallest_val:
            m, e = math.frexp(dtype_info.smallest_normal)
            abs_smallest_val = m * (2 ** (e / small_abs_safety_factor))
    else:
        raise ValueError(
            f"{safety_factor_scale} is not a valid safety factor scale."
            " use 'log' or 'linear'."
        )
    if kind_dtype == "int":
        return int(min_value), int(max_value), None
    return min_value, max_value, abs_smallest_val


# Hypothesis #
# -----------#


# taken from
# https://github.com/data-apis/array-api-tests/array_api_tests/test_manipulation_functions.py
@st.composite
def reshape_shapes(draw, *, shape):
    """
    Draws a random shape with the same number of elements as the given shape.

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
    rshape = draw(
        st.lists(number_helpers.ints(min_value=0)).filter(
            lambda s: math.prod(s) == size
        )
    )
    return tuple(rshape)


# taken from https://github.com/HypothesisWorks/hypothesis/issues/1115
@st.composite
def subsets(draw, *, elements):
    """
    Draws a subset of elements from the given elements.

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
def get_shape(
    draw,
    *,
    allow_none=False,
    min_num_dims=0,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
):
    """
    Draws a tuple of integers drawn randomly from [min_dim_size, max_dim_size] of size
    drawn from min_num_dims to max_num_dims. Useful for randomly drawing the shape of an
    array.

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
                number_helpers.ints(min_value=min_dim_size, max_value=max_dim_size),
                min_size=min_num_dims,
                max_size=max_num_dims,
            )
        )
    else:
        shape = draw(
            st.lists(
                number_helpers.ints(min_value=min_dim_size, max_value=max_dim_size),
                min_size=min_num_dims,
                max_size=max_num_dims,
            )
        )
    if shape is None:
        return shape
    return tuple(shape)


@st.composite
def get_mean_std(draw, *, dtype):
    """
    Draws two integers representing the mean and standard deviation for a given data
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
    none_or_float = none_or_float = number_helpers.floats(dtype=dtype) | st.none()
    values = draw(array_helpers.list_of_size(x=none_or_float, size=2))
    values[1] = abs(values[1]) if values[1] else None
    return values[0], values[1]


@st.composite
def get_bounds(draw, *, dtype):
    """
    Draws two numbers; low and high, for a given data type such that low < high.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    dtype
        data type.

    Returns
    -------
        A strategy that draws a list of two numbers.
    """
    if "int" in dtype:
        values = draw(array_helpers.array_values(dtype=dtype, shape=2))
        values[0], values[1] = abs(values[0]), abs(values[1])
        low, high = min(values), max(values)
        if low == high:
            return draw(get_bounds(dtype=dtype))
    else:
        none_or_float = number_helpers.floats(dtype=dtype) | st.none()
        values = draw(array_helpers.list_of_size(x=none_or_float, size=2))
        if values[0] is not None and values[1] is not None:
            low, high = min(values), max(values)
        else:
            low, high = values[0], values[1]
        if ivy.default(low, 0.0) >= ivy.default(high, 1.0):
            return draw(get_bounds(dtype=dtype))
    return [low, high]


@st.composite
def get_axis(
    draw,
    *,
    shape,
    allow_neg=True,
    allow_none=False,
    sort_values=True,
    unique=True,
    min_size=1,
    max_size=None,
    force_tuple=False,
    force_int=False,
):
    """
    Draws one or more axis for the given shape.

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
    sort_values
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
        A strategy that draws an axis or axes.
    """
    assert not (
        force_int and force_tuple
    ), "Cannot return an int and a tuple. If both are valid then set both to False."

    # Draw values from any strategies given
    if isinstance(shape, st._internal.SearchStrategy):
        shape = draw(shape)
    if isinstance(min_size, st._internal.SearchStrategy):
        min_size = draw(min_size)
    if isinstance(max_size, st._internal.SearchStrategy):
        max_size = draw(max_size)

    axes = len(shape)
    lower_axes_bound = axes if allow_neg else 0

    if max_size is None and unique:
        max_size = max(axes, min_size)

    valid_strategies = []

    if allow_none:
        valid_strategies.append(st.none())

    if min_size > 1:
        force_tuple = True

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
                    unique=unique,
                )
            )

    axis = draw(
        st.one_of(*valid_strategies).filter(
            lambda x: (
                all([i != axes + j for i in x for j in x])
                if (isinstance(x, list) and unique and allow_neg)
                else True
            )
        )
    )

    if type(axis) == list:
        if sort_values:

            def sort_key(ele, max_len):
                if ele < 0:
                    return ele + max_len
                return ele

            axis.sort(key=(lambda ele: sort_key(ele, axes)))
        axis = tuple(axis)
    return axis


@st.composite
def x_and_filters(
    draw,
    dim: int = 2,
    transpose: bool = False,
    depthwise=False,
    mixed_fn_compos=True,
):
    """
    Draws a random x and filters for a convolution.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    dim
        the dimension of the convolution
    transpose
        if True, draw a transpose convolution
    depthwise
        if True, draw a depthwise convolution

    Returns
    -------
        A strategy that draws a random x and filters for a convolution.
    """
    strides = draw(st.integers(min_value=1, max_value=2))
    padding = draw(st.sampled_from(["SAME", "VALID"]))
    batch_size = draw(st.integers(1, 5))
    filter_shape = draw(
        get_shape(min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5)
    )
    input_channels = draw(st.integers(1, 5))
    output_channels = draw(st.integers(1, 5))
    dilations = draw(st.integers(1, 2))
    dtype = draw(
        dtype_helpers.get_dtypes("float", mixed_fn_compos=mixed_fn_compos, full=False)
    )
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
                _deconv_length(x_dim[i], strides, filter_shape[i], padding, dilations)
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
        array_helpers.array_values(
            shape=x_shape,
            dtype=dtype[0],
            large_abs_safety_factor=3,
            small_abs_safety_factor=4,
            safety_factor_scale="log",
        )
    )
    filters = draw(
        array_helpers.array_values(
            shape=filter_shape,
            dtype=dtype[0],
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


@st.composite
def embedding_helper(draw, mixed_fn_compos=True):
    """
    Obtain weights for embeddings, the corresponding indices, the padding indices.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).

    Returns
    -------
        A strategy for generating a tuple
    """
    dtype_weight, weight = draw(
        array_helpers.dtype_and_values(
            available_dtypes=[
                x
                for x in draw(
                    dtype_helpers.get_dtypes("numeric", mixed_fn_compos=mixed_fn_compos)
                )
                if "float" in x or "complex" in x
            ],
            min_num_dims=2,
            max_num_dims=2,
            min_dim_size=1,
            min_value=-1e04,
            max_value=1e04,
        )
    )
    num_embeddings, embedding_dim = weight[0].shape
    dtype_indices, indices = draw(
        array_helpers.dtype_and_values(
            available_dtypes=["int32", "int64"],
            min_num_dims=2,
            min_dim_size=1,
            min_value=0,
            max_value=num_embeddings - 1,
        ).filter(lambda x: x[1][0].shape[-1] == embedding_dim)
    )
    padding_idx = draw(st.integers(min_value=0, max_value=num_embeddings - 1))
    return dtype_indices + dtype_weight, indices[0], weight[0], padding_idx
