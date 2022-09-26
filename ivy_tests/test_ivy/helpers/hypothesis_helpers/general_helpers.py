# global
from hypothesis import strategies as st
import math

# local
import ivy
from . import array_helpers, number_helpers, dtype_helpers


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


# Hypothesis #
# -----------#

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
    rshape = draw(
        st.lists(number_helpers.ints(min_value=0)).filter(
            lambda s: math.prod(s) == size
        )
    )
    # assume(all(side <= MAX_SIDE for side in rshape))
    if len(rshape) != 0 and size > 0 and draw(st.booleans()):
        index = draw(number_helpers.ints(min_value=0, max_value=len(rshape) - 1))
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
    values = draw(number_helpers.none_or_list_of_floats(dtype=dtype, size=2))
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
        values = draw(array_helpers.array_values(dtype=dtype, shape=2))
        values[0], values[1] = abs(values[0]), abs(values[1])
        low, high = min(values), max(values)
        if low == high:
            return draw(get_bounds(dtype=dtype))
    else:
        values = draw(number_helpers.none_or_list_of_floats(dtype=dtype, size=2))
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
    dtype = draw(dtype_helpers.get_dtypes("float", full=False))
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
        array_helpers.array_values(
            dtype=dtype[0],
            shape=x_shape,
            large_abs_safety_factor=3,
            small_abs_safety_factor=4,
            safety_factor_scale="log",
        )
    )
    filters = draw(
        array_helpers.array_values(
            dtype=dtype[0],
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
