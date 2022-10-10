# For Review
"""Collection of tests for manipulation functions."""

# global

import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )
    xs = list()
    input_dtypes = draw(
        helpers.array_dtypes(available_dtypes=draw(helpers.get_dtypes("float")))
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


# concat
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    num_positional_args=helpers.num_positional_args(fn_name="concat"),
)
def test_concat(
    *,
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="concat",
        xs=xs,
        axis=unique_idx,
    )


# expand_dims
@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="value_shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="expand_dims"),
)
def test_expand_dims(
    *,
    dtype_value,
    axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, value = dtype_value

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="expand_dims",
        x=value[0],
        axis=axis,
    )


# flip
@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="flip"),
)
def test_flip(
    *,
    dtype_value,
    axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, value = dtype_value

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="flip",
        x=value[0],
        axis=axis,
    )


@st.composite
def _permute_dims_helper(draw):
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"))
    dims = [x for x in range(len(shape))]
    permutation = draw(st.permutations(dims))
    return permutation


# permute_dims
@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    permutation=_permute_dims_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="permute_dims"),
)
def test_permute_dims(
    *,
    dtype_value,
    permutation,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, value = dtype_value

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="permute_dims",
        x=value[0],
        axes=permutation,
    )


@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    reshape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(), key="value_shape")
    ),
    num_positional_args=helpers.num_positional_args(fn_name="reshape"),
)
def test_reshape(
    *,
    dtype_value,
    reshape,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, value = dtype_value

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="reshape",
        x=value[0],
        shape=reshape,
    )


"""
    roll
    
    dtype_value
        tuple of an array and a data type with a shape under the key of value_shape. Has
        minimum 1 dimension as if there is 0 dimensions then there is no valid input for
        axis. 
    
    shift
        tuple of an array and a data type (always int32). Has either 0 or 1 dimension.
        Always a fixed length under the key shift_len
    axis
        tuple of valid axes for an array with the shape under the key of value_shape.
        Tuple is always a fixed length under the key shift_len
        
    shift and axis must have the same length as per the array API standard for the roll
    function. 
"""


@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    shift=helpers.dtype_and_values(
        available_dtypes=[ivy.int32],
        max_num_dims=1,
        min_dim_size=st.shared(
            helpers.ints(min_value=1, max_value=10),
            key="shift_len",
        ),
        max_dim_size=st.shared(
            helpers.ints(min_value=1, max_value=10),
            key="shift_len",
        ),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
        force_tuple=True,
        unique=False,
        min_size=st.shared(
            helpers.ints(min_value=1, max_value=10),
            key="shift_len",
        ),
        max_size=st.shared(
            helpers.ints(min_value=1, max_value=10),
            key="shift_len",
        ),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="roll"),
)
def test_roll(
    *,
    dtype_value,
    shift,
    axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    value_dtype, value = dtype_value
    shift_dtype, shift_val = shift

    if shift_val[0].ndim == 0:  # If shift is an int
        shift_val = shift_val[0]  # Drop shift's dtype (always int32)
        axis = axis[0]  # Extract an axis value from the tuple
    else:
        # Drop shift's dtype (always int32) and convert list to tuple
        shift_val = tuple(shift_val[0].tolist())

    helpers.test_function(
        input_dtypes=value_dtype + shift_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="roll",
        x=value[0],
        shift=shift_val,
        axis=axis,
    )


# squeeze
@st.composite
def _squeeze_helper(draw):
    shape = draw(st.shared(helpers.get_shape(), key="value_shape"))
    valid_axes = []
    for index, axis in enumerate(shape):
        if axis == 1:
            valid_axes.append(index)
    valid_axes.insert(0, None)
    return draw(st.sampled_from(valid_axes))


@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=_squeeze_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="squeeze"),
)
def test_squeeze(
    *,
    dtype_value,
    axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, value = dtype_value

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="squeeze",
        x=value[0],
        axis=axis,
    )


@st.composite
def _stack_helper(draw):
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="values_shape"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=1, max_value=3), key="num_arrays")
    )
    dtype = draw(st.sampled_from(draw(helpers.get_dtypes("valid"))))
    arrays = []
    dtypes = [dtype for _ in range(num_arrays)]

    for _ in range(num_arrays):
        array = draw(helpers.array_values(dtype=dtype, shape=shape))
        arrays.append(np.asarray(array, dtype=dtype))
    return dtypes, arrays


# stack
@handle_cmd_line_args
@given(
    dtypes_arrays=_stack_helper(),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="values_shape"),
        force_int=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="stack"),
)
def test_stack(
    *,
    dtypes_arrays,
    axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtypes, arrays = dtypes_arrays

    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="stack",
        arrays=arrays,
        axis=axis,
    )


# Extra #
# ------#


@st.composite
def _basic_min_x_max(draw):
    dtype, value = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
        )
    )
    min_val = draw(helpers.array_values(dtype=dtype[0], shape=()))
    max_val = draw(
        helpers.array_values(dtype=dtype[0], shape=()).filter(lambda x: x > min_val)
    )
    return [dtype], (value[0], min_val, max_val)


# clip
@handle_cmd_line_args
@given(
    dtype_x_min_max=_basic_min_x_max(),
    num_positional_args=helpers.num_positional_args(fn_name="clip"),
)
def test_clip(
    *,
    dtype_x_min_max,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtypes, (x_list, min_val, max_val) = dtype_x_min_max
    helpers.test_function(
        input_dtypes=dtypes[0],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="clip",
        x=x_list,
        x_min=min_val,
        x_max=max_val,
    )


@st.composite
def _pad_helper(draw):
    dtype, value, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"), ret_shape=True, min_num_dims=1
        )
    )
    pad_width = tuple(
        draw(
            st.lists(
                st.tuples(
                    helpers.ints(min_value=0, max_value=5),
                    helpers.ints(min_value=0, max_value=5),
                ),
                min_size=len(shape),
                max_size=len(shape),
            )
        )
    )
    return dtype, value, pad_width


# constant_pad
@handle_cmd_line_args
@given(
    dtype_value_pad_width_constant=_pad_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="constant_pad"),
)
def test_constant_pad(
    *,
    dtype_value_pad_width_constant,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, value, pad_width = dtype_value_pad_width_constant
    constant = float(value[0].flat[0])  # just use the first value as fill value
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="constant_pad",
        x=value[0],
        pad_width=pad_width,
        value=constant,
    )


@st.composite
def _repeat_helper(draw):
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"))
    axis = draw(
        st.shared(
            st.one_of(st.none(), helpers.get_axis(shape=shape, max_size=1)), key="axis"
        )
    )

    if not isinstance(axis, int) and axis is not None:
        axis = axis[0]

    repeat_shape = (
        (draw(st.one_of(st.just(1), st.just(shape[axis]))),)
        if axis is not None
        else (1,)
    )
    repeat = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            shape=repeat_shape,
            min_value=0,
            max_value=10,
        )
    )
    return repeat


# repeat
@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    axis=st.shared(
        st.one_of(
            st.none(),
            helpers.get_axis(
                shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
                max_size=1,
            ),
        ),
        key="axis",
    ),
    repeat=st.one_of(st.integers(1, 10), _repeat_helper()),
    num_positional_args=helpers.num_positional_args(fn_name="repeat"),
)
def test_repeat(
    *,
    dtype_value,
    axis,
    repeat,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    value_dtype, value = dtype_value

    if not isinstance(repeat, int):
        repeat_dtype, repeat_list = repeat
        repeat = repeat_list[0]
        value_dtype += repeat_dtype

    if not isinstance(axis, int) and axis is not None:
        axis = axis[0]

    helpers.test_function(
        input_dtypes=value_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="repeat",
        x=value[0],
        repeats=repeat,
        axis=axis,
    )


@st.composite
def _get_splits(draw):
    """
    Generate valid splits, either by generating an integer that evenly divides the axis
    or a list of splits that sum to the length of the axis being split.
    """
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"))
    axis = draw(
        st.shared(helpers.get_axis(shape=shape, force_int=True), key="target_axis")
    )

    @st.composite
    def get_int_split(draw):
        if shape[axis] == 0:
            return 0
        factors = []
        for i in range(1, shape[axis] + 1):
            if shape[axis] % i == 0:
                factors.append(i)
        return draw(st.sampled_from(factors))

    @st.composite
    def get_list_split(draw):
        num_or_size_splits = []
        while sum(num_or_size_splits) < shape[axis]:
            split_value = draw(
                helpers.ints(
                    min_value=1,
                    max_value=shape[axis] - sum(num_or_size_splits),
                )
            )
            num_or_size_splits.append(split_value)
        return num_or_size_splits

    return draw(get_list_split() | get_int_split() | st.none())


@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    axis=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    with_remainder=st.booleans(),
    num_or_size_splits=_get_splits(),
    num_positional_args=helpers.num_positional_args(fn_name="split"),
)
def test_split(
    *,
    dtype_value,
    num_or_size_splits,
    axis,
    with_remainder,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, value = dtype_value

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="split",
        x=value[0],
        num_or_size_splits=num_or_size_splits,
        axis=axis,
        with_remainder=with_remainder,
    )


# swapaxes
@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    axis0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"), force_int=True
    ),
    axis1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"), force_int=True
    ),
    num_positional_args=helpers.num_positional_args(fn_name="swapaxes"),
)
def test_swapaxes(
    *,
    dtype_value,
    axis0,
    axis1,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, value = dtype_value

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="swapaxes",
        x=value[0],
        axis0=axis0,
        axis1=axis1,
    )


"""
    tile
    
    dtype_value
        tuple of a dtype and an array that has the shape with a key of value_shape. Has
        minimum of 1 dimensions
    repeat
        a tuple of integers whose length is the number of dimensions in the shape with a
        key of value_shape. Each integer is between 0 and 10, and represents how many
        time each dimension needs to be tiled 
"""


@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    repeat=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape").map(
            lambda rep: (len(rep),)
        ),
        min_value=0,
        max_value=10,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="tile"),
)
def test_tile(
    *,
    dtype_value,
    repeat,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, value = dtype_value
    repeat_dtype, repeat_list = repeat
    helpers.test_function(
        input_dtypes=dtype + repeat_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="tile",
        x=value[0],
        reps=repeat_list[0],
    )


# zero_pad
@handle_cmd_line_args
@given(
    dtype_value_pad_width=_pad_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="zero_pad"),
)
def test_zero_pad(
    *,
    dtype_value_pad_width,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, value, pad_width = dtype_value_pad_width
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="zero_pad",
        x=value[0],
        pad_width=pad_width,
    )


# unstack
@handle_cmd_line_args
@given(
    x_n_dtype_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=5,
        min_axis=1,
        max_axis=4,
    ),
    keepdims=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="unstack"),
)
def test_unstack(
    x_n_dtype_axis,
    keepdims,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    # smoke test
    dtype, x, axis = x_n_dtype_axis
    if axis >= len(x[0].shape):
        axis = len(x[0].shape) - 1
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="unstack",
        x=x[0],
        axis=axis,
        keepdims=keepdims,
    )
