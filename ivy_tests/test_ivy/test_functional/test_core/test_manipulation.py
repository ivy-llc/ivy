# For Review
"""Collection of tests for manipulation functions."""

# global

import numpy as np
from hypothesis import given, strategies as st, HealthCheck
from hypothesis import settings

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _concat_helper(draw):
    dtype = draw(st.sampled_from(ivy_np.valid_dtypes))
    shape = list(
        draw(helpers.get_shape(min_num_dims=1, max_num_dims=1, max_dim_size=1))
    )
    axis = draw(helpers.get_axis(shape=shape, force_int=True))
    num_arrays = draw(helpers.ints(min_value=1, max_value=5))
    arrays = []
    dtypes = [dtype for _ in range(num_arrays)]

    for i in range(num_arrays):
        array_shape = shape[:]
        array_shape = tuple(array_shape)

        array = draw(helpers.array_values(dtype=dtype, shape=array_shape))
        arrays.append(np.asarray(array, dtype=dtype))
    return dtypes, arrays, axis


# concat
@given(
    dtypes_arrays_axis=_concat_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="concat"),
    data=st.data(),
)
@handle_cmd_line_args
def test_concat(
    *,
    data,
    dtypes_arrays_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtypes, arrays, axis = dtypes_arrays_axis

    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="concat",
        xs=arrays,
        axis=axis,
    )


# expand_dims
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="value_shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="expand_dims"),
    data=st.data(),
)
@handle_cmd_line_args
def test_expand_dims(
    *,
    data,
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
        x=np.asarray(value, dtype=dtype),
        axis=axis,
    )


# flip
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="flip"),
    data=st.data(),
)
@handle_cmd_line_args
def test_flip(
    *,
    data,
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
        x=np.asarray(value, dtype=dtype),
        axis=axis,
    )


@st.composite
def _permute_dims_helper(draw):
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"))
    dims = [x for x in range(len(shape))]
    permutation = draw(st.permutations(dims))
    return permutation


# permute_dims
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    permutation=_permute_dims_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="permute_dims"),
    data=st.data(),
)
@handle_cmd_line_args
def test_permute_dims(
    *,
    data,
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
        x=np.asarray(value, dtype=dtype),
        axes=permutation,
    )


@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    reshape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(), key="value_shape")
    ),
    num_positional_args=helpers.num_positional_args(fn_name="reshape"),
    data=st.data(),
)
@handle_cmd_line_args
def test_reshape(
    *,
    data,
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
        x=np.asarray(value, dtype),
        shape=reshape,
    )


# roll
@settings(
    deadline=1250,
    suppress_health_check=(HealthCheck.data_too_large,),  # jax.roll is very slow
)
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    shift=helpers.dtype_and_values(
        available_dtypes=[ivy.int32],
        max_num_dims=1,
        min_dim_size=st.shared(
            helpers.array_values(dtype="int32", shape=(1,))
            .map(lambda x: abs(x[0]))
            .filter(lambda x: x > 0),
            key="shift_len",
        ),
        max_dim_size=st.shared(
            helpers.array_values(dtype="int32", shape=(1,))
            .map(lambda x: abs(x[0]))
            .filter(lambda x: x > 0),
            key="shift_len",
        ),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
        force_tuple=True,
        unique=False,
        min_size=st.shared(
            helpers.array_values(dtype="int32", shape=(1,))
            .map(lambda x: abs(x[0]))
            .filter(lambda x: x > 0),
            key="shift_len",
        ),
        max_size=st.shared(
            helpers.array_values(dtype="int32", shape=(1,))
            .map(lambda x: abs(x[0]))
            .filter(lambda x: x > 0),
            key="shift_len",
        ),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="roll"),
    data=st.data(),
)
@handle_cmd_line_args
def test_roll(
    *,
    data,
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
    if isinstance(shift[1], int):
        shift = shift[1]
        axis = axis[0]
    else:
        shift = tuple(shift[1])

    helpers.test_function(
        input_dtypes=value_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="roll",
        x=np.asarray(value, dtype=value_dtype),
        shift=shift,
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


@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=_squeeze_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="squeeze"),
    data=st.data(),
)
@handle_cmd_line_args
def test_squeeze(
    *,
    data,
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
        x=np.asarray(value, dtype=dtype),
        axis=axis,
    )


@st.composite
def _stack_helper(draw):
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="values_shape"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=1, max_value=3), key="num_arrays")
    )
    dtype = draw(st.sampled_from(ivy_np.valid_dtypes))
    arrays = []
    dtypes = [dtype for _ in range(num_arrays)]

    for _ in range(num_arrays):
        array = draw(helpers.array_values(dtype=dtype, shape=shape))
        arrays.append(np.asarray(array, dtype=dtype))
    return dtypes, arrays


# stack
@settings(deadline=500)
@given(
    dtypes_arrays=_stack_helper(),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="values_shape"),
        force_int=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="stack"),
    data=st.data(),
)
@handle_cmd_line_args
def test_stack(
    *,
    data,
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


# clip
@settings(deadline=500)
@given(
    x_min_n_max=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=3, shared_dtype=True
    ),
    num_positional_args=helpers.num_positional_args(fn_name="clip"),
    data=st.data(),
)
@handle_cmd_line_args
def test_clip(
    *,
    data,
    x_min_n_max,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    (x_dtype, min_dtype, max_dtype), (x_list, min_val_list, max_val_list) = x_min_n_max
    min_val_raw = np.array(min_val_list, dtype=min_dtype)
    max_val_raw = np.array(max_val_list, dtype=max_dtype)
    min_val = np.asarray(np.minimum(min_val_raw, max_val_raw))
    max_val = np.asarray(np.maximum(min_val_raw, max_val_raw))

    helpers.test_function(
        input_dtypes=[x_dtype, min_dtype, max_dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="clip",
        x=np.asarray(x_list, dtype=x_dtype),
        x_min=min_val,
        x_max=max_val,
    )


@st.composite
def _pad_helper(draw):
    dtype, value, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_dtypes, ret_shape=True, min_num_dims=1
        )
    )
    pad_width = tuple(
        draw(
            st.lists(
                st.tuples(
                    helpers.ints(min_value=0, max_value=100),
                    helpers.ints(min_value=0, max_value=100),
                ),
                min_size=len(shape),
                max_size=len(shape),
            )
        )
    )
    constant = draw(helpers.array_values(dtype=dtype, shape=()))
    return dtype, value, pad_width, constant


# constant_pad
@settings(deadline=1000)
@given(
    dtype_value_pad_width_constant=_pad_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="constant_pad"),
    data=st.data(),
)
@handle_cmd_line_args
def test_constant_pad(
    *,
    data,
    dtype_value_pad_width_constant,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, value, pad_width, constant = dtype_value_pad_width_constant

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
        x=np.asarray(value, dtype=dtype),
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
            available_dtypes=(ivy_np.int8, ivy_np.int16, ivy_np.int32, ivy_np.int64),
            shape=repeat_shape,
            min_value=0,
            max_value=100,
        )
    )
    return repeat


# repeat
@settings(deadline=750)
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
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
    repeat=st.one_of(st.integers(1, 100), _repeat_helper()),
    num_positional_args=helpers.num_positional_args(fn_name="repeat"),
    data=st.data(),
)
@handle_cmd_line_args
def test_repeat(
    *,
    data,
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
    value = np.asarray(value, dtype=value_dtype)

    if not isinstance(repeat, int):
        repeat_dtype, repeat_list = repeat
        repeat = np.asarray(repeat_list, dtype=repeat_dtype)
        value_dtype = [value_dtype, repeat_dtype]

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
        x=value,
        repeats=repeat,
        axis=axis,
    )


@st.composite
def _split_helper(draw):
    noss_is_int = draw(
        st.shared(helpers.ints(min_value=1, max_value=2), key="noss_type").map(
            lambda x: x == 1
        )
    )
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"))
    axis = draw(
        st.shared(helpers.get_axis(shape=shape, force_int=True), key="target_axis")
    )

    if noss_is_int:
        if shape[axis] == 0:
            return 0
        factors = []
        for i in range(1, shape[axis] + 1):
            if shape[axis] % i == 0:
                factors.append(i)
        return draw(st.sampled_from(factors))

    noss_dtype = draw(st.sampled_from(ivy_np.valid_int_dtypes))
    num_or_size_splits = []
    while sum(num_or_size_splits) < shape[axis]:
        split_value = draw(
            helpers.array_values(
                dtype=noss_dtype,
                shape=(1,),
                min_value=0,
                max_value=shape[axis] - sum(num_or_size_splits),
            )
        )
        num_or_size_splits.append(split_value[0])

    return noss_dtype, num_or_size_splits


@given(
    noss_type=st.shared(helpers.ints(min_value=1, max_value=2), key="noss_type"),
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
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
    num_or_size_splits=_split_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="split"),
    data=st.data(),
)
@handle_cmd_line_args
def test_split(
    *,
    data,
    noss_type,
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

    if noss_type == 2:
        num_or_size_splits = num_or_size_splits[1]

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
        x=np.asarray(value, dtype=dtype),
        num_or_size_splits=num_or_size_splits,
        axis=axis,
        with_remainder=with_remainder,
    )


# swapaxes
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    axis0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"), force_int=True
    ),
    axis1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"), force_int=True
    ),
    num_positional_args=helpers.num_positional_args(fn_name="swapaxes"),
    data=st.data(),
)
@handle_cmd_line_args
def test_swapaxes(
    *,
    data,
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
        x=np.asarray(value, dtype=dtype),
        axis0=axis0,
        axis1=axis1,
    )


@st.composite
def _tile_helper(draw):
    dtype, value, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_dtypes, ret_shape=True, min_num_dims=1
        )
    )
    reps = draw(
        helpers.dtype_and_values(
            available_dtypes=(ivy_np.int8, ivy_np.int16, ivy_np.int32, ivy_np.int64),
            shape=(len(shape),),
            min_value=0,
            max_value=10,
        )
    )
    return (dtype, value), reps


# tile
@given(
    dtype_value_repeat=_tile_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="tile"),
    data=st.data(),
)
@handle_cmd_line_args
def test_tile(
    *,
    data,
    dtype_value_repeat,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype_value, repeat = dtype_value_repeat

    dtype, value = dtype_value
    value = np.asarray(value, dtype=dtype)

    repeat_dtype, repeat_list = repeat
    repeat = np.asarray(repeat_list, dtype=repeat_dtype)
    dtype = [dtype, repeat_dtype]

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="tile",
        x=value,
        reps=repeat,
    )


# zero_pad
@settings(deadline=1000)
@given(
    dtype_value_pad_width=_pad_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="zero_pad"),
    data=st.data(),
)
@handle_cmd_line_args
def test_zero_pad(
    *,
    data,
    dtype_value_pad_width,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    # Drop the generated constant as only 0 is used
    dtype, value, pad_width, _ = dtype_value_pad_width

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
        x=np.asarray(value, dtype=dtype),
        pad_width=pad_width,
    )
