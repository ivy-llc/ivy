"""Collection of tests for manipulation functions."""

# global

import numpy as np
from hypothesis import given, strategies as st
from hypothesis import settings


# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(st.integers(1, 4), key="num_dims"))
    num_arrays = draw(st.shared(st.integers(2, 4), key="num_arrays"))
    common_shape = draw(
        helpers.lists(
            arg=st.integers(2, 3), min_size=num_dims - 1, max_size=num_dims - 1
        )
    )
    unique_idx = draw(helpers.integers(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(arg=st.integers(2, 3), min_size=num_arrays, max_size=num_arrays)
    )
    xs = list()
    input_dtypes = draw(helpers.array_dtypes())
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
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    num_positional_args=helpers.num_positional_args(fn_name="concat"),
    data=st.data(),
)
@handle_cmd_line_args
def test_concat(
    *,
    data,
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
    xs = [np.asarray(x, dtype=dt) for x, dt in zip(xs, input_dtypes)]
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

@st.composite
def _dtype_values_axis(draw,min_value=None,max_value=None):
    dtype, values, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_dtypes,
            min_num_dims=1,
            ret_shape=True
        ))

    if min_value is None:
        min_axis = -len(shape)
    elif isinstance(min_value,types.FunctionType):
        min_axis = min_value(len(shape))
    else:
        min_axis = min_value

    if max_value is None:
        max_axis = len(shape) - 1
    elif isinstance(max_value,types.FunctionType):
        max_axis = max_value(len(shape))
    else:
        max_axis = max_value

    axis = draw(st.integers(min_value= min_axis, max_value=max_axis))
    return dtype, values, axis


# expand_dims
@given(
    dtype_array_axis=_dtype_values_axis(min_value = (lambda n: -n - 1)),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="expand_dims"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_expand_dims(
    *,
    data,
    dtype_array_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, array, axis = dtype_array_axis


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
        x=np.asarray(array, dtype=dtype),
        axis=axis,
    )


# flip
@given(
    dtype_array_axis=_dtype_values_axis(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="flip"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_flip(
    *,
    data,
    dtype_array_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, array, axis = dtype_array_axis


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
        x=np.asarray(array, dtype=dtype),
        axis=axis,
    )


@st.composite
def _dtype_array_permutation(draw):
    dtype, array, shape = draw(helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        ret_shape=True,
        min_num_dims=1))
    dims = [x for x in range(len(shape))]
    permutation = draw(st.permutations(dims))
    return dtype, array, permutation

# permute_dims
@given(
    dtype_array_permutation=_dtype_array_permutation(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="permute_dims"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_permute_dims(
    *,
    data,
    dtype_array_permutation,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, array, permutation = dtype_array_permutation

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
        x=np.asarray(array, dtype=dtype),
        axes=permutation,
    )

@st.composite
def _array_dtype_reshape(draw):
    """
    Hypothesis strategy that will return an array, its dtype, and a valid shape for it to be reshaped into
    """
    dtype, array, shape = draw(helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        ret_shape=True))
    return array, dtype, draw(helpers.reshape_shapes(shape=shape))



@given(
    array_dtype_reshape=_array_dtype_reshape(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="reshape"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_reshape(
    *,
    data,
    array_dtype_reshape,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    array, dtype, shape = array_dtype_reshape


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
        x=np.asarray(array,dtype),
        shape=shape
    )


@st.composite
def _roll_helper(draw):
    dtype, array, shape = draw(helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        min_num_dims=1,
        ret_shape=True))
    shift = draw(st.one_of(
        st.integers(min_value=-9223372036854775808, max_value=9223372036854775807),
        st.lists(st.integers(min_value=-9223372036854775808, max_value=9223372036854775807), min_size=1, max_size=len(shape))))
    if isinstance(shift, list):
        axis = draw(st.lists(
            st.integers(
                min_value=-len(shape),
                max_value=len(shape)-1),
            min_size=len(shift),
            max_size=len(shift),
            unique=True))
    else:
        axis = draw(st.one_of(
            st.none(),
            st.integers(
                min_value=-len(shape),
                max_value=len(shape)-1),
            st.lists(
                st.integers(
                    min_value=-len(shape),
                    max_value=len(shape)-1),
                min_size=1,
                max_size=len(shape),
                unique=True)
        ))
    return dtype, array, shift, axis

# roll
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1),
            key='value_shape'
        ),
    ),
    shift=helpers.dtype_and_values(
        available_dtypes=[ivy.int32, ivy.int64],
        max_num_dims=1,
        min_dim_size=st.shared(st.integers(1, 2147483647), key='shift_length'),
        max_dim_size=st.shared(st.integers(1, 2147483647), key='shift_length')
    ),
    axis=helpers.get_axis(
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1),
            key='value_shape'
        ),
        unique=False,
        min_size=st.shared(st.integers(1, 2147483647), key='shift_length'),
        max_size=st.shared(st.integers(1, 2147483647), key='shift_length')
    ),
    as_variable=helpers.array_bools(num_arrays=2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="roll"),
    native_array=helpers.array_bools(num_arrays=2),
    container=helpers.array_bools(num_arrays=2),
    instance_method=st.booleans(),
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
    shift_dtype, shift = shift
    dtypes = [value_dtype]



    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="roll",
        x=np.asarray(value,dtype=value_dtype),
        shift=shift,
        axis=axis,
    )


# squeeze
@st.composite
def _squeeze_helper(draw):
    shape = tuple(draw(st.lists(
        st.integers(min_value=1, max_value=10),
        min_size=0,
        max_size=5
    )))
    valid_axes = []
    for index, axis in enumerate(shape):
        if axis == 1:
            valid_axes.append(index)
    valid_axes.insert(0,None)
    axis = draw(st.sampled_from(valid_axes))
    dtype, value = draw(helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=shape
    ))
    return dtype, value, axis


@given(
    dtype_values_axis=_squeeze_helper(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="squeeze"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_squeeze(
    *,
    data,
    dtype_values_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, values, axis = dtype_values_axis

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
        x=np.asarray(values, dtype=dtype),
        axis=axis,
    )

@st.composite
def _stack_helper(draw):
    shape = tuple(draw(st.lists(
        st.integers(min_value=1, max_value=10),
        min_size=0,
        max_size=5
    )))
    axis = draw(st.integers(min_value=-len(shape), max_value=len(shape)))
    num_arrays = draw(st.shared(st.integers(1, 3), key="num_arrays"))
    dtype = draw(st.sampled_from(ivy_np.valid_dtypes))
    dtypes_arrays = draw(st.lists(
        helpers.dtype_and_values(
            available_dtypes=[dtype],
            shape=shape
        ),
        min_size=num_arrays,
        max_size=num_arrays
    ))
    return dtypes_arrays, axis
# stack
@given(
    dtypes_arrays_axis=_stack_helper(),
    as_variable=helpers.array_bools(
        num_arrays=st.shared(st.integers(1, 3), key="num_arrays")
    ),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="expand_dims"),
    native_array=helpers.array_bools(
        num_arrays=st.shared(st.integers(1, 3), key="num_arrays")
    ),
    container=helpers.array_bools(
        num_arrays=st.shared(st.integers(1, 3), key="num_arrays")
    ),
    instance_method=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_stack(
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

    dtypes = []
    arrays = []
    dtypes_arrays, axis = dtypes_arrays_axis
    for pair in dtypes_arrays:
        dtypes.append(pair[0])
        arrays.append(np.asarray(pair[1], dtype=pair[0]))

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
        x=arrays,
        axis=axis,
    )


# Extra #
# ------#

@st.composite
def _repeat_helper(draw):
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key='value_shape'))
    axis = draw(st.shared(
        st.one_of(
            st.none(),
            helpers.get_axis(
                shape=shape,
                max_size=1))
        , key='axis'))

    if not isinstance(axis,int) and axis is not None:
        axis = axis[0]

    repeat_shape=(draw(st.one_of(st.just(1), st.just(shape[axis]))),) if axis is not None else (1,)
    repeat=draw(
        helpers.dtype_and_values(
            available_dtypes=(ivy_np.int8, ivy_np.int16, ivy_np.int32, ivy_np.int64),
            shape=repeat_shape,
            min_value=0,
            max_value=100
        )
    )
    return repeat

# repeat
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(helpers.get_shape(min_num_dims=1), key='value_shape')
    ),
    axis=st.shared(
        st.one_of(
            st.none(),
            helpers.get_axis(
                shape=st.shared(helpers.get_shape(min_num_dims=1), key='value_shape'),
                max_size=1))
        ,key='axis'),
    repeat=st.one_of(st.integers(1, 100), _repeat_helper()),
    as_variable=helpers.array_bools(num_arrays=2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="repeat"),
    native_array=helpers.array_bools(num_arrays=2),
    container=helpers.array_bools(num_arrays=2),
    instance_method=st.booleans(),
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

    dtype, value = dtype_value
    value = np.asarray(value, dtype=dtype)


    if not isinstance(repeat, int):
        repeat_dtype, repeat_list = repeat
        repeat = np.asarray(repeat_list, dtype=repeat_dtype)
        dtype = [dtype, repeat_dtype]

    if not isinstance(axis,int) and axis is not None:
        axis = axis[0]


    helpers.test_function(
        input_dtypes=dtype,
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
def _tile_helper(draw):
    dtype, value, shape = draw(helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        ret_shape=True,
        min_num_dims=1
    ))
    reps=draw(helpers.dtype_and_values(
        available_dtypes=(ivy_np.int8, ivy_np.int16, ivy_np.int32, ivy_np.int64),
        shape=(len(shape),),
        min_value=0,
        max_value=10
    ))
    return (dtype, value), reps
# tile
@given(
    dtype_value_repeat=_tile_helper(),
    as_variable=helpers.array_bools(
        num_arrays=2
    ),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="tile"),
    native_array=helpers.array_bools(
        num_arrays=2
    ),
    container=helpers.array_bools(
        num_arrays=2
    ),
    instance_method=st.booleans(),
    data = st.data(),
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


@st.composite
def _pad_helper(draw):
    dtype, value, shape = draw(helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        ret_shape=True,
        min_num_dims=1
    ))
    pad_width = tuple(draw(st.lists(st.tuples(st.integers(0,100), st.integers(0,100)), min_size=len(shape), max_size=len(shape))))
    _, constant = draw(helpers.dtype_and_values(
        available_dtypes=[dtype],
        shape=(1,)
    ))
    return dtype, value, pad_width, constant[0]

@settings(
    deadline=500
)
# constant_pad
@given(
    dtype_value_pad_width_constant=_pad_helper(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
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


# zero_pad
@given(
    dtype_value_pad_width=_pad_helper(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="zero_pad"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
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


# swapaxes
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(
            helpers.get_shape(min_num_dims=2),
            key='shape')
        ),
    axis0=helpers.get_axis(
            shape=st.shared(
            helpers.get_shape(min_num_dims=2),
            key='shape')
        ).filter(lambda axis: isinstance(axis, int)),
    axis1=helpers.get_axis(
        shape=st.shared(
            helpers.get_shape(min_num_dims=2),
            key='shape')
        ).filter(lambda axis: isinstance(axis, int)),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="swapaxes"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
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


# clip
@given(
    x_min_n_max=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=3, shared_dtype=True
    ),
    as_variable=helpers.array_bools(num_arrays=3),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="clip"),
    native_array=helpers.array_bools(num_arrays=3),
    container=helpers.array_bools(num_arrays=3),
    instance_method=st.booleans(),
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
def _split_helper(draw):
    noss_is_int = draw(st.shared(st.integers(1, 2), key="noss_type").map(lambda x: x == 1))

    shape = draw(st.shared(
        helpers.get_shape(min_num_dims=1),
        key='value_shape'))

    axis = draw(st.shared(
        helpers.get_axis(shape=shape),
        key='target_axis')
    )

    if not isinstance(axis, int):
        axis = axis[0]

    if noss_is_int:
        if shape[axis] == 0:
            return 0
        factors = []
        for i in range(1, shape[axis]+1):
            if shape[axis] % i == 0:
                factors.append(i)
        return draw(st.sampled_from(factors))

    noss_dtype = draw(st.sampled_from(ivy_np.valid_int_dtypes))
    num_or_size_splits = []
    while sum(num_or_size_splits) < shape[axis]:
        split_value = draw(helpers.array_values(
            dtype=noss_dtype,
            shape=(1,),
            min_value=0,
            max_value=shape[axis] - sum(num_or_size_splits)))
        num_or_size_splits.append(split_value[0])

    return noss_dtype, num_or_size_splits

@given(
    noss_type=st.shared(st.integers(1,2), key="noss_type"),
    dtype_value=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(
            helpers.get_shape(min_num_dims=1),
            key='value_shape'),
    ),
    axis = st.shared(
        helpers.get_axis(
            shape=st.shared(
                helpers.get_shape(min_num_dims=1),
            key='value_shape')
            ),
        key='target_axis'),
    num_or_size_splits=_split_helper(),
    with_remainder=st.booleans(),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="split"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
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
    fw,):

    dtype, value = dtype_value
    x = np.asarray(value, dtype=dtype)

    if not isinstance(axis, int):
        axis = axis[0]

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
        x=x,
        num_or_size_splits=num_or_size_splits,
        axis=axis,
        with_remainder=with_remainder
    )