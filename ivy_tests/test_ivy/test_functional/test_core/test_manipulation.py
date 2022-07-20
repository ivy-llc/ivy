"""Collection of tests for manipulation functions."""

# global
import types
import pytest
import numpy as np
import math
from numbers import Number
from hypothesis import given, strategies as st
from hypothesis import settings


# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


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
    as_variable=helpers.array_bools(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="concat"),
    native_array=helpers.array_bools(),
    container=helpers.array_bools(),
    instance_method=st.booleans(),
)
def test_concat(
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

# For Review
# expand_dims
@settings(
    max_examples=100
)
@given(
    dtype_array_axis=_dtype_values_axis(min_value = (lambda n: -n - 1)),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="expand_dims"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_expand_dims(
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

    print(f"Input: {dtype_array_axis}, "
          f"As Variable Flag: {as_variable}, "
          f"With_out_Flag: {with_out}, "
          f"Number of Positional Arguments: {num_positional_args}, "
          f"Native Array Flag: {native_array}, "
          f"Container Flag: {container}, "
          f"Instance Method Flad: {instance_method}, "
          f"Framework: {fw}")

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
)
def test_flip(
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
)
def test_permute_dims(
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
)
def test_reshape(
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
    dtype_array_shift_axis=_roll_helper(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="roll"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_roll(
    dtype_array_shift_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    dtype, array, shift, axis = dtype_array_shift_axis

    print(dtype_array_shift_axis, type(shift), type(axis))

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="roll",
        x=np.asarray(array,dtype=dtype),
        shift=shift,
        axis=axis,
    )


# squeeze
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 5],
    ).filter(lambda s: 1 in s),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="squeeze"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_squeeze(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))
    squeezable_axes = [i for i, side in enumerate(x.shape) if side == 1]

    valid_axis = st.sampled_from(squeezable_axes) | helpers.subsets(
        elements=squeezable_axes
    )

    axis = data.draw(valid_axis)

    # we need subset of size atleast 1, think of better way to do this
    # right now, we are just ignoring when we sample an empty subset
    if not isinstance(axis, int):
        if len(axis) == 0:
            return
    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    if (
        input_dtype
        in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
        and fw == "torch"
    ):
        return


    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    if input_dtype in [
        ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")
    ] and fw == "torch":
        return

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="squeeze",
        x=x,
        axis=axis,
    )


# stack
@given(
    array_shape=helpers.lists(
        arg=st.integers(0, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[0, 3],
    ),
    num_arrays=st.shared(st.integers(1, 3), key="num_arrays"),
    input_dtype=helpers.array_dtypes(
        num_arrays=st.shared(st.integers(1, 3), key="num_arrays")
    ),
    data=st.data(),
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
)
def test_stack(
    array_shape,
    num_arrays,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    xs = [
        data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype[i]))
        for i in range(num_arrays)
    ]
    ndim = len(xs[0].shape)
    axis = data.draw(st.integers(-ndim, max(0, ndim - 1)))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="stack",
        x=xs,
        axis=axis,
    )


# Extra #
# ------#


# repeat
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="repeat"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_repeat(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    # smoke for torch
    # smoke for tensorflow as well, since it was throwing an error
    # as unint16 not implemented in Tile or something

    input_dtype = _drop_unsupported_torch_dtypes(input_dtype,fw)

    if fw == "tensorflow" and input_dtype in ["uint16"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))
    ndim = len(x.shape)

    valid_axis = st.none() | st.integers(-ndim, ndim - 1)
    axis = data.draw(valid_axis)

    repeats = data.draw(st.integers(1, 3))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="repeat",
        x=x,
        repeats=repeats,
        axis=axis,
    )


# tile
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="tile"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_tile(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype = _drop_unsupported_torch_dtypes(input_dtype,fw)
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))

    # tensorflow needs that reps is exactly of same dimensions as the input
    # other frameworks can broadcast the results
    if fw == "tensorflow":
        if input_dtype == ivy.IntDtype("uint16"):
            return
        reps = data.draw(
            helpers.nph.broadcastable_shapes(
                shape=x.shape, min_dims=len(x.shape), max_dims=len(x.shape)
            )
        )
    else:
        reps = data.draw(
            helpers.nph.broadcastable_shapes(shape=x.shape, min_dims=len(x.shape))
        )

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="tile",
        x=x,
        reps=reps,
    )


# constant_pad
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="constant_pad"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_constant_pad(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype = _drop_unsupported_torch_dtypes(input_dtype,fw)
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))
    pads = [
        (data.draw(st.integers(0, 3)), data.draw(st.integers(0, 3)))
        for _ in range(len(x.shape))
    ]
    constant = data.draw(st.integers(0, 10))

    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    if (
        input_dtype
        in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
        and fw == "torch"
    ):
        return

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="constant_pad",
        x=x,
        pad_width=pads,
        value=constant,
    )


# zero_pad
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="zero_pad"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_zero_pad(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype = _drop_unsupported_torch_dtypes(input_dtype,fw)
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))
    pads = [
        (data.draw(st.integers(0, 3)), data.draw(st.integers(0, 3)))
        for _ in range(len(x.shape))
    ]

    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    """
    if input_dtype in [
        ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")
    ] and fw == "torch":
        return
    """

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="zero_pad",
        x=x,
        pad_width=pads,
    )


# swapaxes
@given(
    array_shape=helpers.lists(
        arg=st.integers(0, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="swapaxes"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_swapaxes(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype = _drop_unsupported_torch_dtypes(input_dtype, fw)


    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))
    valid_axes = st.integers(0, len(x.shape) - 1)
    axis0 = data.draw(valid_axes)
    axis1 = data.draw(valid_axes)




    if (
        input_dtype
        in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
        and fw == "torch"
    ):
        return

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="swapaxes",
        x=x,
        axis0=axis0,
        axis1=axis1,
    )


# clip
@given(
    x_min_n_max=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=3
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="clip"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_clip(
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


# split
@given(
    x_n_noss_n_axis_n_wr=st.sampled_from(
        [
            (1, 1, -1, False),
            ([[0.0, 1.0, 2.0, 3.0]], 2, 1, False),
            ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 2, 0, False),
            ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 2, 1, True),
            ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [2, 1], 1, False),
        ],
    ),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    data=st.data(),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_split(x_n_noss_n_axis_n_wr, dtype, data, tensor_fn, device, call, fw):
    # smoke test
    x, num_or_size_splits, axis, with_remainder = x_n_noss_n_axis_n_wr
    if (
        isinstance(x, Number)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype=dtype, device=device)
    ret = ivy.split(x, num_or_size_splits, axis, with_remainder)
    # type test
    assert isinstance(ret, list)
    # cardinality test
    axis_val = (
        axis % len(x.shape)
        if (axis is not None and len(x.shape) != 0)
        else len(x.shape) - 1
    )
    if x.shape == ():
        expected_shape = ()
    elif isinstance(num_or_size_splits, int):
        expected_shape = tuple(
            [
                math.ceil(item / num_or_size_splits) if i == axis_val else item
                for i, item in enumerate(x.shape)
            ]
        )
    else:
        expected_shape = tuple(
            [
                num_or_size_splits[0] if i == axis_val else item
                for i, item in enumerate(x.shape)
            ]
        )
    assert ret[0].shape == expected_shape
    # value test
    pred_split = call(ivy.split, x, num_or_size_splits, axis, with_remainder)
    true_split = ivy.functional.backends.numpy.split(
        ivy.to_numpy(x), num_or_size_splits, axis, with_remainder
    )
    for pred, true in zip(pred_split, true_split):
        assert np.allclose(pred, true)
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return


# Private Helpers
def _drop_unsupported_torch_dtypes(
        input_dtypes,
        fw
):
    invalid_dtypes = ['uint16', 'uint32', 'uint64'] if fw == 'torch' else []
    if isinstance(input_dtypes, list):
        return ["float32" if d in invalid_dtypes else d for d in input_dtypes]
    else: return "float32" if input_dtypes in invalid_dtypes else input_dtypes