"""Collection of tests for manipulation functions."""

# global
import numpy as np
import math
from numbers import Number
from hypothesis import given, assume, strategies as st


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


# expand_dims
@given(
    array_shape=helpers.lists(
        arg=st.integers(2, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    unique_idx=helpers.integers(min_value=0, max_value="num_dims"),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="expand_dims"),
    seed=st.integers(0, 2**32 - 1),
    data=st.data(),
)
@handle_cmd_line_args
def test_expand_dims(
    *,
    data,
    array_shape,
    unique_idx,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    seed,
    fw,
):
    np.random.seed(seed)

    x = np.random.uniform(size=array_shape).astype(input_dtype)

    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    assume(
        not (
            input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
            and fw == "torch"
        )
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
        fn_name="expand_dims",
        x=x,
        axis=unique_idx,
    )


# flip
@given(
    array_shape=helpers.lists(
        arg=st.integers(2, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 3],
    ),
    axis=helpers.valid_axes(ndim="num_dims", size_bounds=[1, 3]),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="flip"),
    seed=st.integers(0, 2**32 - 1),
    data=st.data(),
)
@handle_cmd_line_args
def test_flip(
    *,
    data,
    array_shape,
    axis,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    seed,
    fw,
):
    np.random.seed(seed)

    x = np.random.uniform(size=array_shape).astype(input_dtype)

    assume(
        not (
            input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
            and fw == "torch"
        )
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
        fn_name="flip",
        x=x,
        axis=axis,
    )


# permute_dims
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 3),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="permute_dims"),
    seed=st.integers(0, 2**32 - 1),
    data=st.data(),
)
@handle_cmd_line_args
def test_permute_dims(
    *,
    data,
    array_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    seed,
    fw,
):

    np.random.seed(seed)

    x = np.random.uniform(size=array_shape).astype(input_dtype)
    axes = np.random.permutation(len(array_shape)).tolist()

    assume(
        not (
            input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
            and fw == "torch"
        )
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
        fn_name="permute_dims",
        x=x,
        axes=axes,
    )


@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 10),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="reshape"),
    data=st.data(),
)
@handle_cmd_line_args
def test_reshapes(
    *,
    data,
    array_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    assume(
        not (
            input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
            and fw == "torch"
        )
    )

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))

    # draw a valid reshape shape
    shape = data.draw(helpers.reshape_shapes(shape=x.shape))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="reshape",
        x=x,
        shape=shape,
    )


# roll
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[1, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="roll"),
    data=st.data(),
)
@handle_cmd_line_args
def test_roll(
    *,
    data,
    array_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):

    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    assume(
        not (
            input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
            and fw == "torch"
        )
    )

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))
    ndim = len(x.shape)

    valid_shifts = st.integers(-5, 5) | st.lists(
        st.integers(-5, 5), min_size=1, max_size=ndim
    )
    shift = data.draw(valid_shifts)

    # shift is just an integer, then axis can be None or any valid axes subset
    if isinstance(shift, int):
        # set min size of tuple to be 1 ?
        # not sure of what Array API standard says on this ?
        valid_axis = (
            st.none()
            | st.integers(-ndim, ndim - 1)
            | helpers.nph.valid_tuple_axes(ndim=ndim, min_size=1)
        )  # to check
    else:
        # need axis of the same length as shift
        valid_axis = helpers.nph.valid_tuple_axes(
            ndim=ndim, min_size=len(shift), max_size=len(shift)
        )

    # draw any valid axis
    axis = data.draw(valid_axis)

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="roll",
        x=x,
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
    num_positional_args=helpers.num_positional_args(fn_name="squeeze"),
    data=st.data(),
)
@handle_cmd_line_args
def test_squeeze(
    *,
    data,
    array_shape,
    input_dtype,
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
    assume(isinstance(axis, int) or len(axis) > 0)
    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    assume(
        not (
            input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
            and fw == "torch"
        )
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
    as_variable=helpers.array_bools(
        num_arrays=st.shared(st.integers(1, 3), key="num_arrays")
    ),
    num_positional_args=helpers.num_positional_args(fn_name="expand_dims"),
    native_array=helpers.array_bools(
        num_arrays=st.shared(st.integers(1, 3), key="num_arrays")
    ),
    container=helpers.array_bools(
        num_arrays=st.shared(st.integers(1, 3), key="num_arrays")
    ),
    data=st.data(),
)
@handle_cmd_line_args
def test_stack(
    *,
    data,
    array_shape,
    num_arrays,
    input_dtype,
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
    num_positional_args=helpers.num_positional_args(fn_name="repeat"),
    data=st.data(),
)
@handle_cmd_line_args
def test_repeat(
    *,
    data,
    array_shape,
    input_dtype,
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
    assume(
        not (fw == "torch" and input_dtype in ["uint16", "uint32", "uint64"])
        or (fw == "tensorflow" and input_dtype in ["uint16"])
    )

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
    num_positional_args=helpers.num_positional_args(fn_name="tile"),
    data=st.data(),
)
@handle_cmd_line_args
def test_tile(
    *,
    data,
    array_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))

    assume(
        not (
            fw == "torch"
            and input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
        )
    )
    # tensorflow needs that reps is exactly of same dimensions as the input
    # other frameworks can broadcast the
    if fw == "tensorflow":

        assume(not (input_dtype == ivy.IntDtype("uint16")))

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
    num_positional_args=helpers.num_positional_args(fn_name="constant_pad"),
    data=st.data(),
)
@handle_cmd_line_args
def test_constant_pad(
    *,
    data,
    array_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))
    pads = [
        (data.draw(st.integers(0, 3)), data.draw(st.integers(0, 3)))
        for _ in range(len(x.shape))
    ]
    constant = data.draw(helpers.array_values(dtype=input_dtype, shape=()))

    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    assume(
        not (
            input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
            and fw == "torch"
        )
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
    num_positional_args=helpers.num_positional_args(fn_name="zero_pad"),
    data=st.data(),
)
@handle_cmd_line_args
def test_zero_pad(
    *,
    data,
    array_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))
    pads = [
        (data.draw(st.integers(0, 3)), data.draw(st.integers(0, 3)))
        for _ in range(len(x.shape))
    ]

    # Torch does not support unsigned integers of more than 8 bits (>uint8)
    assume(
        not (
            input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
            and fw == "torch"
        )
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
    num_positional_args=helpers.num_positional_args(fn_name="swapaxes"),
    data=st.data(),
)
@handle_cmd_line_args
def test_swapaxes(
    *,
    data,
    array_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))
    valid_axes = st.integers(0, len(x.shape) - 1)
    axis0 = data.draw(valid_axes)
    axis1 = data.draw(valid_axes)

    assume(
        not (
            input_dtype
            in [ivy.IntDtype("uint16"), ivy.IntDtype("uint32"), ivy.IntDtype("uint64")]
            and fw == "torch"
        )
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
        fn_name="swapaxes",
        x=x,
        axis0=axis0,
        axis1=axis1,
    )


# clip
@given(
    x_min_n_max=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=3,
        shared_dtype=True,
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
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
    data=st.data(),
)
@handle_cmd_line_args
def test_split(*, data, x_n_noss_n_axis_n_wr, dtype, tensor_fn, device, call, fw):
    # smoke test
    x, num_or_size_splits, axis, with_remainder = x_n_noss_n_axis_n_wr

    # mxnet does not support 0-dimensional variables
    assume(
        not (isinstance(x, Number) and tensor_fn == helpers.var_fn and fw == "mxnet")
    )

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
