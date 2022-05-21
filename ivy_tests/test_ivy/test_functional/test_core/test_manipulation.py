"""Collection of tests for manipulation functions."""

# global
import pytest
import numpy as np
import math
from numbers import Number
from hypothesis import HealthCheck, given, settings, strategies as st


# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# concat
@given(
    common_shape=helpers.lists(
        st.integers(2, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    unique_idx=helpers.integers(0, "num_dims"),
    unique_dims=helpers.lists(
        st.integers(2, 3),
        min_size="num_arrays",
        max_size="num_arrays",
        size_bounds=[2, 3],
    ),
    dtype=helpers.array_dtypes(),
    as_variable=helpers.array_bools(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.array_bools(),
    container=helpers.array_bools(),
    instance_method=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_concat(
    common_shape,
    unique_idx,
    unique_dims,
    dtype,
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
    xs = [
        np.random.uniform(
            size=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:]
        ).astype(dt)
        for ud, dt in zip(unique_dims, dtype)
    ]
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "concat",
        xs=xs,
        axis=unique_idx,
    )


# expand_dims
@given(
    array_shape=helpers.lists(
        st.integers(2, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    unique_idx=helpers.integers(0, "num_dims"),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_expand_dims(
    array_shape,
    unique_idx,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    seed,
    fw,
):
    # smoke this for torch
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    np.random.seed(seed)

    x = np.random.uniform(size=array_shape).astype(dtype)

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "expand_dims",
        x=x,
        axis=unique_idx,
    )


# flip
@given(
    array_shape=helpers.lists(
        st.integers(2, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    axis=helpers.valid_axes(ndim="num_dims", size_bounds=[1, 3]),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_flip(
    array_shape,
    axis,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    seed,
    fw,
):
    # smoke this for torch
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    np.random.seed(seed)

    x = np.random.uniform(size=array_shape).astype(dtype)

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "flip",
        x=x,
        axis=axis,
    )


# permute_dims
@given(
    array_shape=helpers.lists(
        st.integers(1, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_permute_dims(
    array_shape,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    seed,
    fw,
):
    # smoke this for torch
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    np.random.seed(seed)

    x = np.random.uniform(size=array_shape).astype(dtype)
    axes = np.random.permutation(len(array_shape)).tolist()

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "permute_dims",
        x=x,
        axes=axes,
    )


# reshape
@settings(
    suppress_health_check=(HealthCheck.filter_too_much,)
)  # cant figure this out ;-;
@given(
    array_shape=helpers.lists(
        st.integers(1, 10), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_reshape(
    array_shape,
    dtype,
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
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))

    # draw a valid reshape shape
    shape = data.draw(helpers.reshape_shapes(x.shape))

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "reshape",
        x=x,
        shape=shape,
    )


# roll
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 3),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_roll(
    array_shape,
    dtype,
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
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
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

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "roll",
        x=x,
        shift=shift,
        axis=axis,
    )


# squeeze
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ).filter(lambda s: 1 in s),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_squeeze(
    array_shape,
    dtype,
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
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    squeezable_axes = [i for i, side in enumerate(x.shape) if side == 1]

    valid_axis = st.sampled_from(squeezable_axes) | helpers.subsets(squeezable_axes)

    axis = data.draw(valid_axis)

    # we need subset of size atleast 1, think of better way to do this
    # right now, we are just ignoring when we sample an empty subset
    if not isinstance(axis, int):
        if len(axis) == 0:
            return

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "squeeze",
        x=x,
        axis=axis,
    )


# stack
@given(
    array_shape=helpers.lists(
        st.integers(0, 3), min_size="num_dims", max_size="num_dims", size_bounds=[0, 3]
    ),
    num_arrays=st.shared(st.integers(1, 3), key="num_arrays"),
    dtype=helpers.array_dtypes(na=st.shared(st.integers(1, 3), key="num_arrays")),
    data=st.data(),
    as_variable=helpers.array_bools(na=st.shared(st.integers(1, 3), key="num_arrays")),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=helpers.array_bools(na=st.shared(st.integers(1, 3), key="num_arrays")),
    container=helpers.array_bools(na=st.shared(st.integers(1, 3), key="num_arrays")),
    instance_method=st.booleans(),
)
def test_stack(
    array_shape,
    num_arrays,
    dtype,
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
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    xs = [
        data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype[i]))
        for i in range(num_arrays)
    ]
    ndim = len(xs[0].shape)
    axis = data.draw(st.integers(-ndim, max(0, ndim - 1)))

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "stack",
        x=xs,
        axis=axis,
    )


# Extra #
# ------#


# repeat
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 3),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_repeat(
    array_shape,
    dtype,
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
    if (fw == "torch" and dtype in ["uint16", "uint32", "uint64"]) or (
        fw == "tensorflow" and dtype in ["uint16"]
    ):
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    ndim = len(x.shape)

    valid_axis = st.none() | st.integers(-ndim, ndim - 1)
    axis = data.draw(valid_axis)

    repeats = data.draw(st.integers(1, 3))

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "repeat",
        x=x,
        repeats=repeats,
        axis=axis,
    )


# tile
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_tile(
    array_shape,
    dtype,
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
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))

    # tensorflow needs that reps is exactly of same dimensions as the input
    # other frameworks can broadcast the results
    if fw == "tensorflow":
        reps = data.draw(
            helpers.nph.broadcastable_shapes(
                shape=x.shape, min_dims=len(x.shape), max_dims=len(x.shape)
            )
        )
    else:
        reps = data.draw(
            helpers.nph.broadcastable_shapes(shape=x.shape, min_dims=len(x.shape))
        )

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "tile",
        x=x,
        reps=reps,
    )


# constant_pad
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 3),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_constant_pad(
    array_shape,
    dtype,
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
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    pads = [
        (data.draw(st.integers(0, 3)), data.draw(st.integers(0, 3)))
        for _ in range(len(x.shape))
    ]
    constant = data.draw(st.integers(0, 10))

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "constant_pad",
        x=x,
        pad_width=pads,
        value=constant,
    )


# zero_pad
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 3),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_zero_pad(
    array_shape,
    dtype,
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
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    pads = [
        (data.draw(st.integers(0, 3)), data.draw(st.integers(0, 3)))
        for _ in range(len(x.shape))
    ]

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "zero_pad",
        x=x,
        pad_width=pads,
    )


# swapaxes
@given(
    array_shape=helpers.lists(
        st.integers(0, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtype_strs),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 3),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_swapaxes(
    array_shape,
    dtype,
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
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    valid_axes = st.integers(0, len(x.shape) - 1)
    axis0 = data.draw(valid_axes)
    axis1 = data.draw(valid_axes)

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "swapaxes",
        x=x,
        axis0=axis0,
        axis1=axis1,
    )


# clip
@given(
    x_min_n_max=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs, n_arrays=3),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(2, 3),
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
    call,
    fw,
):
    # smoke test
    if (
        (
            isinstance(x_min_n_max[1][0], Number)
            or isinstance(x_min_n_max[1][1], Number)
            or isinstance(x_min_n_max[1][2], Number)
        )
        and as_variable
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        return
    dtype = x_min_n_max[0]
    x = x_min_n_max[1][0]
    min_val1 = np.array(x_min_n_max[1][1], dtype=dtype[1])
    max_val1 = np.array(x_min_n_max[1][2], dtype=dtype[2])
    min_val = np.minimum(min_val1, max_val1)
    max_val = np.maximum(min_val1, max_val1)
    if fw == "torch" and (
        any(d in ["uint16", "uint32", "uint64", "float16"] for d in dtype)
        or any(np.isnan(max_val))
        or len(x) == 0
    ):
        return
    if (
        (len(min_val) != 0 and len(min_val) != 1)
        or (len(max_val) != 0 and len(max_val) != 1)
    ) and call in [helpers.mx_call]:
        # mxnet only supports numbers or 0 or 1 dimensional arrays for min
        # and max while performing clip
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "clip",
        x=np.asarray(x, dtype=dtype[0]),
        x_min=ivy.array(min_val),
        x_max=ivy.array(max_val),
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
    dtype=st.sampled_from(ivy_np.valid_float_dtype_strs),
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
    x = tensor_fn(x, dtype, device)
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
