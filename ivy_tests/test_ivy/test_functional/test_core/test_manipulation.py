"""Collection of tests for manipulation functions."""

# global
import pytest
import numpy as np
import math
from numbers import Number
from hypothesis import given, strategies as st


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
    seed=st.integers(0, 2**32 - 1),
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
    seed,
):
    np.random.seed(seed)

    # smoke for torch
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = np.random.uniform(size=array_shape).astype(dtype)

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
    instance_method=st.booleans()
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
    instance_method=st.booleans()
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
@pytest.mark.parametrize("dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_stack(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([1, 2, 3, 4], dtype=dtype)
    x2 = ivy.array([1, 2, 3, 4], dtype=dtype)
    out = ivy.array([[2, 3, 4, 5], [4, 5, 6, 7]], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x1 = ivy.variable(x1)
        x2 = ivy.variable(x2)
        out = ivy.variable(out)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.stack([x1, x2], 0, out=out)
    else:
        ret = ivy.stack([x1, x2], 0)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# Extra #
# ------#


# repeat
@pytest.mark.parametrize("dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_repeat(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == "tensorflow" and dtype == "uint16":
        pytest.skip("tf repeat doesnt allow uint16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 2, 3], dtype=dtype)
    out = ivy.array([2, 3, 4, 5, 6, 7], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.repeat(x, 2, out=out)
    else:
        ret = ivy.repeat(x, 2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# tile
@pytest.mark.parametrize("dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_tile(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == "tensorflow" and dtype == "uint16":
        pytest.skip("tf tile doesnt allow uint16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 2, 3], dtype=dtype)
    out = ivy.array([2, 3, 4, 5, 6, 7], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.tile(x, 2, out=out)
    else:
        ret = ivy.tile(x, 2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# constant_pad
@pytest.mark.parametrize("dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_constant_pad(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 2, 3], dtype=dtype)
    out = ivy.array([2, 3, 4, 5, 6, 7], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.constant_pad(x, [[2, 1]], out=out)
    else:
        ret = ivy.constant_pad(x, [[2, 1]])
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# zero_pad
@pytest.mark.parametrize("dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_zero_pad(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 2, 3], dtype=dtype)
    out = ivy.array([2, 3, 4, 5, 6, 7], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.zero_pad(x, [[2, 1]], out=out)
    else:
        ret = ivy.zero_pad(x, [[2, 1]])
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# swapaxes
@pytest.mark.parametrize("dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_swapaxes(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([[1, 2], [3, 4]], dtype=dtype)
    out = ivy.array([[2, 3], [4, 5]], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.swapaxes(x, 0, 1, out=out)
    else:
        ret = ivy.swapaxes(x, 0, 1)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


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
@pytest.mark.parametrize(
    "x_n_noss_n_axis_n_wr",
    [
        (1, 1, -1, False),
        ([[0.0, 1.0, 2.0, 3.0]], 2, 1, False),
        ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 2, 0, False),
        ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 2, 1, True),
        ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [2, 1], 1, False),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_split(x_n_noss_n_axis_n_wr, dtype, tensor_fn, device, call):
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
