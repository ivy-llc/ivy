"""Collection of tests for creation functions."""

# global
import numpy as np
import pytest
from numbers import Number
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# array
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_dtypes),
    from_numpy=st.booleans(),
)
def test_array(dtype_and_x, from_numpy, device, call, fw):
    dtype, object_in = dtype_and_x
    # to numpy
    if from_numpy:
        object_in = np.array(object_in)
    # smoke test
    ret = ivy.array(object_in, dtype=dtype, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    helpers.assert_all_close(ivy.to_numpy(ret), np.array(object_in).astype(dtype))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return

# asarray
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="asarray"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_asarray(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "asarray",
        x=x
    )

# native_array
"""
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_dtypes),
    from_numpy=st.booleans(),
)
def test_native_array(dtype_and_x, from_numpy, device, call, fw):
    dtype, object_in = dtype_and_x
    # to numpy
    if from_numpy:
        object_in = np.array(object_in)
    # smoke test
    ret = ivy.native_array(object_in, dtype=dtype, device=device)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    helpers.assert_all_close(ivy.to_numpy(ret), np.array(object_in).astype(dtype))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return
"""
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="native_array"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_native_array(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "native_array",
        x=x
    )

# linspace
"""
@pytest.mark.parametrize(
    "start_n_stop_n_num_n_axis",
    [
        [1, 10, 100, None],
        [[[0.0, 1.0, 2.0]], [[1.0, 2.0, 3.0]], 150, -1],
        [[[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], 6, -2],
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_linspace(start_n_stop_n_num_n_axis, dtype, tensor_fn, device, call):
    # smoke test
    start, stop, num, axis = start_n_stop_n_num_n_axis
    if (
        (isinstance(start, Number) or isinstance(stop, Number))
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    start = tensor_fn(start, dtype=dtype, device=device)
    stop = tensor_fn(stop, dtype=dtype, device=device)
    ret = ivy.linspace(start, stop, num, axis, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    target_shape = list(start.shape)
    target_shape.insert(axis + 1 if (axis and axis != -1) else len(target_shape), num)
    assert ret.shape == tuple(target_shape)
    # value test
    start_np = ivy.to_numpy(start)
    stop_np = ivy.to_numpy(stop)
    ivy.set_backend("numpy")
    np_ret = ivy.linspace(start_np, stop_np, num, axis)
    ivy.unset_backend()
    assert np.allclose(
        call(ivy.linspace, start, stop, num, axis, device=device),
        np_ret,
    )
"""

@given(
    start = st.integers() | st.lists(st.integers()) | st.floats() | st.lists(st.floats()), 
    stop = st.integers() | st.lists(st.integers()) | st.floats() | st.lists(st.floats()),
    num = st.integers(),
    axis = st.none() | st.integers(),
    dtype = st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="linspace"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_linspace(
    start,
    stop,
    num,
    axis,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    if fw == "torch" and dtype == "float16":
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
        "linspace",
        start = start,
        stop = stop,
        num = num,
        axis = axis,
    )



# logspace
"""
@pytest.mark.parametrize(
    "start_n_stop_n_num_n_base_n_axis",
    [
        [1, 10, 100, 10.0, None],
        [[[0.0, 1.0, 2.0]], [[1.0, 2.0, 3.0]], 150, 2.0, -1],
        [[[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], 6, 5.0, -2],
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_logspace(start_n_stop_n_num_n_base_n_axis, dtype, tensor_fn, device, call):
    # smoke test
    start, stop, num, base, axis = start_n_stop_n_num_n_base_n_axis
    if (
        (isinstance(start, Number) or isinstance(stop, Number))
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    start = tensor_fn(start, dtype=dtype, device=device)
    stop = tensor_fn(stop, dtype=dtype, device=device)
    ret = ivy.logspace(start, stop, num, base, axis, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    target_shape = list(start.shape)
    target_shape.insert(axis + 1 if (axis and axis != -1) else len(target_shape), num)
    assert ret.shape == tuple(target_shape)
    # value test
    assert np.allclose(
        call(ivy.logspace, start, stop, num, base, axis, device=device),
        ivy.functional.backends.numpy.logspace(
            ivy.to_numpy(start), ivy.to_numpy(stop), num, base, axis, device=device
        ),
    )
"""

@given(
    start = st.integers() | st.lists(st.integers()), 
    stop = st.integers() | st.lists(st.integers()),
    num = st.integers(),
    base = st.floats(),
    axis = st.none() | st.integers(),
    dtype = st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="logspace"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_logspace(
    start,
    stop,
    num,
    base,
    axis,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    if fw == "torch" and dtype == "float16":
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
        "logspace",
        start = start,
        stop = stop,
        num = num,
        base = base,
        axis = axis,
    )
    
    



# Still to Add #
# ---------------#

# arange()
@given(
    start = st.integers(min_value = 0), 
    stop = st.integers(min_value = 0) | st.none(),
    step = st.integers().filter(lambda x : True if x != 0 else False),
    dtype = st.sampled_from(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="arange"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_arange(
    start,
    stop,
    step,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    if fw == "torch" and dtype == "float16":
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
        "arange",
        start = start,
        stop = stop,
        step = step,
    )


# asarray()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="asarray"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_asarray(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "asarray",
        object_in = x,
    )


# empty()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="empty"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_empty(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "empty",
        shape=np.asarray(x, dtype=dtype),
    )


# empty_like()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="empty_like"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_empty_like(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "empty_like",
        x=np.asarray(x, dtype=dtype),
    )


# eye()
@given(
    n_rows = st.integers(min_value = 0),
    n_cols = st.none() | st.integers(min_value = 0),
    k = st.none() | st.integers(),
    dtype = st.sampled_from(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="eye"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_eye(
    n_rows,
    n_cols,
    k,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    if fw == "torch" and dtype == "float16":
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
        "eye",
        n_rows = n_rows,
        n_cols = n_cols,
        k = k,
    )


# from_dlpack()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="from_dlpack"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_from_dlpack(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "from_dlpack",
        x=np.asarray(x, dtype=dtype),
    )


# full()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="full"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    fill_value= st.integers() | st.floats(),
)
def test_full(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    fill_value,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "full",
        shape=np.asarray(x, dtype=dtype),
        fill_value = fill_value,
    )


# full_like()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="full_like"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    fill_value= st.integers() | st.floats(),
)
def test_full_like(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    fill_value,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "full_like",
        x=np.asarray(x, dtype=dtype),
        fill_value = fill_value,
    )


# meshgrid
@given(
    dtype_and_x=helpers.dtype_and_values(
        ivy_np.valid_int_dtypes,
        st.shared(st.integers(1, 3), key="num_arrays"),
        shared_dtype=True,
    ),
    as_variable=st.booleans(),
    num_positional_args=st.shared(st.integers(1, 3), key="num_arrays"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_meshgrid(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = helpers.as_lists(*dtype_and_x)
    kw = {}
    for i, (dtype_, x_) in enumerate(zip(dtype, x)):
        kw["x{}".format(i)] = np.asarray(x_, dtype=dtype_)
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "meshgrid",
        **kw
    )


# ones()
@given(
dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
as_variable = st.booleans(),
with_out=st.booleans(),
num_positional_args=helpers.num_positional_args(fn_name="ones"),
native_array = st.booleans(),
container=st.booleans(),
instance_method=st.booleans(),
)
def test_ones(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "ones",
        shape=np.asarray(x, dtype=dtype),
    )


# ones_like()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_ones_like(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "ones_like",
        x=np.asarray(x, dtype=dtype),
    )


# tril()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    k=st.integers(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="tril"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_tril(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    k,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "tril",
        x=np.asarray(x, dtype=dtype),
        k=k,
    )


# triu()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    k=st.integers(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="triu"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_triu(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    k,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "triu",
        x=np.asarray(x, dtype=dtype),
        k = k,
    )


# zeros()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="zeros"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_zeros(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "zeros",
        shape=np.asarray(x, dtype=dtype),
    )


# zeros_like()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_zeros_like(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
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
        "zeros_like",
        x=np.asarray(x, dtype=dtype),
    )
    