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
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return
    if call in [helpers.mx_call] and dtype == "int16":
        # mxnet does not support int16
        return
    # to numpy
    if from_numpy:
        object_in = np.array(object_in)
    # smoke test
    ret = ivy.array(object_in, dtype, device)
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


# native_array
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_dtypes),
    from_numpy=st.booleans(),
)
def test_native_array(dtype_and_x, from_numpy, device, call, fw):
    dtype, object_in = dtype_and_x
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return
    if call in [helpers.mx_call] and dtype == "int16":
        # mxnet does not support int16
        return
    # to numpy
    if from_numpy:
        object_in = np.array(object_in)
    # smoke test
    ret = ivy.native_array(object_in, dtype, device)
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


# linspace
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
    start = tensor_fn(start, dtype, device)
    stop = tensor_fn(stop, dtype, device)
    ret = ivy.linspace(start, stop, num, axis, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    target_shape = list(start.shape)
    target_shape.insert(axis + 1 if (axis and axis != -1) else len(target_shape), num)
    assert ret.shape == tuple(target_shape)
    # value test
    assert np.allclose(
        call(ivy.linspace, start, stop, num, axis, device=device),
        np.asarray(
            ivy.functional.backends.numpy.linspace(
                ivy.to_numpy(start), ivy.to_numpy(stop), num, axis
            )
        ),
    )


# logspace
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
    start = tensor_fn(start, dtype, device)
    stop = tensor_fn(stop, dtype, device)
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
            ivy.to_numpy(start), ivy.to_numpy(stop), num, base, axis
        ),
    )


# Still to Add #
# ---------------#

# arange()
# asarray()
# empty()
# empty_like()
# eye()
# from_dlpack()
# full()
# full_like()
# meshgrid()
# ones()
# ones_like()
# tril()
# triu()
# zeros()
# zeros_like()
