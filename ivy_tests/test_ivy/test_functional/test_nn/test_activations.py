"""Collection of tests for unified neural network activation functions."""

# global
import pytest
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# relu
@given(
    x=st.lists(st.floats()),
    dtype=st.sampled_from(ivy.all_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    native_array=st.booleans(),
)
def test_relu(x, dtype, as_variable, with_out, native_array, fw):
    if dtype in ivy.invalid_dtypes:
        return  # invalid dtype
    if dtype == "float16" and fw == "torch":
        return  # torch does not support float16 for relu
    x = ivy.array(x, dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            return  # only floating point variables are supported
        if with_out:
            return  # variables do not support out argument
        x = ivy.variable(x)
    if native_array:
        x = x.data
    ret = ivy.relu(x)
    out = ret
    if with_out:
        if as_variable:
            out = ivy.variable(out)
        if native_array:
            out = out.data
        ret = ivy.relu(x, out=out)
        if not native_array:
            assert ret is out
        if fw in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)
    # value test
    if dtype == "bfloat16":
        return  # bfloat16 is not supported by numpy
    assert np.allclose(
        np.nan_to_num(ivy.to_numpy(ret)), np.nan_to_num(ivy_np.relu(ivy.to_numpy(x)))
    )


# leaky_relu
@pytest.mark.parametrize("x", [[[-1.0, 1.0, 2.0]]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_leaky_relu(x, dtype, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    ret = ivy.leaky_relu(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.leaky_relu, x), ivy_np.leaky_relu(ivy.to_numpy(x)))


# gelu
@pytest.mark.parametrize("x", [[[-1.0, 1.0, 2.0]]])
@pytest.mark.parametrize("approx", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_gelu(x, approx, dtype, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    ret = ivy.gelu(x, approx)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.gelu, x, approx), ivy_np.gelu(ivy.to_numpy(x), approx))
<<<<<<< HEAD
=======

>>>>>>> cde340fc8199e9da5d371eef8cdafcb1f61b866d

# tanh
@pytest.mark.parametrize("x", [[[-1.0, 1.0, 2.0]]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_tanh(x, dtype, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    ret = ivy.tanh(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.tanh, x), ivy_np.tanh(ivy.to_numpy(x)))


# sigmoid
@pytest.mark.parametrize("x", [[[-1.0, 1.0, 2.0]]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_sigmoid(x, dtype, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    ret = ivy.sigmoid(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.sigmoid, x), ivy_np.sigmoid(ivy.to_numpy(x)))


# softmax
@pytest.mark.parametrize("x", [[[-1.0, 1.0, 2.0]]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_softmax(x, dtype, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    ret = ivy.softmax(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.softmax, x), ivy_np.softmax(ivy.to_numpy(x)))


# softplus
@pytest.mark.parametrize("x", [[[-1.0, 1.0, 2.0]]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_softplus(x, dtype, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    ret = ivy.softplus(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.softplus, x), ivy_np.softplus(ivy.to_numpy(x)))
