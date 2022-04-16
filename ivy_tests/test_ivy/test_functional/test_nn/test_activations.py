"""
Collection of tests for unified neural network activation functions
"""

# global
import pytest
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# relu
@pytest.mark.parametrize(
    "x", [[[-1., 1., 2.]]])
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_relu(x, dtype, as_variable, with_out, native_array, fw):
    if dtype in ivy.invalid_dtype_strs:
        return  # invalid dtype
    if dtype == 'float16' and fw == 'torch':
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
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# leaky_relu
@pytest.mark.parametrize(
    "x", [[[-1., 1., 2.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_leaky_relu(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.leaky_relu(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.leaky_relu, x), ivy.functional.backends.numpy.leaky_relu(ivy.to_numpy(x)))
    # docstring test
    helpers.assert_docstring_examples_run(ivy.leaky_relu)


# gelu
@pytest.mark.parametrize(
    "x", [[[-1., 1., 2.]]])
@pytest.mark.parametrize(
    "approx", [True, False])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_gelu(x, approx, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.gelu(x, approx)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.gelu, x, approx), ivy.functional.backends.numpy.gelu(ivy.to_numpy(x), approx))


# tanh
@pytest.mark.parametrize(
    "x", [[[-1., 1., 2.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_tanh(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.tanh(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.tanh, x), ivy.functional.backends.numpy.tanh(ivy.to_numpy(x)))
    # docstring test
    helpers.assert_docstring_examples_run(ivy.tanh)


# sigmoid
@pytest.mark.parametrize(
    "x", [[[-1., 1., 2.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_sigmoid(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.sigmoid(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.sigmoid, x), ivy.functional.backends.numpy.sigmoid(ivy.to_numpy(x)))


# softmax
@pytest.mark.parametrize(
    "x", [[[-1., 1., 2.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_softmax(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.softmax(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.softmax, x), ivy.functional.backends.numpy.softmax(ivy.to_numpy(x)))


# softplus
@pytest.mark.parametrize(
    "x", [[[-1., 1., 2.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_softplus(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.softplus(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.softplus, x), ivy.functional.backends.numpy.softplus(ivy.to_numpy(x)))
    # docstring test
    helpers.assert_docstring_examples_run(ivy.softplus)