"""
Collection of tests for unified neural network activation functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# relu
@pytest.mark.parametrize(
    "x", [[[-1., 1., 2.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_relu(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.relu(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.relu, x), ivy.functional.backends.numpy.relu(ivy.to_numpy(x)))
    # docstring test
    helpers.assert_docstring_examples_run(ivy.relu)


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
