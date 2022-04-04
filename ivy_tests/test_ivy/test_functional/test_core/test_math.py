"""
Collection of tests for unified math functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# cosh
@pytest.mark.parametrize(
    "x", [[[.1, .2, .3]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cosh(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.cosh(x)
    # type test
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.cosh, x), ivy.functional.backends.numpy.cosh(ivy.to_numpy(x)))


# tanh
@pytest.mark.parametrize(
    "x", [[[.1, .2, .3]]])
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


# asinh
@pytest.mark.parametrize(
    "x", [[[.1, .2, .3]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_asinh(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.asinh(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.asinh, x), ivy.functional.backends.numpy.asinh(ivy.to_numpy(x)))




# atanh
@pytest.mark.parametrize(
    "x", [[[-0.5, 0.5]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_atanh(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.atanh(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.atanh, x), ivy.functional.backends.numpy.atanh(ivy.to_numpy(x)))


# log
@pytest.mark.parametrize(
    "x", [[1.], [[1.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_log(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.log(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.log, x), ivy.functional.backends.numpy.log(ivy.to_numpy(x)))


# exp
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_exp(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.exp(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.exp, x), ivy.functional.backends.numpy.exp(ivy.to_numpy(x)))


# erf
@pytest.mark.parametrize(
    "x", [[0.], [[1.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_erf(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.erf(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.erf, x), ivy.functional.backends.numpy.erf(ivy.to_numpy(x)))
