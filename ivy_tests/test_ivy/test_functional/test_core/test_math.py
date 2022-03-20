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


# sin
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_sin(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.sin(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.sin, x), ivy.functional.backends.numpy.sin(ivy.to_numpy(x)))


# cos
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cos(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.cos(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.cos, x), ivy.functional.backends.numpy.cos(ivy.to_numpy(x)))


# tan
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_tan(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.tan(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.tan, x), ivy.functional.backends.numpy.tan(ivy.to_numpy(x)))


# asin
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_asin(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.asin(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.asin, x), ivy.functional.backends.numpy.asin(ivy.to_numpy(x)))


# atan
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_atan(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.atan(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.atan, x), ivy.functional.backends.numpy.atan(ivy.to_numpy(x)))


# atan2
@pytest.mark.parametrize(
    "x1_n_x2", [([0.], [0.]), ([[0.]], [[0.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_atan2(x1_n_x2, dtype, tensor_fn, dev, call):
    # smoke test
    x1, x2 = x1_n_x2
    x1 = tensor_fn(x1, dtype, dev)
    x2 = tensor_fn(x2, dtype, dev)
    ret = ivy.atan2(x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.atan2, x1, x2), ivy.functional.backends.numpy.atan2(ivy.to_numpy(x1), ivy.to_numpy(x2)))


# sinh
@pytest.mark.parametrize(
    "x", [[[.1, .2, .3]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_sinh(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.sinh(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.sinh, x), ivy.functional.backends.numpy.sinh(ivy.to_numpy(x)))


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
    assert ivy.is_array(ret)
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
    assert ivy.is_array(ret)
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
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.asinh, x), ivy.functional.backends.numpy.asinh(ivy.to_numpy(x)))


# acosh
# @pytest.mark.parametrize(
#     "x", [[[1, 1.2, 200, 10000]]])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_acosh(x, dtype, tensor_fn, dev, call):
#     # smoke test
#     x = tensor_fn(x, dtype, dev)
#     ret = ivy.acosh(x)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     assert ret.shape == x.shape
#     # value test
#     assert np.allclose(call(ivy.acosh, x), ivy.functional.backends.numpy.acosh(ivy.to_numpy(x)))


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
    assert ivy.is_array(ret)
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
    assert ivy.is_array(ret)
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
    assert ivy.is_array(ret)
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
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.erf, x), ivy.functional.backends.numpy.erf(ivy.to_numpy(x)))
