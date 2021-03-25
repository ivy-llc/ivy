"""
Collection of tests for templated math functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# sin
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_sin(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.sin(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.sin, x), ivy.numpy.sin(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.sin)


# cos
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cos(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.cos(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.cos, x), ivy.numpy.cos(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.cos)


# tan
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_tan(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.tan(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.tan, x), ivy.numpy.tan(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.tan)


# asin
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_asin(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.asin(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.asin, x), ivy.numpy.asin(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.asin)


# acos
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_acos(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.acos(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.acos, x), ivy.numpy.acos(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.acos)


# atan
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_atan(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.atan(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.atan, x), ivy.numpy.atan(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.atan)


# atan2
@pytest.mark.parametrize(
    "x1_n_x2", [([0.], [0.]), ([[0.]], [[0.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_atan2(x1_n_x2, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x1, x2 = x1_n_x2
    x1 = tensor_fn(x1, dtype_str, dev_str)
    x2 = tensor_fn(x2, dtype_str, dev_str)
    ret = ivy.atan2(x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.atan2, x1, x2), ivy.numpy.atan2(ivy.to_numpy(x1), ivy.to_numpy(x2)))
    # compilation test
    helpers.assert_compilable(ivy.atan2)


# sinh
@pytest.mark.parametrize(
    "x", [[[.1, .2, .3]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_sinh(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.sinh(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.sinh, x), ivy.numpy.sinh(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.sinh)


# cosh
@pytest.mark.parametrize(
    "x", [[[.1, .2, .3]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cosh(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.cosh(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.cosh, x), ivy.numpy.cosh(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.cosh)


# tanh
@pytest.mark.parametrize(
    "x", [[[.1, .2, .3]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_tanh(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.tanh(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.tanh, x), ivy.numpy.tanh(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.tanh)


# asinh
@pytest.mark.parametrize(
    "x", [[[.1, .2, .3]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_asinh(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.asinh(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.asinh, x), ivy.numpy.asinh(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.asinh)


# acosh
@pytest.mark.parametrize(
    "x", [[[1, 1.2, 200, 10000]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_acosh(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.acosh(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.acosh, x), ivy.numpy.acosh(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.acosh)


# atanh
@pytest.mark.parametrize(
    "x", [[[-0.5, 0.5]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_atanh(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.atanh(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.atanh, x), ivy.numpy.atanh(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.atanh)


# log
@pytest.mark.parametrize(
    "x", [[1.], [[1.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_log(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.log(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.log, x), ivy.numpy.log(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.log)


# exp
@pytest.mark.parametrize(
    "x", [[0.], [[0.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_exp(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.exp(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.exp, x), ivy.numpy.exp(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.exp)
