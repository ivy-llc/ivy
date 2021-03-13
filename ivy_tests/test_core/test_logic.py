"""
Collection of tests for templated logic functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# logical_and
@pytest.mark.parametrize(
    "x1_n_x2", [([True, True], [False, True]), ([[0.]], [[1.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_logical_and(x1_n_x2, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x1, x2 = x1_n_x2
    x1 = tensor_fn(x1, dtype_str, dev_str)
    x2 = tensor_fn(x2, dtype_str, dev_str)
    ret = ivy.logical_and(x1, x2)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.logical_and, x1, x2), ivy.numpy.logical_and(ivy.to_numpy(x1), ivy.to_numpy(x2)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    helpers.assert_compilable(ivy.logical_and)


# logical_or
@pytest.mark.parametrize(
    "x1_n_x2", [([True, True], [False, True]), ([[0.]], [[1.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_logical_or(x1_n_x2, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x1, x2 = x1_n_x2
    x1 = tensor_fn(x1, dtype_str, dev_str)
    x2 = tensor_fn(x2, dtype_str, dev_str)
    ret = ivy.logical_or(x1, x2)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.logical_or, x1, x2), ivy.numpy.logical_or(ivy.to_numpy(x1), ivy.to_numpy(x2)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    helpers.assert_compilable(ivy.logical_or)


# logical_not
@pytest.mark.parametrize(
    "x", [[True, True], [[0.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_logical_not(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.logical_not(x)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.logical_not, x), ivy.numpy.logical_not(ivy.to_numpy(x)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    helpers.assert_compilable(ivy.logical_not)
