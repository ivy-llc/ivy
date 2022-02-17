"""
Collection of tests for unified logic functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# logical_and
@pytest.mark.parametrize(
    "x1_n_x2", [([True, True], [False, True]), ([[0.]], [[1.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_logical_and(x1_n_x2, dtype, tensor_fn, dev, call):
    # smoke test
    x1, x2 = x1_n_x2
    x1 = tensor_fn(x1, dtype, dev)
    x2 = tensor_fn(x2, dtype, dev)
    ret = ivy.logical_and(x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.logical_and, x1, x2), ivy.functional.backends.numpy.logical_and(ivy.to_numpy(x1), ivy.to_numpy(x2)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    if not ivy.array_mode():
        helpers.assert_compilable(ivy.logical_and)


# logical_or
@pytest.mark.parametrize(
    "x1_n_x2", [([True, True], [False, True]), ([[0.]], [[1.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_logical_or(x1_n_x2, dtype, tensor_fn, dev, call):
    # smoke test
    x1, x2 = x1_n_x2
    x1 = tensor_fn(x1, dtype, dev)
    x2 = tensor_fn(x2, dtype, dev)
    ret = ivy.logical_or(x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.logical_or, x1, x2), ivy.functional.backends.numpy.logical_or(ivy.to_numpy(x1), ivy.to_numpy(x2)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    if not ivy.array_mode():
        helpers.assert_compilable(ivy.logical_or)


# logical_not
@pytest.mark.parametrize(
    "x", [[True, True], [[0.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_logical_not(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.logical_not(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.logical_not, x), ivy.functional.backends.numpy.logical_not(ivy.to_numpy(x)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    if not ivy.array_mode():
        helpers.assert_compilable(ivy.logical_not)
