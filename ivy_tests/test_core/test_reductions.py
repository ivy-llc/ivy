"""
Collection of tests for templated reduction functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# reduce_sum
@pytest.mark.parametrize(
    "x", [[1., 2., 3.], [[1., 2., 3.]]])
@pytest.mark.parametrize(
    "axis", [None, 0, -1, (0,), (-1,)])
@pytest.mark.parametrize(
    "kd", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_reduce_sum(x, axis, kd, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.reduce_sum(x, axis, kd)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if axis is None:
        expected_shape = [1]*len(x.shape) if kd else []
    else:
        axis_ = [axis] if isinstance(axis, int) else axis
        axis_ = [item % len(x.shape) for item in axis_]
        expected_shape = list(x.shape)
        if kd:
            expected_shape = [1 if i % len(x.shape) in axis_ else item for i, item in enumerate(expected_shape)]
        else:
            [expected_shape.pop(item) for item in axis_]
    expected_shape = [1] if expected_shape == [] else expected_shape
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.reduce_sum, x), ivy.numpy.reduce_sum(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.reduce_sum)


# reduce_prod
@pytest.mark.parametrize(
    "x", [[1., 2., 3.], [[1., 2., 3.]]])
@pytest.mark.parametrize(
    "axis", [None, 0, -1, (0,), (-1,)])
@pytest.mark.parametrize(
    "kd", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_reduce_prod(x, axis, kd, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.reduce_prod(x, axis, kd)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if axis is None:
        expected_shape = [1]*len(x.shape) if kd else []
    else:
        axis_ = [axis] if isinstance(axis, int) else axis
        axis_ = [item % len(x.shape) for item in axis_]
        expected_shape = list(x.shape)
        if kd:
            expected_shape = [1 if i % len(x.shape) in axis_ else item for i, item in enumerate(expected_shape)]
        else:
            [expected_shape.pop(item) for item in axis_]
    expected_shape = [1] if expected_shape == [] else expected_shape
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.reduce_prod, x), ivy.numpy.reduce_prod(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.reduce_prod)


# reduce_mean
@pytest.mark.parametrize(
    "x", [[1., 2., 3.], [[1., 2., 3.]]])
@pytest.mark.parametrize(
    "axis", [None, 0, -1, (0,), (-1,)])
@pytest.mark.parametrize(
    "kd", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_reduce_mean(x, axis, kd, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.reduce_mean(x, axis, kd)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if axis is None:
        expected_shape = [1]*len(x.shape) if kd else []
    else:
        axis_ = [axis] if isinstance(axis, int) else axis
        axis_ = [item % len(x.shape) for item in axis_]
        expected_shape = list(x.shape)
        if kd:
            expected_shape = [1 if i % len(x.shape) in axis_ else item for i, item in enumerate(expected_shape)]
        else:
            [expected_shape.pop(item) for item in axis_]
    expected_shape = [1] if expected_shape == [] else expected_shape
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.reduce_mean, x), ivy.numpy.reduce_mean(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.reduce_mean)


# reduce_min
@pytest.mark.parametrize(
    "x", [[1., 2., 3.], [[1., 2., 3.]]])
@pytest.mark.parametrize(
    "axis", [None, 0, -1, (0,), (-1,)])
@pytest.mark.parametrize(
    "kd", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_reduce_min(x, axis, kd, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.reduce_min(x, axis, kd)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if axis is None:
        expected_shape = [1]*len(x.shape) if kd else []
    else:
        axis_ = [axis] if isinstance(axis, int) else axis
        axis_ = [item % len(x.shape) for item in axis_]
        expected_shape = list(x.shape)
        if kd:
            expected_shape = [1 if i % len(x.shape) in axis_ else item for i, item in enumerate(expected_shape)]
        else:
            [expected_shape.pop(item) for item in axis_]
    expected_shape = [1] if expected_shape == [] else expected_shape
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.reduce_min, x), ivy.numpy.reduce_min(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.reduce_min)


# reduce_max
@pytest.mark.parametrize(
    "x", [[1., 2., 3.], [[1., 2., 3.]]])
@pytest.mark.parametrize(
    "axis", [None, 0, -1, (0,), (-1,)])
@pytest.mark.parametrize(
    "kd", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_reduce_max(x, axis, kd, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.reduce_max(x, axis, kd)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if axis is None:
        expected_shape = [1]*len(x.shape) if kd else []
    else:
        axis_ = [axis] if isinstance(axis, int) else axis
        axis_ = [item % len(x.shape) for item in axis_]
        expected_shape = list(x.shape)
        if kd:
            expected_shape = [1 if i % len(x.shape) in axis_ else item for i, item in enumerate(expected_shape)]
        else:
            [expected_shape.pop(item) for item in axis_]
    expected_shape = [1] if expected_shape == [] else expected_shape
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.reduce_max, x), ivy.numpy.reduce_max(ivy.to_numpy(x)))
    # compilation test
    helpers.assert_compilable(ivy.reduce_max)
