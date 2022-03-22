"""
Collection of tests for unified reduction functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# einsum
@pytest.mark.parametrize(
    "eq_n_op_n_shp", [("ii", (np.arange(25).reshape(5, 5),), ()),
                      ("ii->i", (np.arange(25).reshape(5, 5),), (5,)),
                      ("ij,j", (np.arange(25).reshape(5, 5), np.arange(5)), (5,))])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_einsum(eq_n_op_n_shp, dtype, tensor_fn, dev, call):
    # smoke test
    eq, operands, true_shape = eq_n_op_n_shp
    operands = [tensor_fn(op, dtype, dev) for op in operands]
    ret = ivy.einsum(eq, *operands)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_shape
    # value test
    assert np.allclose(call(ivy.einsum, eq, *operands),
                       ivy.functional.backends.numpy.einsum(eq, *[ivy.to_numpy(op) for op in operands]))


# all
@pytest.mark.parametrize(
    "x", [[1., 2., 3.], [[1., 2., 3.]]])
@pytest.mark.parametrize(
    "axis", [None, 0, -1, (0,), (-1,)])
@pytest.mark.parametrize(
    "kd", [True, False])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_all(x, axis, kd, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.all(x, axis, kd)
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
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.all, x),
                       ivy.functional.backends.numpy.all(ivy.to_numpy(x)))
