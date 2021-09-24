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
    if ivy.wrapped_mode():
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
    if ivy.wrapped_mode():
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
    if ivy.wrapped_mode():
        helpers.assert_compilable(ivy.reduce_mean)


# reduce_var
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
def test_reduce_var(x, axis, kd, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.reduce_var(x, axis, kd)
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
    assert np.allclose(call(ivy.reduce_var, x), ivy.numpy.reduce_var(ivy.to_numpy(x)))
    # compilation test
    if ivy.wrapped_mode():
        helpers.assert_compilable(ivy.reduce_var)


# reduce_std
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
def test_reduce_std(x, axis, kd, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.reduce_std(x, axis, kd)
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
    assert np.allclose(call(ivy.reduce_std, x), ivy.numpy.reduce_var(ivy.to_numpy(x)) ** 0.5)
    # compilation test
    if call is helpers.torch_call:
        # PyTorch cannot yet compile ivy.core only functions, without a direct backend implementation
        return
    if ivy.wrapped_mode():
        helpers.assert_compilable(ivy.reduce_std)


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
    if ivy.wrapped_mode():
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
    if ivy.wrapped_mode():
        helpers.assert_compilable(ivy.reduce_max)


# einsum
@pytest.mark.parametrize(
    "eq_n_op_n_shp", [("ii", (np.arange(25).reshape(5, 5),), (1,)),
                      ("ii->i", (np.arange(25).reshape(5, 5),), (5,)),
                      ("ij,j", (np.arange(25).reshape(5, 5), np.arange(5)), (5,))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_einsum(eq_n_op_n_shp, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    eq, operands, true_shape = eq_n_op_n_shp
    operands = [tensor_fn(op, dtype_str, dev_str) for op in operands]
    ret = ivy.einsum(eq, *operands)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_shape
    # value test
    assert np.allclose(call(ivy.einsum, eq, *operands),
                       ivy.numpy.einsum(eq, *[ivy.to_numpy(op) for op in operands]))
    # compilation test
    if call is helpers.torch_call:
        # torch.jit functions can't take variable number of arguments
        return
    if ivy.wrapped_mode():
        helpers.assert_compilable(ivy.einsum)
