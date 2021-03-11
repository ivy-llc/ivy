"""
Collection of tests for templated general functions
"""

# global
import pytest
import ivy.numpy
import numpy as np
from operator import mul as _mul
from functools import reduce as _reduce

# local
import ivy
import ivy_tests.helpers as helpers

from collections.abc import Sequence


# Helpers #
# --------#

def _get_shape_of_list(lst, shape=()):
    if not lst:
        return []
    if not isinstance(lst, Sequence):
        return shape
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)
    shape += (len(lst),)
    shape = _get_shape_of_list(lst[0], shape)
    return shape


def np_scatter(indices, updates, shape, reduction='sum'):
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    if reduction == 'sum':
        target = np.zeros(shape, dtype=updates.dtype)
        np.add.at(target, indices_tuple, updates)
    elif reduction == 'min':
        target = np.ones(shape, dtype=updates.dtype)*1e12
        np.minimum.at(target, indices_tuple, updates)
        target = np.where(target == 1e12, 0., target)
    elif reduction == 'max':
        target = np.ones(shape, dtype=updates.dtype)*-1e12
        np.maximum.at(target, indices_tuple, updates)
        target = np.where(target == -1e12, 0., target)
    else:
        raise Exception('Invalid reduction selected')
    return target


# Tests #
# ------#

# tensor
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
def test_tensor(object_in, dtype_str, dev_str, call):
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    # smoke test
    ret = ivy.tensor(object_in, dtype_str, dev_str)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    assert np.allclose(call(ivy.tensor, object_in, dtype_str, dev_str), np.array(object_in).astype(dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return
    helpers.assert_compilable(ivy.tensor)


# to_numpy
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
def test_to_numpy(object_in, dtype_str, dev_str, call):
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call in [helpers.tf_graph_call]:
        # to_numpy() requires eager execution
        pytest.skip()
    # smoke test
    ret = ivy.to_numpy(ivy.tensor(object_in, dtype_str, dev_str))
    # type test
    assert isinstance(ret, np.ndarray)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    assert np.allclose(ivy.to_numpy(ivy.tensor(object_in, dtype_str, dev_str)), np.array(object_in).astype(dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support numpy conversion
        return
    helpers.assert_compilable(ivy.to_numpy)


# to_list
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
def test_to_list(object_in, dtype_str, dev_str, call):
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call in [helpers.tf_graph_call]:
        # to_list() requires eager execution
        pytest.skip()
    # smoke test
    ret = ivy.to_list(ivy.tensor(object_in, dtype_str, dev_str))
    # type test
    assert isinstance(ret, list)
    # cardinality test
    assert _get_shape_of_list(ret) == _get_shape_of_list(object_in)
    # value test
    assert np.allclose(np.asarray(ivy.to_list(ivy.tensor(object_in, dtype_str, dev_str))),
                       np.array(object_in).astype(dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support list conversion
        return
    helpers.assert_compilable(ivy.to_list)


# shape
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float32'])
@pytest.mark.parametrize(
    "as_tensor", [None, True, False])
def test_shape(object_in, dtype_str, as_tensor, dev_str, call):
    # smoke test
    ret = ivy.shape(ivy.tensor(object_in, dtype_str, dev_str), as_tensor)
    # type test
    if as_tensor:
        assert isinstance(ret, ivy.Tensor)
    else:
        assert isinstance(ret, tuple)
        ret = ivy.tensor(ret)
    # cardinality test
    assert ret.shape[0] == len(np.asarray(object_in).shape)
    # value test
    assert np.array_equal(ivy.to_numpy(ret), np.asarray(np.asarray(object_in).shape, np.int32))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support Union
        return
    helpers.assert_compilable(ivy.shape)


# get_num_dims
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float32'])
@pytest.mark.parametrize(
    "as_tensor", [None, True, False])
def test_get_num_dims(object_in, dtype_str, as_tensor, dev_str, call):
    # smoke test
    ret = ivy.get_num_dims(ivy.tensor(object_in, dtype_str, dev_str), as_tensor)
    # type test
    if as_tensor:
        assert isinstance(ret, ivy.Tensor)
    else:
        assert isinstance(ret, int)
        ret = ivy.tensor(ret)
    # cardinality test
    assert list(ret.shape) == []
    # value test
    assert np.array_equal(ivy.to_numpy(ret), np.asarray(len(np.asarray(object_in).shape), np.int32))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support Union
        return
    helpers.assert_compilable(ivy.shape)


# minimum
@pytest.mark.parametrize(
    "xy", [([0.7], [0.5]), ([0.7], 0.5), (0.5, [0.7]), ([[0.8, 1.2], [1.5, 0.2]], [0., 1.])])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float32'])
def test_minimum(xy, dtype_str, dev_str, call):
    # smoke test
    x = ivy.tensor(xy[0], dtype_str, dev_str)
    y = ivy.tensor(xy[1], dtype_str, dev_str)
    if call in [helpers.mx_call] and x.shape != y.shape:
        pytest.skip()
    ret = ivy.minimum(x, y)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    if len(x.shape) > len(y.shape):
        assert ret.shape == x.shape
    else:
        assert ret.shape == y.shape
    # value test
    assert np.array_equal(call(ivy.minimum, x, y), ivy.numpy.minimum(ivy.to_numpy(x), ivy.to_numpy(y)))
    # compilation test
    helpers.assert_compilable(ivy.minimum)


# maximum
@pytest.mark.parametrize(
    "xy", [([0.7], [0.5]), ([0.7], 0.5), (0.5, [0.7]), ([[0.8, 1.2], [1.5, 0.2]], [0., 1.])])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float32'])
def test_maximum(xy, dtype_str, dev_str, call):
    # smoke test
    x = ivy.tensor(xy[0], dtype_str, dev_str)
    y = ivy.tensor(xy[1], dtype_str, dev_str)
    if call in [helpers.mx_call] and x.shape != y.shape:
        pytest.skip()
    ret = ivy.maximum(x, y)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    if len(x.shape) > len(y.shape):
        assert ret.shape == x.shape
    else:
        assert ret.shape == y.shape
    # value test
    assert np.array_equal(call(ivy.maximum, x, y), ivy.numpy.maximum(ivy.to_numpy(x), ivy.to_numpy(y)))
    # compilation test
    helpers.assert_compilable(ivy.maximum)


def test_clip(dev_str, call):
    assert np.array_equal(call(ivy.clip, ivy.tensor([0.]), 0, 1), np.clip(np.array([0.]), 0, 1))
    assert np.array_equal(call(ivy.clip, ivy.tensor([[0.]]), 0, 1), np.clip(np.array([[0.]]), 0, 1))
    helpers.assert_compilable(ivy.clip)


def test_round(dev_str, call):
    assert np.array_equal(call(ivy.round, ivy.tensor([0.3])), np.round(np.array([0.3])))
    assert np.array_equal(call(ivy.round, ivy.tensor([[0.51]])), np.array([[1.]]))
    helpers.assert_compilable(ivy.round)


def test_floormod(dev_str, call):
    assert np.allclose(call(ivy.floormod, ivy.tensor([3.3]), ivy.tensor([3.])),
                       np.array([0.3]), atol=1e-6)
    assert np.allclose(call(ivy.floormod, ivy.tensor([[10.7]]), ivy.tensor([[5.]])),
                       np.array([[0.7]]), atol=1e-6)
    helpers.assert_compilable(ivy.floormod)


def test_floor(dev_str, call):
    assert np.array_equal(call(ivy.floor, ivy.tensor([0.3])), np.floor(np.array([0.3])))
    assert np.array_equal(call(ivy.floor, ivy.tensor([[0.7]])), np.floor(np.array([[0.7]])))
    helpers.assert_compilable(ivy.floor)


def test_ceil(dev_str, call):
    assert np.array_equal(call(ivy.ceil, ivy.tensor([0.3])), np.ceil(np.array([0.3])))
    assert np.array_equal(call(ivy.ceil, ivy.tensor([[0.7]])), np.ceil(np.array([[0.7]])))
    helpers.assert_compilable(ivy.ceil)


def test_abs(dev_str, call):
    assert np.allclose(call(ivy.abs, ivy.tensor([-0.3])), np.array([0.3]), atol=1e-6)
    assert np.allclose(call(ivy.abs, ivy.tensor([[-0.7]])), np.array([[0.7]]), atol=1e-6)
    helpers.assert_compilable(ivy.abs)


def test_argmax(dev_str, call):
    assert np.allclose(call(ivy.argmax, ivy.tensor([-0.3, 0.1])), np.array([1]), atol=1e-6)
    assert np.allclose(call(ivy.argmax, ivy.tensor([[1.3, -0.7], [0.1, 2.5]])),
                       np.array([0, 1]), atol=1e-6)
    helpers.assert_compilable(ivy.argmax)


def test_argmin(dev_str, call):
    assert np.allclose(call(ivy.argmin, ivy.tensor([-0.3, 0.1])), np.array([0]), atol=1e-6)
    assert np.allclose(call(ivy.argmin, ivy.tensor([[1.3, -0.7], [0.1, 2.5]])),
                       np.array([1, 0]), atol=1e-6)
    helpers.assert_compilable(ivy.argmin)


def test_cast(dev_str, call):
    assert np.array_equal(call(ivy.cast, ivy.tensor([0]), 'float32'),
                          np.array([0]).astype(np.float32))
    assert np.array_equal(call(ivy.cast, ivy.tensor([[0]]), 'float32'),
                          np.array([[0]]).astype(np.float32))
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    helpers.assert_compilable(ivy.cast)


def test_arange(dev_str, call):
    assert np.array_equal(call(ivy.arange, 10), np.arange(10))
    assert np.array_equal(call(ivy.arange, 10, 2), np.arange(2, 10))
    assert np.array_equal(call(ivy.arange, 10, 2, 2), np.arange(2, 10, 2))
    assert np.array_equal(call(ivy.arange, 10, 2, 2, 'float32'), np.arange(2, 10, 2, dtype=np.float32))
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    helpers.assert_compilable(ivy.arange)


def test_linspace(dev_str, call):
    assert np.allclose(call(ivy.linspace, 1, 10, 100), np.linspace(1, 10, 100), atol=1e-6)
    start = ivy.tensor([[0., 1., 2.]])
    stop = ivy.tensor([[1., 2., 3.]])
    assert np.allclose(call(ivy.linspace, start, stop, 100),
                       np.linspace(np.array([[0., 1., 2.]]), np.array([[1., 2., 3.]]), 100, axis=-1), atol=1e-6)
    start = ivy.tensor([[[-0.1471, 0.4477, 0.2214]]])
    stop = ivy.tensor([[[-0.3048, 0.3308, 0.2721]]])
    res = np.array([[[[-0.1471,  0.4477,  0.2214],
                      [-0.1786,  0.4243,  0.2316],
                      [-0.2102,  0.4009,  0.2417],
                      [-0.2417,  0.3776,  0.2518],
                      [-0.2732,  0.3542,  0.2620],
                      [-0.3048,  0.3308,  0.2721]]]])
    assert np.allclose(call(ivy.linspace, start, stop, 6, axis=-2), res, atol=1e-4)
    if call is helpers.torch_call:
        start = ivy.variable(start)
        stop = ivy.variable(stop)
        assert np.allclose(ivy.linspace(start, stop, 6, axis=-2).detach().numpy(), res, atol=1e-4)
    if call in [helpers.torch_call]:
        # pytorch scripting does not support numpy conversion
        return
    helpers.assert_compilable(ivy.linspace)


def test_concatenate(dev_str, call):
    assert np.array_equal(call(ivy.concatenate, (ivy.tensor([0.]), ivy.tensor([0.])), 0),
                          np.concatenate((np.array([0.]), np.array([0.])), 0))
    assert np.array_equal(call(ivy.concatenate,
                               (ivy.tensor([[0.]]), ivy.tensor([[0.]])), 0),
                          np.concatenate((np.array([[0.]]), np.array([[0.]])), 0))
    helpers.assert_compilable(ivy.concatenate)


def test_flip(dev_str, call):
    assert np.array_equal(call(ivy.flip, ivy.tensor([0., 1.]), batch_shape=[2]),
                          np.flip(np.array([0., 1.])))
    assert np.array_equal(call(ivy.flip, ivy.tensor([0., 1.]), -1, batch_shape=[2]),
                          np.flip(np.array([0., 1.])))
    assert np.array_equal(call(ivy.flip, ivy.tensor([[0., 1.]]), -1, batch_shape=[1, 2]),
                          np.flip(np.array([[0., 1.]])))
    helpers.assert_compilable(ivy.flip)


def test_stack(dev_str, call):
    assert np.array_equal(call(ivy.stack, [ivy.tensor([0.]), ivy.tensor([0.])], 0),
                          np.stack([np.array([0.]), np.array([0.])]))
    assert np.array_equal(call(ivy.stack, [ivy.tensor([[0.]]), ivy.tensor([[0.]])], 0),
                          np.stack([np.array([[0.]]), np.array([[0.]])]))
    helpers.assert_compilable(ivy.stack)


def test_unstack(dev_str, call):
    x = np.swapaxes(np.array([[0.]]), 0, 0)
    true = [np.array(item) for item in x.tolist()]
    pred = call(ivy.unstack, ivy.tensor([[0.]]), 0, num_outputs=1)
    assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
    x = np.swapaxes(np.array([[[0.]]]), 0, 0)
    true = [np.array(item) for item in x.tolist()]
    pred = call(ivy.unstack, ivy.tensor([[[0.]]]), 0, num_outputs=1)
    assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
    helpers.assert_compilable(ivy.unstack)


def test_split(dev_str, call):
    assert np.array_equal(call(ivy.split, ivy.tensor([[0., 1.]]), 2, -1),
                          np.split(np.array([[0., 1.]]), 2, -1))
    assert np.array_equal(call(ivy.split, ivy.tensor([[[0., 1.]]]), 2, -1),
                          np.split(np.array([[[0., 1.]]]), 2, -1))
    helpers.assert_compilable(ivy.split)


def test_tile(dev_str, call):
    assert np.array_equal(call(ivy.tile, ivy.tensor([[0.]]), [1, 2]),
                          np.tile(np.array([[0.]]), [1, 2]))
    assert np.array_equal(call(ivy.tile, ivy.tensor([[[0.]]]), [1, 2, 3]),
                          np.tile(np.array([[[0.]]]), [1, 2, 3]))
    helpers.assert_compilable(ivy.tile)


def test_zero_pad(dev_str, call):
    assert np.array_equal(call(ivy.zero_pad, ivy.tensor([[0.]]), [[0, 1], [1, 2]], x_shape=[1, 1]),
                          np.pad(np.array([[0.]]), [[0, 1], [1, 2]]))
    assert np.array_equal(call(ivy.zero_pad, ivy.tensor([[[0.]]]), [[0, 0], [1, 1], [2, 3]],
                               x_shape=[1, 1, 1]),
                          np.pad(np.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]]))
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    helpers.assert_compilable(ivy.zero_pad)


def test_constant_pad(dev_str, call):
    assert np.array_equal(call(ivy.constant_pad, ivy.tensor([[0.]]), [[0, 1], [1, 2]], 2.,
                               x_shape=[1, 1]), np.pad(np.array([[0.]]), [[0, 1], [1, 2]], constant_values=2.))
    assert np.array_equal(call(ivy.constant_pad, ivy.tensor([[[0.]]]), [[0, 0], [1, 1], [2, 3]],
                               3., x_shape=[1, 1, 1]),
                          np.pad(np.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]], constant_values=3.))
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    helpers.assert_compilable(ivy.constant_pad)


def test_swapaxes(dev_str, call):
    assert np.array_equal(call(ivy.swapaxes, ivy.tensor([[0., 0.]]), 1, 0),
                          np.swapaxes(np.array([[0., 0.]]), 1, 0))
    assert np.array_equal(call(ivy.swapaxes, ivy.tensor([[0., 0.]]), -1, -2),
                          np.swapaxes(np.array([[0., 0.]]), -1, -2))
    helpers.assert_compilable(ivy.swapaxes)


def test_transpose(dev_str, call):
    assert np.array_equal(call(ivy.transpose, ivy.tensor([[0., 0.]]), [1, 0]),
                          np.transpose(np.array([[0., 0.]]), [1, 0]))
    assert np.array_equal(call(ivy.transpose, ivy.tensor([[[0., 0.]]]), [2, 0, 1]),
                          np.transpose(np.array([[[0., 0.]]]), [2, 0, 1]))
    helpers.assert_compilable(ivy.transpose)


def test_expand_dims(dev_str, call):
    assert np.array_equal(call(ivy.expand_dims, ivy.tensor([[0., 0.]]), 0),
                          np.expand_dims(np.array([[0., 0.]]), 0))
    assert np.array_equal(call(ivy.expand_dims, ivy.tensor([[[0., 0.]]]), -1),
                          np.expand_dims(np.array([[[0., 0.]]]), -1))
    helpers.assert_compilable(ivy.expand_dims)


def test_where(dev_str, call):
    assert np.array_equal(call(ivy.where, ivy.tensor([[0., 1.]]) > 0,
                               ivy.tensor([[1., 1.]]), ivy.tensor([[2., 2.]]),
                               condition_shape=[1, 2], x_shape=[1, 2]),
                          np.where(np.array([[0., 1.]]) > 0, np.array([[0., 1.]]), np.array([[2., 2.]])))
    assert np.array_equal(call(ivy.where, ivy.tensor([[[1., 0.]]]) > 0,
                               ivy.tensor([[[1., 1.]]]), ivy.tensor([[[2., 2.]]]),
                               condition_shape=[1, 1, 2], x_shape=[1, 1, 2]),
                          np.where(np.array([[[1., 0.]]]) > 0, np.array([[[1., 1.]]]), np.array([[[2., 2.]]])))
    helpers.assert_compilable(ivy.where)


def test_indices_where(dev_str, call):
    assert np.array_equal(call(ivy.indices_where, ivy.tensor([[False, True],
                                                              [True, False],
                                                              [True, True]])),
                          np.array([[0, 1], [1, 0], [2, 0], [2, 1]]))
    assert np.array_equal(call(ivy.indices_where, ivy.tensor([[[False, True],
                                                               [True, False],
                                                               [True, True]]])),
                          np.array([[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 2, 1]]))
    helpers.assert_compilable(ivy.indices_where)


def test_reshape(dev_str, call):
    assert np.array_equal(call(ivy.reshape, ivy.tensor([[0., 1.]]), (-1,)),
                          np.reshape(np.array([[0., 1.]]), (-1,)))
    assert np.array_equal(call(ivy.reshape, ivy.tensor([[[1., 0.]]]), (1, 2)),
                          np.reshape(np.array([[[1., 0.]]]), (1, 2)))
    helpers.assert_compilable(ivy.reshape)


def test_squeeze(dev_str, call):
    assert np.array_equal(call(ivy.squeeze, ivy.tensor([[0., 1.]])),
                          np.squeeze(np.array([[0., 1.]])))
    assert np.array_equal(call(ivy.squeeze, ivy.tensor([[[1., 0.]]]), 1),
                          np.squeeze(np.array([[[1., 0.]]]), 1))
    helpers.assert_compilable(ivy.squeeze)


def test_zeros(dev_str, call):
    assert np.array_equal(call(ivy.zeros, (1, 2)), np.zeros((1, 2)))
    assert np.array_equal(call(ivy.zeros, (1, 2), 'int64'), np.zeros((1, 2), np.int64))
    assert np.array_equal(call(ivy.zeros, (1, 2, 3)), np.zeros((1, 2, 3)))
    helpers.assert_compilable(ivy.zeros)


def test_zeros_like(dev_str, call):
    assert np.array_equal(call(ivy.zeros_like, ivy.tensor([[0., 1.]])),
                          np.zeros_like(np.array([[0., 1.]])))
    assert np.array_equal(call(ivy.zeros_like, ivy.tensor([[[1., 0.]]])),
                          np.zeros_like(np.array([[[1., 0.]]])))
    helpers.assert_compilable(ivy.zeros_like)


def test_ones(dev_str, call):
    assert np.array_equal(call(ivy.ones, (1, 2)), np.ones((1, 2)))
    assert np.array_equal(call(ivy.ones, (1, 2), 'int64'), np.ones((1, 2), np.int64))
    assert np.array_equal(call(ivy.ones, (1, 2, 3)), np.ones((1, 2, 3)))
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return
    helpers.assert_compilable(ivy.ones)


def test_ones_like(dev_str, call):
    assert np.array_equal(call(ivy.ones_like, ivy.tensor([[0., 1.]])),
                          np.ones_like(np.array([[0., 1.]])))
    assert np.array_equal(call(ivy.ones_like, ivy.tensor([[[1., 0.]]])),
                          np.ones_like(np.array([[[1., 0.]]])))
    helpers.assert_compilable(ivy.ones_like)


def test_one_hot(dev_str, call):
    np_one_hot = helpers._ivy_np.one_hot(np.array([0, 1, 2]), 3)
    assert np.array_equal(call(ivy.one_hot, ivy.tensor([0, 1, 2]), 3), np_one_hot)
    helpers.assert_compilable(ivy.one_hot)


def test_cross(dev_str, call):
    assert np.array_equal(call(ivy.cross, ivy.tensor([0., 0., 0.]),
                               ivy.tensor([0., 0., 0.])),
                          np.cross(np.array([0., 0., 0.]), np.array([0., 0., 0.])))
    assert np.array_equal(call(ivy.cross, ivy.tensor([[0., 0., 0.]]),
                               ivy.tensor([[0., 0., 0.]])),
                          np.cross(np.array([[0., 0., 0.]]), np.array([[0., 0., 0.]])))
    helpers.assert_compilable(ivy.cross)


def test_matmul(dev_str, call):
    assert np.array_equal(call(ivy.matmul, ivy.tensor([[1., 0.], [0., 1.]]),
                               ivy.tensor([[1., 0.], [0., 1.]]), batch_shape=[]),
                          np.matmul(np.array([[1., 0.], [0., 1.]]), np.array([[1., 0.], [0., 1.]])))
    assert np.array_equal(call(ivy.matmul, ivy.tensor([[[[1., 0.], [0., 1.]]]]),
                               ivy.tensor([[[[1., 0.], [0., 1.]]]]), batch_shape=[1, 1]),
                          np.matmul(np.array([[[[1., 0.], [0., 1.]]]]), np.array([[[[1., 0.], [0., 1.]]]])))
    helpers.assert_compilable(ivy.matmul)


def test_cumsum(dev_str, call):
    assert np.array_equal(call(ivy.cumsum, ivy.tensor([[0., 1., 2., 3.]]), 1),
                          np.array([[0., 1., 3., 6.]]))
    assert np.array_equal(call(ivy.cumsum, ivy.tensor([[0., 1., 2.], [0., 1., 2.]]), 0),
                          np.array([[0., 1., 2.], [0., 2., 4.]]))
    helpers.assert_compilable(ivy.cumsum)


def test_identity(dev_str, call):
    assert np.array_equal(call(ivy.identity, 1), np.identity(1))
    assert np.array_equal(call(ivy.identity, 2, 'int64'), np.identity(2, np.int64))
    call(ivy.identity, 2, 'int64', (1, 2))
    helpers.assert_compilable(ivy.identity)


def test_scatter_flat_sum(dev_str, call):
    assert np.array_equal(call(ivy.scatter_flat, ivy.tensor([0, 4, 1, 2]),
                               ivy.tensor([1, 2, 3, 4]), 8),
                          np.array([1, 3, 4, 0, 2, 0, 0, 0]))
    if call in [helpers.mx_call]:
        # mxnet scatter does not support sum reduction
        return
    assert np.array_equal(call(ivy.scatter_flat, ivy.tensor([0, 4, 1, 2, 0]),
                               ivy.tensor([1, 2, 3, 4, 5]), 8),
                          np.array([6, 3, 4, 0, 2, 0, 0, 0]))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_flat)


def test_scatter_flat_min(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support max reduction for scatter flat
        pytest.skip()
    assert np.array_equal(call(ivy.scatter_flat, ivy.tensor([0, 4, 1, 2]),
                               ivy.tensor([1, 2, 3, 4]), 8, 'min'),
                          np.array([1, 3, 4, 0, 2, 0, 0, 0]))
    assert np.array_equal(call(ivy.scatter_flat, ivy.tensor([0, 4, 1, 2, 0]),
                               ivy.tensor([1, 2, 3, 4, 5]), 8, 'min'),
                          np.array([1, 3, 4, 0, 2, 0, 0, 0]))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_flat)


def test_scatter_flat_max(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support max reduction for scatter flat
        pytest.skip()
    assert np.array_equal(call(ivy.scatter_flat, ivy.tensor([0, 4, 1, 2]),
                               ivy.tensor([1, 2, 3, 4]), 8, 'max'),
                          np.array([1, 3, 4, 0, 2, 0, 0, 0]))
    assert np.array_equal(call(ivy.scatter_flat, ivy.tensor([0, 4, 1, 2, 0]),
                               ivy.tensor([1, 2, 3, 4, 5]), 8, 'max'),
                          np.array([5, 3, 4, 0, 2, 0, 0, 0]))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_flat)


def test_scatter_sum_nd(dev_str, call):
    assert np.array_equal(call(ivy.scatter_nd, ivy.tensor([[4], [3], [1], [7]]),
                               ivy.tensor([9, 10, 11, 12]), [8], 2),
                          np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]),
                                             [8]))
    assert np.array_equal(call(ivy.scatter_nd, ivy.tensor([[0, 1, 2]]),
                               ivy.tensor([1]), [3, 3, 3], 2),
                          np_scatter(np.array([[0, 1, 2]]), np.array([1]),
                                             [3, 3, 3]))
    assert np.array_equal(call(ivy.scatter_nd, ivy.tensor([[0], [2]]),
                               ivy.tensor([[[5, 5, 5, 5], [6, 6, 6, 6],
                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                               [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 2),
                          np_scatter(np.array([[0], [2]]),
                                             np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]],
                                                       [[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4]))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_nd)


def test_scatter_min_nd(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support min reduction for scatter nd
        pytest.skip()
    assert np.array_equal(call(ivy.scatter_nd, ivy.tensor([[4], [3], [1], [7]]),
                               ivy.tensor([9, 10, 11, 12]), [8], 'min'),
                          np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'min'))
    assert np.array_equal(call(ivy.scatter_nd, ivy.tensor([[0, 1, 2]]),
                               ivy.tensor([1]), [3, 3, 3], 'min'),
                          np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'min'))
    assert np.array_equal(call(ivy.scatter_nd, ivy.tensor([[0], [2]]),
                               ivy.tensor([[[5, 5, 5, 5], [6, 6, 6, 6],
                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                               [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'min'),
                          np_scatter(np.array([[0], [2]]),
                                             np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]],
                                                       [[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'min'))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_nd)


def test_scatter_max_nd(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support max reduction for scatter nd
        pytest.skip()
    assert np.array_equal(call(ivy.scatter_nd, ivy.tensor([[4], [3], [1], [7]]),
                               ivy.tensor([9, 10, 11, 12]), [8], 'max'),
                          np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'max'))
    assert np.array_equal(call(ivy.scatter_nd, ivy.tensor([[0, 1, 2]]),
                               ivy.tensor([1]), [3, 3, 3], 'max'),
                          np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'max'))
    assert np.array_equal(call(ivy.scatter_nd, ivy.tensor([[0], [2]]),
                               ivy.tensor([[[5, 5, 5, 5], [6, 6, 6, 6],
                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                               [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'max'),
                          np_scatter(np.array([[0], [2]]),
                                             np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]],
                                                       [[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'max'))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_nd)


def test_gather_flat(dev_str, call):
    assert np.allclose(call(ivy.gather_flat, ivy.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
                            ivy.tensor([0, 4, 7])), np.array([9, 5, 2]), atol=1e-6)
    helpers.assert_compilable(ivy.gather_flat)


def test_gather_nd(dev_str, call):
    assert np.allclose(call(ivy.gather_nd, ivy.tensor([[[0.0, 1.0], [2.0, 3.0]],
                                                       [[0.1, 1.1], [2.1, 3.1]]]),
                            ivy.tensor([[0, 1], [1, 0]]), indices_shape=[2, 2]),
                       np.array([[2.0, 3.0], [0.1, 1.1]]), atol=1e-6)
    assert np.allclose(call(ivy.gather_nd, ivy.tensor([[[0.0, 1.0], [2.0, 3.0]],
                                                       [[0.1, 1.1], [2.1, 3.1]]]),
                            ivy.tensor([[[0, 1]], [[1, 0]]]), indices_shape=[2, 1, 2]),
                       np.array([[[2.0, 3.0]], [[0.1, 1.1]]]), atol=1e-6)
    assert np.allclose(call(ivy.gather_nd, ivy.tensor([[[0.0, 1.0], [2.0, 3.0]],
                                                       [[0.1, 1.1], [2.1, 3.1]]]),
                            ivy.tensor([[[0, 1, 0]], [[1, 0, 1]]]),
                            indices_shape=[2, 1, 3]), np.array([[2.0], [1.1]]), atol=1e-6)
    if call in [helpers.torch_call]:
        # torch scripting does not support builtins
        return
    helpers.assert_compilable(ivy.gather_nd)


def test_dev(dev_str, call):
    assert ivy.dev(ivy.tensor([1.]))
    helpers.assert_compilable(ivy.dev)


def test_dev_to_str(dev_str, call):
    assert 'cpu' in ivy.dev_to_str(ivy.dev(ivy.tensor([0.]))).lower()
    helpers.assert_compilable(ivy.dev_to_str)


def test_dev_str(dev_str, call):
    assert 'cpu' in ivy.dev_str(ivy.tensor([0.])).lower()
    helpers.assert_compilable(ivy.dev_str)


def test_dtype(dev_str, call):
    assert ivy.dtype(ivy.tensor([0.])) == ivy.tensor([0.]).dtype
    helpers.assert_compilable(ivy.dtype)


def test_dtype_to_str(dev_str, call):
    assert ivy.dtype_to_str(ivy.tensor([0.], dtype_str='float32').dtype) == 'float32'
    helpers.assert_compilable(ivy.dtype_to_str)


def test_dtype_str(dev_str, call):
    assert ivy.dtype_str(ivy.tensor([0.], dtype_str='float32')) == 'float32'
    helpers.assert_compilable(ivy.dtype_str)


def test_compile_fn(dev_str, call):
    some_fn = lambda x: x**2
    example_inputs = ivy.tensor([2.])
    new_fn = ivy.compile_fn(some_fn, False, example_inputs)
    assert np.allclose(call(new_fn, example_inputs), np.array([4.]))
