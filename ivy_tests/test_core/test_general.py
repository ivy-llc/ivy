"""
Collection of tests for templated general functions
"""

# global
import numpy as np
from functools import reduce as _reduce
from operator import mul as _mul

# local
import ivy_tests.helpers as helpers
import ivy.core.general as ivy_gen
import ivy.core.gradients as ivy_grad


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


def test_array():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.array, [0.], f=lib), np.array([0.]))
        assert np.array_equal(call(ivy_gen.array, [0.], 'float32', f=lib), np.array([0.], dtype=np.float32))
        assert np.array_equal(call(ivy_gen.array, [[0.]], f=lib), np.array([[0.]]))


def test_to_numpy():
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # to_numpy() requires eager execution
            continue
        assert call(ivy_gen.to_numpy, ivy_gen.array([0.], f=lib), f=lib) == np.array([0.])
        assert call(ivy_gen.to_numpy, ivy_gen.array([0.], 'float32', f=lib), f=lib) == np.array([0.])
        assert call(ivy_gen.to_numpy, ivy_gen.array([[0.]], f=lib), f=lib) == np.array([[0.]])


def test_to_list():
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # to_list() requires eager execution
            continue
        assert call(ivy_gen.to_list, ivy_gen.array([0.], f=lib), f=lib) == [0.]
        assert call(ivy_gen.to_list, ivy_gen.array([0.], 'float32', f=lib), f=lib) == [0.]
        assert call(ivy_gen.to_list, ivy_gen.array([[0.]], f=lib), f=lib) == [[0.]]


def test_shape():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.shape, ivy_gen.array([0.], f=lib), f=lib), np.array([1]))
        assert np.array_equal(call(ivy_gen.shape, ivy_gen.array([[0.]], f=lib), f=lib), np.array([1, 1]))


def test_get_num_dims():
    for lib, call in helpers.calls:
        assert call(ivy_gen.get_num_dims, ivy_gen.array([0.], f=lib), f=lib) == np.array([1])
        assert call(ivy_gen.get_num_dims, ivy_gen.array([[0.]], f=lib), f=lib) == np.array([2])


def test_minimum():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_gen.minimum, ivy_gen.array([0.7], f=lib), 0.5), np.minimum(np.array([0.7]), 0.5))
        if call is helpers.mx_graph_call:
            # mxnet symbolic minimum does not support varying array shapes
            continue
        assert np.allclose(call(ivy_gen.minimum, ivy_gen.array([[0.8, 1.2], [1.5, 0.2]], f=lib),
                                ivy_gen.array([0., 1.], f=lib)),
                           np.minimum(np.array([[0.8, 1.2], [1.5, 0.2]]), np.array([0., 1.])))


def test_maximum():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_gen.maximum, ivy_gen.array([0.7], f=lib), 0.5), np.maximum(np.array([0.7]), 0.5))
        if call is helpers.mx_graph_call:
            # mxnet symbolic maximum does not support varying array shapes
            continue
        assert np.allclose(call(ivy_gen.maximum, ivy_gen.array([[0.8, 1.2], [1.5, 0.2]], f=lib),
                                ivy_gen.array([0., 1.], f=lib)),
                           np.maximum(np.array([[0.8, 1.2], [1.5, 0.2]]), np.array([0., 1.])))


def test_clip():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.clip, ivy_gen.array([0.], f=lib), 0, 1), np.clip(np.array([0.]), 0, 1))
        assert np.array_equal(call(ivy_gen.clip, ivy_gen.array([[0.]], f=lib), 0, 1), np.clip(np.array([[0.]]), 0, 1))


def test_round():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.round, ivy_gen.array([0.3], f=lib)), np.round(np.array([0.3])))
        assert np.array_equal(call(ivy_gen.round, ivy_gen.array([[0.51]], f=lib)), np.array([[1.]]))


def test_floormod():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_gen.floormod, ivy_gen.array([3.3], f=lib), ivy_gen.array([3.], f=lib)),
                           np.array([0.3]), atol=1e-6)
        assert np.allclose(call(ivy_gen.floormod, ivy_gen.array([[10.7]], f=lib), ivy_gen.array([[5.]], f=lib)),
                           np.array([[0.7]]), atol=1e-6)


def test_floor():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.floor, ivy_gen.array([0.3], f=lib)), np.floor(np.array([0.3])))
        assert np.array_equal(call(ivy_gen.floor, ivy_gen.array([[0.7]], f=lib)), np.floor(np.array([[0.7]])))


def test_ceil():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.ceil, ivy_gen.array([0.3], f=lib)), np.ceil(np.array([0.3])))
        assert np.array_equal(call(ivy_gen.ceil, ivy_gen.array([[0.7]], f=lib)), np.ceil(np.array([[0.7]])))


def test_abs():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_gen.abs, ivy_gen.array([-0.3], f=lib)), np.array([0.3]), atol=1e-6)
        assert np.allclose(call(ivy_gen.abs, ivy_gen.array([[-0.7]], f=lib)), np.array([[0.7]]), atol=1e-6)


def test_argmax():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_gen.argmax, ivy_gen.array([-0.3, 0.1], f=lib)), np.array([1]), atol=1e-6)
        assert np.allclose(call(ivy_gen.argmax, ivy_gen.array([[1.3, -0.7], [0.1, 2.5]], f=lib)),
                           np.array([0, 1]), atol=1e-6)


def test_argmin():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_gen.argmin, ivy_gen.array([-0.3, 0.1], f=lib)), np.array([0]), atol=1e-6)
        assert np.allclose(call(ivy_gen.argmin, ivy_gen.array([[1.3, -0.7], [0.1, 2.5]], f=lib)),
                           np.array([1, 0]), atol=1e-6)


def test_cast():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.cast, ivy_gen.array([0], f=lib), 'float32'),
                              np.array([0]).astype(np.float32))
        assert np.array_equal(call(ivy_gen.cast, ivy_gen.array([[0]], f=lib), 'float32'),
                              np.array([[0]]).astype(np.float32))


def test_arange():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.arange, 10, f=lib), np.arange(10))
        assert np.array_equal(call(ivy_gen.arange, 10, 2, f=lib), np.arange(2, 10))
        assert np.array_equal(call(ivy_gen.arange, 10, 2, 2, f=lib), np.arange(2, 10, 2))
        assert np.array_equal(call(ivy_gen.arange, 10, 2, 2, 'float32', f=lib), np.arange(2, 10, 2, dtype=np.float32))


def test_linspace():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support linspace
            continue
        assert np.allclose(call(ivy_gen.linspace, 1, 10, 100, f=lib), np.linspace(1, 10, 100), atol=1e-6)
        start = ivy_gen.array([[0., 1., 2.]], f=lib)
        stop = ivy_gen.array([[1., 2., 3.]], f=lib)
        assert np.allclose(call(ivy_gen.linspace, start, stop, 100, f=lib),
                           np.linspace(np.array([[0., 1., 2.]]), np.array([[1., 2., 3.]]), 100, axis=-1), atol=1e-6)
        start = ivy_gen.array([[[-0.1471,  0.4477,  0.2214]]], f=lib)
        stop = ivy_gen.array([[[-0.3048,  0.3308,  0.2721]]], f=lib)
        res = np.array([[[[-0.1471,  0.4477,  0.2214],
                          [-0.1786,  0.4243,  0.2316],
                          [-0.2102,  0.4009,  0.2417],
                          [-0.2417,  0.3776,  0.2518],
                          [-0.2732,  0.3542,  0.2620],
                          [-0.3048,  0.3308,  0.2721]]]])
        assert np.allclose(call(ivy_gen.linspace, start, stop, 6, axis=-2, f=lib), res, atol=1e-4)
        if call is helpers.torch_call:
            start = ivy_grad.variable(start)
            stop = ivy_grad.variable(stop)
            res = ivy_grad.variable(res)
            assert np.allclose(ivy_gen.linspace(start, stop, 6, axis=-2, f=lib).detach().numpy(), res, atol=1e-4)


def test_concatenate():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.concatenate, (ivy_gen.array([0.], f=lib), ivy_gen.array([0.], f=lib)), 0),
                              np.concatenate((np.array([0.]), np.array([0.])), 0))
        assert np.array_equal(call(ivy_gen.concatenate,
                                   (ivy_gen.array([[0.]], f=lib), ivy_gen.array([[0.]], f=lib)), 0),
                              np.concatenate((np.array([[0.]]), np.array([[0.]])), 0))


def test_flip():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.flip, ivy_gen.array([0., 1.], f=lib), batch_shape=[2]),
                              np.flip(np.array([0., 1.])))
        assert np.array_equal(call(ivy_gen.flip, ivy_gen.array([0., 1.], f=lib), -1, batch_shape=[2]),
                              np.flip(np.array([0., 1.])))
        assert np.array_equal(call(ivy_gen.flip, ivy_gen.array([[0., 1.]], f=lib), -1, batch_shape=[1, 2]),
                              np.flip(np.array([[0., 1.]])))


def test_stack():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.stack, [ivy_gen.array([0.], f=lib), ivy_gen.array([0.], f=lib)], 0),
                              np.stack([np.array([0.]), np.array([0.])]))
        assert np.array_equal(call(ivy_gen.stack, [ivy_gen.array([[0.]], f=lib), ivy_gen.array([[0.]], f=lib)], 0),
                              np.stack([np.array([[0.]]), np.array([[0.]])]))


def test_unstack():
    for lib, call in helpers.calls:
        x = np.swapaxes(np.array([[0.]]), 0, 0)
        true = [np.array(item) for item in x.tolist()]
        pred = call(ivy_gen.unstack, ivy_gen.array([[0.]], f=lib), 0, num_outputs=1)
        assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
        x = np.swapaxes(np.array([[[0.]]]), 0, 0)
        true = [np.array(item) for item in x.tolist()]
        pred = call(ivy_gen.unstack, ivy_gen.array([[[0.]]], f=lib), 0, num_outputs=1)
        assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1


def test_split():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.split, ivy_gen.array([[0., 1.]], f=lib), 2, -1),
                              np.split(np.array([[0., 1.]]), 2, -1))
        assert np.array_equal(call(ivy_gen.split, ivy_gen.array([[[0., 1.]]], f=lib), 2, -1),
                              np.split(np.array([[[0., 1.]]]), 2, -1))


def test_tile():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.tile, ivy_gen.array([[0.]], f=lib), [1, 2]),
                              np.tile(np.array([[0.]]), [1, 2]))
        assert np.array_equal(call(ivy_gen.tile, ivy_gen.array([[[0.]]], f=lib), [1, 2, 3]),
                              np.tile(np.array([[[0.]]]), [1, 2, 3]))


def test_zero_pad():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.zero_pad, ivy_gen.array([[0.]], f=lib), [[0, 1], [1, 2]], x_shape=[1, 1]),
                              np.pad(np.array([[0.]]), [[0, 1], [1, 2]]))
        assert np.array_equal(call(ivy_gen.zero_pad, ivy_gen.array([[[0.]]], f=lib), [[0, 0], [1, 1], [2, 3]],
                                   x_shape=[1, 1, 1]),
                              np.pad(np.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]]))


def test_swapaxes():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.swapaxes, ivy_gen.array([[0., 0.]], f=lib), 1, 0),
                              np.swapaxes(np.array([[0., 0.]]), 1, 0))
        assert np.array_equal(call(ivy_gen.swapaxes, ivy_gen.array([[0., 0.]], f=lib), -1, -2),
                              np.swapaxes(np.array([[0., 0.]]), -1, -2))


def test_transpose():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.transpose, ivy_gen.array([[0., 0.]], f=lib), [1, 0]),
                              np.transpose(np.array([[0., 0.]]), [1, 0]))
        assert np.array_equal(call(ivy_gen.transpose, ivy_gen.array([[[0., 0.]]], f=lib), [2, 0, 1]),
                              np.transpose(np.array([[[0., 0.]]]), [2, 0, 1]))


def test_expand_dims():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.expand_dims, ivy_gen.array([[0., 0.]], f=lib), 0),
                              np.expand_dims(np.array([[0., 0.]]), 0))
        assert np.array_equal(call(ivy_gen.expand_dims, ivy_gen.array([[[0., 0.]]], f=lib), -1),
                              np.expand_dims(np.array([[[0., 0.]]]), -1))


def test_where():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.where, ivy_gen.array([[0., 1.]], f=lib) > 0,
                                   ivy_gen.array([[1., 1.]], f=lib), ivy_gen.array([[2., 2.]], f=lib),
                                   condition_shape=[1, 2], x_shape=[1, 2]),
                              np.where(np.array([[0., 1.]]) > 0, np.array([[0., 1.]]), np.array([[2., 2.]])))
        assert np.array_equal(call(ivy_gen.where, ivy_gen.array([[[1., 0.]]], f=lib) > 0,
                                   ivy_gen.array([[[1., 1.]]], f=lib), ivy_gen.array([[[2., 2.]]], f=lib),
                                   condition_shape=[1, 1, 2], x_shape=[1, 1, 2]),
                              np.where(np.array([[[1., 0.]]]) > 0, np.array([[[1., 1.]]]), np.array([[[2., 2.]]])))


def test_indices_where():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support indices_where
            continue
        assert np.array_equal(call(ivy_gen.indices_where, ivy_gen.array([[False, True],
                                                                         [True, False],
                                                                         [True, True]], f=lib)),
                              np.array([[0, 1], [1, 0], [2, 0], [2, 1]]))
        assert np.array_equal(call(ivy_gen.indices_where, ivy_gen.array([[[False, True],
                                                                          [True, False],
                                                                          [True, True]]], f=lib)),
                              np.array([[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 2, 1]]))


def test_reshape():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.reshape, ivy_gen.array([[0., 1.]], f=lib), (-1,), f=lib),
                              np.reshape(np.array([[0., 1.]]), (-1,)))
        assert np.array_equal(call(ivy_gen.reshape, ivy_gen.array([[[1., 0.]]], f=lib), (1, 2), f=lib),
                              np.reshape(np.array([[[1., 0.]]]), (1, 2)))


def test_squeeze():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.squeeze, ivy_gen.array([[0., 1.]], f=lib), f=lib),
                              np.squeeze(np.array([[0., 1.]])))
        assert np.array_equal(call(ivy_gen.squeeze, ivy_gen.array([[[1., 0.]]], f=lib), 1, f=lib),
                              np.squeeze(np.array([[[1., 0.]]]), 1))


def test_zeros():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.zeros, (1, 2), f=lib), np.zeros((1, 2)))
        assert np.array_equal(call(ivy_gen.zeros, (1, 2), 'int64', f=lib), np.zeros((1, 2), np.int64))
        assert np.array_equal(call(ivy_gen.zeros, (1, 2, 3), f=lib), np.zeros((1, 2, 3)))


def test_zeros_like():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.zeros_like, ivy_gen.array([[0., 1.]], f=lib), f=lib),
                              np.zeros_like(np.array([[0., 1.]])))
        assert np.array_equal(call(ivy_gen.zeros_like, ivy_gen.array([[[1., 0.]]], f=lib), f=lib),
                              np.zeros_like(np.array([[[1., 0.]]])))


def test_ones():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.ones, (1, 2), f=lib), np.ones((1, 2)))
        assert np.array_equal(call(ivy_gen.ones, (1, 2), 'int64', f=lib), np.ones((1, 2), np.int64))
        assert np.array_equal(call(ivy_gen.ones, (1, 2, 3), f=lib), np.ones((1, 2, 3)))


def test_ones_like():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.ones_like, ivy_gen.array([[0., 1.]], f=lib), f=lib),
                              np.ones_like(np.array([[0., 1.]])))
        assert np.array_equal(call(ivy_gen.ones_like, ivy_gen.array([[[1., 0.]]], f=lib), f=lib),
                              np.ones_like(np.array([[[1., 0.]]])))


def test_one_hot():
    np_one_hot = ivy_gen.one_hot(np.array([0, 1, 2]), 3, f=helpers._ivy_np)
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.one_hot, ivy_gen.array([0, 1, 2], f=lib), 3, f=lib), np_one_hot)


def test_cross():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.array_equal(call(ivy_gen.cross, ivy_gen.array([0., 0., 0.], f=lib),
                                   ivy_gen.array([0., 0., 0.], f=lib)),
                              np.cross(np.array([0., 0., 0.]), np.array([0., 0., 0.])))
        assert np.array_equal(call(ivy_gen.cross, ivy_gen.array([[0., 0., 0.]], f=lib),
                                   ivy_gen.array([[0., 0., 0.]], f=lib)),
                              np.cross(np.array([[0., 0., 0.]]), np.array([[0., 0., 0.]])))


def test_matmul():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.matmul, ivy_gen.array([[1., 0.], [0., 1.]], f=lib),
                                   ivy_gen.array([[1., 0.], [0., 1.]], f=lib), batch_shape=[]),
                              np.matmul(np.array([[1., 0.], [0., 1.]]), np.array([[1., 0.], [0., 1.]])))
        assert np.array_equal(call(ivy_gen.matmul, ivy_gen.array([[[[1., 0.], [0., 1.]]]], f=lib),
                                   ivy_gen.array([[[[1., 0.], [0., 1.]]]], f=lib), batch_shape=[1, 1]),
                              np.matmul(np.array([[[[1., 0.], [0., 1.]]]]), np.array([[[[1., 0.], [0., 1.]]]])))


def test_cumsum():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.cumsum, ivy_gen.array([[0., 1., 2., 3.]], f=lib), 1, f=lib),
                              np.array([[0., 1., 3., 6.]]))
        assert np.array_equal(call(ivy_gen.cumsum, ivy_gen.array([[0., 1., 2.], [0., 1., 2.]], f=lib), 0, f=lib),
                              np.array([[0., 1., 2.], [0., 2., 4.]]))


def test_identity():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.identity, 1, f=lib), np.identity(1))
        assert np.array_equal(call(ivy_gen.identity, 2, 'int64', f=lib), np.identity(2, np.int64))
        call(ivy_gen.identity, 2, 'int64', (1, 2), f=lib)


def test_scatter_flat_sum():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.scatter_flat, ivy_gen.array([0, 4, 1, 2], f=lib),
                                   ivy_gen.array([1, 2, 3, 4], f=lib), 8, f=lib),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet scatter does not support sum reduction
            continue
        assert np.array_equal(call(ivy_gen.scatter_flat, ivy_gen.array([0, 4, 1, 2, 0], f=lib),
                                   ivy_gen.array([1, 2, 3, 4, 5], f=lib), 8, f=lib),
                              np.array([6, 3, 4, 0, 2, 0, 0, 0]))


def test_scatter_flat_min():
    for lib, call in helpers.calls:
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support max reduction for scatter flat
            continue
        assert np.array_equal(call(ivy_gen.scatter_flat, ivy_gen.array([0, 4, 1, 2], f=lib),
                                   ivy_gen.array([1, 2, 3, 4], f=lib), 8, 'min', f=lib),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        assert np.array_equal(call(ivy_gen.scatter_flat, ivy_gen.array([0, 4, 1, 2, 0], f=lib),
                                   ivy_gen.array([1, 2, 3, 4, 5], f=lib), 8, 'min', f=lib),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))


def test_scatter_flat_max():
    for lib, call in helpers.calls:
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support max reduction for scatter flat
            continue
        assert np.array_equal(call(ivy_gen.scatter_flat, ivy_gen.array([0, 4, 1, 2], f=lib),
                                   ivy_gen.array([1, 2, 3, 4], f=lib), 8, 'max', f=lib),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        assert np.array_equal(call(ivy_gen.scatter_flat, ivy_gen.array([0, 4, 1, 2, 0], f=lib),
                                   ivy_gen.array([1, 2, 3, 4, 5], f=lib), 8, 'max', f=lib),
                              np.array([5, 3, 4, 0, 2, 0, 0, 0]))


def test_scatter_sum_nd():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.scatter_nd, ivy_gen.array([[4], [3], [1], [7]], f=lib),
                                   ivy_gen.array([9, 10, 11, 12], f=lib), [8], 2, f=lib),
                              np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]),
                                                 [8]))
        assert np.array_equal(call(ivy_gen.scatter_nd, ivy_gen.array([[0, 1, 2]], f=lib),
                                   ivy_gen.array([1], f=lib), [3, 3, 3], 2, f=lib),
                              np_scatter(np.array([[0, 1, 2]]), np.array([1]),
                                                 [3, 3, 3]))
        assert np.array_equal(call(ivy_gen.scatter_nd, ivy_gen.array([[0], [2]], f=lib),
                                   ivy_gen.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]],
                                                  [[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]]], f=lib), [4, 4, 4], 2, f=lib),
                              np_scatter(np.array([[0], [2]]),
                                                 np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4]))


def test_scatter_min_nd():
    for lib, call in helpers.calls:
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support min reduction for scatter nd
            continue
        assert np.array_equal(call(ivy_gen.scatter_nd, ivy_gen.array([[4], [3], [1], [7]], f=lib),
                                   ivy_gen.array([9, 10, 11, 12], f=lib), [8], 'min', f=lib),
                              np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'min'))
        assert np.array_equal(call(ivy_gen.scatter_nd, ivy_gen.array([[0, 1, 2]], f=lib),
                                   ivy_gen.array([1], f=lib), [3, 3, 3], 'min', f=lib),
                              np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'min'))
        assert np.array_equal(call(ivy_gen.scatter_nd, ivy_gen.array([[0], [2]], f=lib),
                                   ivy_gen.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]],
                                                  [[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]]], f=lib), [4, 4, 4], 'min', f=lib),
                              np_scatter(np.array([[0], [2]]),
                                                 np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'min'))


def test_scatter_max_nd():
    for lib, call in helpers.calls:
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support max reduction for scatter nd
            continue
        assert np.array_equal(call(ivy_gen.scatter_nd, ivy_gen.array([[4], [3], [1], [7]], f=lib),
                                   ivy_gen.array([9, 10, 11, 12], f=lib), [8], 'max', f=lib),
                              np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'max'))
        assert np.array_equal(call(ivy_gen.scatter_nd, ivy_gen.array([[0, 1, 2]], f=lib),
                                   ivy_gen.array([1], f=lib), [3, 3, 3], 'max', f=lib),
                              np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'max'))
        assert np.array_equal(call(ivy_gen.scatter_nd, ivy_gen.array([[0], [2]], f=lib),
                                   ivy_gen.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]],
                                                  [[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]]], f=lib), [4, 4, 4], 'max', f=lib),
                              np_scatter(np.array([[0], [2]]),
                                                 np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'max'))


def test_gather_flat():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_gen.gather_flat, ivy_gen.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], f=lib),
                                ivy_gen.array([0, 4, 7], f=lib), f=lib), np.array([9, 5, 2]), atol=1e-6)


def test_gather_nd():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_gen.gather_nd, ivy_gen.array([[[0.0, 1.0], [2.0, 3.0]],
                                                                  [[0.1, 1.1], [2.1, 3.1]]], f=lib),
                                ivy_gen.array([[0, 1], [1, 0]], f=lib), indices_shape=[2, 2], f=lib),
                           np.array([[2.0, 3.0], [0.1, 1.1]]), atol=1e-6)
        assert np.allclose(call(ivy_gen.gather_nd, ivy_gen.array([[[0.0, 1.0], [2.0, 3.0]],
                                                                  [[0.1, 1.1], [2.1, 3.1]]], f=lib),
                                ivy_gen.array([[[0, 1]], [[1, 0]]], f=lib), indices_shape=[2, 1, 2], f=lib),
                           np.array([[[2.0, 3.0]], [[0.1, 1.1]]]), atol=1e-6)
        assert np.allclose(call(ivy_gen.gather_nd, ivy_gen.array([[[0.0, 1.0], [2.0, 3.0]],
                                                                  [[0.1, 1.1], [2.1, 3.1]]], f=lib),
                                ivy_gen.array([[[0, 1, 0]], [[1, 0, 1]]], f=lib),
                                indices_shape=[2, 1, 3], f=lib), np.array([[2.0], [1.1]]), atol=1e-6)


def test_get_device():
    for lib, call in helpers.calls:
        if call in [helpers.mx_graph_call]:
            # mxnet symbolic tensors do not have a context
            continue
        assert 'cpu' in ivy_gen.get_device(ivy_gen.array([0.], f=lib)).lower()


def test_dtype():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # MXNet symbolic does not support dtype
            continue
        assert ivy_gen.dtype(ivy_gen.array([0.], f=lib)) == ivy_gen.array([0.], f=lib).dtype


def test_compile_fn():
    for lib, call in helpers.calls:
        some_fn = lambda x: x**2
        example_inputs = lib.array([2.])
        new_fn = ivy_gen.compile_fn(some_fn, example_inputs, lib)
        assert np.allclose(call(new_fn, example_inputs), np.array([4.]))
