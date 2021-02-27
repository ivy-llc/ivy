"""
Collection of tests for templated general functions
"""

# global
import numpy as np
from operator import mul as _mul
from functools import reduce as _reduce

# local
import ivy
import ivy_tests.helpers as helpers


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
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.array, [0.], f=lib), np.array([0.]))
        assert np.array_equal(call(ivy.array, [0.], 'float32', f=lib), np.array([0.], dtype=np.float32))
        assert np.array_equal(call(ivy.array, [[0.]], f=lib), np.array([[0.]]))
        helpers.assert_compilable('array', lib)


def test_to_numpy():
    for lib, call in helpers.calls():
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # to_numpy() requires eager execution
            continue
        assert call(ivy.to_numpy, ivy.array([0.], f=lib), f=lib) == np.array([0.])
        assert call(ivy.to_numpy, ivy.array([0.], 'float32', f=lib), f=lib) == np.array([0.])
        assert call(ivy.to_numpy, ivy.array([[0.]], f=lib), f=lib) == np.array([[0.]])
        if call in [helpers.torch_call]:
            # pytorch scripting does not support numpy conversion
            continue
        helpers.assert_compilable('to_numpy', lib)


def test_to_list():
    for lib, call in helpers.calls():
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # to_list() requires eager execution
            continue
        assert call(ivy.to_list, ivy.array([0.], f=lib), f=lib) == [0.]
        assert call(ivy.to_list, ivy.array([0.], 'float32', f=lib), f=lib) == [0.]
        assert call(ivy.to_list, ivy.array([[0.]], f=lib), f=lib) == [[0.]]
        if call in [helpers.torch_call]:
            # pytorch scripting does not support list conversion
            continue
        helpers.assert_compilable('to_list', lib)


def test_shape():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.shape, ivy.array([0.], f=lib), f=lib), np.array([1]))
        assert np.array_equal(call(ivy.shape, ivy.array([[0.]], f=lib), f=lib), np.array([1, 1]))
        helpers.assert_compilable('shape', lib)


def test_get_num_dims():
    for lib, call in helpers.calls():
        assert call(ivy.get_num_dims, ivy.array([0.], f=lib), f=lib) == np.array([1])
        assert call(ivy.get_num_dims, ivy.array([[0.]], f=lib), f=lib) == np.array([2])
        helpers.assert_compilable('get_num_dims', lib)


def test_minimum():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.minimum, ivy.array([0.7], f=lib), 0.5), np.minimum(np.array([0.7]), 0.5))
        if call is helpers.mx_graph_call:
            # mxnet symbolic minimum does not support varying array shapes
            continue
        assert np.allclose(call(ivy.minimum, ivy.array([[0.8, 1.2], [1.5, 0.2]], f=lib),
                                ivy.array([0., 1.], f=lib)),
                           np.minimum(np.array([[0.8, 1.2], [1.5, 0.2]]), np.array([0., 1.])))
        helpers.assert_compilable('minimum', lib)


def test_maximum():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.maximum, ivy.array([0.7], f=lib), 0.5), np.maximum(np.array([0.7]), 0.5))
        if call is helpers.mx_graph_call:
            # mxnet symbolic maximum does not support varying array shapes
            continue
        assert np.allclose(call(ivy.maximum, ivy.array([[0.8, 1.2], [1.5, 0.2]], f=lib),
                                ivy.array([0., 1.], f=lib)),
                           np.maximum(np.array([[0.8, 1.2], [1.5, 0.2]]), np.array([0., 1.])))
        helpers.assert_compilable('maximum', lib)


def test_clip():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.clip, ivy.array([0.], f=lib), 0, 1), np.clip(np.array([0.]), 0, 1))
        assert np.array_equal(call(ivy.clip, ivy.array([[0.]], f=lib), 0, 1), np.clip(np.array([[0.]]), 0, 1))
        helpers.assert_compilable('clip', lib)


def test_round():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.round, ivy.array([0.3], f=lib)), np.round(np.array([0.3])))
        assert np.array_equal(call(ivy.round, ivy.array([[0.51]], f=lib)), np.array([[1.]]))
        helpers.assert_compilable('round', lib)


def test_floormod():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.floormod, ivy.array([3.3], f=lib), ivy.array([3.], f=lib)),
                           np.array([0.3]), atol=1e-6)
        assert np.allclose(call(ivy.floormod, ivy.array([[10.7]], f=lib), ivy.array([[5.]], f=lib)),
                           np.array([[0.7]]), atol=1e-6)
        helpers.assert_compilable('floormod', lib)


def test_floor():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.floor, ivy.array([0.3], f=lib)), np.floor(np.array([0.3])))
        assert np.array_equal(call(ivy.floor, ivy.array([[0.7]], f=lib)), np.floor(np.array([[0.7]])))
        helpers.assert_compilable('floor', lib)


def test_ceil():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.ceil, ivy.array([0.3], f=lib)), np.ceil(np.array([0.3])))
        assert np.array_equal(call(ivy.ceil, ivy.array([[0.7]], f=lib)), np.ceil(np.array([[0.7]])))
        helpers.assert_compilable('ceil', lib)


def test_abs():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.abs, ivy.array([-0.3], f=lib)), np.array([0.3]), atol=1e-6)
        assert np.allclose(call(ivy.abs, ivy.array([[-0.7]], f=lib)), np.array([[0.7]]), atol=1e-6)
        helpers.assert_compilable('abs', lib)


def test_argmax():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.argmax, ivy.array([-0.3, 0.1], f=lib)), np.array([1]), atol=1e-6)
        assert np.allclose(call(ivy.argmax, ivy.array([[1.3, -0.7], [0.1, 2.5]], f=lib)),
                           np.array([0, 1]), atol=1e-6)
        helpers.assert_compilable('argmax', lib)


def test_argmin():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.argmin, ivy.array([-0.3, 0.1], f=lib)), np.array([0]), atol=1e-6)
        assert np.allclose(call(ivy.argmin, ivy.array([[1.3, -0.7], [0.1, 2.5]], f=lib)),
                           np.array([1, 0]), atol=1e-6)
        helpers.assert_compilable('argmin', lib)


def test_cast():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.cast, ivy.array([0], f=lib), 'float32'),
                              np.array([0]).astype(np.float32))
        assert np.array_equal(call(ivy.cast, ivy.array([[0]], f=lib), 'float32'),
                              np.array([[0]]).astype(np.float32))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support .type() method
            continue
        helpers.assert_compilable('cast', lib)


def test_arange():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.arange, 10, f=lib), np.arange(10))
        assert np.array_equal(call(ivy.arange, 10, 2, f=lib), np.arange(2, 10))
        assert np.array_equal(call(ivy.arange, 10, 2, 2, f=lib), np.arange(2, 10, 2))
        assert np.array_equal(call(ivy.arange, 10, 2, 2, 'float32', f=lib), np.arange(2, 10, 2, dtype=np.float32))
        if call is helpers.torch_call:
            # pytorch scripting does not support Union or Numbers for type hinting
            continue
        helpers.assert_compilable('arange', lib)


def test_linspace():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support linspace
            continue
        assert np.allclose(call(ivy.linspace, 1, 10, 100, f=lib), np.linspace(1, 10, 100), atol=1e-6)
        start = ivy.array([[0., 1., 2.]], f=lib)
        stop = ivy.array([[1., 2., 3.]], f=lib)
        assert np.allclose(call(ivy.linspace, start, stop, 100, f=lib),
                           np.linspace(np.array([[0., 1., 2.]]), np.array([[1., 2., 3.]]), 100, axis=-1), atol=1e-6)
        start = ivy.array([[[-0.1471,  0.4477,  0.2214]]], f=lib)
        stop = ivy.array([[[-0.3048,  0.3308,  0.2721]]], f=lib)
        res = np.array([[[[-0.1471,  0.4477,  0.2214],
                          [-0.1786,  0.4243,  0.2316],
                          [-0.2102,  0.4009,  0.2417],
                          [-0.2417,  0.3776,  0.2518],
                          [-0.2732,  0.3542,  0.2620],
                          [-0.3048,  0.3308,  0.2721]]]])
        assert np.allclose(call(ivy.linspace, start, stop, 6, axis=-2, f=lib), res, atol=1e-4)
        if call is helpers.torch_call:
            start = ivy.variable(start)
            stop = ivy.variable(stop)
            res = ivy.variable(res)
            assert np.allclose(ivy.linspace(start, stop, 6, axis=-2, f=lib).detach().numpy(), res, atol=1e-4)
        if call in [helpers.torch_call]:
            # pytorch scripting does not support numpy conversion
            continue
        helpers.assert_compilable('linspace', lib)


def test_concatenate():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.concatenate, (ivy.array([0.], f=lib), ivy.array([0.], f=lib)), 0),
                              np.concatenate((np.array([0.]), np.array([0.])), 0))
        assert np.array_equal(call(ivy.concatenate,
                                   (ivy.array([[0.]], f=lib), ivy.array([[0.]], f=lib)), 0),
                              np.concatenate((np.array([[0.]]), np.array([[0.]])), 0))
        helpers.assert_compilable('concatenate', lib)


def test_flip():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.flip, ivy.array([0., 1.], f=lib), batch_shape=[2]),
                              np.flip(np.array([0., 1.])))
        assert np.array_equal(call(ivy.flip, ivy.array([0., 1.], f=lib), -1, batch_shape=[2]),
                              np.flip(np.array([0., 1.])))
        assert np.array_equal(call(ivy.flip, ivy.array([[0., 1.]], f=lib), -1, batch_shape=[1, 2]),
                              np.flip(np.array([[0., 1.]])))
        helpers.assert_compilable('flip', lib)


def test_stack():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.stack, [ivy.array([0.], f=lib), ivy.array([0.], f=lib)], 0),
                              np.stack([np.array([0.]), np.array([0.])]))
        assert np.array_equal(call(ivy.stack, [ivy.array([[0.]], f=lib), ivy.array([[0.]], f=lib)], 0),
                              np.stack([np.array([[0.]]), np.array([[0.]])]))
        helpers.assert_compilable('stack', lib)


def test_unstack():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            # mxsymbolic split returns either list or tensor depending on number of splits
            continue
        x = np.swapaxes(np.array([[0.]]), 0, 0)
        true = [np.array(item) for item in x.tolist()]
        pred = call(ivy.unstack, ivy.array([[0.]], f=lib), 0, num_outputs=1)
        assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
        x = np.swapaxes(np.array([[[0.]]]), 0, 0)
        true = [np.array(item) for item in x.tolist()]
        pred = call(ivy.unstack, ivy.array([[[0.]]], f=lib), 0, num_outputs=1)
        assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
        helpers.assert_compilable('unstack', lib)


def test_split():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.split, ivy.array([[0., 1.]], f=lib), 2, -1),
                              np.split(np.array([[0., 1.]]), 2, -1))
        assert np.array_equal(call(ivy.split, ivy.array([[[0., 1.]]], f=lib), 2, -1),
                              np.split(np.array([[[0., 1.]]]), 2, -1))
        helpers.assert_compilable('split', lib)


def test_tile():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.tile, ivy.array([[0.]], f=lib), [1, 2]),
                              np.tile(np.array([[0.]]), [1, 2]))
        assert np.array_equal(call(ivy.tile, ivy.array([[[0.]]], f=lib), [1, 2, 3]),
                              np.tile(np.array([[[0.]]]), [1, 2, 3]))
        helpers.assert_compilable('tile', lib)


def test_zero_pad():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.zero_pad, ivy.array([[0.]], f=lib), [[0, 1], [1, 2]], x_shape=[1, 1]),
                              np.pad(np.array([[0.]]), [[0, 1], [1, 2]]))
        assert np.array_equal(call(ivy.zero_pad, ivy.array([[[0.]]], f=lib), [[0, 0], [1, 1], [2, 3]],
                                   x_shape=[1, 1, 1]),
                              np.pad(np.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]]))
        if call is helpers.torch_call:
            # pytorch scripting does not support Union or Numbers for type hinting
            continue
        helpers.assert_compilable('zero_pad', lib)


def test_constant_pad():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.constant_pad, ivy.array([[0.]], f=lib), [[0, 1], [1, 2]], 2.,
                                   x_shape=[1, 1]), np.pad(np.array([[0.]]), [[0, 1], [1, 2]], constant_values=2.))
        assert np.array_equal(call(ivy.constant_pad, ivy.array([[[0.]]], f=lib), [[0, 0], [1, 1], [2, 3]],
                                   3., x_shape=[1, 1, 1]),
                              np.pad(np.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]], constant_values=3.))
        if call is helpers.torch_call:
            # pytorch scripting does not support Union or Numbers for type hinting
            continue
        helpers.assert_compilable('constant_pad', lib)


def test_swapaxes():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.swapaxes, ivy.array([[0., 0.]], f=lib), 1, 0),
                              np.swapaxes(np.array([[0., 0.]]), 1, 0))
        assert np.array_equal(call(ivy.swapaxes, ivy.array([[0., 0.]], f=lib), -1, -2),
                              np.swapaxes(np.array([[0., 0.]]), -1, -2))
        helpers.assert_compilable('swapaxes', lib)


def test_transpose():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.transpose, ivy.array([[0., 0.]], f=lib), [1, 0]),
                              np.transpose(np.array([[0., 0.]]), [1, 0]))
        assert np.array_equal(call(ivy.transpose, ivy.array([[[0., 0.]]], f=lib), [2, 0, 1]),
                              np.transpose(np.array([[[0., 0.]]]), [2, 0, 1]))
        helpers.assert_compilable('transpose', lib)


def test_expand_dims():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.expand_dims, ivy.array([[0., 0.]], f=lib), 0),
                              np.expand_dims(np.array([[0., 0.]]), 0))
        assert np.array_equal(call(ivy.expand_dims, ivy.array([[[0., 0.]]], f=lib), -1),
                              np.expand_dims(np.array([[[0., 0.]]]), -1))
        helpers.assert_compilable('expand_dims', lib)


def test_where():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.where, ivy.array([[0., 1.]], f=lib) > 0,
                                   ivy.array([[1., 1.]], f=lib), ivy.array([[2., 2.]], f=lib),
                                   condition_shape=[1, 2], x_shape=[1, 2]),
                              np.where(np.array([[0., 1.]]) > 0, np.array([[0., 1.]]), np.array([[2., 2.]])))
        assert np.array_equal(call(ivy.where, ivy.array([[[1., 0.]]], f=lib) > 0,
                                   ivy.array([[[1., 1.]]], f=lib), ivy.array([[[2., 2.]]], f=lib),
                                   condition_shape=[1, 1, 2], x_shape=[1, 1, 2]),
                              np.where(np.array([[[1., 0.]]]) > 0, np.array([[[1., 1.]]]), np.array([[[2., 2.]]])))
        helpers.assert_compilable('where', lib)


def test_indices_where():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support indices_where
            continue
        assert np.array_equal(call(ivy.indices_where, ivy.array([[False, True],
                                                                         [True, False],
                                                                         [True, True]], f=lib)),
                              np.array([[0, 1], [1, 0], [2, 0], [2, 1]]))
        assert np.array_equal(call(ivy.indices_where, ivy.array([[[False, True],
                                                                          [True, False],
                                                                          [True, True]]], f=lib)),
                              np.array([[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 2, 1]]))
        helpers.assert_compilable('indices_where', lib)


def test_reshape():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.reshape, ivy.array([[0., 1.]], f=lib), (-1,), f=lib),
                              np.reshape(np.array([[0., 1.]]), (-1,)))
        assert np.array_equal(call(ivy.reshape, ivy.array([[[1., 0.]]], f=lib), (1, 2), f=lib),
                              np.reshape(np.array([[[1., 0.]]]), (1, 2)))
        helpers.assert_compilable('reshape', lib)


def test_squeeze():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.squeeze, ivy.array([[0., 1.]], f=lib), f=lib),
                              np.squeeze(np.array([[0., 1.]])))
        assert np.array_equal(call(ivy.squeeze, ivy.array([[[1., 0.]]], f=lib), 1, f=lib),
                              np.squeeze(np.array([[[1., 0.]]]), 1))
        helpers.assert_compilable('squeeze', lib)


def test_zeros():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.zeros, (1, 2), f=lib), np.zeros((1, 2)))
        assert np.array_equal(call(ivy.zeros, (1, 2), 'int64', f=lib), np.zeros((1, 2), np.int64))
        assert np.array_equal(call(ivy.zeros, (1, 2, 3), f=lib), np.zeros((1, 2, 3)))
        helpers.assert_compilable('zeros', lib)


def test_zeros_like():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.zeros_like, ivy.array([[0., 1.]], f=lib), f=lib),
                              np.zeros_like(np.array([[0., 1.]])))
        assert np.array_equal(call(ivy.zeros_like, ivy.array([[[1., 0.]]], f=lib), f=lib),
                              np.zeros_like(np.array([[[1., 0.]]])))
        helpers.assert_compilable('zeros_like', lib)


def test_ones():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.ones, (1, 2), f=lib), np.ones((1, 2)))
        assert np.array_equal(call(ivy.ones, (1, 2), 'int64', f=lib), np.ones((1, 2), np.int64))
        assert np.array_equal(call(ivy.ones, (1, 2, 3), f=lib), np.ones((1, 2, 3)))
        helpers.assert_compilable('ones', lib)


def test_ones_like():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.ones_like, ivy.array([[0., 1.]], f=lib), f=lib),
                              np.ones_like(np.array([[0., 1.]])))
        assert np.array_equal(call(ivy.ones_like, ivy.array([[[1., 0.]]], f=lib), f=lib),
                              np.ones_like(np.array([[[1., 0.]]])))
        helpers.assert_compilable('ones_like', lib)


def test_one_hot():
    np_one_hot = ivy.one_hot(np.array([0, 1, 2]), 3, f=helpers._ivy_np)
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.one_hot, ivy.array([0, 1, 2], f=lib), 3, f=lib), np_one_hot)
        helpers.assert_compilable('one_hot', lib)


def test_cross():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.array_equal(call(ivy.cross, ivy.array([0., 0., 0.], f=lib),
                                   ivy.array([0., 0., 0.], f=lib)),
                              np.cross(np.array([0., 0., 0.]), np.array([0., 0., 0.])))
        assert np.array_equal(call(ivy.cross, ivy.array([[0., 0., 0.]], f=lib),
                                   ivy.array([[0., 0., 0.]], f=lib)),
                              np.cross(np.array([[0., 0., 0.]]), np.array([[0., 0., 0.]])))
        helpers.assert_compilable('cross', lib)


def test_matmul():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.matmul, ivy.array([[1., 0.], [0., 1.]], f=lib),
                                   ivy.array([[1., 0.], [0., 1.]], f=lib), batch_shape=[]),
                              np.matmul(np.array([[1., 0.], [0., 1.]]), np.array([[1., 0.], [0., 1.]])))
        assert np.array_equal(call(ivy.matmul, ivy.array([[[[1., 0.], [0., 1.]]]], f=lib),
                                   ivy.array([[[[1., 0.], [0., 1.]]]], f=lib), batch_shape=[1, 1]),
                              np.matmul(np.array([[[[1., 0.], [0., 1.]]]]), np.array([[[[1., 0.], [0., 1.]]]])))
        helpers.assert_compilable('matmul', lib)


def test_cumsum():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.cumsum, ivy.array([[0., 1., 2., 3.]], f=lib), 1, f=lib),
                              np.array([[0., 1., 3., 6.]]))
        assert np.array_equal(call(ivy.cumsum, ivy.array([[0., 1., 2.], [0., 1., 2.]], f=lib), 0, f=lib),
                              np.array([[0., 1., 2.], [0., 2., 4.]]))
        helpers.assert_compilable('cumsum', lib)


def test_identity():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.identity, 1, f=lib), np.identity(1))
        assert np.array_equal(call(ivy.identity, 2, 'int64', f=lib), np.identity(2, np.int64))
        call(ivy.identity, 2, 'int64', (1, 2), f=lib)
        helpers.assert_compilable('identity', lib)


def test_scatter_flat_sum():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2], f=lib),
                                   ivy.array([1, 2, 3, 4], f=lib), 8, f=lib),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet scatter does not support sum reduction
            continue
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2, 0], f=lib),
                                   ivy.array([1, 2, 3, 4, 5], f=lib), 8, f=lib),
                              np.array([6, 3, 4, 0, 2, 0, 0, 0]))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_flat', lib)


def test_scatter_flat_min():
    for lib, call in helpers.calls():
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support max reduction for scatter flat
            continue
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2], f=lib),
                                   ivy.array([1, 2, 3, 4], f=lib), 8, 'min', f=lib),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2, 0], f=lib),
                                   ivy.array([1, 2, 3, 4, 5], f=lib), 8, 'min', f=lib),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_flat', lib)


def test_scatter_flat_max():
    for lib, call in helpers.calls():
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support max reduction for scatter flat
            continue
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2], f=lib),
                                   ivy.array([1, 2, 3, 4], f=lib), 8, 'max', f=lib),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2, 0], f=lib),
                                   ivy.array([1, 2, 3, 4, 5], f=lib), 8, 'max', f=lib),
                              np.array([5, 3, 4, 0, 2, 0, 0, 0]))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_flat', lib)


def test_scatter_sum_nd():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[4], [3], [1], [7]], f=lib),
                                   ivy.array([9, 10, 11, 12], f=lib), [8], 2, f=lib),
                              np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]),
                                                 [8]))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0, 1, 2]], f=lib),
                                   ivy.array([1], f=lib), [3, 3, 3], 2, f=lib),
                              np_scatter(np.array([[0, 1, 2]]), np.array([1]),
                                                 [3, 3, 3]))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0], [2]], f=lib),
                                   ivy.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]],
                                                  [[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]]], f=lib), [4, 4, 4], 2, f=lib),
                              np_scatter(np.array([[0], [2]]),
                                                 np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4]))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_nd', lib)


def test_scatter_min_nd():
    for lib, call in helpers.calls():
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support min reduction for scatter nd
            continue
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[4], [3], [1], [7]], f=lib),
                                   ivy.array([9, 10, 11, 12], f=lib), [8], 'min', f=lib),
                              np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'min'))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0, 1, 2]], f=lib),
                                   ivy.array([1], f=lib), [3, 3, 3], 'min', f=lib),
                              np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'min'))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0], [2]], f=lib),
                                   ivy.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]],
                                                  [[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]]], f=lib), [4, 4, 4], 'min', f=lib),
                              np_scatter(np.array([[0], [2]]),
                                                 np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'min'))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_nd', lib)


def test_scatter_max_nd():
    for lib, call in helpers.calls():
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support max reduction for scatter nd
            continue
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[4], [3], [1], [7]], f=lib),
                                   ivy.array([9, 10, 11, 12], f=lib), [8], 'max', f=lib),
                              np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'max'))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0, 1, 2]], f=lib),
                                   ivy.array([1], f=lib), [3, 3, 3], 'max', f=lib),
                              np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'max'))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0], [2]], f=lib),
                                   ivy.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]],
                                                  [[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]]], f=lib), [4, 4, 4], 'max', f=lib),
                              np_scatter(np.array([[0], [2]]),
                                                 np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'max'))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_nd', lib)


def test_gather_flat():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.gather_flat, ivy.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], f=lib),
                                ivy.array([0, 4, 7], f=lib), f=lib), np.array([9, 5, 2]), atol=1e-6)
        helpers.assert_compilable('gather_flat', lib)


def test_gather_nd():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.gather_nd, ivy.array([[[0.0, 1.0], [2.0, 3.0]],
                                                                  [[0.1, 1.1], [2.1, 3.1]]], f=lib),
                                ivy.array([[0, 1], [1, 0]], f=lib), indices_shape=[2, 2], f=lib),
                           np.array([[2.0, 3.0], [0.1, 1.1]]), atol=1e-6)
        assert np.allclose(call(ivy.gather_nd, ivy.array([[[0.0, 1.0], [2.0, 3.0]],
                                                                  [[0.1, 1.1], [2.1, 3.1]]], f=lib),
                                ivy.array([[[0, 1]], [[1, 0]]], f=lib), indices_shape=[2, 1, 2], f=lib),
                           np.array([[[2.0, 3.0]], [[0.1, 1.1]]]), atol=1e-6)
        assert np.allclose(call(ivy.gather_nd, ivy.array([[[0.0, 1.0], [2.0, 3.0]],
                                                                  [[0.1, 1.1], [2.1, 3.1]]], f=lib),
                                ivy.array([[[0, 1, 0]], [[1, 0, 1]]], f=lib),
                                indices_shape=[2, 1, 3], f=lib), np.array([[2.0], [1.1]]), atol=1e-6)
        if call in [helpers.torch_call]:
            # torch scripting does not support builtins
            continue
        helpers.assert_compilable('gather_nd', lib)


def test_dev():
    for lib, call in helpers.calls():
        if call in [helpers.mx_graph_call]:
            # mxnet symbolic tensors do not have a context
            continue
        assert ivy.dev(ivy.array([1.], f=lib))
        helpers.assert_compilable('dev', lib)


def test_dev_to_str():
    for lib, call in helpers.calls():
        if call in [helpers.mx_graph_call]:
            # mxnet symbolic tensors do not have a context
            continue
        assert 'cpu' in ivy.dev_to_str(ivy.dev(ivy.array([0.], f=lib)), f=lib).lower()
        helpers.assert_compilable('dev_to_str', lib)


def test_dev_str():
    for lib, call in helpers.calls():
        if call in [helpers.mx_graph_call]:
            # mxnet symbolic tensors do not have a context
            continue
        assert 'cpu' in ivy.dev_str(ivy.array([0.], f=lib)).lower()
        helpers.assert_compilable('dev_str', lib)


def test_dtype():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            # MXNet symbolic does not support dtype
            continue
        assert ivy.dtype(ivy.array([0.], f=lib)) == ivy.array([0.], f=lib).dtype
        helpers.assert_compilable('dtype', lib)


def test_dtype_to_str():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            # MXNet symbolic does not support dtype_str
            continue
        assert ivy.dtype_to_str(ivy.array([0.], dtype_str='float32', f=lib).dtype, f=lib) == 'float32'
        helpers.assert_compilable('dtype_to_str', lib)


def test_dtype_str():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            # MXNet symbolic does not support dtype_str
            continue
        assert ivy.dtype_str(ivy.array([0.], dtype_str='float32', f=lib), f=lib) == 'float32'
        helpers.assert_compilable('dtype_str', lib)


def test_compile_fn():
    for lib, call in helpers.calls():
        some_fn = lambda x: x**2
        example_inputs = lib.array([2.])
        new_fn = ivy.compile_fn(some_fn, False, example_inputs, lib)
        assert np.allclose(call(new_fn, example_inputs), np.array([4.]))
