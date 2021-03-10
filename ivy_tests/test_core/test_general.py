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
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.array, [0.], f=f), np.array([0.]))
        assert np.array_equal(call(ivy.array, [0.], 'float32', f=f), np.array([0.], dtype=np.float32))
        assert np.array_equal(call(ivy.array, [[0.]], f=f), np.array([[0.]]))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support string devices
            continue
        helpers.assert_compilable('array', f)


def test_to_numpy():
    for f, call in helpers.f_n_calls():
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # to_numpy() requires eager execution
            continue
        assert call(ivy.to_numpy, ivy.array([0.], f=f), f=f) == np.array([0.])
        assert call(ivy.to_numpy, ivy.array([0.], 'float32', f=f), f=f) == np.array([0.])
        assert call(ivy.to_numpy, ivy.array([[0.]], f=f), f=f) == np.array([[0.]])
        if call in [helpers.torch_call]:
            # pytorch scripting does not support numpy conversion
            continue
        helpers.assert_compilable('to_numpy', f)


def test_to_list():
    for f, call in helpers.f_n_calls():
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # to_list() requires eager execution
            continue
        assert call(ivy.to_list, ivy.array([0.], f=f), f=f) == [0.]
        assert call(ivy.to_list, ivy.array([0.], 'float32', f=f), f=f) == [0.]
        assert call(ivy.to_list, ivy.array([[0.]], f=f), f=f) == [[0.]]
        if call in [helpers.torch_call]:
            # pytorch scripting does not support list conversion
            continue
        helpers.assert_compilable('to_list', f)


def test_shape():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.shape, ivy.array([0.], f=f), f=f), np.array([1]))
        assert np.array_equal(call(ivy.shape, ivy.array([[0.]], f=f), f=f), np.array([1, 1]))
        helpers.assert_compilable('shape', f)


def test_get_num_dims():
    for f, call in helpers.f_n_calls():
        assert call(ivy.get_num_dims, ivy.array([0.], f=f), f=f) == np.array([1])
        assert call(ivy.get_num_dims, ivy.array([[0.]], f=f), f=f) == np.array([2])
        helpers.assert_compilable('get_num_dims', f)


def test_minimum():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.minimum, ivy.array([0.7], f=f), 0.5), np.minimum(np.array([0.7]), 0.5))
        if call is helpers.mx_graph_call:
            # mxnet symbolic minimum does not support varying array shapes
            continue
        assert np.allclose(call(ivy.minimum, ivy.array([[0.8, 1.2], [1.5, 0.2]], f=f),
                                ivy.array([0., 1.], f=f)),
                           np.minimum(np.array([[0.8, 1.2], [1.5, 0.2]]), np.array([0., 1.])))
        helpers.assert_compilable('minimum', f)


def test_maximum():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.maximum, ivy.array([0.7], f=f), 0.5), np.maximum(np.array([0.7]), 0.5))
        if call is helpers.mx_graph_call:
            # mxnet symbolic maximum does not support varying array shapes
            continue
        assert np.allclose(call(ivy.maximum, ivy.array([[0.8, 1.2], [1.5, 0.2]], f=f),
                                ivy.array([0., 1.], f=f)),
                           np.maximum(np.array([[0.8, 1.2], [1.5, 0.2]]), np.array([0., 1.])))
        helpers.assert_compilable('maximum', f)


def test_clip():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.clip, ivy.array([0.], f=f), 0, 1), np.clip(np.array([0.]), 0, 1))
        assert np.array_equal(call(ivy.clip, ivy.array([[0.]], f=f), 0, 1), np.clip(np.array([[0.]]), 0, 1))
        helpers.assert_compilable('clip', f)


def test_round():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.round, ivy.array([0.3], f=f)), np.round(np.array([0.3])))
        assert np.array_equal(call(ivy.round, ivy.array([[0.51]], f=f)), np.array([[1.]]))
        helpers.assert_compilable('round', f)


def test_floormod():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.floormod, ivy.array([3.3], f=f), ivy.array([3.], f=f)),
                           np.array([0.3]), atol=1e-6)
        assert np.allclose(call(ivy.floormod, ivy.array([[10.7]], f=f), ivy.array([[5.]], f=f)),
                           np.array([[0.7]]), atol=1e-6)
        helpers.assert_compilable('floormod', f)


def test_floor():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.floor, ivy.array([0.3], f=f)), np.floor(np.array([0.3])))
        assert np.array_equal(call(ivy.floor, ivy.array([[0.7]], f=f)), np.floor(np.array([[0.7]])))
        helpers.assert_compilable('floor', f)


def test_ceil():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.ceil, ivy.array([0.3], f=f)), np.ceil(np.array([0.3])))
        assert np.array_equal(call(ivy.ceil, ivy.array([[0.7]], f=f)), np.ceil(np.array([[0.7]])))
        helpers.assert_compilable('ceil', f)


def test_abs():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.abs, ivy.array([-0.3], f=f)), np.array([0.3]), atol=1e-6)
        assert np.allclose(call(ivy.abs, ivy.array([[-0.7]], f=f)), np.array([[0.7]]), atol=1e-6)
        helpers.assert_compilable('abs', f)


def test_argmax():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.argmax, ivy.array([-0.3, 0.1], f=f)), np.array([1]), atol=1e-6)
        assert np.allclose(call(ivy.argmax, ivy.array([[1.3, -0.7], [0.1, 2.5]], f=f)),
                           np.array([0, 1]), atol=1e-6)
        helpers.assert_compilable('argmax', f)


def test_argmin():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.argmin, ivy.array([-0.3, 0.1], f=f)), np.array([0]), atol=1e-6)
        assert np.allclose(call(ivy.argmin, ivy.array([[1.3, -0.7], [0.1, 2.5]], f=f)),
                           np.array([1, 0]), atol=1e-6)
        helpers.assert_compilable('argmin', f)


def test_cast():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.cast, ivy.array([0], f=f), 'float32'),
                              np.array([0]).astype(np.float32))
        assert np.array_equal(call(ivy.cast, ivy.array([[0]], f=f), 'float32'),
                              np.array([[0]]).astype(np.float32))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support .type() method
            continue
        helpers.assert_compilable('cast', f)


def test_arange():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.arange, 10, f=f), np.arange(10))
        assert np.array_equal(call(ivy.arange, 10, 2, f=f), np.arange(2, 10))
        assert np.array_equal(call(ivy.arange, 10, 2, 2, f=f), np.arange(2, 10, 2))
        assert np.array_equal(call(ivy.arange, 10, 2, 2, 'float32', f=f), np.arange(2, 10, 2, dtype=np.float32))
        if call is helpers.torch_call:
            # pytorch scripting does not support Union or Numbers for type hinting
            continue
        helpers.assert_compilable('arange', f)


def test_linspace():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support linspace
            continue
        assert np.allclose(call(ivy.linspace, 1, 10, 100, f=f), np.linspace(1, 10, 100), atol=1e-6)
        start = ivy.array([[0., 1., 2.]], f=f)
        stop = ivy.array([[1., 2., 3.]], f=f)
        assert np.allclose(call(ivy.linspace, start, stop, 100, f=f),
                           np.linspace(np.array([[0., 1., 2.]]), np.array([[1., 2., 3.]]), 100, axis=-1), atol=1e-6)
        start = ivy.array([[[-0.1471,  0.4477,  0.2214]]], f=f)
        stop = ivy.array([[[-0.3048,  0.3308,  0.2721]]], f=f)
        res = np.array([[[[-0.1471,  0.4477,  0.2214],
                          [-0.1786,  0.4243,  0.2316],
                          [-0.2102,  0.4009,  0.2417],
                          [-0.2417,  0.3776,  0.2518],
                          [-0.2732,  0.3542,  0.2620],
                          [-0.3048,  0.3308,  0.2721]]]])
        assert np.allclose(call(ivy.linspace, start, stop, 6, axis=-2, f=f), res, atol=1e-4)
        if call is helpers.torch_call:
            start = ivy.variable(start)
            stop = ivy.variable(stop)
            res = ivy.variable(res)
            assert np.allclose(ivy.linspace(start, stop, 6, axis=-2, f=f).detach().numpy(), res, atol=1e-4)
        if call in [helpers.torch_call]:
            # pytorch scripting does not support numpy conversion
            continue
        helpers.assert_compilable('linspace', f)


def test_concatenate():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.concatenate, (ivy.array([0.], f=f), ivy.array([0.], f=f)), 0),
                              np.concatenate((np.array([0.]), np.array([0.])), 0))
        assert np.array_equal(call(ivy.concatenate,
                                   (ivy.array([[0.]], f=f), ivy.array([[0.]], f=f)), 0),
                              np.concatenate((np.array([[0.]]), np.array([[0.]])), 0))
        helpers.assert_compilable('concatenate', f)


def test_flip():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.flip, ivy.array([0., 1.], f=f), batch_shape=[2]),
                              np.flip(np.array([0., 1.])))
        assert np.array_equal(call(ivy.flip, ivy.array([0., 1.], f=f), -1, batch_shape=[2]),
                              np.flip(np.array([0., 1.])))
        assert np.array_equal(call(ivy.flip, ivy.array([[0., 1.]], f=f), -1, batch_shape=[1, 2]),
                              np.flip(np.array([[0., 1.]])))
        helpers.assert_compilable('flip', f)


def test_stack():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.stack, [ivy.array([0.], f=f), ivy.array([0.], f=f)], 0),
                              np.stack([np.array([0.]), np.array([0.])]))
        assert np.array_equal(call(ivy.stack, [ivy.array([[0.]], f=f), ivy.array([[0.]], f=f)], 0),
                              np.stack([np.array([[0.]]), np.array([[0.]])]))
        helpers.assert_compilable('stack', f)


def test_unstack():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # mxsymbolic split returns either list or tensor depending on number of splits
            continue
        x = np.swapaxes(np.array([[0.]]), 0, 0)
        true = [np.array(item) for item in x.tolist()]
        pred = call(ivy.unstack, ivy.array([[0.]], f=f), 0, num_outputs=1)
        assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
        x = np.swapaxes(np.array([[[0.]]]), 0, 0)
        true = [np.array(item) for item in x.tolist()]
        pred = call(ivy.unstack, ivy.array([[[0.]]], f=f), 0, num_outputs=1)
        assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
        helpers.assert_compilable('unstack', f)


def test_split():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.split, ivy.array([[0., 1.]], f=f), 2, -1),
                              np.split(np.array([[0., 1.]]), 2, -1))
        assert np.array_equal(call(ivy.split, ivy.array([[[0., 1.]]], f=f), 2, -1),
                              np.split(np.array([[[0., 1.]]]), 2, -1))
        helpers.assert_compilable('split', f)


def test_tile():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.tile, ivy.array([[0.]], f=f), [1, 2]),
                              np.tile(np.array([[0.]]), [1, 2]))
        assert np.array_equal(call(ivy.tile, ivy.array([[[0.]]], f=f), [1, 2, 3]),
                              np.tile(np.array([[[0.]]]), [1, 2, 3]))
        helpers.assert_compilable('tile', f)


def test_zero_pad():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.zero_pad, ivy.array([[0.]], f=f), [[0, 1], [1, 2]], x_shape=[1, 1]),
                              np.pad(np.array([[0.]]), [[0, 1], [1, 2]]))
        assert np.array_equal(call(ivy.zero_pad, ivy.array([[[0.]]], f=f), [[0, 0], [1, 1], [2, 3]],
                                   x_shape=[1, 1, 1]),
                              np.pad(np.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]]))
        if call is helpers.torch_call:
            # pytorch scripting does not support Union or Numbers for type hinting
            continue
        helpers.assert_compilable('zero_pad', f)


def test_constant_pad():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.constant_pad, ivy.array([[0.]], f=f), [[0, 1], [1, 2]], 2.,
                                   x_shape=[1, 1]), np.pad(np.array([[0.]]), [[0, 1], [1, 2]], constant_values=2.))
        assert np.array_equal(call(ivy.constant_pad, ivy.array([[[0.]]], f=f), [[0, 0], [1, 1], [2, 3]],
                                   3., x_shape=[1, 1, 1]),
                              np.pad(np.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]], constant_values=3.))
        if call is helpers.torch_call:
            # pytorch scripting does not support Union or Numbers for type hinting
            continue
        helpers.assert_compilable('constant_pad', f)


def test_swapaxes():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.swapaxes, ivy.array([[0., 0.]], f=f), 1, 0),
                              np.swapaxes(np.array([[0., 0.]]), 1, 0))
        assert np.array_equal(call(ivy.swapaxes, ivy.array([[0., 0.]], f=f), -1, -2),
                              np.swapaxes(np.array([[0., 0.]]), -1, -2))
        helpers.assert_compilable('swapaxes', f)


def test_transpose():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.transpose, ivy.array([[0., 0.]], f=f), [1, 0]),
                              np.transpose(np.array([[0., 0.]]), [1, 0]))
        assert np.array_equal(call(ivy.transpose, ivy.array([[[0., 0.]]], f=f), [2, 0, 1]),
                              np.transpose(np.array([[[0., 0.]]]), [2, 0, 1]))
        helpers.assert_compilable('transpose', f)


def test_expand_dims():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.expand_dims, ivy.array([[0., 0.]], f=f), 0),
                              np.expand_dims(np.array([[0., 0.]]), 0))
        assert np.array_equal(call(ivy.expand_dims, ivy.array([[[0., 0.]]], f=f), -1),
                              np.expand_dims(np.array([[[0., 0.]]]), -1))
        helpers.assert_compilable('expand_dims', f)


def test_where():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.where, ivy.array([[0., 1.]], f=f) > 0,
                                   ivy.array([[1., 1.]], f=f), ivy.array([[2., 2.]], f=f),
                                   condition_shape=[1, 2], x_shape=[1, 2]),
                              np.where(np.array([[0., 1.]]) > 0, np.array([[0., 1.]]), np.array([[2., 2.]])))
        assert np.array_equal(call(ivy.where, ivy.array([[[1., 0.]]], f=f) > 0,
                                   ivy.array([[[1., 1.]]], f=f), ivy.array([[[2., 2.]]], f=f),
                                   condition_shape=[1, 1, 2], x_shape=[1, 1, 2]),
                              np.where(np.array([[[1., 0.]]]) > 0, np.array([[[1., 1.]]]), np.array([[[2., 2.]]])))
        helpers.assert_compilable('where', f)


def test_indices_where():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support indices_where
            continue
        assert np.array_equal(call(ivy.indices_where, ivy.array([[False, True],
                                                                         [True, False],
                                                                         [True, True]], f=f)),
                              np.array([[0, 1], [1, 0], [2, 0], [2, 1]]))
        assert np.array_equal(call(ivy.indices_where, ivy.array([[[False, True],
                                                                          [True, False],
                                                                          [True, True]]], f=f)),
                              np.array([[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 2, 1]]))
        helpers.assert_compilable('indices_where', f)


def test_reshape():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.reshape, ivy.array([[0., 1.]], f=f), (-1,), f=f),
                              np.reshape(np.array([[0., 1.]]), (-1,)))
        assert np.array_equal(call(ivy.reshape, ivy.array([[[1., 0.]]], f=f), (1, 2), f=f),
                              np.reshape(np.array([[[1., 0.]]]), (1, 2)))
        helpers.assert_compilable('reshape', f)


def test_squeeze():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.squeeze, ivy.array([[0., 1.]], f=f), f=f),
                              np.squeeze(np.array([[0., 1.]])))
        assert np.array_equal(call(ivy.squeeze, ivy.array([[[1., 0.]]], f=f), 1, f=f),
                              np.squeeze(np.array([[[1., 0.]]]), 1))
        helpers.assert_compilable('squeeze', f)


def test_zeros():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.zeros, (1, 2), f=f), np.zeros((1, 2)))
        assert np.array_equal(call(ivy.zeros, (1, 2), 'int64', f=f), np.zeros((1, 2), np.int64))
        assert np.array_equal(call(ivy.zeros, (1, 2, 3), f=f), np.zeros((1, 2, 3)))
        helpers.assert_compilable('zeros', f)


def test_zeros_like():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.zeros_like, ivy.array([[0., 1.]], f=f), f=f),
                              np.zeros_like(np.array([[0., 1.]])))
        assert np.array_equal(call(ivy.zeros_like, ivy.array([[[1., 0.]]], f=f), f=f),
                              np.zeros_like(np.array([[[1., 0.]]])))
        helpers.assert_compilable('zeros_like', f)


def test_ones():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.ones, (1, 2), f=f), np.ones((1, 2)))
        assert np.array_equal(call(ivy.ones, (1, 2), 'int64', f=f), np.ones((1, 2), np.int64))
        assert np.array_equal(call(ivy.ones, (1, 2, 3), f=f), np.ones((1, 2, 3)))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support string devices
            continue
        helpers.assert_compilable('ones', f)


def test_ones_like():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.ones_like, ivy.array([[0., 1.]], f=f), f=f),
                              np.ones_like(np.array([[0., 1.]])))
        assert np.array_equal(call(ivy.ones_like, ivy.array([[[1., 0.]]], f=f), f=f),
                              np.ones_like(np.array([[[1., 0.]]])))
        helpers.assert_compilable('ones_like', f)


def test_one_hot():
    np_one_hot = ivy.one_hot(np.array([0, 1, 2]), 3, f=helpers._ivy_np)
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.one_hot, ivy.array([0, 1, 2], f=f), 3, f=f), np_one_hot)
        helpers.assert_compilable('one_hot', f)


def test_cross():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.array_equal(call(ivy.cross, ivy.array([0., 0., 0.], f=f),
                                   ivy.array([0., 0., 0.], f=f)),
                              np.cross(np.array([0., 0., 0.]), np.array([0., 0., 0.])))
        assert np.array_equal(call(ivy.cross, ivy.array([[0., 0., 0.]], f=f),
                                   ivy.array([[0., 0., 0.]], f=f)),
                              np.cross(np.array([[0., 0., 0.]]), np.array([[0., 0., 0.]])))
        helpers.assert_compilable('cross', f)


def test_matmul():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.matmul, ivy.array([[1., 0.], [0., 1.]], f=f),
                                   ivy.array([[1., 0.], [0., 1.]], f=f), batch_shape=[]),
                              np.matmul(np.array([[1., 0.], [0., 1.]]), np.array([[1., 0.], [0., 1.]])))
        assert np.array_equal(call(ivy.matmul, ivy.array([[[[1., 0.], [0., 1.]]]], f=f),
                                   ivy.array([[[[1., 0.], [0., 1.]]]], f=f), batch_shape=[1, 1]),
                              np.matmul(np.array([[[[1., 0.], [0., 1.]]]]), np.array([[[[1., 0.], [0., 1.]]]])))
        helpers.assert_compilable('matmul', f)


def test_cumsum():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.cumsum, ivy.array([[0., 1., 2., 3.]], f=f), 1, f=f),
                              np.array([[0., 1., 3., 6.]]))
        assert np.array_equal(call(ivy.cumsum, ivy.array([[0., 1., 2.], [0., 1., 2.]], f=f), 0, f=f),
                              np.array([[0., 1., 2.], [0., 2., 4.]]))
        helpers.assert_compilable('cumsum', f)


def test_identity():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.identity, 1, f=f), np.identity(1))
        assert np.array_equal(call(ivy.identity, 2, 'int64', f=f), np.identity(2, np.int64))
        call(ivy.identity, 2, 'int64', (1, 2), f=f)
        helpers.assert_compilable('identity', f)


def test_scatter_flat_sum():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2], f=f),
                                   ivy.array([1, 2, 3, 4], f=f), 8, f=f),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet scatter does not support sum reduction
            continue
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2, 0], f=f),
                                   ivy.array([1, 2, 3, 4, 5], f=f), 8, f=f),
                              np.array([6, 3, 4, 0, 2, 0, 0, 0]))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_flat', f)


def test_scatter_flat_min():
    for f, call in helpers.f_n_calls():
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support max reduction for scatter flat
            continue
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2], f=f),
                                   ivy.array([1, 2, 3, 4], f=f), 8, 'min', f=f),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2, 0], f=f),
                                   ivy.array([1, 2, 3, 4, 5], f=f), 8, 'min', f=f),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_flat', f)


def test_scatter_flat_max():
    for f, call in helpers.f_n_calls():
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support max reduction for scatter flat
            continue
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2], f=f),
                                   ivy.array([1, 2, 3, 4], f=f), 8, 'max', f=f),
                              np.array([1, 3, 4, 0, 2, 0, 0, 0]))
        assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2, 0], f=f),
                                   ivy.array([1, 2, 3, 4, 5], f=f), 8, 'max', f=f),
                              np.array([5, 3, 4, 0, 2, 0, 0, 0]))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_flat', f)


def test_scatter_sum_nd():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[4], [3], [1], [7]], f=f),
                                   ivy.array([9, 10, 11, 12], f=f), [8], 2, f=f),
                              np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]),
                                                 [8]))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0, 1, 2]], f=f),
                                   ivy.array([1], f=f), [3, 3, 3], 2, f=f),
                              np_scatter(np.array([[0, 1, 2]]), np.array([1]),
                                                 [3, 3, 3]))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0], [2]], f=f),
                                   ivy.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]],
                                                  [[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]]], f=f), [4, 4, 4], 2, f=f),
                              np_scatter(np.array([[0], [2]]),
                                                 np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4]))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_nd', f)


def test_scatter_min_nd():
    for f, call in helpers.f_n_calls():
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support min reduction for scatter nd
            continue
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[4], [3], [1], [7]], f=f),
                                   ivy.array([9, 10, 11, 12], f=f), [8], 'min', f=f),
                              np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'min'))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0, 1, 2]], f=f),
                                   ivy.array([1], f=f), [3, 3, 3], 'min', f=f),
                              np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'min'))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0], [2]], f=f),
                                   ivy.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]],
                                                  [[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]]], f=f), [4, 4, 4], 'min', f=f),
                              np_scatter(np.array([[0], [2]]),
                                                 np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'min'))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_nd', f)


def test_scatter_max_nd():
    for f, call in helpers.f_n_calls():
        if call in [helpers.mx_call, helpers.mx_graph_call]:
            # mxnet does not support max reduction for scatter nd
            continue
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[4], [3], [1], [7]], f=f),
                                   ivy.array([9, 10, 11, 12], f=f), [8], 'max', f=f),
                              np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'max'))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0, 1, 2]], f=f),
                                   ivy.array([1], f=f), [3, 3, 3], 'max', f=f),
                              np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'max'))
        assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0], [2]], f=f),
                                   ivy.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]],
                                                  [[5, 5, 5, 5], [6, 6, 6, 6],
                                                   [7, 7, 7, 7], [8, 8, 8, 8]]], f=f), [4, 4, 4], 'max', f=f),
                              np_scatter(np.array([[0], [2]]),
                                                 np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                                            [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'max'))
        if call in [helpers.torch_call]:
            # global torch_scatter var not supported when scripting
            continue
        helpers.assert_compilable('scatter_nd', f)


def test_gather_flat():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.gather_flat, ivy.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], f=f),
                                ivy.array([0, 4, 7], f=f), f=f), np.array([9, 5, 2]), atol=1e-6)
        helpers.assert_compilable('gather_flat', f)


def test_gather_nd():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.gather_nd, ivy.array([[[0.0, 1.0], [2.0, 3.0]],
                                                                  [[0.1, 1.1], [2.1, 3.1]]], f=f),
                                ivy.array([[0, 1], [1, 0]], f=f), indices_shape=[2, 2], f=f),
                           np.array([[2.0, 3.0], [0.1, 1.1]]), atol=1e-6)
        assert np.allclose(call(ivy.gather_nd, ivy.array([[[0.0, 1.0], [2.0, 3.0]],
                                                                  [[0.1, 1.1], [2.1, 3.1]]], f=f),
                                ivy.array([[[0, 1]], [[1, 0]]], f=f), indices_shape=[2, 1, 2], f=f),
                           np.array([[[2.0, 3.0]], [[0.1, 1.1]]]), atol=1e-6)
        assert np.allclose(call(ivy.gather_nd, ivy.array([[[0.0, 1.0], [2.0, 3.0]],
                                                                  [[0.1, 1.1], [2.1, 3.1]]], f=f),
                                ivy.array([[[0, 1, 0]], [[1, 0, 1]]], f=f),
                                indices_shape=[2, 1, 3], f=f), np.array([[2.0], [1.1]]), atol=1e-6)
        if call in [helpers.torch_call]:
            # torch scripting does not support builtins
            continue
        helpers.assert_compilable('gather_nd', f)


def test_dev():
    for f, call in helpers.f_n_calls():
        if call in [helpers.mx_graph_call]:
            # mxnet symbolic tensors do not have a context
            continue
        assert ivy.dev(ivy.array([1.], f=f))
        helpers.assert_compilable('dev', f)


def test_dev_to_str():
    for f, call in helpers.f_n_calls():
        if call in [helpers.mx_graph_call]:
            # mxnet symbolic tensors do not have a context
            continue
        assert 'cpu' in ivy.dev_to_str(ivy.dev(ivy.array([0.], f=f)), f=f).lower()
        helpers.assert_compilable('dev_to_str', f)


def test_dev_str():
    for f, call in helpers.f_n_calls():
        if call in [helpers.mx_graph_call]:
            # mxnet symbolic tensors do not have a context
            continue
        assert 'cpu' in ivy.dev_str(ivy.array([0.], f=f)).lower()
        helpers.assert_compilable('dev_str', f)


def test_dtype():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # MXNet symbolic does not support dtype
            continue
        assert ivy.dtype(ivy.array([0.], f=f)) == ivy.array([0.], f=f).dtype
        helpers.assert_compilable('dtype', f)


def test_dtype_to_str():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # MXNet symbolic does not support dtype_str
            continue
        assert ivy.dtype_to_str(ivy.array([0.], dtype_str='float32', f=f).dtype, f=f) == 'float32'
        helpers.assert_compilable('dtype_to_str', f)


def test_dtype_str():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # MXNet symbolic does not support dtype_str
            continue
        assert ivy.dtype_str(ivy.array([0.], dtype_str='float32', f=f), f=f) == 'float32'
        helpers.assert_compilable('dtype_str', f)


def test_compile_fn():
    for f, call in helpers.f_n_calls():
        some_fn = lambda x: x**2
        example_inputs = f.array([2.])
        new_fn = ivy.compile_fn(some_fn, False, example_inputs, f)
        assert np.allclose(call(new_fn, example_inputs), np.array([4.]))
