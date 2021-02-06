"""
Collection of tests for templated neural network layers
"""

# global
import numpy as np

# local
import ivy.neural_net.layers as ivy_layers
import ivy.core.general as ivy_gen
import ivy_tests.helpers as helpers


def test_conv1d():
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 1d convolutions
            continue
        x = ivy_gen.array([[[0.], [3.], [0.]]], f=lib)
        x_batched = ivy_gen.tile(x, (5, 1, 1), f=lib)
        filters = ivy_gen.array([[[0.]], [[1.]], [[0.]]], f=lib)
        result_same = np.array([[[0.], [3.], [0.]]])
        result_same_batched = np.tile(result_same, (5, 1, 1))
        result_valid = np.array([[[3.]]])
        assert np.allclose(call(ivy_layers.conv1d, x, filters, 1, "SAME", filter_shape=[3], num_filters=1, f=lib),
                           result_same)
        assert np.allclose(call(ivy_layers.conv1d, x_batched, filters, 1, "SAME",  filter_shape=[3], num_filters=1,
                                f=lib), result_same_batched)
        assert np.allclose(call(ivy_layers.conv1d, x, filters, 1, "VALID",  filter_shape=[3], num_filters=1, f=lib),
                           result_valid)


def test_conv1d_transpose():
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 1d transpose convolutions
            continue
        x = ivy_gen.array([[[0.], [3.], [0.]]], f=lib)
        x_batched = ivy_gen.tile(x, (5, 1, 1))
        filters = ivy_gen.array([[[0.]], [[1.]], [[0.]]], f=lib)
        result_same = np.array([[[0.], [3.], [0.]]])
        result_same_batched = np.tile(result_same, (5, 1, 1))
        result_valid = np.array([[[0.], [0.], [3.], [0.], [0.]]])
        assert np.allclose(call(ivy_layers.conv1d_transpose, x, filters, 1, "SAME", (1, 3, 1),
                                filter_shape=[3], num_filters=1, f=lib), result_same)
        assert np.allclose(call(ivy_layers.conv1d_transpose, x_batched, filters, 1, "SAME", (5, 3, 1),
                                filter_shape=[3], num_filters=1, f=lib), result_same_batched)
        assert np.allclose(call(ivy_layers.conv1d_transpose, x, filters, 1, "VALID", (1, 5, 1),
                                filter_shape=[3], num_filters=1, f=lib), result_valid)


def test_conv2d():
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 2d convolutions
            continue
        x = ivy_gen.array([[[[0.], [0.], [0.]],
                            [[0.], [3.], [0.]],
                            [[0.], [0.], [0.]]]], f=lib)
        x_batched = ivy_gen.tile(x, (5, 1, 1, 1), f=lib)
        filters = ivy_gen.array([[[[0.]], [[1.]], [[0.]]],
                                 [[[1.]], [[1.]], [[1.]]],
                                 [[[0.]], [[1.]], [[0.]]]], f=lib)
        result_same = np.array([[[[0.], [3.], [0.]],
                                 [[3.], [3.], [3.]],
                                 [[0.], [3.], [0.]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1))
        result_valid = np.array([[[[3.]]]])
        assert np.allclose(call(ivy_layers.conv2d, x, filters, 1, "SAME", filter_shape=[3, 3], num_filters=1, f=lib),
                           result_same)
        assert np.allclose(call(ivy_layers.conv2d, x_batched, filters, 1, "SAME",  filter_shape=[3, 3], num_filters=1,
                                f=lib), result_same_batched)
        assert np.allclose(call(ivy_layers.conv2d, x, filters, 1, "VALID",  filter_shape=[3, 3], num_filters=1, f=lib),
                           result_valid)


def test_conv2d_transpose():
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 2d transpose convolutions
            continue
        x = ivy_gen.array([[[[0.], [0.], [0.]],
                            [[0.], [3.], [0.]],
                            [[0.], [0.], [0.]]]], f=lib)
        x_batched = ivy_gen.tile(x, (5, 1, 1, 1))
        filters = ivy_gen.array([[[[0.]], [[1.]], [[0.]]],
                                 [[[1.]], [[1.]], [[1.]]],
                                 [[[0.]], [[1.]], [[0.]]]], f=lib)
        result_same = np.array([[[[0.], [3.], [0.]],
                                 [[3.], [3.], [3.]],
                                 [[0.], [3.], [0.]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1))
        result_valid = np.array([[[[0.], [0.], [0.], [0.], [0.]],
                                  [[0.], [0.], [3.], [0.], [0.]],
                                  [[0.], [3.], [3.], [3.], [0.]],
                                  [[0.], [0.], [3.], [0.], [0.]],
                                  [[0.], [0.], [0.], [0.], [0.]]]])
        assert np.allclose(call(ivy_layers.conv2d_transpose, x, filters, 1, "SAME", (1, 3, 3, 1),
                                filter_shape=[3, 3], num_filters=1, f=lib), result_same)
        assert np.allclose(call(ivy_layers.conv2d_transpose, x_batched, filters, 1, "SAME", (5, 3, 3, 1),
                                filter_shape=[3, 3], num_filters=1, f=lib), result_same_batched)
        assert np.allclose(call(ivy_layers.conv2d_transpose, x, filters, 1, "VALID", (1, 5, 5, 1),
                                filter_shape=[3, 3], num_filters=1, f=lib), result_valid)


def test_depthwise_conv2d():
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support depthwise 2d convolutions
            continue
        x1 = ivy_gen.array([[[[0.], [0.], [0.]],
                             [[0.], [3.], [0.]],
                             [[0.], [0.], [0.]]]], f=lib)
        x2 = ivy_gen.concatenate((x1, x1), -1)
        x1_batched = ivy_gen.tile(x1, (5, 1, 1, 1), f=lib)
        x2_batched = ivy_gen.tile(x2, (5, 1, 1, 1), f=lib)
        filters1 = ivy_gen.array([[[0.], [1.], [0.]],
                                  [[1.], [1.], [1.]],
                                  [[0.], [1.], [0.]]], f=lib)
        filters2 = ivy_gen.concatenate((filters1, filters1), -1)
        result_same = np.array([[[[0.], [3.], [0.]],
                                 [[3.], [3.], [3.]],
                                 [[0.], [3.], [0.]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1))
        result_valid = np.array([[[[3.]]]])
        call(ivy_layers.depthwise_conv2d, x2, filters2, 1, "SAME", filter_shape=[3, 3], num_filters=2,
                                num_channels=2, f=lib)
        assert np.allclose(call(ivy_layers.depthwise_conv2d, x1, filters1, 1, "SAME", filter_shape=[3, 3],
                                num_filters=1, num_channels=1, f=lib), result_same)
        assert np.allclose(call(ivy_layers.depthwise_conv2d, x1_batched, filters1, 1, "SAME",
                                filter_shape=[3, 3], num_filters=1, num_channels=1, f=lib), result_same_batched)
        assert np.allclose(call(ivy_layers.depthwise_conv2d, x1, filters1, 1, "VALID", filter_shape=[3, 3],
                                num_filters=1, num_channels=1, f=lib), result_valid)


def test_conv3d():
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 3d convolutions
            continue
        x = ivy_gen.array([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
                            [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
                            [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]], f=lib)
        x_batched = ivy_gen.tile(x, (5, 1, 1, 1, 1), f=lib)
        filters = ivy_gen.array([[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
                                 [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
                                 [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]], f=lib)
        result_same = np.array([[[[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]],
                                 [[[3.], [3.], [3.]], [[3.], [3.], [3.]], [[3.], [3.], [3.]]],
                                 [[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1, 1))
        result_valid = np.array([[[[[3.]]]]])
        assert np.allclose(call(ivy_layers.conv3d, x, filters, 1, "SAME", filter_shape=[3, 3, 3], num_filters=1,
                                f=lib), result_same)
        assert np.allclose(call(ivy_layers.conv3d, x_batched, filters, 1, "SAME",  filter_shape=[3, 3, 3],
                                num_filters=1, f=lib), result_same_batched)
        assert np.allclose(call(ivy_layers.conv3d, x, filters, 1, "VALID",  filter_shape=[3, 3, 3],
                                num_filters=1, f=lib), result_valid)


def test_conv3d_transpose():
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call, helpers.mx_graph_call]:
            # numpy and jax do not yet support 3d convolutions, and mxnet only supports with CUDNN
            continue
        x = ivy_gen.array([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
                            [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
                            [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]], f=lib)
        x_batched = ivy_gen.tile(x, (5, 1, 1, 1, 1), f=lib)
        filters = ivy_gen.array([[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
                                 [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
                                 [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]], f=lib)
        result_same = np.array([[[[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]],
                                 [[[3.], [3.], [3.]], [[3.], [3.], [3.]], [[3.], [3.], [3.]]],
                                 [[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1, 1))
        result_valid = np.array([[[[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
                                  [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
                                  [[[0.], [0.], [0.], [0.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
                                  [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
                                  [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]]]])
        assert np.allclose(call(ivy_layers.conv3d_transpose, x, filters, 1, "SAME", (1, 3, 3, 3, 1),
                                filter_shape=[3, 3, 3], num_filters=1, f=lib), result_same)
        assert np.allclose(call(ivy_layers.conv3d_transpose, x_batched, filters, 1, "SAME", (5, 3, 3, 3, 1),
                                filter_shape=[3, 3, 3], num_filters=1, f=lib), result_same_batched)
        assert np.allclose(call(ivy_layers.conv3d_transpose, x, filters, 1, "VALID", (1, 5, 5, 5, 1),
                                filter_shape=[3, 3, 3], num_filters=1, f=lib), result_valid)


def test_linear():
    for lib, call in helpers.calls:
        x = ivy_gen.array([[1., 2., 3.]], f=lib)
        weight = ivy_gen.array([[1., 1., 1.], [1., 1., 1.]], f=lib)
        bias = ivy_gen.array([2., 2.], f=lib)
        res = np.array([[8., 8.]])
        assert np.allclose(call(ivy_layers.linear, x, weight, bias, num_hidden=2, f=lib), res)
