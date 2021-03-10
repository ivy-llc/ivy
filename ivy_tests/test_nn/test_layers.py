"""
Collection of tests for templated neural network layers
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_conv1d():
    for f, call in helpers.f_n_calls():
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 1d convolutions
            continue
        x = ivy.array([[[0.], [3.], [0.]]], f=f)
        x_batched = ivy.tile(x, (5, 1, 1), f=f)
        filters = ivy.array([[[0.]], [[1.]], [[0.]]], f=f)
        result_same = np.array([[[0.], [3.], [0.]]])
        result_same_batched = np.tile(result_same, (5, 1, 1))
        result_valid = np.array([[[3.]]])
        assert np.allclose(call(ivy.conv1d, x, filters, 1, "SAME", filter_shape=[3], num_filters=1, f=f),
                           result_same)
        assert np.allclose(call(ivy.conv1d, x_batched, filters, 1, "SAME", filter_shape=[3], num_filters=1,
                                f=f), result_same_batched)
        assert np.allclose(call(ivy.conv1d, x, filters, 1, "VALID", filter_shape=[3], num_filters=1, f=f),
                           result_valid)
        helpers.assert_compilable('conv1d', f)


def test_conv1d_transpose():
    for f, call in helpers.f_n_calls():
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 1d transpose convolutions
            continue
        x = ivy.array([[[0.], [3.], [0.]]], f=f)
        x_batched = ivy.tile(x, (5, 1, 1))
        filters = ivy.array([[[0.]], [[1.]], [[0.]]], f=f)
        result_same = np.array([[[0.], [3.], [0.]]])
        result_same_batched = np.tile(result_same, (5, 1, 1))
        result_valid = np.array([[[0.], [0.], [3.], [0.], [0.]]])
        assert np.allclose(call(ivy.conv1d_transpose, x, filters, 1, "SAME", (1, 3, 1),
                                filter_shape=[3], num_filters=1, f=f), result_same)
        assert np.allclose(call(ivy.conv1d_transpose, x_batched, filters, 1, "SAME", (5, 3, 1),
                                filter_shape=[3], num_filters=1, f=f), result_same_batched)
        assert np.allclose(call(ivy.conv1d_transpose, x, filters, 1, "VALID", (1, 5, 1),
                                filter_shape=[3], num_filters=1, f=f), result_valid)
        helpers.assert_compilable('conv1d_transpose', f)


def test_conv2d():
    for f, call in helpers.f_n_calls():
        x = ivy.array([[[[1.], [2.], [3.], [4.], [5.]],
                        [[6.], [7.], [8.], [9.], [10.]],
                        [[11.], [12.], [13.], [14.], [15.]],
                        [[16.], [17.], [18.], [19.], [20.]],
                        [[21.], [22.], [23.], [24.], [25.]]]], f=f)
        x_batched = ivy.tile(x, (5, 1, 1, 1), f=f)
        filters = ivy.array([[[[0.]], [[1.]], [[0.]]],
                             [[[1.]], [[1.]], [[1.]]],
                             [[[0.]], [[1.]], [[0.]]]], f=f)
        result_same = np.array([[[[9.], [13.], [17.], [21.], [19.]],
                                 [[25.], [35.], [40.], [45.], [39.]],
                                 [[45.], [60.], [65.], [70.], [59.]],
                                 [[65.], [85.], [90.], [95.], [79.]],
                                 [[59.], [83.], [87.], [91.], [69.]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1))
        result_valid = np.array([[[[35.], [40.], [45.]],
                                  [[60.], [65.], [70.]],
                                  [[85.], [90.], [95.]]]])
        assert np.allclose(call(ivy.conv2d, x, filters, 1, "SAME", filter_shape=[3, 3], num_filters=1, f=f),
                           result_same)
        assert np.allclose(call(ivy.conv2d, x_batched, filters, 1, "SAME", filter_shape=[3, 3], num_filters=1,
                                f=f), result_same_batched)
        assert np.allclose(call(ivy.conv2d, x, filters, 1, "VALID", filter_shape=[3, 3], num_filters=1, f=f),
                           result_valid)
        helpers.assert_compilable('conv2d', f)


def test_conv2d_transpose():
    for f, call in helpers.f_n_calls():
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 2d transpose convolutions
            continue
        x = ivy.array([[[[0.], [0.], [0.]],
                        [[0.], [3.], [0.]],
                        [[0.], [0.], [0.]]]], f=f)
        x_batched = ivy.tile(x, (5, 1, 1, 1))
        filters = ivy.array([[[[0.]], [[1.]], [[0.]]],
                             [[[1.]], [[1.]], [[1.]]],
                             [[[0.]], [[1.]], [[0.]]]], f=f)
        result_same = np.array([[[[0.], [3.], [0.]],
                                 [[3.], [3.], [3.]],
                                 [[0.], [3.], [0.]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1))
        result_valid = np.array([[[[0.], [0.], [0.], [0.], [0.]],
                                  [[0.], [0.], [3.], [0.], [0.]],
                                  [[0.], [3.], [3.], [3.], [0.]],
                                  [[0.], [0.], [3.], [0.], [0.]],
                                  [[0.], [0.], [0.], [0.], [0.]]]])
        assert np.allclose(call(ivy.conv2d_transpose, x, filters, 1, "SAME", (1, 3, 3, 1),
                                filter_shape=[3, 3], num_filters=1, f=f), result_same)
        assert np.allclose(call(ivy.conv2d_transpose, x_batched, filters, 1, "SAME", (5, 3, 3, 1),
                                filter_shape=[3, 3], num_filters=1, f=f), result_same_batched)
        assert np.allclose(call(ivy.conv2d_transpose, x, filters, 1, "VALID", (1, 5, 5, 1),
                                filter_shape=[3, 3], num_filters=1, f=f), result_valid)
        helpers.assert_compilable('conv2d_transpose', f)


def test_depthwise_conv2d():
    for f, call in helpers.f_n_calls():
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support depthwise 2d convolutions
            continue
        x1 = ivy.array([[[[0.], [0.], [0.]],
                         [[0.], [3.], [0.]],
                         [[0.], [0.], [0.]]]], f=f)
        x2 = ivy.concatenate((x1, x1), -1)
        x1_batched = ivy.tile(x1, (5, 1, 1, 1), f=f)
        x2_batched = ivy.tile(x2, (5, 1, 1, 1), f=f)
        filters1 = ivy.array([[[0.], [1.], [0.]],
                              [[1.], [1.], [1.]],
                              [[0.], [1.], [0.]]], f=f)
        filters2 = ivy.concatenate((filters1, filters1), -1)
        result_same = np.array([[[[0.], [3.], [0.]],
                                 [[3.], [3.], [3.]],
                                 [[0.], [3.], [0.]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1))
        result_valid = np.array([[[[3.]]]])
        call(ivy.depthwise_conv2d, x2, filters2, 1, "SAME", filter_shape=[3, 3], num_filters=2,
             num_channels=2, f=f)
        assert np.allclose(call(ivy.depthwise_conv2d, x1, filters1, 1, "SAME", filter_shape=[3, 3],
                                num_filters=1, num_channels=1, f=f), result_same)
        assert np.allclose(call(ivy.depthwise_conv2d, x1_batched, filters1, 1, "SAME",
                                filter_shape=[3, 3], num_filters=1, num_channels=1, f=f), result_same_batched)
        assert np.allclose(call(ivy.depthwise_conv2d, x1, filters1, 1, "VALID", filter_shape=[3, 3],
                                num_filters=1, num_channels=1, f=f), result_valid)
        helpers.assert_compilable('depthwise_conv2d', f)


def test_conv3d():
    for f, call in helpers.f_n_calls():
        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 3d convolutions
            continue
        x = ivy.array([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
                        [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
                        [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]], f=f)
        x_batched = ivy.tile(x, (5, 1, 1, 1, 1), f=f)
        filters = ivy.array([[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
                             [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
                             [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]], f=f)
        result_same = np.array([[[[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]],
                                 [[[3.], [3.], [3.]], [[3.], [3.], [3.]], [[3.], [3.], [3.]]],
                                 [[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1, 1))
        result_valid = np.array([[[[[3.]]]]])
        assert np.allclose(call(ivy.conv3d, x, filters, 1, "SAME", filter_shape=[3, 3, 3], num_filters=1,
                                f=f), result_same)
        assert np.allclose(call(ivy.conv3d, x_batched, filters, 1, "SAME", filter_shape=[3, 3, 3],
                                num_filters=1, f=f), result_same_batched)
        assert np.allclose(call(ivy.conv3d, x, filters, 1, "VALID", filter_shape=[3, 3, 3],
                                num_filters=1, f=f), result_valid)
        helpers.assert_compilable('conv3d', f)


def test_conv3d_transpose():
    for f, call in helpers.f_n_calls():
        if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call, helpers.mx_graph_call]:
            # numpy and jax do not yet support 3d convolutions, and mxnet only supports with CUDNN
            continue
        x = ivy.array([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
                        [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
                        [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]], f=f)
        x_batched = ivy.tile(x, (5, 1, 1, 1, 1), f=f)
        filters = ivy.array([[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
                             [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
                             [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]], f=f)
        result_same = np.array([[[[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]],
                                 [[[3.], [3.], [3.]], [[3.], [3.], [3.]], [[3.], [3.], [3.]]],
                                 [[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]]]])
        result_same_batched = np.tile(result_same, (5, 1, 1, 1, 1))
        result_valid = np.array([[[[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
                                  [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
                                  [[[0.], [0.], [0.], [0.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
                                  [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
                                  [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]]]])
        assert np.allclose(call(ivy.conv3d_transpose, x, filters, 1, "SAME", (1, 3, 3, 3, 1),
                                filter_shape=[3, 3, 3], num_filters=1, f=f), result_same)
        assert np.allclose(call(ivy.conv3d_transpose, x_batched, filters, 1, "SAME", (5, 3, 3, 3, 1),
                                filter_shape=[3, 3, 3], num_filters=1, f=f), result_same_batched)
        assert np.allclose(call(ivy.conv3d_transpose, x, filters, 1, "VALID", (1, 5, 5, 5, 1),
                                filter_shape=[3, 3, 3], num_filters=1, f=f), result_valid)
        helpers.assert_compilable('conv3d_transpose', f)


def test_linear():
    for f, call in helpers.f_n_calls():
        x = ivy.array([[1., 2., 3.]], f=f)
        weight = ivy.array([[1., 1., 1.], [1., 1., 1.]], f=f)
        bias = ivy.array([2., 2.], f=f)
        res = np.array([[8., 8.]])
        assert np.allclose(call(ivy.linear, x, weight, bias, num_hidden=2, f=f), res)
        helpers.assert_compilable('linear', f)
