"""
Collection of tests for templated neural network layers
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


# conv1d
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_res", [
        ([[[0.], [3.], [0.]]],
         [[[0.]], [[1.]], [[0.]]],
         "SAME",
         [[[0.], [3.], [0.]]]),

        ([[[0.], [3.], [0.]] for _ in range(5)],
         [[[0.]], [[1.]], [[0.]]],
         "SAME",
         [[[0.], [3.], [0.]] for _ in range(5)]),

        ([[[0.], [3.], [0.]]],
         [[[0.]], [[1.]], [[0.]]],
         "VALID",
         [[[3.]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv1d(x_n_filters_n_pad_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    filters = tensor_fn(filters, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.conv1d(x, filters, 1, padding)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv1d, x, filters, 1, padding),
                       ivy.to_numpy(true_res))
    # compilation test
    helpers.assert_compilable(ivy.conv1d)


# conv1d_transpose
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_outshp_n_res", [
        ([[[0.], [3.], [0.]]],
         [[[0.]], [[1.]], [[0.]]],
         "SAME",
         (1, 3, 1),
         [[[0.], [3.], [0.]]]),

        ([[[0.], [3.], [0.]] for _ in range(5)],
         [[[0.]], [[1.]], [[0.]]],
         "SAME",
         (5, 3, 1),
         [[[0.], [3.], [0.]] for _ in range(5)]),

        ([[[0.], [3.], [0.]]],
         [[[0.]], [[1.]], [[0.]]],
         "VALID",
         (1, 5, 1),
         [[[0.], [0.], [3.], [0.], [0.]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv1d_transpose(x_n_filters_n_pad_n_outshp_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d_transpose
        pytest.skip()
    x, filters, padding, output_shape, true_res = x_n_filters_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    filters = tensor_fn(filters, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.conv1d_transpose(x, filters, 1, padding, output_shape)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv1d_transpose, x, filters, 1, padding, output_shape), ivy.to_numpy(true_res))
    # compilation test
    helpers.assert_compilable(ivy.conv1d_transpose)


# conv2d
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_res", [([[[[1.], [2.], [3.], [4.], [5.]],
                                   [[6.], [7.], [8.], [9.], [10.]],
                                   [[11.], [12.], [13.], [14.], [15.]],
                                   [[16.], [17.], [18.], [19.], [20.]],
                                   [[21.], [22.], [23.], [24.], [25.]]]],
                                 [[[[0.]], [[1.]], [[0.]]],
                                  [[[1.]], [[1.]], [[1.]]],
                                  [[[0.]], [[1.]], [[0.]]]],
                                 "SAME",
                                 [[[[9.], [13.], [17.], [21.], [19.]],
                                   [[25.], [35.], [40.], [45.], [39.]],
                                   [[45.], [60.], [65.], [70.], [59.]],
                                   [[65.], [85.], [90.], [95.], [79.]],
                                   [[59.], [83.], [87.], [91.], [69.]]]]),

                                ([[[[1.], [2.], [3.], [4.], [5.]],
                                   [[6.], [7.], [8.], [9.], [10.]],
                                   [[11.], [12.], [13.], [14.], [15.]],
                                   [[16.], [17.], [18.], [19.], [20.]],
                                   [[21.], [22.], [23.], [24.], [25.]]] for _ in range(5)],
                                 [[[[0.]], [[1.]], [[0.]]],
                                  [[[1.]], [[1.]], [[1.]]],
                                  [[[0.]], [[1.]], [[0.]]]],
                                 "SAME",
                                 [[[[9.], [13.], [17.], [21.], [19.]],
                                   [[25.], [35.], [40.], [45.], [39.]],
                                   [[45.], [60.], [65.], [70.], [59.]],
                                   [[65.], [85.], [90.], [95.], [79.]],
                                   [[59.], [83.], [87.], [91.], [69.]]] for _ in range(5)]),

                                ([[[[1.], [2.], [3.], [4.], [5.]],
                                   [[6.], [7.], [8.], [9.], [10.]],
                                   [[11.], [12.], [13.], [14.], [15.]],
                                   [[16.], [17.], [18.], [19.], [20.]],
                                   [[21.], [22.], [23.], [24.], [25.]]]],
                                 [[[[0.]], [[1.]], [[0.]]],
                                  [[[1.]], [[1.]], [[1.]]],
                                  [[[0.]], [[1.]], [[0.]]]],
                                 "VALID",
                                 [[[[35.], [40.], [45.]],
                                   [[60.], [65.], [70.]],
                                   [[85.], [90.], [95.]]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv2d(x_n_filters_n_pad_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    filters = tensor_fn(filters, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.conv2d(x, filters, 1, padding)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv2d, x, filters, 1, padding), ivy.to_numpy(true_res))
    # compilation test
    helpers.assert_compilable(ivy.conv2d)


# conv2d_transpose
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_outshp_n_res", [
        ([[[[0.], [0.], [0.]],
           [[0.], [3.], [0.]],
           [[0.], [0.], [0.]]]],
         [[[[0.]], [[1.]], [[0.]]],
          [[[1.]], [[1.]], [[1.]]],
          [[[0.]], [[1.]], [[0.]]]],
         "SAME",
         (1, 3, 3, 1),
         [[[[0.], [3.], [0.]],
           [[3.], [3.], [3.]],
           [[0.], [3.], [0.]]]]),

        ([[[[0.], [0.], [0.]],
           [[0.], [3.], [0.]],
           [[0.], [0.], [0.]]] for _ in range(5)],
         [[[[0.]], [[1.]], [[0.]]],
          [[[1.]], [[1.]], [[1.]]],
          [[[0.]], [[1.]], [[0.]]]],
         "SAME",
         (5, 3, 3, 1),
         [[[[0.], [3.], [0.]],
           [[3.], [3.], [3.]],
           [[0.], [3.], [0.]]] for _ in range(5)]),

        ([[[[0.], [0.], [0.]],
           [[0.], [3.], [0.]],
           [[0.], [0.], [0.]]]],
         [[[[0.]], [[1.]], [[0.]]],
          [[[1.]], [[1.]], [[1.]]],
          [[[0.]], [[1.]], [[0.]]]],
         "VALID",
         (1, 5, 5, 1),
         [[[[0.], [0.], [0.], [0.], [0.]],
           [[0.], [0.], [3.], [0.], [0.]],
           [[0.], [3.], [3.], [3.], [0.]],
           [[0.], [0.], [3.], [0.], [0.]],
           [[0.], [0.], [0.], [0.], [0.]]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv2d_transpose(x_n_filters_n_pad_n_outshp_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv2d_transpose
        pytest.skip()
    x, filters, padding, output_shape, true_res = x_n_filters_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    filters = tensor_fn(filters, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.conv2d_transpose(x, filters, 1, padding, output_shape)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv2d_transpose, x, filters, 1, padding, output_shape), ivy.to_numpy(true_res))
    # compilation test
    helpers.assert_compilable(ivy.conv2d_transpose)


# depthwise_conv2d
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_res", [([[[[0.], [0.], [0.]],
                                   [[0.], [3.], [0.]],
                                   [[0.], [0.], [0.]]]],
                                 [[[0.], [1.], [0.]],
                                  [[1.], [1.], [1.]],
                                  [[0.], [1.], [0.]]],
                                 "SAME",
                                 [[[[0.], [3.], [0.]],
                                   [[3.], [3.], [3.]],
                                   [[0.], [3.], [0.]]]]),

                                ([[[[0.], [0.], [0.]],
                                   [[0.], [3.], [0.]],
                                   [[0.], [0.], [0.]]] for _ in range(5)],
                                 [[[0.], [1.], [0.]],
                                  [[1.], [1.], [1.]],
                                  [[0.], [1.], [0.]]],
                                 "SAME",
                                 [[[[0.], [3.], [0.]],
                                   [[3.], [3.], [3.]],
                                   [[0.], [3.], [0.]]] for _ in range(5)]),

                                ([[[[0.], [0.], [0.]],
                                   [[0.], [3.], [0.]],
                                   [[0.], [0.], [0.]]]],
                                 [[[0.], [1.], [0.]],
                                  [[1.], [1.], [1.]],
                                  [[0.], [1.], [0.]]],
                                 "VALID",
                                 [[[[3.]]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_depthwise_conv2d(x_n_filters_n_pad_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support depthwise 2d convolutions
        pytest.skip()
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    filters = tensor_fn(filters, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.depthwise_conv2d(x, filters, 1, padding)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.depthwise_conv2d, x, filters, 1, padding), ivy.to_numpy(true_res))
    # compilation test
    helpers.assert_compilable(ivy.depthwise_conv2d)


# conv3d
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_res", [([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
                                   [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
                                   [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
                                 [[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
                                  [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
                                  [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]],
                                 "SAME",
                                 [[[[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]],
                                   [[[3.], [3.], [3.]], [[3.], [3.], [3.]], [[3.], [3.], [3.]]],
                                   [[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]]]]),

                                ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
                                   [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
                                   [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]] for _ in range(5)],
                                 [[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
                                  [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
                                  [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]],
                                 "SAME",
                                 [[[[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]],
                                   [[[3.], [3.], [3.]], [[3.], [3.], [3.]], [[3.], [3.], [3.]]],
                                   [[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]]] for _ in range(5)]),

                                ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
                                   [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
                                   [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
                                 [[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
                                  [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
                                  [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]],
                                 "VALID",
                                 [[[[[3.]]]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv3d(x_n_filters_n_pad_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support 3d convolutions
        pytest.skip()
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    filters = tensor_fn(filters, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.conv3d(x, filters, 1, padding)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv3d, x, filters, 1, padding), ivy.to_numpy(true_res))
    # compilation test
    helpers.assert_compilable(ivy.conv3d)


# conv3d_transpose
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_outshp_n_res", [
        ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
         [[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
          [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
          [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]],
         "SAME",
         (1, 3, 3, 3, 1),
         [[[[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]],
           [[[3.], [3.], [3.]], [[3.], [3.], [3.]], [[3.], [3.], [3.]]],
           [[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]]]]),

        ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]] for _ in range(5)],
         [[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
          [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
          [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]],
         "SAME",
         (5, 3, 3, 3, 1),
         [[[[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]],
           [[[3.], [3.], [3.]], [[3.], [3.], [3.]], [[3.], [3.], [3.]]],
           [[[0.], [0.], [0.]], [[3.], [3.], [3.]], [[0.], [0.], [0.]]]] for _ in range(5)]),

        ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
         [[[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]],
          [[[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]], [[[1.]], [[1.]], [[1.]]]],
          [[[[0.]], [[0.]], [[0.]]], [[[1.]], [[1.]], [[1.]]], [[[0.]], [[0.]], [[0.]]]]],
         "VALID",
         (1, 5, 5, 5, 1),
         [[[[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]],
            [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
           [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [3.], [3.], [3.], [0.]],
            [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
           [[[0.], [0.], [0.], [0.], [0.]], [[0.], [3.], [3.], [3.], [0.]], [[0.], [3.], [3.], [3.], [0.]],
            [[0.], [3.], [3.], [3.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
           [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [3.], [3.], [3.], [0.]],
            [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
           [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]],
            [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv3d_transpose(x_n_filters_n_pad_n_outshp_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # numpy and jax do not yet support 3d transpose convolutions, and mxnet only supports with CUDNN
        pytest.skip()
    if call in [helpers.mx_call] and 'cpu' in dev_str:
        # mxnet only supports 3d transpose convolutions with CUDNN
        pytest.skip()
    x, filters, padding, output_shape, true_res = x_n_filters_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    filters = tensor_fn(filters, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.conv3d_transpose(x, filters, 1, padding, output_shape)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv3d_transpose, x, filters, 1, padding, output_shape), ivy.to_numpy(true_res))
    # compilation test
    helpers.assert_compilable(ivy.conv3d_transpose)


# linear
@pytest.mark.parametrize(
    "x_n_w_n_b_n_res", [
        ([[1., 2., 3.]],
         [[1., 1., 1.], [1., 1., 1.]],
         [2., 2.],
         [[8., 8.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_linear(x_n_w_n_b_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, weight, bias, true_res = x_n_w_n_b_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    weight = tensor_fn(weight, dtype_str, dev_str)
    bias = tensor_fn(bias, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.linear(x, weight, bias)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.linear, x, weight, bias), ivy.to_numpy(true_res))
    # compilation test
    helpers.assert_compilable(ivy.linear)
