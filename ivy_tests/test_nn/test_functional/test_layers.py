"""
Collection of tests for templated neural network layers
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


# Linear #
# -------#

# linear
@pytest.mark.parametrize(
    "x_n_w_n_b_n_res", [

        ([[1., 2., 3.]],
         [[1., 1., 1.], [1., 1., 1.]],
         [2., 2.],
         [[8., 8.]]),

        ([[[1., 2., 3.]]],
         [[1., 1., 1.], [1., 1., 1.]],
         [2., 2.],
         [[[8., 8.]]]),

        ([[[1., 2., 3.]], [[4., 5., 6.]]],
         [[[1., 1., 1.], [1., 1., 1.]], [[2., 2., 2.], [2., 2., 2.]]],
         [[2., 2.], [4., 4.]],
         [[[8., 8.]], [[34., 34.]]])
    ])
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
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.linear, x, weight, bias), ivy.to_numpy(true_res))
    # compilation test
    if call in [helpers.torch_call]:
        # optional Tensors in framework agnostic implementations not supported by torch.jit
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.linear)


# Dropout #
# --------#

# dropout
@pytest.mark.parametrize(
    "x", [([[1., 2., 3.]]),
          ([[[1., 2., 3.]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_dropout(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.dropout(x, 0.9)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    ivy.seed(0)
    assert np.min(call(ivy.dropout, x, 0.9)) == 0.
    # compilation test
    if call in [helpers.torch_call]:
        # str_to_dev not supported by torch.jit due to Device and Str not seen as the same
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dropout)


# Attention #
# ----------#

# scaled_dot_product_attention
@pytest.mark.parametrize(
    "q_n_k_n_v_n_s_n_m_n_gt", [([[1.]], [[2.]], [[3.]], 2., [[1.]], [[3.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_scaled_dot_product_attention(q_n_k_n_v_n_s_n_m_n_gt, dtype_str, tensor_fn, dev_str, call):
    q, k, v, scale, mask, ground_truth = q_n_k_n_v_n_s_n_m_n_gt
    # smoke test
    q = tensor_fn(q, dtype_str, dev_str)
    k = tensor_fn(k, dtype_str, dev_str)
    v = tensor_fn(v, dtype_str, dev_str)
    mask = tensor_fn(mask, dtype_str, dev_str)
    ret = ivy.scaled_dot_product_attention(q, k, v, scale, mask)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == q.shape
    # value test
    assert np.allclose(call(ivy.scaled_dot_product_attention, q, k, v, scale, mask), np.array(ground_truth))
    # compilation test
    if call in [helpers.torch_call]:
        # torch.jit compiled functions can't take variable number of arguments, which torch.einsum takes
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.scaled_dot_product_attention)


# multi_head_attention
@pytest.mark.parametrize(
    "x_n_s_n_m_n_c_n_gt", [([[3.]], 2., [[1.]], [[4., 5.]], [[4., 5., 4., 5.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_multi_head_attention(x_n_s_n_m_n_c_n_gt, dtype_str, tensor_fn, dev_str, call):
    x, scale, mask, context, ground_truth = x_n_s_n_m_n_c_n_gt
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    context = tensor_fn(context, dtype_str, dev_str)
    mask = tensor_fn(mask, dtype_str, dev_str)
    fn = lambda x_, v: ivy.tile(x_, (1, 2))
    ret = ivy.multi_head_attention(x, scale, 2, context, mask, fn, fn, fn)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == list(np.array(ground_truth).shape)
    # value test
    assert np.allclose(call(ivy.multi_head_attention, x, scale, 2, context, mask, fn, fn, fn), np.array(ground_truth))
    # compilation test
    if call in [helpers.torch_call]:
        # torch.jit compiled functions can't take variable number of arguments, which torch.einsum takes
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.multi_head_attention)


# Convolutions #
# -------------#

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
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    filters = tensor_fn(filters, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.conv1d(x, filters, 1, padding)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv1d, x, filters, 1, padding),
                       ivy.to_numpy(true_res))
    # compilation test
    if not ivy.wrapped_mode():
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
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d transpose does not work when CUDA is installed, but array is on CPU
        pytest.skip()
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
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv1d_transpose, x, filters, 1, padding, output_shape), ivy.to_numpy(true_res))
    # compilation test
    if not ivy.wrapped_mode():
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
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv2d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    # smoke test
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    filters = tensor_fn(filters, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.conv2d(x, filters, 1, padding)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv2d, x, filters, 1, padding), ivy.to_numpy(true_res))
    # compilation test
    if not ivy.wrapped_mode():
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
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv2d transpose does not work when CUDA is installed, but array is on CPU
        pytest.skip()
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
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv2d_transpose, x, filters, 1, padding, output_shape), ivy.to_numpy(true_res))
    # compilation test
    if not ivy.wrapped_mode():
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
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf depthwise conv2d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
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
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.depthwise_conv2d, x, filters, 1, padding), ivy.to_numpy(true_res))
    # compilation test
    if not ivy.wrapped_mode():
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
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv3d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
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
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv3d, x, filters, 1, padding), ivy.to_numpy(true_res))
    # compilation test
    if not ivy.wrapped_mode():
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
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv3d transpose does not work when CUDA is installed, but array is on CPU
        pytest.skip()
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
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv3d_transpose, x, filters, 1, padding, output_shape), ivy.to_numpy(true_res))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.conv3d_transpose)


# LSTM #
# -----#

# lstm
@pytest.mark.parametrize(
    "b_t_ic_hc_otf_sctv", [
        (2, 3, 4, 5, [0.93137765, 0.9587628, 0.96644664, 0.93137765, 0.9587628, 0.96644664], 3.708991),
    ])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_lstm(b_t_ic_hc_otf_sctv, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    b, t, input_channels, hidden_channels, output_true_flat, state_c_true_val = b_t_ic_hc_otf_sctv
    x = ivy.cast(ivy.linspace(ivy.zeros([b, t]), ivy.ones([b, t]), input_channels), 'float32')
    init_h = ivy.ones([b, hidden_channels])
    init_c = ivy.ones([b, hidden_channels])
    kernel = ivy.variable(ivy.ones([input_channels, 4*hidden_channels]))*0.5
    recurrent_kernel = ivy.variable(ivy.ones([hidden_channels, 4*hidden_channels]))*0.5
    output, state_c = ivy.lstm_update(x, init_h, init_c, kernel, recurrent_kernel)
    # type test
    assert ivy.is_array(output)
    assert ivy.is_array(state_c)
    # cardinality test
    assert output.shape == (b, t, hidden_channels)
    assert state_c.shape == (b, hidden_channels)
    # value test
    output_true = np.tile(np.asarray(output_true_flat).reshape((b, t, 1)), (1, 1, hidden_channels))
    state_c_true = np.ones([b, hidden_channels]) * state_c_true_val
    output, state_c = call(ivy.lstm_update, x, init_h, init_c, kernel, recurrent_kernel)
    assert np.allclose(output, output_true, atol=1e-6)
    assert np.allclose(state_c, state_c_true, atol=1e-6)
    # compilation test
    if call in [helpers.torch_call]:
        # this is not a backend implemented function
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.lstm_update)
