"""Collection of tests for unified neural network layers."""

# global
import pytest
import numpy as np
from hypothesis import given, strategies as st
# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np

# Linear #
# -------#

# linear
@pytest.mark.parametrize(
    "x_n_w_n_b_n_res",
    [
        (
            [[1.0, 2.0, 3.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [2.0, 2.0],
            [[8.0, 8.0]],
        ),
        (
            [[[1.0, 2.0, 3.0]]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [2.0, 2.0],
            [[[8.0, 8.0]]],
        ),
        (
            [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]],
            [[2.0, 2.0], [4.0, 4.0]],
            [[[8.0, 8.0]], [[34.0, 34.0]]],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_linear(x_n_w_n_b_n_res, dtype, tensor_fn, device, call):
    # smoke test
    x, weight, bias, true_res = x_n_w_n_b_n_res
    x = tensor_fn(x, dtype, device)
    weight = tensor_fn(weight, dtype, device)
    bias = tensor_fn(bias, dtype, device)
    true_res = tensor_fn(true_res, dtype, device)
    ret = ivy.linear(x, weight, bias)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.linear, x, weight, bias), ivy.to_numpy(true_res))


# Dropout #
# --------#

# dropout
@given(
    array_shape=helpers.lists(
        st.integers(1, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans()
)
def test_dropout(array_shape, dtype, as_variable, fw, device, call):
    if (fw == 'tensorflow' or fw == 'torch') and 'int' in dtype:
        return
    x = np.random.uniform(size=tuple(array_shape)).astype(dtype)
    x = ivy.asarray(x)
    if as_variable:
        x = ivy.variable(x)
    # smoke test
    ret = ivy.dropout(x, 0.9)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    ivy.seed(0)
    assert np.min(call(ivy.dropout, x, 0.9)) == 0.0


# Attention #
# ----------#

# # scaled_dot_product_attention
@pytest.mark.parametrize(
    "q_n_k_n_v_n_s_n_m_n_gt", [([[1.0]], [[2.0]], [[3.0]], 2.0, [[1.0]], [[3.0]])]
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_scaled_dot_product_attention(
    q_n_k_n_v_n_s_n_m_n_gt, dtype, tensor_fn, device, call
):
    q, k, v, scale, mask, ground_truth = q_n_k_n_v_n_s_n_m_n_gt
    # smoke test
    q = tensor_fn(q, dtype, device)
    k = tensor_fn(k, dtype, device)
    v = tensor_fn(v, dtype, device)
    mask = tensor_fn(mask, dtype, device)
    ret = ivy.scaled_dot_product_attention(q, k, v, scale, mask)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == q.shape
    # value test
    assert np.allclose(
        call(ivy.scaled_dot_product_attention, q, k, v, scale, mask),
        np.array(ground_truth),
    )


# multi_head_attention
@pytest.mark.parametrize(
    "x_n_s_n_m_n_c_n_gt",
    [([[3.0]], 2.0, [[1.0]], [[4.0, 5.0]], [[4.0, 5.0, 4.0, 5.0]])],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_multi_head_attention(x_n_s_n_m_n_c_n_gt, dtype, tensor_fn, device, call):
    x, scale, mask, context, ground_truth = x_n_s_n_m_n_c_n_gt
    # smoke test
    x = tensor_fn(x, dtype, device)
    context = tensor_fn(context, dtype, device)
    mask = tensor_fn(mask, dtype, device)
    fn = lambda x_, v: ivy.tile(x_, (1, 2))
    ret = ivy.multi_head_attention(x, scale, 2, context, mask, fn, fn, fn)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert list(ret.shape) == list(np.array(ground_truth).shape)
    # value test
    assert np.allclose(
        call(ivy.multi_head_attention, x, scale, 2, context, mask, fn, fn, fn),
        np.array(ground_truth),
    )


# Convolutions #
# -------------#

# conv1d
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_res",
    [
        (
            [[[0.0], [3.0], [0.0]]],
            [[[0.0]], [[1.0]], [[0.0]]],
            "SAME",
            [[[0.0], [3.0], [0.0]]],
        ),
        (
            [[[0.0], [3.0], [0.0]] for _ in range(5)],
            [[[0.0]], [[1.0]], [[0.0]]],
            "SAME",
            [[[0.0], [3.0], [0.0]] for _ in range(5)],
        ),
        ([[[0.0], [3.0], [0.0]]], [[[0.0]], [[1.0]], [[0.0]]], "VALID", [[[3.0]]]),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_conv1d(x_n_filters_n_pad_n_res, dtype, tensor_fn, device, call):
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype, device)
    filters = tensor_fn(filters, dtype, device)
    true_res = tensor_fn(true_res, dtype, device)
    ret = ivy.conv1d(x, filters, 1, padding)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv1d, x, filters, 1, padding), ivy.to_numpy(true_res))


# conv1d_transpose
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_outshp_n_res",
    [
        (
            [[[0.0], [3.0], [0.0]]],
            [[[0.0]], [[1.0]], [[0.0]]],
            "SAME",
            (1, 3, 1),
            [[[0.0], [3.0], [0.0]]],
        ),
        (
            [[[0.0], [3.0], [0.0]] for _ in range(5)],
            [[[0.0]], [[1.0]], [[0.0]]],
            "SAME",
            (5, 3, 1),
            [[[0.0], [3.0], [0.0]] for _ in range(5)],
        ),
        (
            [[[0.0], [3.0], [0.0]]],
            [[[0.0]], [[1.0]], [[0.0]]],
            "VALID",
            (1, 5, 1),
            [[[0.0], [0.0], [3.0], [0.0], [0.0]]],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_conv1d_transpose(
    x_n_filters_n_pad_n_outshp_n_res, dtype, tensor_fn, device, call
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d transpose does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d_transpose
        pytest.skip()
    x, filters, padding, output_shape, true_res = x_n_filters_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype, device)
    filters = tensor_fn(filters, dtype, device)
    true_res = tensor_fn(true_res, dtype, device)
    ret = ivy.conv1d_transpose(x, filters, 1, padding, output_shape)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(
        call(ivy.conv1d_transpose, x, filters, 1, padding, output_shape),
        ivy.to_numpy(true_res),
    )


# conv2d
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_res",
    [
        (
            [
                [
                    [[1.0], [2.0], [3.0], [4.0], [5.0]],
                    [[6.0], [7.0], [8.0], [9.0], [10.0]],
                    [[11.0], [12.0], [13.0], [14.0], [15.0]],
                    [[16.0], [17.0], [18.0], [19.0], [20.0]],
                    [[21.0], [22.0], [23.0], [24.0], [25.0]],
                ]
            ],
            [
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[1.0]], [[1.0]], [[1.0]]],
                [[[0.0]], [[1.0]], [[0.0]]],
            ],
            "SAME",
            [
                [
                    [[9.0], [13.0], [17.0], [21.0], [19.0]],
                    [[25.0], [35.0], [40.0], [45.0], [39.0]],
                    [[45.0], [60.0], [65.0], [70.0], [59.0]],
                    [[65.0], [85.0], [90.0], [95.0], [79.0]],
                    [[59.0], [83.0], [87.0], [91.0], [69.0]],
                ]
            ],
        ),
        (
            [
                [
                    [[1.0], [2.0], [3.0], [4.0], [5.0]],
                    [[6.0], [7.0], [8.0], [9.0], [10.0]],
                    [[11.0], [12.0], [13.0], [14.0], [15.0]],
                    [[16.0], [17.0], [18.0], [19.0], [20.0]],
                    [[21.0], [22.0], [23.0], [24.0], [25.0]],
                ]
                for _ in range(5)
            ],
            [
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[1.0]], [[1.0]], [[1.0]]],
                [[[0.0]], [[1.0]], [[0.0]]],
            ],
            "SAME",
            [
                [
                    [[9.0], [13.0], [17.0], [21.0], [19.0]],
                    [[25.0], [35.0], [40.0], [45.0], [39.0]],
                    [[45.0], [60.0], [65.0], [70.0], [59.0]],
                    [[65.0], [85.0], [90.0], [95.0], [79.0]],
                    [[59.0], [83.0], [87.0], [91.0], [69.0]],
                ]
                for _ in range(5)
            ],
        ),
        (
            [
                [
                    [[1.0], [2.0], [3.0], [4.0], [5.0]],
                    [[6.0], [7.0], [8.0], [9.0], [10.0]],
                    [[11.0], [12.0], [13.0], [14.0], [15.0]],
                    [[16.0], [17.0], [18.0], [19.0], [20.0]],
                    [[21.0], [22.0], [23.0], [24.0], [25.0]],
                ]
            ],
            [
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[1.0]], [[1.0]], [[1.0]]],
                [[[0.0]], [[1.0]], [[0.0]]],
            ],
            "VALID",
            [
                [
                    [[35.0], [40.0], [45.0]],
                    [[60.0], [65.0], [70.0]],
                    [[85.0], [90.0], [95.0]],
                ]
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_conv2d(x_n_filters_n_pad_n_res, dtype, tensor_fn, device, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv2d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    # smoke test
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype, device)
    filters = tensor_fn(filters, dtype, device)
    true_res = tensor_fn(true_res, dtype, device)
    ret = ivy.conv2d(x, filters, 1, padding)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv2d, x, filters, 1, padding), ivy.to_numpy(true_res))


# conv2d_transpose
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_outshp_n_res",
    [
        (
            [[[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]],
            [
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[1.0]], [[1.0]], [[1.0]]],
                [[[0.0]], [[1.0]], [[0.0]]],
            ],
            "SAME",
            (1, 3, 3, 1),
            [[[[0.0], [3.0], [0.0]], [[3.0], [3.0], [3.0]], [[0.0], [3.0], [0.0]]]],
        ),
        (
            [
                [[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]
                for _ in range(5)
            ],
            [
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[1.0]], [[1.0]], [[1.0]]],
                [[[0.0]], [[1.0]], [[0.0]]],
            ],
            "SAME",
            (5, 3, 3, 1),
            [
                [[[0.0], [3.0], [0.0]], [[3.0], [3.0], [3.0]], [[0.0], [3.0], [0.0]]]
                for _ in range(5)
            ],
        ),
        (
            [[[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]],
            [
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[1.0]], [[1.0]], [[1.0]]],
                [[[0.0]], [[1.0]], [[0.0]]],
            ],
            "VALID",
            (1, 5, 5, 1),
            [
                [
                    [[0.0], [0.0], [0.0], [0.0], [0.0]],
                    [[0.0], [0.0], [3.0], [0.0], [0.0]],
                    [[0.0], [3.0], [3.0], [3.0], [0.0]],
                    [[0.0], [0.0], [3.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0], [0.0], [0.0]],
                ]
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_conv2d_transpose(
    x_n_filters_n_pad_n_outshp_n_res, dtype, tensor_fn, device, call
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv2d transpose does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv2d_transpose
        pytest.skip()
    x, filters, padding, output_shape, true_res = x_n_filters_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype, device)
    filters = tensor_fn(filters, dtype, device)
    true_res = tensor_fn(true_res, dtype, device)
    ret = ivy.conv2d_transpose(x, filters, 1, padding, output_shape)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(
        call(ivy.conv2d_transpose, x, filters, 1, padding, output_shape),
        ivy.to_numpy(true_res),
    )


# depthwise_conv2d
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_res",
    [
        (
            [[[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]],
            [[[0.0], [1.0], [0.0]], [[1.0], [1.0], [1.0]], [[0.0], [1.0], [0.0]]],
            "SAME",
            [[[[0.0], [3.0], [0.0]], [[3.0], [3.0], [3.0]], [[0.0], [3.0], [0.0]]]],
        ),
        (
            [
                [[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]
                for _ in range(5)
            ],
            [[[0.0], [1.0], [0.0]], [[1.0], [1.0], [1.0]], [[0.0], [1.0], [0.0]]],
            "SAME",
            [
                [[[0.0], [3.0], [0.0]], [[3.0], [3.0], [3.0]], [[0.0], [3.0], [0.0]]]
                for _ in range(5)
            ],
        ),
        (
            [[[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]],
            [[[0.0], [1.0], [0.0]], [[1.0], [1.0], [1.0]], [[0.0], [1.0], [0.0]]],
            "VALID",
            [[[[3.0]]]],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_depthwise_conv2d(x_n_filters_n_pad_n_res, dtype, tensor_fn, device, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf depthwise conv2d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support depthwise 2d convolutions
        pytest.skip()
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype, device)
    filters = tensor_fn(filters, dtype, device)
    true_res = tensor_fn(true_res, dtype, device)
    ret = ivy.depthwise_conv2d(x, filters, 1, padding)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(
        call(ivy.depthwise_conv2d, x, filters, 1, padding), ivy.to_numpy(true_res)
    )


# conv3d
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_res",
    [
        (
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
            ],
            [
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
                [
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                ],
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
            ],
            "SAME",
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[3.0], [3.0], [3.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[3.0], [3.0], [3.0]],
                        [[3.0], [3.0], [3.0]],
                        [[3.0], [3.0], [3.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[3.0], [3.0], [3.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
            ],
        ),
        (
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
                for _ in range(5)
            ],
            [
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
                [
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                ],
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
            ],
            "SAME",
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[3.0], [3.0], [3.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[3.0], [3.0], [3.0]],
                        [[3.0], [3.0], [3.0]],
                        [[3.0], [3.0], [3.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[3.0], [3.0], [3.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
                for _ in range(5)
            ],
        ),
        (
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
            ],
            [
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
                [
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                ],
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
            ],
            "VALID",
            [[[[[3.0]]]]],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_conv3d(x_n_filters_n_pad_n_res, dtype, tensor_fn, device, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv3d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support 3d convolutions
        pytest.skip()
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x = tensor_fn(x, dtype, device)
    filters = tensor_fn(filters, dtype, device)
    true_res = tensor_fn(true_res, dtype, device)
    ret = ivy.conv3d(x, filters, 1, padding)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.conv3d, x, filters, 1, padding), ivy.to_numpy(true_res))


# conv3d_transpose
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_outshp_n_res",
    [
        (
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
            ],
            [
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
                [
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                ],
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
            ],
            "SAME",
            (1, 3, 3, 3, 1),
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[3.0], [3.0], [3.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[3.0], [3.0], [3.0]],
                        [[3.0], [3.0], [3.0]],
                        [[3.0], [3.0], [3.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[3.0], [3.0], [3.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
            ],
        ),
        (
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
                for _ in range(5)
            ],
            [
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
                [
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                ],
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
            ],
            "SAME",
            (5, 3, 3, 3, 1),
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[3.0], [3.0], [3.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[3.0], [3.0], [3.0]],
                        [[3.0], [3.0], [3.0]],
                        [[3.0], [3.0], [3.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[3.0], [3.0], [3.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
                for _ in range(5)
            ],
        ),
        (
            [
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                ]
            ],
            [
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
                [
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                ],
                [
                    [[[0.0]], [[0.0]], [[0.0]]],
                    [[[1.0]], [[1.0]], [[1.0]]],
                    [[[0.0]], [[0.0]], [[0.0]]],
                ],
            ],
            "VALID",
            (1, 5, 5, 5, 1),
            [
                [
                    [
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [3.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [3.0], [3.0], [0.0]],
                        [[0.0], [3.0], [3.0], [3.0], [0.0]],
                        [[0.0], [3.0], [3.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [3.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                    ],
                    [
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                    ],
                ]
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_conv3d_transpose(
    x_n_filters_n_pad_n_outshp_n_res, dtype, tensor_fn, device, call
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv3d transpose does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # numpy and jax do not yet support 3d transpose convolutions, and mxnet only
        # supports with CUDNN
        pytest.skip()
    if call in [helpers.mx_call] and "cpu" in device:
        # mxnet only supports 3d transpose convolutions with CUDNN
        pytest.skip()
    x, filters, padding, output_shape, true_res = x_n_filters_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype, device)
    filters = tensor_fn(filters, dtype, device)
    true_res = tensor_fn(true_res, dtype, device)
    ret = ivy.conv3d_transpose(x, filters, 1, padding, output_shape)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(
        call(ivy.conv3d_transpose, x, filters, 1, padding, output_shape),
        ivy.to_numpy(true_res),
    )


# LSTM #
# -----#

# lstm
@pytest.mark.parametrize(
    "b_t_ic_hc_otf_sctv",
    [
        (
            2,
            3,
            4,
            5,
            [0.93137765, 0.9587628, 0.96644664, 0.93137765, 0.9587628, 0.96644664],
            3.708991,
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_lstm(b_t_ic_hc_otf_sctv, dtype, tensor_fn, device, call):
    # smoke test
    (
        b,
        t,
        input_channels,
        hidden_channels,
        output_true_flat,
        state_c_true_val,
    ) = b_t_ic_hc_otf_sctv
    x = ivy.asarray(
        ivy.linspace(ivy.zeros([b, t]), ivy.ones([b, t]), input_channels), "float32"
    )
    init_h = ivy.ones([b, hidden_channels])
    init_c = ivy.ones([b, hidden_channels])
    kernel = ivy.variable(ivy.ones([input_channels, 4 * hidden_channels])) * 0.5
    recurrent_kernel = (
        ivy.variable(ivy.ones([hidden_channels, 4 * hidden_channels])) * 0.5
    )
    output, state_c = ivy.lstm_update(x, init_h, init_c, kernel, recurrent_kernel)
    # type test
    assert ivy.is_ivy_array(output)
    assert ivy.is_ivy_array(state_c)
    # cardinality test
    assert output.shape == (b, t, hidden_channels)
    assert state_c.shape == (b, hidden_channels)
    # value test
    output_true = np.tile(
        np.asarray(output_true_flat).reshape((b, t, 1)), (1, 1, hidden_channels)
    )
    state_c_true = np.ones([b, hidden_channels]) * state_c_true_val
    output, state_c = call(ivy.lstm_update, x, init_h, init_c, kernel, recurrent_kernel)
    assert np.allclose(output, output_true, atol=1e-6)
    assert np.allclose(state_c, state_c_true, atol=1e-6)
