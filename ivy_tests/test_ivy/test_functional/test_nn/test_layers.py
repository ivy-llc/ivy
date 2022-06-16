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
@given(
    outer_batch_shape=helpers.lists(st.integers(2, 5), min_size=1, max_size=3),
    inner_batch_shape=helpers.lists(st.integers(2, 5), min_size=1, max_size=3),
    num_out_feats=st.integers(min_value=1, max_value=5),
    num_in_feats=st.integers(min_value=1, max_value=5),
    dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 3),
    as_variable=helpers.list_of_length(st.booleans(), 3),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="linear"),
    native_array=helpers.list_of_length(st.booleans(), 3),
    container=helpers.list_of_length(st.booleans(), 3),
    instance_method=st.booleans(),
)
def test_linear(
    outer_batch_shape,
    inner_batch_shape,
    num_out_feats,
    num_in_feats,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):

    x = np.random.uniform(
        size=outer_batch_shape + inner_batch_shape + [num_in_feats]
    ).astype(dtype[0])
    weight = np.random.uniform(
        size=outer_batch_shape + [num_out_feats] + [num_in_feats]
    ).astype(dtype[1])
    bias = np.random.uniform(size=weight.shape[:-1]).astype(dtype[2])

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "linear",
        x=x,
        weight=weight,
        bias=bias,
    )


# Dropout #
# --------#

# dropout
@given(
    array_shape=helpers.lists(
        st.integers(1, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
)
def test_dropout(array_shape, dtype, as_variable, fw, device, call):
    if (fw == "tensorflow" or fw == "torch") and "int" in dtype:
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
    q = tensor_fn(q, dtype=dtype, device=device)
    k = tensor_fn(k, dtype=dtype, device=device)
    v = tensor_fn(v, dtype=dtype, device=device)
    mask = tensor_fn(mask, dtype=dtype, device=device)
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
    x = tensor_fn(x, dtype=dtype, device=device)
    context = tensor_fn(context, dtype=dtype, device=device)
    mask = tensor_fn(mask, dtype=dtype, device=device)
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
    x = tensor_fn(x, dtype=dtype, device=device)
    filters = tensor_fn(filters, dtype=dtype, device=device)
    true_res = tensor_fn(true_res, dtype=dtype, device=device)
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
    x = tensor_fn(x, dtype=dtype, device=device)
    filters = tensor_fn(filters, dtype=dtype, device=device)
    true_res = tensor_fn(true_res, dtype=dtype, device=device)
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
@given(
    array_shape=helpers.lists(st.integers(1, 5), min_size=3, max_size=3),
    filter_shape=st.integers(min_value=1, max_value=5),
    stride=st.integers(min_value=1, max_value=3),
    pad=st.sampled_from(["VALID", "SAME"]),
    data_format=st.sampled_from(["NHWC", "NCHW"]),
    dilations=st.integers(min_value=1, max_value=5),
    dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=helpers.num_positional_args(fn_name="conv2d"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_conv2d(
    array_shape,
    filter_shape,
    stride,
    pad,
    data_format,
    dilations,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    if fw in ["tensorflow"] and "cpu" in device:
        # tf conv2d does not work when CUDA is installed, but array is on CPU
        return

    x = np.random.uniform(size=array_shape).astype(dtype[0])
    x = np.expand_dims(x, (-1))
    filters = np.random.uniform(size=(filter_shape, filter_shape, 1, 1)).astype(
        dtype[1]
    )
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "conv2d",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


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
    filters = tensor_fn(filters, dtype=dtype, device=device)
    true_res = tensor_fn(true_res, dtype=dtype, device=device)
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
    x = tensor_fn(x, dtype=dtype, device=device)
    filters = tensor_fn(filters, dtype=dtype, device=device)
    true_res = tensor_fn(true_res, dtype=dtype, device=device)
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
    x = tensor_fn(x, dtype=dtype, device=device)
    filters = tensor_fn(filters, dtype=dtype, device=device)
    true_res = tensor_fn(true_res, dtype=dtype, device=device)
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
    x = tensor_fn(x, dtype=dtype, device=device)
    filters = tensor_fn(filters, dtype=dtype, device=device)
    true_res = tensor_fn(true_res, dtype=dtype, device=device)
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
