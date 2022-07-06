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

    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "linear",
        test_rtol=1e-03,
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
@given(
    batch_shape=helpers.lists(
        st.integers(1, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    num_queries=st.integers(min_value=1, max_value=5),
    num_keys=st.integers(min_value=1, max_value=5),
    feat_dim=st.integers(min_value=1, max_value=5),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 5),
    num_positional_args=helpers.num_positional_args(
        fn_name="scaled_dot_product_attention"
    ),
    with_out=st.booleans(),
    native_array=helpers.list_of_length(st.booleans(), 5),
    container=helpers.list_of_length(st.booleans(), 5),
    instance_method=st.booleans(),
)
def test_scaled_dot_product_attention(
    batch_shape,
    num_queries,
    num_keys,
    feat_dim,
    dtype,
    as_variable,
    num_positional_args,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
    device,
):

    dtype = [dtype] * 5
    if fw == "torch" and "float16" in dtype:
        return
    q = np.random.uniform(size=batch_shape + [num_queries] + [feat_dim]).astype(
        dtype[0]
    )
    k = np.random.uniform(size=batch_shape + [num_keys] + [feat_dim]).astype(dtype[1])
    v = np.random.uniform(size=batch_shape + [num_keys] + [feat_dim]).astype(dtype[2])
    mask = np.random.uniform(size=batch_shape + [num_queries] + [num_keys]).astype(
        dtype[3]
    )
    scale = np.random.uniform(size=[1]).astype(dtype[4])

    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "scaled_dot_product_attention",
        q=q,
        k=k,
        v=v,
        scale=scale,
        mask=mask,
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
    helpers.test_function(
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
@given(
    array_shape=helpers.lists(st.integers(1, 5), min_size=3, max_size=3),
    filter_shape=st.integers(min_value=1, max_value=5),
    stride=st.integers(min_value=1, max_value=3),
    pad=st.sampled_from(["VALID", "SAME"]),
    output_shape=helpers.lists(st.integers(1, 5), min_size=4, max_size=4),
    data_format=st.sampled_from(["NHWC", "NCHW"]),
    dilations=st.integers(min_value=1, max_value=5),
    dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=helpers.num_positional_args(fn_name="conv2d_transpose"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_conv2d_transpose(
    array_shape,
    filter_shape,
    stride,
    pad,
    output_shape,
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
    if fw == "tensorflow" and "cpu" in device:
        # tf conv2d transpose does not work when CUDA is installed, but array is on CPU
        return
    if fw in ["numpy", "jax"]:
        # numpy and jax do not yet support conv2d_transpose
        return
    if fw == "torch" and ("16" in dtype[0] or "16" in dtype[1]):
        # not implemented for Half
        return
    x = np.random.uniform(size=array_shape).astype(dtype[0])
    x = np.expand_dims(x, (-1))
    filters = np.random.uniform(size=(filter_shape, filter_shape, 1, 1)).astype(
        dtype[1]
    )
    helpers.test_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "conv2d_transpose",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=tuple(output_shape),
        data_format=data_format,
        dilations=dilations,
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
@given(
    array_shape=helpers.lists(st.integers(1, 5), min_size=4, max_size=4),
    filter_shape=st.integers(min_value=1, max_value=5),
    stride=st.integers(min_value=1, max_value=3),
    pad=st.sampled_from(["VALID", "SAME"]),
    output_shape=helpers.lists(st.integers(1, 5), min_size=5, max_size=5),
    data_format=st.sampled_from(["NHWC", "NCHW"]),
    dilations=st.integers(min_value=1, max_value=5),
    dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=helpers.num_positional_args(fn_name="conv3d_transpose"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_conv3d_transpose(
    array_shape,
    filter_shape,
    stride,
    pad,
    output_shape,
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
    if fw == "tensorflow" and "cpu" in device:
        # tf conv3d transpose does not work when CUDA is installed, but array is on CPU
        return
    # smoke test
    if fw in ["numpy", "jax"]:
        # numpy and jax do not yet support 3d transpose convolutions, and mxnet only
        # supports with CUDNN
        return
    if fw == "mxnet" and "cpu" in device:
        # mxnet only supports 3d transpose convolutions with CUDNN
        return
    if fw == "torch" and ("16" in dtype[0] or "16" in dtype[1]):
        # not implemented for half
        return
    x = np.random.uniform(size=array_shape).astype(dtype[0])
    x = np.expand_dims(x, (-1))
    filters = np.random.uniform(
        size=(filter_shape, filter_shape, filter_shape, 1, 1)
    ).astype(dtype[1])
    helpers.test_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "conv3d_transpose",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


# LSTM #
# -----#

# lstm
@given(
    b=helpers.lists(st.integers(1, 5), min_size=4, max_size=4),
    t=st.integers(min_value=1, max_value=5),
    input_channel=st.integers(min_value=1, max_value=5),
    hidden_channel=st.integers(min_value=1, max_value=5),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 7),
    num_positional_args=helpers.num_positional_args(fn_name="lstm_update"),
    native_array=helpers.list_of_length(st.booleans(), 7),
    container=helpers.list_of_length(st.booleans(), 7),
    instance_method=st.booleans(),
)
def test_lstm(
    b,
    t,
    input_channel,
    hidden_channel,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype = [dtype] * 7

    # smoke test
    if fw == 'torch' and device == 'cpu' and 'float16' in dtype:
        # "sigmoid_cpu" not implemented for 'Half'
        return

    x = np.random.uniform(size=b + [t] + [input_channel]).astype(dtype[0])
    init_h = np.ones(b + [hidden_channel]).astype(dtype[1])
    init_c = np.ones(b + [hidden_channel]).astype(dtype[2])

    kernel = np.array(
        np.ones([input_channel, 4 * hidden_channel])
    ).astype(dtype[3]) * 0.5

    recurrent_kernel = (
        np.array(
            np.ones([hidden_channel, 4 * hidden_channel])
        ).astype(dtype[4]) * 0.5
    )

    bias = np.random.uniform(size=[4 * hidden_channel]).astype(dtype[5])

    recurrent_bias = np.random.uniform(
        size=[4 * hidden_channel]
    ).astype(dtype[6])

    helpers.test_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "lstm_update",
        x=x,
        init_h=init_h,
        init_c=init_c,
        kernel=kernel,
        recurrent_kernel=recurrent_kernel,
        bias=bias,
        recurrent_bias=recurrent_bias
    )
