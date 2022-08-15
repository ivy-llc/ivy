"""Collection of tests for unified neural network layers."""

# global
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy
from ivy.container import Container
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np

# Linear #
# -------#


# linear
@given(
    bs_ic_oc_target=st.sampled_from(
        [
            (
                [1, 2],
                4,
                5,
                [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]],
            ),
        ]
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_linear_layer(
    bs_ic_oc_target, with_v, dtype, as_variable, device, compile_graph, call
):
    # smoke test
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    batch_shape, input_channels, output_channels, target = bs_ic_oc_target
    if as_variable:
        x = ivy.variable(
            ivy.asarray(
                ivy.linspace(
                    ivy.zeros(batch_shape),
                    ivy.ones(batch_shape),
                    input_channels,
                    axis=None,
                    device=device,
                ),
                dtype=dtype,
            )
        )
    else:
        x = ivy.asarray(
            ivy.linspace(
                ivy.zeros(batch_shape),
                ivy.ones(batch_shape),
                input_channels,
                axis=None,
                device=device,
            ),
            dtype=dtype,
        )

    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.asarray(
                np.random.uniform(-wlim, wlim, (output_channels, input_channels)),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(
            ivy.array(
                ivy.zeros([output_channels], device=device), dtype=dtype, device=device
            )
        )
        v = Container({"w": w, "b": b})
    else:
        v = None

    linear_layer = ivy.Linear(
        input_channels, output_channels, device=device, v=v, dtype=dtype
    )
    ret = linear_layer(x)

    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == tuple(batch_shape + [output_channels])

    # value test
    if not with_v:
        return
    assert np.allclose(
        call(linear_layer, x), np.array(target, dtype=dtype), rtol=tolerance_dict[dtype]
    )
    # compilation test
    if ivy.current_backend_str() == "torch":
        # pytest scripting does not **kwargs
        return


# Dropout #
# --------#

# dropout
@given(
    x_shape=st.sampled_from([(1, 2, 3)]),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_dropout_layer(x_shape, dtype, as_variable, device, compile_graph, call):
    # smoke test
    if as_variable:
        x = ivy.variable(ivy.array(ivy.random_uniform(shape=x_shape), dtype=dtype))
    else:
        x = ivy.array(ivy.random_uniform(shape=x_shape), dtype=dtype)
    dropout_layer = ivy.Dropout(0.9)
    ret = dropout_layer(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    ivy.seed(seed_value=0)
    assert np.min(call(dropout_layer, x)) == 0.0
    # compilation test
    if ivy.current_backend_str() == "torch":
        # pytest scripting does not **kwargs
        return


# Attention #
# ----------#

# multi_head_attention
@given(
    x_n_s_n_m_n_c_n_gt=st.sampled_from(
        [([[3.0]], 2.0, [[1.0]], [[4.0, 5.0]], [[0.8066473]])]
    ),
    with_v=st.booleans(),
    build_mode=st.sampled_from(["on_init", "explicit", "on_call"]),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_multi_head_attention_layer(
    x_n_s_n_m_n_c_n_gt,
    with_v,
    build_mode,
    as_variable,
    device,
    compile_graph,
    call,
    dtype,
):
    x, scale, mask, context, ground_truth = x_n_s_n_m_n_c_n_gt
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    # smoke test
    if as_variable:
        x = ivy.variable(ivy.array(x, dtype=dtype, device=device))
        context = ivy.variable(ivy.array(context, dtype=dtype, device=device))
        mask = ivy.variable(ivy.array(mask, dtype=dtype, device=device))
    else:
        x = ivy.array(x, dtype=dtype, device=device)
        context = ivy.array(context, dtype=dtype, device=device)
        mask = ivy.array(mask, dtype=dtype, device=device)
    query_dim = x.shape[-1]
    context_dim = context.shape[-1]
    if with_v:
        inner_dim = 64 * 8
        np.random.seed(0)
        wlim = (6 / (inner_dim + query_dim)) ** 0.5
        w_to_q = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, (inner_dim, query_dim)),
                dtype=dtype,
                device=device,
            )
        )
        wlim = (6 / (inner_dim * 2 + context_dim)) ** 0.5
        w_to_k = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, (inner_dim, context_dim)),
                dtype=dtype,
                device=device,
            )
        )
        w_to_v = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, (inner_dim, context_dim)),
                dtype=dtype,
                device=device,
            )
        )
        wlim = (6 / (query_dim + inner_dim)) ** 0.5
        w_to_out = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, (query_dim, inner_dim)),
                dtype=dtype,
                device=device,
            )
        )
        b_to_out = ivy.variable(ivy.zeros([query_dim], device=device))
        v = Container(
            {
                "to_q": {"w": w_to_q},
                "to_kv": {"k": {"w": w_to_k}, "v": {"w": w_to_v}},
                "to_out": {"submodules": {"v0": {"w": w_to_out, "b": b_to_out}}},
            }
        )
    else:
        v = None
    multi_head_attention_layer = ivy.MultiHeadAttention(
        query_dim,
        context_dim=context_dim,
        scale=scale,
        device=device,
        v=v,
        build_mode=build_mode,
    )
    if build_mode == "explicit":
        multi_head_attention_layer.build()
    ret = multi_head_attention_layer(x, context, mask)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert list(ret.shape) == list(np.array(ground_truth).shape)
    # value test
    if not with_v:
        return
    assert np.allclose(
        call(multi_head_attention_layer, x, context, mask),
        np.array(ground_truth),
        rtol=tolerance_dict[dtype],
    )
    # compilation test
    if ivy.current_backend_str() == "torch":
        # torch.jit compiled functions can't take variable number of arguments,
        # which torch.einsum takes
        return


# Convolutions #
# -------------#

# conv1d
@given(
    x_n_fs_n_pad_n_res=st.sampled_from(
        [
            (
                [[[0.0], [3.0], [0.0]]],
                3,
                "SAME",
                [[[1.0679483], [2.2363136], [0.5072848]]],
            ),
            (
                [[[0.0], [3.0], [0.0]] for _ in range(5)],
                3,
                "SAME",
                [[[1.0679483], [2.2363136], [0.5072848]] for _ in range(5)],
            ),
            ([[[0.0], [3.0], [0.0]]], 3, "VALID", [[[2.2363136]]]),
        ]
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv1d_layer(
    x_n_fs_n_pad_n_res, with_v, dtype, as_variable, device, compile_graph, call
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv1d
        # doesn't seem to be able to handle it
        return
    # smoke test
    x, filter_size, padding, target = x_n_fs_n_pad_n_res
    if as_variable:
        x = ivy.variable(ivy.array(x, dtype=dtype, device=device))
    else:
        x = ivy.array(x, dtype=dtype, device=device)

    target = np.array(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]

    width = x.shape[1]
    if with_v and not dtype:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, (filter_size, output_channels, input_channels)
                ),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, output_channels], device=device, dtype="float32")
        )
        v = Container({"w": w, "b": b})
    elif with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, (filter_size, output_channels, input_channels)
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([1, 1, output_channels], device=device, dtype=dtype))
        v = Container({"w": w, "b": b})
    else:
        v = None
    conv1d_layer = ivy.Conv1D(
        input_channels,
        output_channels,
        filter_size,
        1,
        padding,
        device=device,
        v=v,
        dtype=dtype,
    )
    ret = conv1d_layer(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    new_width = width if padding == "SAME" else width - filter_size + 1
    assert ret.shape == (batch_size, new_width, output_channels)
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv1d_layer, x), target, rtol=tolerance_dict[dtype])
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return


# conv1d transpose
@given(
    x_n_fs_n_pad_n_outshp_n_res=st.sampled_from(
        [
            (
                [[[0.0], [3.0], [0.0]]],
                3,
                "SAME",
                (1, 3, 1),
                [[[0.5072848], [2.2363136], [1.0679483]]],
            ),
            (
                [[[0.0], [3.0], [0.0]] for _ in range(5)],
                3,
                "SAME",
                (5, 3, 1),
                [[[0.5072848], [2.2363136], [1.0679483]] for _ in range(5)],
            ),
            (
                [[[0.0], [3.0], [0.0]]],
                3,
                "VALID",
                (1, 5, 1),
                [[[0.0], [0.5072848], [2.2363136], [1.0679483], [0.0]]],
            ),
        ]
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv1d_transpose_layer(
    x_n_fs_n_pad_n_outshp_n_res, with_v, dtype, as_variable, device, compile_graph, call
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv2d
        # doesn't seem to be able to handle it
        return
    # smoke test
    x, filter_size, padding, out_shape, target = x_n_fs_n_pad_n_outshp_n_res
    if as_variable:
        x = ivy.variable(ivy.array(x, dtype=dtype, device=device))
    else:
        x = ivy.array(x, dtype=dtype, device=device)

    target = np.array(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    width = x.shape[1]
    if with_v and not dtype:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, (filter_size, output_channels, input_channels)
                ),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, output_channels], device=device, dtype="float32")
        )
        v = Container({"w": w, "b": b})
    elif with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, (filter_size, output_channels, input_channels)
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([1, 1, output_channels], device=device, dtype=dtype))
        v = Container({"w": w, "b": b})
    else:
        v = None
    conv1d_trans_layer = ivy.Conv1DTranspose(
        input_channels,
        output_channels,
        filter_size,
        1,
        padding,
        output_shape=out_shape,
        device=device,
        v=v,
        dtype=dtype,
    )
    ret = conv1d_trans_layer(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    new_width = width if padding == "SAME" else width + filter_size - 1
    assert ret.shape == (batch_size, new_width, output_channels)
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv1d_trans_layer, x), target, rtol=tolerance_dict[dtype])
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return


# # conv2d
@given(
    x_n_fs_n_pad_n_res=st.sampled_from(
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
                [3, 3],
                "SAME",
                [
                    [
                        [
                            [20.132391],
                            [22.194885],
                            [25.338402],
                            [28.481918],
                            [10.9251585],
                        ],
                        [[37.611], [40.64039], [45.05442], [49.468452], [20.488476]],
                        [[59.139305], [62.71055], [67.12458], [71.53861], [30.220888]],
                        [[80.66761], [84.78071], [89.19474], [93.60877], [39.9533]],
                        [[23.54352], [30.85646], [32.52338], [34.1903], [15.24139]],
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
                [3, 3],
                "SAME",
                [
                    [
                        [
                            [20.132391],
                            [22.194885],
                            [25.338402],
                            [28.481918],
                            [10.9251585],
                        ],
                        [[37.611], [40.64039], [45.05442], [49.468452], [20.488476]],
                        [[59.139305], [62.71055], [67.12458], [71.53861], [30.220888]],
                        [[80.66761], [84.78071], [89.19474], [93.60877], [39.9533]],
                        [[23.54352], [30.85646], [32.52338], [34.1903], [15.24139]],
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
                [3, 3],
                "VALID",
                [
                    [
                        [[40.64039], [45.05442], [49.468452]],
                        [[62.71055], [67.12458], [71.53861]],
                        [[84.78071], [89.19474], [93.60877]],
                    ]
                ],
            ),
        ],
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv2d_layer(
    x_n_fs_n_pad_n_res, with_v, dtype, as_variable, device, compile_graph, call
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv2d
        # doesn't seem to be able to handle it
        return
    # smoke test
    x, filter_shape, padding, target = x_n_fs_n_pad_n_res
    if as_variable:
        x = ivy.variable(ivy.array(x, dtype=dtype, device=device))
    else:
        x = ivy.array(x, dtype=dtype, device=device)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:3])
    if with_v and not dtype:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, output_channels], device=device, dtype="float32")
        )
        v = Container({"w": w, "b": b})
    elif with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, output_channels], device=device, dtype=dtype)
        )
        v = Container({"w": w, "b": b})
    else:
        v = None
    conv2d_layer = ivy.Conv2D(
        input_channels,
        output_channels,
        filter_shape,
        1,
        padding,
        device=device,
        v=v,
        dtype=dtype,
    )
    ret = conv2d_layer(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    new_shape = (
        input_shape
        if padding == "SAME"
        else [item - filter_shape[i] + 1 for i, item in enumerate(input_shape)]
    )
    assert ret.shape == tuple([batch_size] + new_shape + [output_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv2d_layer, x), target, rtol=tolerance_dict[dtype])
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return


# # conv2d transpose
@given(
    x_n_fs_n_pad_n_outshp_n_res=st.sampled_from(
        [
            (
                [[[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]],
                [3, 3],
                "SAME",
                (1, 3, 3, 1),
                [
                    [
                        [[0.5072848], [2.2363136], [1.0679483]],
                        [[0.46643972], [-0.7934026], [1.516176]],
                        [[-0.64861274], [4.0714245], [4.818525]],
                    ]
                ],
            ),
            (
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ]
                    for _ in range(5)
                ],
                [3, 3],
                "SAME",
                (5, 3, 3, 1),
                [
                    [
                        [[0.5072848], [2.2363136], [1.0679483]],
                        [[0.46643972], [-0.7934026], [1.516176]],
                        [[-0.64861274], [4.0714245], [4.818525]],
                    ]
                    for _ in range(5)
                ],
            ),
            (
                [[[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]],
                [3, 3],
                "VALID",
                (1, 5, 5, 1),
                [
                    [
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.5072848], [2.2363136], [1.0679483], [0.0]],
                        [[0.0], [0.46643972], [-0.7934026], [1.516176], [0.0]],
                        [[0.0], [-0.64861274], [4.0714245], [4.818525], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                    ]
                ],
            ),
        ]
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv2d_transpose_layer(
    x_n_fs_n_pad_n_outshp_n_res, with_v, dtype, as_variable, device, compile_graph, call
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv_transpose2d
        # doesn't seem to be able to handle it
        return
    # smoke test
    x, filter_shape, padding, out_shape, target = x_n_fs_n_pad_n_outshp_n_res
    if as_variable:
        x = ivy.variable(ivy.array(x, dtype=dtype, device=device))
    else:
        x = ivy.array(x, dtype=dtype, device=device)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:3])
    if with_v and not dtype:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, output_channels], device=device, dtype="float32")
        )
        v = Container({"w": w, "b": b})
    elif with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, output_channels], device=device, dtype=dtype)
        )
        v = Container({"w": w, "b": b})
    else:
        v = None
    conv2d_transpose_layer = ivy.Conv2DTranspose(
        input_channels,
        output_channels,
        filter_shape,
        1,
        padding,
        output_shape=out_shape,
        device=device,
        v=v,
        dtype=dtype,
    )
    ret = conv2d_transpose_layer(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    new_shape = (
        input_shape
        if padding == "SAME"
        else [item + filter_shape[i] - 1 for i, item in enumerate(input_shape)]
    )
    assert ret.shape == tuple([batch_size] + new_shape + [output_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(
        call(conv2d_transpose_layer, x), target, rtol=tolerance_dict[dtype]
    )
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return


# # depthwise conv2d
@given(
    x_n_fs_n_pad_n_res=st.sampled_from(
        [
            (
                [[[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]],
                [3, 3],
                "SAME",
                [
                    [
                        [[4.818525], [4.0714245], [-0.64861274]],
                        [[1.516176], [-0.7934026], [0.46643972]],
                        [[1.0679483], [2.2363136], [0.5072848]],
                    ]
                ],
            ),
            (
                [
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [3.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ]
                    for _ in range(5)
                ],
                [3, 3],
                "SAME",
                [
                    [
                        [[4.818525], [4.0714245], [-0.64861274]],
                        [[1.516176], [-0.7934026], [0.46643972]],
                        [[1.0679483], [2.2363136], [0.5072848]],
                    ]
                    for _ in range(5)
                ],
            ),
            (
                [[[[0.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [0.0]]]],
                [3, 3],
                "VALID",
                [[[[-0.7934026]]]],
            ),
        ]
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_depthwise_conv2d_layer(
    x_n_fs_n_pad_n_res, with_v, dtype, as_variable, device, compile_graph, call
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv2d
        # doesn't seem to be able to handle it
        return
    # smoke test
    x, filter_shape, padding, target = x_n_fs_n_pad_n_res
    if as_variable:
        x = ivy.variable(ivy.array(x, dtype=dtype, device=device))
    else:
        x = ivy.array(x, dtype=dtype, device=device)
    target = np.asarray(target)
    num_channels = x.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:3])
    if with_v and not dtype:
        np.random.seed(0)
        wlim = (6 / (num_channels * 2)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, tuple(filter_shape + [num_channels])),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, num_channels], device=device, dtype="float32")
        )
        v = Container({"w": w, "b": b})
    elif with_v:
        np.random.seed(0)
        wlim = (6 / (num_channels * 2)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, tuple(filter_shape + [num_channels])),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([1, 1, num_channels], device=device, dtype=dtype))
        v = Container({"w": w, "b": b})

    else:
        v = None
    depthwise_conv2d_layer = ivy.DepthwiseConv2D(
        num_channels, filter_shape, 1, padding, device=device, v=v, dtype=dtype
    )
    ret = depthwise_conv2d_layer(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    new_shape = (
        input_shape
        if padding == "SAME"
        else [item - filter_shape[i] + 1 for i, item in enumerate(input_shape)]
    )
    assert ret.shape == tuple([batch_size] + new_shape + [num_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(
        call(depthwise_conv2d_layer, x), target, rtol=tolerance_dict[dtype]
    )


#
# # conv3d
@given(
    x_n_fs_n_pad_n_res=st.sampled_from(
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
                [3, 3, 3],
                "SAME",
                [
                    [
                        [
                            [[-3.7063813], [1.4541019], [-3.9670086]],
                            [[2.9153447], [-0.4003182], [3.108947]],
                            [[4.9739475], [3.8452792], [2.8906898]],
                        ],
                        [
                            [[3.456687], [-4.986037], [-4.290678]],
                            [[-4.457924], [4.4229302], [0.70713985]],
                            [[0.3002848], [3.0316954], [-1.2113112]],
                        ],
                        [
                            [[4.818525], [4.0714245], [-0.64861274]],
                            [[1.516176], [-0.7934026], [0.46643972]],
                            [[1.0679483], [2.2363136], [0.5072848]],
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
                [3, 3, 3],
                "SAME",
                [
                    [
                        [
                            [[-3.7063813], [1.4541019], [-3.9670086]],
                            [[2.9153447], [-0.4003182], [3.108947]],
                            [[4.9739475], [3.8452792], [2.8906898]],
                        ],
                        [
                            [[3.456687], [-4.986037], [-4.290678]],
                            [[-4.457924], [4.4229302], [0.70713985]],
                            [[0.3002848], [3.0316954], [-1.2113112]],
                        ],
                        [
                            [[4.818525], [4.0714245], [-0.64861274]],
                            [[1.516176], [-0.7934026], [0.46643972]],
                            [[1.0679483], [2.2363136], [0.5072848]],
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
                [3, 3, 3],
                "VALID",
                [[[[[4.4229302]]]]],
            ),
        ]
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv3d_layer(
    x_n_fs_n_pad_n_res, with_v, dtype, as_variable, device, compile_graph, call
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv3d
        # doesn't seem to be able to handle it
        return
    # smoke test
    x, filter_shape, padding, target = x_n_fs_n_pad_n_res
    if as_variable:
        x = ivy.variable(ivy.array(x, dtype=dtype, device=device))
    else:
        x = ivy.array(x, dtype=dtype, device=device)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:4])
    if with_v and not dtype:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, 1, output_channels], device=device, dtype="float32")
        )
        v = Container({"w": w, "b": b})
    elif with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, 1, output_channels], device=device, dtype=dtype)
        )
        v = Container({"w": w, "b": b})
    else:
        v = None
    conv3d_layer = ivy.Conv3D(
        input_channels,
        output_channels,
        filter_shape,
        1,
        padding,
        device=device,
        v=v,
        dtype=dtype,
    )
    ret = conv3d_layer(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    new_shape = (
        input_shape
        if padding == "SAME"
        else [item - filter_shape[i] + 1 for i, item in enumerate(input_shape)]
    )
    assert ret.shape == tuple([batch_size] + new_shape + [output_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv3d_layer, x), target, rtol=tolerance_dict[dtype])


# # conv3d transpose
@given(
    x_n_fs_n_pad_n_outshp_n_res=st.sampled_from(
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
                [3, 3, 3],
                "SAME",
                (1, 3, 3, 3, 1),
                [
                    [
                        [
                            [[0.5072848], [2.2363136], [1.0679483]],
                            [[0.46643972], [-0.7934026], [1.516176]],
                            [[-0.64861274], [4.0714245], [4.818525]],
                        ],
                        [
                            [[-1.2113112], [3.0316954], [0.3002848]],
                            [[0.70713985], [4.4229302], [-4.457924]],
                            [[-4.290678], [-4.986037], [3.456687]],
                        ],
                        [
                            [[2.8906898], [3.8452792], [4.9739475]],
                            [[3.108947], [-0.4003182], [2.9153447]],
                            [[-3.9670086], [1.4541019], [-3.7063813]],
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
                [3, 3, 3],
                "SAME",
                (5, 3, 3, 3, 1),
                [
                    [
                        [
                            [[0.5072848], [2.2363136], [1.0679483]],
                            [[0.46643972], [-0.7934026], [1.516176]],
                            [[-0.64861274], [4.0714245], [4.818525]],
                        ],
                        [
                            [[-1.2113112], [3.0316954], [0.3002848]],
                            [[0.70713985], [4.4229302], [-4.457924]],
                            [[-4.290678], [-4.986037], [3.456687]],
                        ],
                        [
                            [[2.8906898], [3.8452792], [4.9739475]],
                            [[3.108947], [-0.4003182], [2.9153447]],
                            [[-3.9670086], [1.4541019], [-3.7063813]],
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
                [3, 3, 3],
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
                            [[0.0], [0.5072848], [2.2363136], [1.0679483], [0.0]],
                            [[0.0], [0.46643972], [-0.7934026], [1.516176], [0.0]],
                            [[0.0], [-0.64861274], [4.0714245], [4.818525], [0.0]],
                            [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        ],
                        [
                            [[0.0], [0.0], [0.0], [0.0], [0.0]],
                            [[0.0], [-1.2113112], [3.0316954], [0.3002848], [0.0]],
                            [[0.0], [0.70713985], [4.4229302], [-4.457924], [0.0]],
                            [[0.0], [-4.290678], [-4.986037], [3.456687], [0.0]],
                            [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        ],
                        [
                            [[0.0], [0.0], [0.0], [0.0], [0.0]],
                            [[0.0], [2.8906898], [3.8452792], [4.9739475], [0.0]],
                            [[0.0], [3.108947], [-0.4003182], [2.9153447], [0.0]],
                            [[0.0], [-3.9670086], [1.4541019], [-3.7063813], [0.0]],
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
        ]
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv3d_transpose_layer(
    x_n_fs_n_pad_n_outshp_n_res, with_v, dtype, as_variable, device, compile_graph, call
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.mx_call] and "cpu" in device:
        # mxnet only supports 3d transpose convolutions with CUDNN
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv_transpose3d
        # doesn't seem to be able to handle it
        return
    # smoke test
    x, filter_shape, padding, out_shape, target = x_n_fs_n_pad_n_outshp_n_res
    if as_variable:
        x = ivy.variable(ivy.array(x, dtype=dtype, device=device))
    x = ivy.array(x, dtype=dtype, device=device)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:4])
    if with_v and not dtype:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, 1, output_channels], device=device, dtype="float32")
        )
        v = Container({"w": w, "b": b})
    elif with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, 1, output_channels], device=device, dtype=dtype)
        )
        v = Container({"w": w, "b": b})
    else:
        v = None
    conv3d_transpose_layer = ivy.Conv3DTranspose(
        input_channels,
        output_channels,
        filter_shape,
        1,
        padding,
        output_shape=out_shape,
        device=device,
        v=v,
        dtype=dtype,
    )
    ret = conv3d_transpose_layer(x)

    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    new_shape = (
        input_shape
        if padding == "SAME"
        else [item + filter_shape[i] - 1 for i, item in enumerate(input_shape)]
    )
    assert ret.shape == tuple([batch_size] + new_shape + [output_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(
        call(conv3d_transpose_layer, x), target, rtol=tolerance_dict[dtype]
    )


#
# # LSTM #
@given(
    b_t_ic_hc_otf_sctv=st.sampled_from(
        [
            (
                2,
                3,
                4,
                5,
                [0.93137765, 0.9587628, 0.96644664, 0.93137765, 0.9587628, 0.96644664],
                3.708991,
            ),
        ]
    ),
    with_v=st.booleans(),
    with_initial_state=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_lstm_layer(
    b_t_ic_hc_otf_sctv,
    with_v,
    with_initial_state,
    dtype,
    as_variable,
    device,
    compile_graph,
    call,
):
    if call in [helpers.torch_call] and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv3d
        # doesn't seem to be able to handle it
        return
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    # smoke test
    (
        b,
        t,
        input_channels,
        hidden_channels,
        output_true_flat,
        state_c_true_val,
    ) = b_t_ic_hc_otf_sctv
    if as_variable:
        x = ivy.variable(
            ivy.asarray(
                ivy.linspace(ivy.zeros([b, t]), ivy.ones([b, t]), input_channels),
                dtype=dtype,
            )
        )
    else:
        x = ivy.asarray(
            ivy.linspace(ivy.zeros([b, t]), ivy.ones([b, t]), input_channels),
            dtype=dtype,
        )
    if with_initial_state:
        init_h = ivy.ones([b, hidden_channels], dtype=dtype)
        init_c = ivy.ones([b, hidden_channels], dtype=dtype)
        initial_state = ([init_h], [init_c])
    else:
        initial_state = None
    if with_v:
        kernel = ivy.variable(
            ivy.ones([input_channels, 4 * hidden_channels], device=device, dtype=dtype)
            * 0.5
        )
        recurrent_kernel = ivy.variable(
            ivy.ones([hidden_channels, 4 * hidden_channels], device=device, dtype=dtype)
            * 0.5
        )
        v = Container(
            {
                "input": {"layer_0": {"w": kernel}},
                "recurrent": {"layer_0": {"w": recurrent_kernel}},
            }
        )
    else:
        v = None
    lstm_layer = ivy.LSTM(
        input_channels, hidden_channels, device=device, v=v, dtype=dtype
    )
    output, (state_h, state_c) = lstm_layer(x, initial_state=initial_state)
    # type test
    assert ivy.is_ivy_array(output)
    assert ivy.is_ivy_array(state_h[0])
    assert ivy.is_ivy_array(state_c[0])
    # cardinality test
    assert output.shape == (b, t, hidden_channels)
    assert state_h[0].shape == (b, hidden_channels)
    assert state_c[0].shape == (b, hidden_channels)
    # value test
    if not with_v or not with_initial_state:
        return
    output_true = np.tile(
        np.asarray(output_true_flat).reshape((b, t, 1)), (1, 1, hidden_channels)
    )
    state_c_true = np.ones([b, hidden_channels]) * state_c_true_val
    output, (state_h, state_c) = call(lstm_layer, x, initial_state=initial_state)
    assert np.allclose(output, output_true, atol=1e-6, rtol=tolerance_dict[dtype])
    assert np.allclose(state_c, state_c_true, atol=1e-6, rtol=tolerance_dict[dtype])


# # Sequential #
@given(
    bs_c_target=st.sampled_from(
        [
            (
                [1, 2],
                5,
                [
                    [
                        [-0.34784955, 0.47909835, 0.7241975, -0.82175905, -0.43836743],
                        [-0.34784955, 0.47909835, 0.7241975, -0.82175905, -0.43836743],
                    ]
                ],
            )
        ]
    ),
    with_v=st.booleans(),
    seq_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_sequential_layer(
    bs_c_target, with_v, seq_v, dtype, as_variable, device, compile_graph, call
):
    # smoke test
    batch_shape, channels, target = bs_c_target
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if as_variable:
        x = ivy.variable(
            ivy.asarray(
                ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), channels),
                dtype=dtype,
            )
        )
    else:
        x = ivy.asarray(
            ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), channels),
            dtype=dtype,
        )
    if with_v:
        np.random.seed(0)
        wlim = (6 / (channels + channels)) ** 0.5
        v = Container(
            {
                "submodules": {
                    "v0": {
                        "w": ivy.variable(
                            ivy.array(
                                np.random.uniform(-wlim, wlim, (channels, channels)),
                                dtype=dtype,
                                device=device,
                            )
                        ),
                        "b": ivy.variable(
                            ivy.zeros([channels], device=device, dtype=dtype)
                        ),
                    },
                    "v2": {
                        "w": ivy.variable(
                            ivy.array(
                                np.random.uniform(-wlim, wlim, (channels, channels)),
                                dtype=dtype,
                                device=device,
                            )
                        ),
                        "b": ivy.variable(
                            ivy.zeros([channels], device=device, dtype=dtype)
                        ),
                    },
                }
            }
        )
    else:
        v = None
    if seq_v:
        seq = ivy.Sequential(
            ivy.Linear(channels, channels, device=device, dtype=dtype),
            ivy.Dropout(0.0),
            ivy.Linear(channels, channels, device=device, dtype=dtype),
            device=device,
            v=v if with_v else None,
            dtype=dtype,
        )
    else:
        seq = ivy.Sequential(
            ivy.Linear(
                channels,
                channels,
                device=device,
                v=v["submodules"]["v0"] if with_v else None,
                dtype=dtype,
            ),
            ivy.Dropout(0.0),
            ivy.Linear(
                channels,
                channels,
                device=device,
                v=v["submodules"]["v2"] if with_v else None,
                dtype=dtype,
            ),
            device=device,
        )
    ret = seq(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == tuple(batch_shape + [channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(seq, x), np.array(target), rtol=tolerance_dict[dtype])
