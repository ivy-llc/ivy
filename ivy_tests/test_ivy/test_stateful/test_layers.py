"""Collection of tests for unified neural network layers."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
from ivy.container import Container
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# Helpers #
# --------#

all_constant_initializers = (ivy.Zeros, ivy.Ones)
all_uniform_initializers = (ivy.GlorotUniform, ivy.FirstLayerSiren, ivy.Siren)
all_gaussian_initializers = (ivy.KaimingNormal, ivy.Siren)
all_initializers = (
    all_constant_initializers + all_uniform_initializers + all_gaussian_initializers
)


@st.composite
def _sample_initializer(draw):
    return draw(st.sampled_from(all_initializers))()


# Linear #
# -------#


@st.composite
def _bias_flag_and_initializer(draw):
    with_bias = draw(st.booleans())
    if with_bias:
        return with_bias, draw(_sample_initializer())
    return with_bias, None


@st.composite
def _input_channels_and_dtype_and_values(draw):
    input_channels = draw(st.integers(min_value=1, max_value=10))
    x_shape = draw(helpers.get_shape()) + (input_channels,)
    dtype, vals = draw(
        helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_float_dtypes, shape=x_shape
        )
    )
    return input_channels, dtype, vals


# linear
@handle_cmd_line_args
@given(
    ic_n_dtype_n_vals=_input_channels_and_dtype_and_values(),
    output_channels=st.shared(
        st.integers(min_value=1, max_value=10), key="output_channels"
    ),
    weight_initializer=_sample_initializer(),
    wb_n_b_init=_bias_flag_and_initializer(),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="Linear.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="Linear._forward"),
    seed=helpers.seed(),
)
def test_linear_layer(
    *,
    ic_n_dtype_n_vals,
    output_channels,
    weight_initializer,
    wb_n_b_init,
    init_with_v,
    method_with_v,
    num_positional_args_init,
    num_positional_args_method,
    seed,
    as_variable,
    native_array,
    container,
    fw,
    device,
):
    ivy.seed(seed_value=seed)
    input_channels, input_dtype, x = ic_n_dtype_n_vals
    with_bias, bias_initializer = wb_n_b_init
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        all_as_kwargs_np_init={
            "input_channels": input_channels,
            "output_channels": output_channels,
            "weight_initializer": weight_initializer,
            "bias_initializer": bias_initializer,
            "with_bias": with_bias,
            "device": device,
            "dtype": input_dtype,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"x": np.asarray(x, dtype=input_dtype)},
        fw=fw,
        class_name="Linear",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
    )


# Dropout #
# --------#

# dropout
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    prob=helpers.floats(min_value=0, max_value=0.9, width=64),
    scale=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="Dropout.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="Dropout._forward"),
)
def test_dropout_layer(
    *,
    dtype_and_x,
    prob,
    scale,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
    device,
):
    input_dtype, x = dtype_and_x
    x = np.asarray(x, dtype=input_dtype)
    ret = helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        all_as_kwargs_np_init={
            "prob": prob,
            "scale": scale,
            "dtype": input_dtype,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"inputs": x},
        fw=fw,
        class_name="Dropout",
        test_values=False,
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    for u in ret:
        # cardinality test
        assert u.shape == x.shape


# Attention #
# ----------#
@st.composite
def x_and_mha(draw, dtypes):
    dtype = draw(dtypes)
    query_dim = draw(helpers.ints(min_value=1, max_value=10))
    num_heads = draw(helpers.ints(min_value=1, max_value=100))
    head_dim = draw(helpers.ints(min_value=1, max_value=100))
    dropout_rate = draw(helpers.ints(min_value=1, max_value=3))
    context_dim = draw(helpers.ints(min_value=1, max_value=3))
    with_to_q_fn=st
    x_mha_shape = (num_queries,) + (feat_dim * num_heads,)
    context_shape = (num_keys,) + (2 * feat_dim * num_heads,)
    mask_shape = (num_queries,) + (num_keys,)
    scale = draw(helpers.floats(min_value=0.1, max_value=1, width=64))
    x_mha = draw(
        helpers.array_values(
            dtype=dtype,
            shape=x_mha_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    context = draw(
        helpers.array_values(
            dtype=dtype,
            shape=context_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    mask = draw(
        helpers.array_values(
            dtype=dtype,
            shape=mask_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    return dtype, x_mha, scale, num_heads, context, mask

# multi_head_attention
@given(
    dtype_qdim=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    num_heads=st.integers(min_value=1, max_value=100),
    head_dim=st.integers(min_value=1, max_value=100),
    dropout_rate=st.integers(min_value=1, max_value=3),
    context_dim=st.integers(min_value=1, max_value=3),
    with_to_q_fn=st.booleans(),
    with_to_kv_fn=st.booleans(),
    with_to_out_fn=st.booleans(),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="multi_head_attention.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="multi_head_attention._forward"),
    build_mode=st.sampled_from(["on_init", "explicit", "on_call"]),
)
def test_multi_head_attention_layer(
    dtype_qdim,
    num_heads,
    head_dim,
    dropout_rate,
    context_dim,
    with_to_q_fn,
    with_to_kv_fn,
    with_to_out_fn,
    init_with_v,
    method_with_v,
    num_positional_args_init,
    num_positional_args_method,
    build_mode,
    as_variable,
    native_array,
    container,
    fw,
    device,
):
    input_dtype, query_dim = dtype_qdim
    num_heads = num_heads
    head_dim = head_dim
    dropout_rate = dropout_rate
    context_dim = context_dim
    with_to_q_fn = with_to_q_fn
    with_to_kv_fn = with_to_kv_fn
    with_to_out_fn = with_to_out_fn
    build_mode = build_mode
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        all_as_kwargs_np_init={
            "query_dim": query_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dropout_rate": dropout_rate,
            "context_dim": context_dim,
            "with_to_q_fn": with_to_q_fn,
            "with_to_kv_fn": with_to_kv_fn,
            "with_to_out_fn":with_to_out_fn,
            "build_mode":build_mode,
            "device": device,
            "dtype": input_dtype,

        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"inputs": np.asarray(query_dim, dtype=input_dtype)},
        fw=fw,
        class_name="multi_head_attention",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
    )

    # x, scale, mask, context, ground_truth = x_n_s_n_m_n_c_n_gt
    # tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    # # smoke test
    # if as_variable:
    #     x = ivy.variable(ivy.array(x, dtype=dtype, device=device))
    #     context = ivy.variable(ivy.array(context, dtype=dtype, device=device))
    #     mask = ivy.variable(ivy.array(mask, dtype=dtype, device=device))
    # else:
    #     x = ivy.array(x, dtype=dtype, device=device)
    #     context = ivy.array(context, dtype=dtype, device=device)
    #     mask = ivy.array(mask, dtype=dtype, device=device)
    # query_dim = x.shape[-1]
    # context_dim = context.shape[-1]
    # if with_v:
    #     inner_dim = 64 * 8
    #     np.random.seed(0)
    #     wlim = (6 / (inner_dim + query_dim)) ** 0.5
    #     w_to_q = ivy.variable(
    #         ivy.array(
    #             np.random.uniform(-wlim, wlim, (inner_dim, query_dim)),
    #             dtype=dtype,
    #             device=device,
    #         )
    #     )
    #     wlim = (6 / (inner_dim * 2 + context_dim)) ** 0.5
    #     w_to_k = ivy.variable(
    #         ivy.array(
    #             np.random.uniform(-wlim, wlim, (inner_dim, context_dim)),
    #             dtype=dtype,
    #             device=device,
    #         )
    #     )
    #     w_to_v = ivy.variable(
    #         ivy.array(
    #             np.random.uniform(-wlim, wlim, (inner_dim, context_dim)),
    #             dtype=dtype,
    #             device=device,
    #         )
    #     )
    #     wlim = (6 / (query_dim + inner_dim)) ** 0.5
    #     w_to_out = ivy.variable(
    #         ivy.array(
    #             np.random.uniform(-wlim, wlim, (query_dim, inner_dim)),
    #             dtype=dtype,
    #             device=device,
    #         )
    #     )
    #     b_to_out = ivy.variable(ivy.zeros([query_dim], device=device))
    #     v = Container(
    #         {
    #             "to_q": {"w": w_to_q},
    #             "to_kv": {"k": {"w": w_to_k}, "v": {"w": w_to_v}},
    #             "to_out": {"submodules": {"v0": {"w": w_to_out, "b": b_to_out}}},
    #         }
    #     )
    # else:
    #     v = None
    # multi_head_attention_layer = ivy.MultiHeadAttention(
    #     query_dim,
    #     context_dim=context_dim,
    #     scale=scale,
    #     device=device,
    #     v=v,
    #     build_mode=build_mode,
    # )
    # if build_mode == "explicit":
    #     multi_head_attention_layer.build()
    # ret = multi_head_attention_layer(x, context, mask)
    # # type test
    # assert ivy.is_ivy_array(ret)
    # # cardinality test
    # assert list(ret.shape) == list(np.array(ground_truth).shape)
    # # value test
    # if not with_v:
    #     return
    # assert np.allclose(
    #     ivy.to_numpy(multi_head_attention_layer(x, context, mask)),
    #     np.array(ground_truth),
    #     rtol=tolerance_dict[dtype],
    # )
    # # compilation test
    # if ivy.current_backend_str() == "torch":
    #     # torch.jit compiled functions can't take variable number of arguments,
    #     # which torch.einsum takes
    #     return


# Convolutions #
# -------------#


@st.composite
def _x_ic_oc_f_d_df(draw, dim: int = 2, transpose: bool = False):
    strides = draw(st.integers(min_value=1, max_value=4))
    padding = draw(st.sampled_from(["SAME", "VALID"]))
    batch_size = draw(st.integers(1, 5))
    filter_shape = draw(helpers.get_shape(min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5))
    input_channels = draw(st.integers(1, 5))
    output_channels = draw(st.integers(1, 5))
    dilations = draw(st.integers(1, 4))
    x_dim = []
    for i in range(dim):
        min_x = filter_shape[i] + (filter_shape[i] - 1) * (dilations - 1)
        x_dim.append(draw(st.integers(min_x, 100)))
    if dim == 2:
        data_format = draw(st.sampled_from(["NCHW"]))
    elif dim == 1:
        data_format = draw(st.sampled_from(["NWC", "NCW"]))
    else:
        data_format = draw(st.sampled_from(["NDHWC", "NCDHW"]))
    if data_format == "NHWC" or data_format == "NWC" or data_format == "NDHWC":
        x_shape = [batch_size] + x_dim + [input_channels]
    else:
        x_shape = [batch_size] + [input_channels] + x_dim

    if transpose:
        output_shape = []
        for i in range(dim):
            output_shape.append(ivy.deconv_length(x_dim[i], strides, filter_shape[i], padding, dilations))
        # output_shape = [batch_size] + output_shape + [output_channels]
    if dim == 1:
        filter_shape = filter_shape[0]
    dtype, vals = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float", full=True), shape=x_shape
        )
    )
    if transpose:
        return dtype, vals, input_channels, output_channels, filter_shape, strides, dilations, data_format, padding, output_shape
    return dtype, vals, input_channels, output_channels, filter_shape, strides, dilations, data_format, padding



# conv1d
@handle_cmd_line_args
@given(
    _x_ic_oc_f_s_d_df_p=_x_ic_oc_f_d_df(dim=1),
    weight_initializer=_sample_initializer(),
    bias_initializer=_sample_initializer(),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="Conv1D.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="Conv1D._forward"),
)
def test_conv1d_layer(
    _x_ic_oc_f_s_d_df_p,
    weight_initializer,
    bias_initializer,
    init_with_v,
    method_with_v,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
    device,
):
    input_dtype, vals, input_channels, output_channels, filter_shape, strides, dilations, data_format, padding = _x_ic_oc_f_s_d_df_p
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        all_as_kwargs_np_init={
            "input_channels": input_channels,
            "output_channels": output_channels,
            "filter_shape": filter_shape,
            "strides": strides,
            "padding": padding,
            "weight_initializer": weight_initializer,
            "bias_initializer": bias_initializer,
            "data_format": data_format,
            "dilations": dilations,
            "device": device,
            "dtype": input_dtype,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"inputs": np.asarray(vals, dtype=input_dtype)},
        fw=fw,
        class_name="Conv1D",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
    )


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
    x_n_fs_n_pad_n_outshp_n_res, with_v, dtype, as_variable, device, compile_graph
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if ivy.current_backend_str() == "tensorflow" and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if ivy.current_backend_str() in ("numpy", "jax"):
        # numpy and jax do not yet support conv1d
        return
    if ivy.current_backend_str() == "torch" and (dtype == "float16"):
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
    assert np.allclose(
        ivy.to_numpy(conv1d_trans_layer(x)), target, rtol=tolerance_dict[dtype]
    )
    # compilation test
    if ivy.current_backend_str() == "torch":
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
    x_n_fs_n_pad_n_res, with_v, dtype, as_variable, device, compile_graph
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if ivy.current_backend_str() == "tensorflow" and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if ivy.current_backend_str() in ("numpy", "jax"):
        # numpy and jax do not yet support conv1d
        return
    if ivy.current_backend_str() == "torch" and (dtype == "float16"):
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
    assert np.allclose(
        ivy.to_numpy(conv2d_layer(x)), target, rtol=tolerance_dict[dtype]
    )
    # compilation test
    if ivy.current_backend_str() == "torch":
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
    x_n_fs_n_pad_n_outshp_n_res, with_v, dtype, as_variable, device, compile_graph
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if ivy.current_backend_str() == "tensorflow" and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if ivy.current_backend_str() in ("numpy", "jax"):
        # numpy and jax do not yet support conv1d
        return
    if ivy.current_backend_str() == "torch" and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv2d
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
        ivy.to_numpy(conv2d_transpose_layer(x)), target, rtol=tolerance_dict[dtype]
    )
    # compilation test
    if ivy.current_backend_str() == "torch":
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
    x_n_fs_n_pad_n_res, with_v, dtype, as_variable, device, compile_graph
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if ivy.current_backend_str() == "tensorflow" and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if ivy.current_backend_str() in ("numpy", "jax"):
        # numpy and jax do not yet support conv1d
        return
    if ivy.current_backend_str() == "torch" and (dtype == "float16"):
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
        ivy.to_numpy(depthwise_conv2d_layer(x)), target, rtol=tolerance_dict[dtype]
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
    x_n_fs_n_pad_n_res, with_v, dtype, as_variable, device, compile_graph
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if ivy.current_backend_str() == "tensorflow" and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if ivy.current_backend_str() in ("numpy", "jax"):
        # numpy and jax do not yet support conv1d
        return
    if ivy.current_backend_str() == "torch" and (dtype == "float16"):
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
    assert np.allclose(
        ivy.to_numpy(conv3d_layer(x)), target, rtol=tolerance_dict[dtype]
    )


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
    x_n_fs_n_pad_n_outshp_n_res, with_v, dtype, as_variable, device, compile_graph
):
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    if ivy.current_backend_str() == "tensorflow" and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if ivy.current_backend_str() in ("numpy", "jax"):
        # numpy and jax do not yet support conv1d
        return
    if ivy.current_backend_str() == "mxnet" and "cpu" in device:
        # mxnet only supports 3d transpose convolutions with CUDNN
        return
    if ivy.current_backend_str() == "torch" and (dtype == "float16"):
        # we are skipping for float16 as it torch.nn.functional.conv2d
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
        ivy.to_numpy(conv3d_transpose_layer(x)), target, rtol=tolerance_dict[dtype]
    )

#LSTM

@st.composite
def _x_ic_oc_f_d_df(draw, dim: int = 2, transpose: bool = False):
    strides = draw(st.integers(min_value=1, max_value=4))
    padding = draw(st.sampled_from(["SAME", "VALID"]))
    batch_size = draw(st.integers(1, 5))
    filter_shape = draw(helpers.get_shape(min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5))
    input_channels = draw(st.integers(1, 5))
    output_channels = draw(st.integers(1, 5))
    dilations = draw(st.integers(1, 4))
    x_dim = []
    for i in range(dim):
        min_x = filter_shape[i] + (filter_shape[i] - 1) * (dilations - 1)
        x_dim.append(draw(st.integers(min_x, 100)))
    if dim == 2:
        data_format = draw(st.sampled_from(["NCHW"]))
    elif dim == 1:
        data_format = draw(st.sampled_from(["NWC", "NCW"]))
    else:
        data_format = draw(st.sampled_from(["NDHWC", "NCDHW"]))
    if data_format == "NHWC" or data_format == "NWC" or data_format == "NDHWC":
        x_shape = [batch_size] + x_dim + [input_channels]
    else:
        x_shape = [batch_size] + [input_channels] + x_dim

    if transpose:
        output_shape = []
        for i in range(dim):
            output_shape.append(ivy.deconv_length(x_dim[i], strides, filter_shape[i], padding, dilations))
        # output_shape = [batch_size] + output_shape + [output_channels]
    if dim == 1:
        filter_shape = filter_shape[0]
    dtype, vals = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float", full=True), shape=x_shape
        )
    )
    if transpose:
        return dtype, vals, input_channels, output_channels, filter_shape, strides, dilations, data_format, padding, output_shape
    return dtype, vals, input_channels, output_channels, filter_shape, strides, dilations, data_format, padding


#
# # LSTM #

@handle_cmd_line_args
@given(
    input_dtype_val=_input_channels_and_dtype_and_values(),
    output_channels=st.shared(
        st.integers(min_value=1, max_value=10), key="output_channels"
    ),
    weight_initializer=_sample_initializer(),
    num_layers=st.integers(min_value=1, max_value=3),
    return_sequence=st.booleans(),
    return_state=st.booleans(),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="LSTM.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="LSTM._forward"),
)
def test_lstm_layer(
    input_dtype_val,
    output_channels,
    weight_initializer,
    num_layers,
    return_sequence,
    return_state,
    init_with_v,
    method_with_v,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
    device,
):
    input_channels, input_dtype, vals = input_dtype_val
    return_sequence = return_sequence
    return_state = return_state
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        all_as_kwargs_np_init={
            "input_channels": input_channels,
            "output_channels": output_channels,
            "num_layers": num_layers,
            "return_sequence": return_sequence,
            "return_state": return_state,
            "weight_initializer": weight_initializer,
            "device": device,
            "dtype": input_dtype,

        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"inputs": np.asarray(vals, dtype=input_dtype)},
        fw=fw,
        class_name="LSTM",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
    )


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
    bs_c_target, with_v, seq_v, dtype, as_variable, device, compile_graph
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
    assert np.allclose(
        ivy.to_numpy(seq(x)), np.array(target), rtol=tolerance_dict[dtype]
    )
