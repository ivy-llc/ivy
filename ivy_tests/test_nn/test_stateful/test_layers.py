"""
Collection of tests for templated neural network layers
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers
from ivy.core.container import Container


# Linear #
# -------#

# linear
@pytest.mark.parametrize(
    "bs_ic_oc_target", [
        ([1, 2], 4, 5, [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]]),
    ])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_linear_layer(bs_ic_oc_target, with_v, dtype_str, tensor_fn, dev_str, compile_graph, call):
    # smoke test
    batch_shape, input_channels, output_channels, target = bs_ic_oc_target
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(-wlim, wlim, (output_channels, input_channels)),
                                   'float32', dev_str=dev_str))
        b = ivy.variable(ivy.zeros([output_channels], dev_str=dev_str))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    linear_layer = ivy.Linear(input_channels, output_channels, dev_str=dev_str, v=v)
    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        linear_layer.compile_graph(x)
    ret = linear_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == tuple(batch_shape + [output_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(linear_layer, x), np.array(target))
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(linear_layer)


# Dropout #
# --------#

# dropout
@pytest.mark.parametrize(
    "x_shape", [(1, 2, 3)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_dropout_layer(x_shape, dtype_str, tensor_fn, dev_str, compile_graph, call):
    # smoke test
    x = ivy.random_uniform(shape=x_shape)
    dropout_layer = ivy.Dropout(0.9)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        dropout_layer.compile_graph(x)
    ret = dropout_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    ivy.seed(0)
    assert np.min(call(dropout_layer, x)) == 0.
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(dropout_layer)


# Attention #
# ----------#

# multi_head_attention
@pytest.mark.parametrize(
    "x_n_s_n_m_n_c_n_gt", [([[3.]], 2., [[1.]], [[4., 5.]], [[0.8066473]])])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "build_mode", ['on_init', 'explicit', 'on_call'])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_multi_head_attention_layer(x_n_s_n_m_n_c_n_gt, with_v, build_mode, dtype_str, tensor_fn, dev_str,
                                    compile_graph, call):
    x, scale, mask, context, ground_truth = x_n_s_n_m_n_c_n_gt
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    context = tensor_fn(context, dtype_str, dev_str)
    mask = tensor_fn(mask, dtype_str, dev_str)
    query_dim = x.shape[-1]
    context_dim = context.shape[-1]
    if with_v:
        inner_dim = 64 * 8
        np.random.seed(0)
        wlim = (6 / (inner_dim + query_dim)) ** 0.5
        w_to_q = ivy.variable(ivy.array(np.random.uniform(-wlim, wlim, (inner_dim, query_dim)),
                                        'float32', dev_str=dev_str))
        wlim = (6 / (inner_dim * 2 + context_dim)) ** 0.5
        w_to_k = ivy.variable(ivy.array(np.random.uniform(-wlim, wlim, (inner_dim, context_dim)),
                                         'float32', dev_str=dev_str))
        w_to_v = ivy.variable(ivy.array(np.random.uniform(-wlim, wlim, (inner_dim, context_dim)),
                                         'float32', dev_str=dev_str))
        wlim = (6 / (query_dim + inner_dim)) ** 0.5
        w_to_out = ivy.variable(ivy.array(np.random.uniform(-wlim, wlim, (query_dim, inner_dim)),
                                          'float32', dev_str=dev_str))
        b_to_out = ivy.variable(ivy.zeros([query_dim], dev_str=dev_str))
        v = Container({'to_q': {'w': w_to_q},
                       'to_kv': {'k': {'w': w_to_k}, 'v': {'w': w_to_v}},
                       'to_out': {'submodules': {'v0': {'w': w_to_out,
                                                        'b': b_to_out}}}})
    else:
        v = None
    multi_head_attention_layer = ivy.MultiHeadAttention(
        query_dim, context_dim=context_dim, scale=scale, dev_str=dev_str, v=v, build_mode=build_mode)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        multi_head_attention_layer.compile_graph(x, context, mask)
    if build_mode == 'explicit':
        multi_head_attention_layer.build()
    ret = multi_head_attention_layer(x, context, mask)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == list(np.array(ground_truth).shape)
    # value test
    if not with_v:
        return
    assert np.allclose(call(multi_head_attention_layer, x, context, mask), np.array(ground_truth))
    # compilation test
    if call in [helpers.torch_call]:
        # torch.jit compiled functions can't take variable number of arguments, which torch.einsum takes
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(multi_head_attention_layer)


# Convolutions #
# -------------#

# conv1d
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_res", [
        ([[[0.], [3.], [0.]]],
         3,
         "SAME",
         [[[1.0679483],
           [2.2363136],
           [0.5072848]]]),

        ([[[0.], [3.], [0.]] for _ in range(5)],
         3,
         "SAME",
         [[[1.0679483], [2.2363136], [0.5072848]] for _ in range(5)]),

        ([[[0.], [3.], [0.]]],
         3,
         "VALID",
         [[[2.2363136]]])])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv1d_layer(x_n_fs_n_pad_n_res, with_v, dtype_str, tensor_fn, dev_str, compile_graph, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_size, padding, target = x_n_fs_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    width = x.shape[1]
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, (filter_size, output_channels, input_channels)), 'float32', dev_str=dev_str))
        b = ivy.variable(ivy.zeros([1, 1, output_channels], dev_str=dev_str))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv1d_layer = ivy.Conv1D(input_channels, output_channels, filter_size, 1, padding, dev_str=dev_str, v=v)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        conv1d_layer.compile_graph(x)
    ret = conv1d_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    new_width = width if padding == 'SAME' else width - filter_size + 1
    assert ret.shape == (batch_size, new_width, output_channels)
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv1d_layer, x), target)
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(conv1d_layer)


# conv1d transpose
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_outshp_n_res", [
        ([[[0.], [3.], [0.]]],
         3,
         "SAME",
         (1, 3, 1),
         [[[0.5072848], [2.2363136], [1.0679483]]]),

        ([[[0.], [3.], [0.]] for _ in range(5)],
         3,
         "SAME",
         (5, 3, 1),
         [[[0.5072848], [2.2363136], [1.0679483]] for _ in range(5)]),

        ([[[0.], [3.], [0.]]],
         3,
         "VALID",
         (1, 5, 1),
         [[[0.], [0.5072848], [2.2363136], [1.0679483], [0.]]])])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv1d_transpose_layer(x_n_fs_n_pad_n_outshp_n_res, with_v, dtype_str, tensor_fn, dev_str, compile_graph,
                                call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_size, padding, out_shape, target = x_n_fs_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    width = x.shape[1]
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, (filter_size, output_channels, input_channels)), 'float32', dev_str=dev_str))
        b = ivy.variable(ivy.zeros([1, 1, output_channels], dev_str=dev_str))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv1d_trans_layer = ivy.Conv1DTranspose(input_channels, output_channels, filter_size, 1, padding,
                                             output_shape=out_shape, dev_str=dev_str, v=v)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        conv1d_trans_layer.compile_graph(x)
    ret = conv1d_trans_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    new_width = width if padding == 'SAME' else width + filter_size - 1
    assert ret.shape == (batch_size, new_width, output_channels)
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv1d_trans_layer, x), target)
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(conv1d_trans_layer)


# conv2d
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_res", [([[[[1.], [2.], [3.], [4.], [5.]],
                              [[6.], [7.], [8.], [9.], [10.]],
                              [[11.], [12.], [13.], [14.], [15.]],
                              [[16.], [17.], [18.], [19.], [20.]],
                              [[21.], [22.], [23.], [24.], [25.]]]],
                            [3, 3],
                            "SAME",
                            [[[[20.132391], [22.194885], [25.338402], [28.481918], [10.9251585]],
                              [[37.611], [40.64039], [45.05442], [49.468452], [20.488476]],
                              [[59.139305], [62.71055], [67.12458], [71.53861], [30.220888]],
                              [[80.66761], [84.78071], [89.19474], [93.60877], [39.9533]],
                              [[23.54352], [30.85646], [32.52338], [34.1903], [15.24139]]]]),

                           ([[[[1.], [2.], [3.], [4.], [5.]],
                              [[6.], [7.], [8.], [9.], [10.]],
                              [[11.], [12.], [13.], [14.], [15.]],
                              [[16.], [17.], [18.], [19.], [20.]],
                              [[21.], [22.], [23.], [24.], [25.]]] for _ in range(5)],
                            [3, 3],
                            "SAME",
                            [[[[20.132391], [22.194885], [25.338402], [28.481918], [10.9251585]],
                              [[37.611], [40.64039], [45.05442], [49.468452], [20.488476]],
                              [[59.139305], [62.71055], [67.12458], [71.53861], [30.220888]],
                              [[80.66761], [84.78071], [89.19474], [93.60877], [39.9533]],
                              [[23.54352], [30.85646], [32.52338], [34.1903], [15.24139]]] for _ in range(5)]),

                           ([[[[1.], [2.], [3.], [4.], [5.]],
                              [[6.], [7.], [8.], [9.], [10.]],
                              [[11.], [12.], [13.], [14.], [15.]],
                              [[16.], [17.], [18.], [19.], [20.]],
                              [[21.], [22.], [23.], [24.], [25.]]]],
                            [3, 3],
                            "VALID",
                            [[[[40.64039], [45.05442], [49.468452]],
                              [[62.71055], [67.12458], [71.53861]],
                              [[84.78071], [89.19474], [93.60877]]]])])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv2d_layer(x_n_fs_n_pad_n_res, with_v, dtype_str, tensor_fn, dev_str, compile_graph, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_shape, padding, target = x_n_fs_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:3])
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])), 'float32', dev_str=dev_str))
        b = ivy.variable(ivy.zeros([1, 1, 1, output_channels], dev_str=dev_str))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv2d_layer = ivy.Conv2D(input_channels, output_channels, filter_shape, 1, padding, dev_str=dev_str, v=v)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        conv2d_layer.compile_graph(x)
    ret = conv2d_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    new_shape = input_shape if padding == 'SAME' else [item - filter_shape[i] + 1 for i, item in enumerate(input_shape)]
    assert ret.shape == tuple([batch_size] + new_shape + [output_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv2d_layer, x), target)
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(conv2d_layer)


# conv2d transpose
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_outshp_n_res", [
        ([[[[0.], [0.], [0.]],
           [[0.], [3.], [0.]],
           [[0.], [0.], [0.]]]],
         [3, 3],
         "SAME",
         (1, 3, 3, 1),
         [[[[0.5072848], [2.2363136], [1.0679483]],
           [[0.46643972], [-0.7934026], [1.516176]],
           [[-0.64861274], [4.0714245], [4.818525]]]]),

        ([[[[0.], [0.], [0.]],
           [[0.], [3.], [0.]],
           [[0.], [0.], [0.]]] for _ in range(5)],
         [3, 3],
         "SAME",
         (5, 3, 3, 1),
         [[[[0.5072848], [2.2363136], [1.0679483]],
           [[0.46643972], [-0.7934026], [1.516176]],
           [[-0.64861274], [4.0714245], [4.818525]]] for _ in range(5)]),

        ([[[[0.], [0.], [0.]],
           [[0.], [3.], [0.]],
           [[0.], [0.], [0.]]]],
         [3, 3],
         "VALID",
         (1, 5, 5, 1),
         [[[[0.], [0.], [0.], [0.], [0.]],
           [[0.], [0.5072848], [2.2363136], [1.0679483], [0.]],
           [[0.], [0.46643972], [-0.7934026], [1.516176], [0.]],
           [[0.], [-0.64861274], [4.0714245], [4.818525], [0.]],
           [[0.], [0.], [0.], [0.], [0.]]]])])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv2d_transpose_layer(x_n_fs_n_pad_n_outshp_n_res, with_v, dtype_str, tensor_fn, dev_str, compile_graph,
                                call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_shape, padding, out_shape, target = x_n_fs_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:3])
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])), 'float32', dev_str=dev_str))
        b = ivy.variable(ivy.zeros([1, 1, 1, output_channels], dev_str=dev_str))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv2d_transpose_layer = ivy.Conv2DTranspose(input_channels, output_channels, filter_shape, 1, padding,
                                                 output_shape=out_shape, dev_str=dev_str, v=v)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        conv2d_transpose_layer.compile_graph(x)
    ret = conv2d_transpose_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    new_shape = input_shape if padding == 'SAME' else [item + filter_shape[i] - 1 for i, item in enumerate(input_shape)]
    assert ret.shape == tuple([batch_size] + new_shape + [output_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv2d_transpose_layer, x), target)
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(conv2d_transpose_layer)


# depthwise conv2d
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_res", [
                                ([[[[0.], [0.], [0.]],
                                   [[0.], [3.], [0.]],
                                   [[0.], [0.], [0.]]]],
                                 [3, 3],
                                 "SAME",
                                 [[[[4.818525], [4.0714245], [-0.64861274]],
                                   [[1.516176], [-0.7934026], [0.46643972]],
                                   [[1.0679483], [2.2363136], [0.5072848]]]]),

                                ([[[[0.], [0.], [0.]],
                                   [[0.], [3.], [0.]],
                                   [[0.], [0.], [0.]]] for _ in range(5)],
                                 [3, 3],
                                 "SAME",
                                 [[[[4.818525], [4.0714245], [-0.64861274]],
                                   [[1.516176], [-0.7934026], [0.46643972]],
                                   [[1.0679483], [2.2363136], [0.5072848]]] for _ in range(5)]),

                                ([[[[0.], [0.], [0.]],
                                   [[0.], [3.], [0.]],
                                   [[0.], [0.], [0.]]]],
                                 [3, 3],
                                 "VALID",
                                 [[[[-0.7934026]]]])])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_depthwise_conv2d_layer(x_n_fs_n_pad_n_res, with_v, dtype_str, tensor_fn, dev_str, compile_graph, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_shape, padding, target = x_n_fs_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    target = np.asarray(target)
    num_channels = x.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:3])
    if with_v:
        np.random.seed(0)
        wlim = (6 / (num_channels*2)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, tuple(filter_shape + [num_channels])), 'float32', dev_str=dev_str))
        b = ivy.variable(ivy.zeros([1, 1, num_channels], dev_str=dev_str))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    depthwise_conv2d_layer = ivy.DepthwiseConv2D(num_channels, filter_shape, 1, padding, dev_str=dev_str, v=v)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        depthwise_conv2d_layer.compile_graph(x)
    ret = depthwise_conv2d_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    new_shape = input_shape if padding == 'SAME' else [item - filter_shape[i] + 1 for i, item in enumerate(input_shape)]
    assert ret.shape == tuple([batch_size] + new_shape + [num_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(depthwise_conv2d_layer, x), target)
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(depthwise_conv2d_layer)


# conv3d
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_res",
    [([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
      [3, 3, 3],
      "SAME",
      [[[[[-3.7063813], [1.4541019], [-3.9670086]], [[2.9153447], [-0.4003182], [3.108947]], [[4.9739475], [3.8452792], [2.8906898]]],
        [[[3.456687], [-4.986037], [-4.290678]], [[-4.457924], [4.4229302], [0.70713985]], [[0.3002848], [3.0316954], [-1.2113112]]],
        [[[4.818525], [4.0714245], [-0.64861274]], [[1.516176], [-0.7934026], [0.46643972]], [[1.0679483], [2.2363136], [0.5072848]]]]]),

     ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]] for _ in range(5)],
      [3, 3, 3],
      "SAME",
      [[[[[-3.7063813], [1.4541019], [-3.9670086]], [[2.9153447], [-0.4003182], [3.108947]], [[4.9739475], [3.8452792], [2.8906898]]],
        [[[3.456687], [-4.986037], [-4.290678]], [[-4.457924], [4.4229302], [0.70713985]], [[0.3002848], [3.0316954], [-1.2113112]]],
        [[[4.818525], [4.0714245], [-0.64861274]], [[1.516176], [-0.7934026], [0.46643972]], [[1.0679483], [2.2363136], [0.5072848]]]]
       for _ in range(5)]),

     ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
      [3, 3, 3],
      "VALID",
      [[[[[4.4229302]]]]])])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv3d_layer(x_n_fs_n_pad_n_res, with_v, dtype_str, tensor_fn, dev_str, compile_graph, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_shape, padding, target = x_n_fs_n_pad_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:4])
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])), 'float32', dev_str=dev_str))
        b = ivy.variable(ivy.zeros([1, 1, 1, 1, output_channels], dev_str=dev_str))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv3d_layer = ivy.Conv3D(input_channels, output_channels, filter_shape, 1, padding, dev_str=dev_str, v=v)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        conv3d_layer.compile_graph(x)
    ret = conv3d_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    new_shape = input_shape if padding == 'SAME' else [item - filter_shape[i] + 1 for i, item in enumerate(input_shape)]
    assert ret.shape == tuple([batch_size] + new_shape + [output_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv3d_layer, x), target)
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(conv3d_layer)


# conv3d transpose
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_outshp_n_res", [
        ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
         [3, 3, 3],
         "SAME",
         (1, 3, 3, 3, 1),
         [[[[[0.5072848], [2.2363136], [1.0679483]], [[0.46643972], [-0.7934026], [1.516176]], [[-0.64861274], [4.0714245], [4.818525]]],
           [[[-1.2113112], [3.0316954], [0.3002848]], [[0.70713985], [4.4229302], [-4.457924]], [[-4.290678], [-4.986037], [3.456687]]],
           [[[2.8906898], [3.8452792], [4.9739475]], [[3.108947], [-0.4003182], [2.9153447]], [[-3.9670086], [1.4541019], [-3.7063813]]]]]),

        ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]] for _ in range(5)],
         [3, 3, 3],
         "SAME",
         (5, 3, 3, 3, 1),
         [[[[[0.5072848], [2.2363136], [1.0679483]], [[0.46643972], [-0.7934026], [1.516176]], [[-0.64861274], [4.0714245], [4.818525]]],
           [[[-1.2113112], [3.0316954], [0.3002848]], [[0.70713985], [4.4229302], [-4.457924]], [[-4.290678], [-4.986037], [3.456687]]],
           [[[2.8906898], [3.8452792], [4.9739475]], [[3.108947], [-0.4003182], [2.9153447]], [[-3.9670086], [1.4541019], [-3.7063813]]]] for _ in range(5)]),

        ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
         [3, 3, 3],
         "VALID",
         (1, 5, 5, 5, 1),
         [[[[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]],
            [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
           [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.5072848], [2.2363136], [1.0679483], [0.]], [[0.], [0.46643972], [-0.7934026], [1.516176], [0.]],
            [[0.], [-0.64861274], [4.0714245], [4.818525], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
           [[[0.], [0.], [0.], [0.], [0.]], [[0.], [-1.2113112], [3.0316954], [0.3002848], [0.]], [[0.], [0.70713985], [4.4229302], [-4.457924], [0.]],
            [[0.], [-4.290678], [-4.986037], [3.456687], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
           [[[0.], [0.], [0.], [0.], [0.]], [[0.], [2.8906898], [3.8452792], [4.9739475], [0.]], [[0.], [3.108947], [-0.4003182], [2.9153447], [0.]],
            [[0.], [-3.9670086], [1.4541019], [-3.7063813], [0.]], [[0.], [0.], [0.], [0.], [0.]]],
           [[[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]],
            [[0.], [0.], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.], [0.]]]]])])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv3d_transpose_layer(x_n_fs_n_pad_n_outshp_n_res, with_v, dtype_str, tensor_fn, dev_str, compile_graph,
                                call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    if call in [helpers.mx_call] and 'cpu' in dev_str:
        # mxnet only supports 3d transpose convolutions with CUDNN
        pytest.skip()
    # smoke test
    x, filter_shape, padding, out_shape, target = x_n_fs_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    target = np.asarray(target)
    input_channels = x.shape[-1]
    output_channels = target.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:4])
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])), 'float32', dev_str=dev_str))
        b = ivy.variable(ivy.zeros([1, 1, 1, 1, output_channels], dev_str=dev_str))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv3d_transpose_layer = ivy.Conv3DTranspose(
        input_channels, output_channels, filter_shape, 1, padding, output_shape=out_shape, dev_str=dev_str, v=v)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        conv3d_transpose_layer.compile_graph(x)
    ret = conv3d_transpose_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    new_shape = input_shape if padding == 'SAME' else [item + filter_shape[i] - 1 for i, item in enumerate(input_shape)]
    assert ret.shape == tuple([batch_size] + new_shape + [output_channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(conv3d_transpose_layer, x), target)
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(conv3d_transpose_layer)


# LSTM #
# -----#

@pytest.mark.parametrize(
    "b_t_ic_hc_otf_sctv", [
        (2, 3, 4, 5, [0.93137765, 0.9587628, 0.96644664, 0.93137765, 0.9587628, 0.96644664], 3.708991),
    ])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "with_initial_state", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_lstm_layer(b_t_ic_hc_otf_sctv, with_v, with_initial_state, dtype_str, tensor_fn, dev_str, compile_graph, call):
    # smoke test
    b, t, input_channels, hidden_channels, output_true_flat, state_c_true_val = b_t_ic_hc_otf_sctv
    x = ivy.cast(ivy.linspace(ivy.zeros([b, t]), ivy.ones([b, t]), input_channels), 'float32')
    if with_initial_state:
        init_h = ivy.ones([b, hidden_channels])
        init_c = ivy.ones([b, hidden_channels])
        initial_state = ([init_h], [init_c])
    else:
        initial_state = None
    if with_v:
        kernel = ivy.variable(ivy.ones([input_channels, 4*hidden_channels], dev_str=dev_str)*0.5)
        recurrent_kernel = ivy.variable(ivy.ones([hidden_channels, 4*hidden_channels], dev_str=dev_str)*0.5)
        v = Container({'input': {'layer_0': {'w': kernel}},
                       'recurrent': {'layer_0': {'w': recurrent_kernel}}})
    else:
        v = None
    lstm_layer = ivy.LSTM(input_channels, hidden_channels, dev_str=dev_str, v=v)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        lstm_layer.compile_graph(x, initial_state=initial_state)
    output, (state_h, state_c) = lstm_layer(x, initial_state=initial_state)
    # type test
    assert ivy.is_array(output)
    assert ivy.is_array(state_h[0])
    assert ivy.is_array(state_c[0])
    # cardinality test
    assert output.shape == (b, t, hidden_channels)
    assert state_h[0].shape == (b, hidden_channels)
    assert state_c[0].shape == (b, hidden_channels)
    # value test
    if not with_v or not with_initial_state:
        return
    output_true = np.tile(np.asarray(output_true_flat).reshape((b, t, 1)), (1, 1, hidden_channels))
    state_c_true = np.ones([b, hidden_channels]) * state_c_true_val
    output, (state_h, state_c) = call(lstm_layer, x, initial_state=initial_state)
    assert np.allclose(output, output_true, atol=1e-6)
    assert np.allclose(state_c, state_c_true, atol=1e-6)
    # compilation test
    if call in [helpers.torch_call]:
        # this is not a backend implemented function
        pytest.skip()
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.lstm_update)


# Sequential #
# -----------#

# sequential
@pytest.mark.parametrize(
    "bs_c_target", [
        ([1, 2], 5, [[[-0.34784955,  0.47909835,  0.7241975 , -0.82175905, -0.43836743],
                      [-0.34784955,  0.47909835,  0.7241975 , -0.82175905, -0.43836743]]])])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "seq_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_sequential_layer(bs_c_target, with_v, seq_v, dtype_str, tensor_fn, dev_str, compile_graph, call):
    # smoke test
    batch_shape, channels, target = bs_c_target
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), channels), 'float32')
    if with_v:
        np.random.seed(0)
        wlim = (6 / (channels + channels)) ** 0.5
        v = Container(
            {'submodules':
                      {'v0':
                           {'w': ivy.variable(ivy.array(np.random.uniform(
                               -wlim, wlim, (channels, channels)), 'float32', dev_str=dev_str)),
                               'b': ivy.variable(ivy.zeros([channels], dev_str=dev_str))},
                       'v2':
                           {'w': ivy.variable(ivy.array(np.random.uniform(
                               -wlim, wlim, (channels, channels)), 'float32', dev_str=dev_str)),
                               'b': ivy.variable(ivy.zeros([channels], dev_str=dev_str))}}})
    else:
        v = None
    if seq_v:
        seq = ivy.Sequential(ivy.Linear(channels, channels, dev_str=dev_str),
                             ivy.Dropout(0.),
                             ivy.Linear(channels, channels, dev_str=dev_str),
                             dev_str=dev_str, v=v if with_v else None)
    else:
        seq = ivy.Sequential(ivy.Linear(channels, channels, dev_str=dev_str,
                                        v=v['submodules']['v0'] if with_v else None),
                             ivy.Dropout(0.),
                             ivy.Linear(channels, channels, dev_str=dev_str,
                                        v=v['submodules']['v2'] if with_v else None), dev_str=dev_str)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        seq.compile_graph(x)
    ret = seq(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == tuple(batch_shape + [channels])
    # value test
    if not with_v:
        return
    assert np.allclose(call(seq, x), np.array(target))
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(seq)
