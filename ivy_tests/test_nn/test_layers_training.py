"""
Collection of tests for training templated neural network layers
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
def test_linear_layer_training(bs_ic_oc_target, with_v, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels, target = bs_ic_oc_target
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(-wlim, wlim, (output_channels, input_channels)), 'float32'))
        b = ivy.variable(ivy.zeros([output_channels]))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    linear_layer = ivy.Linear(input_channels, output_channels, v=v)

    def loss_fn(v_):
        out = linear_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
        linear_layer.v = ivy.gradient_descent_update(linear_layer.v, grads, 1e-3)
        assert loss < loss_tm1
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert ivy.reduce_max(ivy.abs(grads.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    helpers.assert_compilable(loss_fn)


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
def test_conv1d_layer_training(x_n_fs_n_pad_n_res, with_v, dtype_str, tensor_fn, dev_str, call):
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
            -wlim, wlim, (filter_size, output_channels, input_channels)), 'float32'))
        b = ivy.variable(ivy.zeros([1, 1, output_channels]))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv1d_layer = ivy.Conv1D(input_channels, output_channels, filter_size, 1, padding, v=v)

    def loss_fn(v_):
        out = conv1d_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, conv1d_layer.v)
        conv1d_layer.v = ivy.gradient_descent_update(conv1d_layer.v, grads, 1e-3)
        assert loss < loss_tm1
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert ivy.reduce_max(ivy.abs(grads.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    helpers.assert_compilable(loss_fn)


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
def test_conv1d_transpose_layer_training(x_n_fs_n_pad_n_outshp_n_res, with_v, dtype_str, tensor_fn, dev_str, call):
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
            -wlim, wlim, (filter_size, output_channels, input_channels)), 'float32'))
        b = ivy.variable(ivy.zeros([1, 1, output_channels]))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv1d_trans_layer = ivy.Conv1DTranspose(input_channels, output_channels, filter_size, 1, padding, out_shape, v=v)

    def loss_fn(v_):
        out = conv1d_trans_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, conv1d_trans_layer.v)
        conv1d_trans_layer.v = ivy.gradient_descent_update(conv1d_trans_layer.v, grads, 1e-3)
        assert loss < loss_tm1
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert ivy.reduce_max(ivy.abs(grads.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    helpers.assert_compilable(loss_fn)


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
def test_conv2d_layer_training(x_n_fs_n_pad_n_res, with_v, dtype_str, tensor_fn, dev_str, call):
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
            -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])), 'float32'))
        b = ivy.variable(ivy.zeros([1, 1, 1, output_channels]))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv2d_layer = ivy.Conv2D(input_channels, output_channels, filter_shape, 1, padding, v=v)

    def loss_fn(v_):
        out = conv2d_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, conv2d_layer.v)
        conv2d_layer.v = ivy.gradient_descent_update(conv2d_layer.v, grads, 1e-3)
        assert loss < loss_tm1
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert ivy.reduce_max(ivy.abs(grads.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    helpers.assert_compilable(loss_fn)


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
def test_conv2d_transpose_layer_training(x_n_fs_n_pad_n_outshp_n_res, with_v, dtype_str, tensor_fn, dev_str, call):
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
            -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])), 'float32'))
        b = ivy.variable(ivy.zeros([1, 1, 1, output_channels]))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv2d_transpose_layer = ivy.Conv2DTranspose(
        input_channels, output_channels, filter_shape, 1, padding, out_shape, v=v)

    def loss_fn(v_):
        out = conv2d_transpose_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, conv2d_transpose_layer.v)
        conv2d_transpose_layer.v = ivy.gradient_descent_update(conv2d_transpose_layer.v, grads, 1e-3)
        assert loss < loss_tm1
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert ivy.reduce_max(ivy.abs(grads.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    helpers.assert_compilable(loss_fn)


# LSTM #
# -----#


@pytest.mark.parametrize(
    "b_t_ic_hc_otf_sctv", [
        (2, 3, 4, 5, [0.93137765, 0.9587628, 0.96644664, 0.93137765, 0.9587628, 0.96644664], 3.708991),
    ])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_lstm_layer_training(b_t_ic_hc_otf_sctv, with_v, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    # smoke test
    b, t, input_channels, hidden_channels, output_true_flat, state_c_true_val = b_t_ic_hc_otf_sctv
    x = ivy.cast(ivy.linspace(ivy.zeros([b, t]), ivy.ones([b, t]), input_channels), 'float32')
    if with_v:
        kernel = ivy.variable(ivy.ones([input_channels, 4*hidden_channels])*0.5)
        recurrent_kernel = ivy.variable(ivy.ones([hidden_channels, 4*hidden_channels])*0.5)
        v = Container({'input': {'layer_0': {'w': kernel}},
                       'recurrent': {'layer_0': {'w': recurrent_kernel}}})
    else:
        v = None
    lstm_layer = ivy.LSTM(input_channels, hidden_channels, v=v)

    def loss_fn(v_):
        out, (state_h, state_c) = lstm_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, lstm_layer.v)
        lstm_layer.v = ivy.gradient_descent_update(lstm_layer.v, grads, 1e-3)
        assert loss < loss_tm1
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    for key, val in grads.to_iterator():
        assert ivy.reduce_max(ivy.abs(val)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    helpers.assert_compilable(loss_fn)
