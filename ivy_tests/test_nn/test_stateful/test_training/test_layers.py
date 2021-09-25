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


# linear
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_linear_layer_training(bs_ic_oc, with_v, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# Convolutions #
# -------------#


# conv1d
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_oc", [
        ([[[0.], [3.], [0.]]],
         3,
         "SAME",
         1),

        ([[[0.], [3.], [0.]] for _ in range(5)],
         3,
         "SAME",
         1),

        ([[[0.], [3.], [0.]]],
         3,
         "VALID",
         1)])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv1d_layer_training(x_n_fs_n_pad_n_oc, with_v, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_size, padding, output_channels = x_n_fs_n_pad_n_oc
    x = tensor_fn(x, dtype_str, dev_str)
    input_channels = x.shape[-1]
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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# conv1d transpose
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_outshp_n_oc", [
        ([[[0.], [3.], [0.]]],
         3,
         "SAME",
         (1, 3, 1),
         1),

        ([[[0.], [3.], [0.]] for _ in range(5)],
         3,
         "SAME",
         (5, 3, 1),
         1),

        ([[[0.], [3.], [0.]]],
         3,
         "VALID",
         (1, 5, 1),
         1)])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv1d_transpose_layer_training(x_n_fs_n_pad_n_outshp_n_oc, with_v, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_size, padding, out_shape, output_channels = x_n_fs_n_pad_n_outshp_n_oc
    x = tensor_fn(x, dtype_str, dev_str)
    input_channels = x.shape[-1]
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
    conv1d_trans_layer = ivy.Conv1DTranspose(input_channels, output_channels, filter_size, 1, padding,
                                             output_shape=out_shape, v=v)

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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# conv2d
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_oc",  [([[[[1.], [2.], [3.], [4.], [5.]],
                              [[6.], [7.], [8.], [9.], [10.]],
                              [[11.], [12.], [13.], [14.], [15.]],
                              [[16.], [17.], [18.], [19.], [20.]],
                              [[21.], [22.], [23.], [24.], [25.]]]],
                            [3, 3],
                            "SAME",
                            1),

                           ([[[[1.], [2.], [3.], [4.], [5.]],
                              [[6.], [7.], [8.], [9.], [10.]],
                              [[11.], [12.], [13.], [14.], [15.]],
                              [[16.], [17.], [18.], [19.], [20.]],
                              [[21.], [22.], [23.], [24.], [25.]]] for _ in range(5)],
                            [3, 3],
                            "SAME",
                            1),

                           ([[[[1.], [2.], [3.], [4.], [5.]],
                              [[6.], [7.], [8.], [9.], [10.]],
                              [[11.], [12.], [13.], [14.], [15.]],
                              [[16.], [17.], [18.], [19.], [20.]],
                              [[21.], [22.], [23.], [24.], [25.]]]],
                            [3, 3],
                            "VALID",
                            1)])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv2d_layer_training(x_n_fs_n_pad_n_oc, with_v, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_shape, padding, output_channels = x_n_fs_n_pad_n_oc
    x = tensor_fn(x, dtype_str, dev_str)
    input_channels = x.shape[-1]
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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# conv2d transpose
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_outshp_n_oc", [
        ([[[[0.], [0.], [0.]],
           [[0.], [3.], [0.]],
           [[0.], [0.], [0.]]]],
         [3, 3],
         "SAME",
         (1, 3, 3, 1),
         1),

        ([[[[0.], [0.], [0.]],
           [[0.], [3.], [0.]],
           [[0.], [0.], [0.]]] for _ in range(5)],
         [3, 3],
         "SAME",
         (5, 3, 3, 1),
         1),

        ([[[[0.], [0.], [0.]],
           [[0.], [3.], [0.]],
           [[0.], [0.], [0.]]]],
         [3, 3],
         "VALID",
         (1, 5, 5, 1),
         1)])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv2d_transpose_layer_training(x_n_fs_n_pad_n_outshp_n_oc, with_v, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_shape, padding, out_shape, output_channels = x_n_fs_n_pad_n_outshp_n_oc
    x = tensor_fn(x, dtype_str, dev_str)
    input_channels = x.shape[-1]
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
        input_channels, output_channels, filter_shape, 1, padding, output_shape=out_shape, v=v)

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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# depthwise conv2d
@pytest.mark.parametrize(
    "x_n_fs_n_pad", [
                                ([[[[0.], [0.], [0.]],
                                   [[0.], [3.], [0.]],
                                   [[0.], [0.], [0.]]]],
                                 [3, 3],
                                 "SAME"),

                                ([[[[0.], [0.], [0.]],
                                   [[0.], [3.], [0.]],
                                   [[0.], [0.], [0.]]] for _ in range(5)],
                                 [3, 3],
                                 "SAME"),

                                ([[[[0.], [0.], [0.]],
                                   [[0.], [3.], [0.]],
                                   [[0.], [0.], [0.]]]],
                                 [3, 3],
                                 "VALID")])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_depthwise_conv2d_layer_training(x_n_fs_n_pad, with_v, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_shape, padding = x_n_fs_n_pad
    x = tensor_fn(x, dtype_str, dev_str)
    num_channels = x.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:3])
    if with_v:
        np.random.seed(0)
        wlim = (6 / (num_channels*2)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, tuple(filter_shape + [num_channels])), 'float32'))
        b = ivy.variable(ivy.zeros([1, 1, num_channels]))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    depthwise_conv2d_layer = ivy.DepthwiseConv2D(num_channels, filter_shape, 1, padding, v=v)

    def loss_fn(v_):
        out = depthwise_conv2d_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, depthwise_conv2d_layer.v)
        depthwise_conv2d_layer.v = ivy.gradient_descent_update(depthwise_conv2d_layer.v, grads, 1e-3)
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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# conv3d
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_oc",
    [([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
      [3, 3, 3],
      "SAME",
      1),

     ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]] for _ in range(5)],
      [3, 3, 3],
      "SAME",
      1),

     ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
        [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
      [3, 3, 3],
      "VALID",
      1)])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv3d_layer_training(x_n_fs_n_pad_n_oc, with_v, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        pytest.skip()
    # smoke test
    x, filter_shape, padding, output_channels = x_n_fs_n_pad_n_oc
    x = tensor_fn(x, dtype_str, dev_str)
    input_channels = x.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:4])
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])), 'float32'))
        b = ivy.variable(ivy.zeros([1, 1, 1, 1, output_channels]))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv3d_layer = ivy.Conv3D(input_channels, output_channels, filter_shape, 1, padding, v=v)

    def loss_fn(v_):
        out = conv3d_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, conv3d_layer.v)
        conv3d_layer.v = ivy.gradient_descent_update(conv3d_layer.v, grads, 1e-3)
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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# conv3d transpose
@pytest.mark.parametrize(
    "x_n_fs_n_pad_n_outshp_n_oc", [
        ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
         [3, 3, 3],
         "SAME",
         (1, 3, 3, 3, 1),
         1),

        ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]] for _ in range(5)],
         [3, 3, 3],
         "SAME",
         (5, 3, 3, 3, 1),
         1),

        ([[[[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [3.], [0.]], [[0.], [0.], [0.]]],
           [[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]]]],
         [3, 3, 3],
         "VALID",
         (1, 5, 5, 5, 1),
         1)])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_conv3d_transpose_layer_training(x_n_fs_n_pad_n_outshp_n_oc, with_v, dtype_str, tensor_fn, dev_str, call):
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
    x, filter_shape, padding, out_shape, output_channels = x_n_fs_n_pad_n_outshp_n_oc
    x = tensor_fn(x, dtype_str, dev_str)
    input_channels = x.shape[-1]
    batch_size = x.shape[0]
    input_shape = list(x.shape[1:4])
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(ivy.array(np.random.uniform(
            -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])), 'float32'))
        b = ivy.variable(ivy.zeros([1, 1, 1, 1, output_channels]))
        v = Container({'w': w, 'b': b})
    else:
        v = None
    conv3d_transpose_layer = ivy.Conv3DTranspose(
        input_channels, output_channels, filter_shape, 1, padding, output_shape=out_shape, v=v)

    def loss_fn(v_):
        out = conv3d_transpose_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, conv3d_transpose_layer.v)
        conv3d_transpose_layer.v = ivy.gradient_descent_update(conv3d_transpose_layer.v, grads, 1e-3)
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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# Sequential #
# -----------#


# sequential
@pytest.mark.parametrize(
    "bs_c", [([1, 2], 5)])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "seq_v", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_sequential_layer_training(bs_c, with_v, seq_v, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, channels = bs_c
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), channels), 'float32')
    if with_v:
        np.random.seed(0)
        wlim = (6 / (channels + channels)) ** 0.5
        v = Container(
            {'submodules':
                      {'v0':
                           {'w': ivy.variable(ivy.array(np.random.uniform(
                               -wlim, wlim, (channels, channels)), 'float32')),
                               'b': ivy.variable(ivy.zeros([channels]))},
                       'v1':
                           {'w': ivy.variable(ivy.array(np.random.uniform(
                               -wlim, wlim, (channels, channels)), 'float32')),
                               'b': ivy.variable(ivy.zeros([channels]))},
                       'v2':
                           {'w': ivy.variable(ivy.array(np.random.uniform(
                               -wlim, wlim, (channels, channels)), 'float32')),
                               'b': ivy.variable(ivy.zeros([channels]))}}})
    else:
        v = None
    if seq_v:
        seq = ivy.Sequential(ivy.Linear(channels, channels),
                             ivy.Linear(channels, channels),
                             ivy.Linear(channels, channels),
                             v=v if with_v else None)
    else:
        seq = ivy.Sequential(ivy.Linear(channels, channels, v=v['submodules']['v0'] if with_v else None),
                             ivy.Linear(channels, channels, v=v['submodules']['v1'] if with_v else None),
                             ivy.Linear(channels, channels, v=v['submodules']['v2'] if with_v else None))

    def loss_fn(v_):
        out = seq(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, seq.v)
        seq.v = ivy.gradient_descent_update(seq.v, grads, 1e-3)
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
    assert (abs(grads).reduce_max() > 0).all_true()
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)
