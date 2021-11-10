"""
Collection of tests for Ivy optimizers
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers
from ivy.core.container import Container


# sgd
@pytest.mark.parametrize(
    "bs_ic_oc_target", [
        ([1, 2], 4, 5, [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]]),
    ])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "inplace", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
def test_sgd_optimizer(bs_ic_oc_target, with_v, inplace, dtype_str, dev_str, compile_graph, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
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

    def loss_fn(v_):
        out = linear_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # optimizer
    optimizer = ivy.SGD(inplace=inplace)

    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        optimizer.compile_graph(linear_layer.v, )

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
        linear_layer.v = optimizer.step(linear_layer.v, grads)
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# lars
@pytest.mark.parametrize(
    "bs_ic_oc_target", [
        ([1, 2], 4, 5, [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]]),
    ])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "inplace", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
def test_lars_optimizer(bs_ic_oc_target, with_v, inplace, dtype_str, dev_str, compile_graph, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
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

    def loss_fn(v_):
        out = linear_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # optimizer
    optimizer = ivy.LARS(inplace=inplace)

    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        optimizer.compile_graph(linear_layer.v)

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
        linear_layer.v = optimizer.step(linear_layer.v, grads)
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# adam
@pytest.mark.parametrize(
    "bs_ic_oc_target", [
        ([1, 2], 4, 5, [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]]),
    ])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "inplace", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
def test_adam_optimizer(bs_ic_oc_target, with_v, inplace, dtype_str, dev_str, compile_graph, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
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

    def loss_fn(v_):
        out = linear_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # optimizer
    optimizer = ivy.Adam(dev_str=dev_str, inplace=inplace)

    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        optimizer.compile_on_next_step()

    # train
    loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
    linear_layer.v = optimizer.step(linear_layer.v, grads)
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
        linear_layer.v = optimizer.step(linear_layer.v, grads)
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


# lamb
@pytest.mark.parametrize(
    "bs_ic_oc_target", [
        ([1, 2], 4, 5, [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]]),
    ])
@pytest.mark.parametrize(
    "with_v", [True, False])
@pytest.mark.parametrize(
    "inplace", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
def test_lamb_optimizer(bs_ic_oc_target, with_v, inplace, dtype_str, dev_str, compile_graph, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
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

    def loss_fn(v_):
        out = linear_layer(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # optimizer
    optimizer = ivy.LAMB(dev_str=dev_str, inplace=inplace)

    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        optimizer.compile_on_next_step()

    # train
    loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
    linear_layer.v = optimizer.step(linear_layer.v, grads)
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
        linear_layer.v = optimizer.step(linear_layer.v, grads)
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)
