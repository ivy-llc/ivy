"""
Collection of tests for Ivy modules
"""

# global
import pytest

# local
import ivy
import ivy_tests.helpers as helpers


class TrainableModule(ivy.Module):

    def __init__(self, in_size, out_size, dev_str=None, hidden_size=64):
        self._linear0 = ivy.Linear(in_size, hidden_size, dev_str=dev_str)
        self._linear1 = ivy.Linear(hidden_size, hidden_size, dev_str=dev_str)
        self._linear2 = ivy.Linear(hidden_size, out_size, dev_str=dev_str)
        ivy.Module.__init__(self, dev_str)

    def _forward(self, x):
        x = ivy.expand_dims(x, 0)
        x = ivy.tanh(self._linear0(x))
        x = ivy.tanh(self._linear1(x))
        return ivy.tanh(self._linear2(x))[0]


# module training
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_module_training(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    module = TrainableModule(input_channels, output_channels, dev_str=dev_str)

    def loss_fn(v_):
        out = module(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, module.v)
        module.v = ivy.gradient_descent_update(module.v, grads, 1e-3)
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
    assert ivy.reduce_max(ivy.abs(grads.linear0.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear0.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear1.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear1.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear2.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear2.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not support **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


class TrainableModuleWithList(ivy.Module):

    def __init__(self, in_size, out_size, dev_str=None, hidden_size=64):
        linear0 = ivy.Linear(in_size, hidden_size, dev_str=dev_str)
        linear1 = ivy.Linear(hidden_size, hidden_size, dev_str=dev_str)
        linear2 = ivy.Linear(hidden_size, out_size, dev_str=dev_str)
        self._layers = [linear0, linear1, linear2]
        ivy.Module.__init__(self, dev_str)

    def _forward(self, x):
        x = ivy.expand_dims(x, 0)
        x = ivy.tanh(self._layers[0](x))
        x = ivy.tanh(self._layers[1](x))
        return ivy.tanh(self._layers[2](x))[0]


# module with list training
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_module_w_list_training(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    module = TrainableModuleWithList(input_channels, output_channels, dev_str=dev_str)

    def loss_fn(v_):
        out = module(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, module.v)
        module.v = ivy.gradient_descent_update(module.v, grads, 1e-3)
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
    assert ivy.reduce_max(ivy.abs(grads.layers.v0.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.v0.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.v1.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.v1.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.v2.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.v2.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not support **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


class ModuleWithNoneAttribute(ivy.Module):

    def __init__(self, dev_str=None, hidden_size=64):
        self.some_attribute = None
        ivy.Module.__init__(self, dev_str)

    def _forward(self, x):
        return x


# module with none attribute
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_module_w_none_attribute(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    module = ModuleWithNoneAttribute(dev_str=dev_str)


class TrainableModuleWithDuplicate(ivy.Module):

    def __init__(self, channels, same_layer, dev_str=None):
        if same_layer:
            linear = ivy.Linear(channels, channels, dev_str=dev_str)
            self._linear0 = linear
            self._linear1 = linear
        else:
            w = ivy.variable(ivy.ones((channels, channels)))
            b0 = ivy.variable(ivy.ones((channels,)))
            b1 = ivy.variable(ivy.ones((channels,)))
            v0 = ivy.Container({'w': w, 'b': b0})
            v1 = ivy.Container({'w': w, 'b': b1})
            self._linear0 = ivy.Linear(channels, channels, dev_str=dev_str, v=v0)
            self._linear1 = ivy.Linear(channels, channels,dev_str=dev_str, v=v1)
        ivy.Module.__init__(self)

    def _forward(self, x):
        x = self._linear0(x)
        return self._linear1(x)


# module training with duplicate
@pytest.mark.parametrize(
    "bs_c", [([1, 2], 64)])
@pytest.mark.parametrize(
    "same_layer", [True, False])
def test_module_training_with_duplicate(bs_c, same_layer, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, channels = bs_c
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), channels), 'float32')
    module = TrainableModuleWithDuplicate(channels, same_layer, dev_str=dev_str)

    def loss_fn(v_):
        out = module(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, module.v)
        module.v = ivy.gradient_descent_update(module.v, grads, 1e-3)
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
    assert ivy.reduce_max(ivy.abs(grads.linear0.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.linear0.w)) > 0
    if not same_layer:
        assert ivy.reduce_max(ivy.abs(grads.linear1.b)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not support **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)


class TrainableModuleWithDict(ivy.Module):

    def __init__(self, in_size, out_size, dev_str=None, hidden_size=64):
        linear0 = ivy.Linear(in_size, hidden_size, dev_str=dev_str)
        linear1 = ivy.Linear(hidden_size, hidden_size, dev_str=dev_str)
        linear2 = ivy.Linear(hidden_size, out_size, dev_str=dev_str)
        self._layers = {'linear0': linear0, 'linear1': linear1, 'linear2': linear2}
        ivy.Module.__init__(self, dev_str)

    def _forward(self, x):
        x = ivy.expand_dims(x, 0)
        x = ivy.tanh(self._layers['linear0'](x))
        x = ivy.tanh(self._layers['linear1'](x))
        return ivy.tanh(self._layers['linear2'](x))[0]


# module with dict training
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_module_w_dict_training(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    module = TrainableModuleWithDict(input_channels, output_channels, dev_str=dev_str)

    def loss_fn(v_):
        out = module(x, v=v_)
        return ivy.reduce_mean(out)[0]

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, module.v)
        module.v = ivy.gradient_descent_update(module.v, grads, 1e-3)
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
    assert ivy.reduce_max(ivy.abs(grads.layers.linear0.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.linear0.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.linear1.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.linear1.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.linear2.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers.linear2.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not support **kwargs
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(loss_fn)
