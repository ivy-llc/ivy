"""
Collection of tests for Ivy modules
"""

# global
import pytest

# local
import ivy
import ivy_tests.helpers as helpers


class TrainableModule(ivy.Module):

    def __init__(self, in_size, out_size, dev_str='cpu', hidden_size=64):
        self._linear0 = ivy.Linear(in_size, hidden_size)
        self._linear1 = ivy.Linear(hidden_size, hidden_size)
        self._linear2 = ivy.Linear(hidden_size, out_size)
        ivy.Module.__init__(self, dev_str)

    def _forward(self, x):
        x = ivy.expand_dims(x, 0)
        x = ivy.tanh(self._linear0(x))
        x = ivy.tanh(self._linear1(x))
        return ivy.tanh(self._linear2(x))[0]


class TrainableModuleWithList(ivy.Module):

    def __init__(self, in_size, out_size, dev_str='cpu', hidden_size=64):
        linear0 = ivy.Linear(in_size, hidden_size)
        linear1 = ivy.Linear(hidden_size, hidden_size)
        linear2 = ivy.Linear(hidden_size, out_size)
        self._layers = [linear0, linear1, linear2]
        ivy.Module.__init__(self, dev_str)

    def _forward(self, x):
        x = ivy.expand_dims(x, 0)
        x = ivy.tanh(self._layers[0](x))
        x = ivy.tanh(self._layers[1](x))
        return ivy.tanh(self._layers[2](x))[0]


class ModuleWithNoneAttribute(ivy.Module):

    def __init__(self, dev_str='cpu', hidden_size=64):
        self.some_attribute = None
        ivy.Module.__init__(self, dev_str)

    def _forward(self, x):
        return x


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
    module = TrainableModule(input_channels, output_channels)

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
    helpers.assert_compilable(loss_fn)


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
    module = TrainableModuleWithList(input_channels, output_channels)

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
    assert ivy.reduce_max(ivy.abs(grads.layers0.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers0.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers1.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers1.w)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers2.b)) > 0
    assert ivy.reduce_max(ivy.abs(grads.layers2.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not support **kwargs
        return
    helpers.assert_compilable(loss_fn)


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
    module = ModuleWithNoneAttribute()
