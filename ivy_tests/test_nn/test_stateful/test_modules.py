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
def test_module_training(bs_ic_oc, dev_str, compile_graph, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    module = TrainableModule(input_channels, output_channels, dev_str=dev_str)
    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        module.compile_graph(x)

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
def test_module_w_list_training(bs_ic_oc, dev_str, compile_graph, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    module = TrainableModuleWithList(input_channels, output_channels, dev_str=dev_str)
    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        module.compile_graph(x)

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
def test_module_w_none_attribute(bs_ic_oc, dev_str, compile_graph, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    module = ModuleWithNoneAttribute(dev_str=dev_str)
    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        module.compile_graph(x)
    module(x)


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
def test_module_training_with_duplicate(bs_c, same_layer, dev_str, compile_graph, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, channels = bs_c
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), channels), 'float32')
    module = TrainableModuleWithDuplicate(channels, same_layer, dev_str=dev_str)
    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        module.compile_graph(x)

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
def test_module_w_dict_training(bs_ic_oc, dev_str, compile_graph, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    module = TrainableModuleWithDict(input_channels, output_channels, dev_str=dev_str)
    # compile if this mode is set
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        module.compile_graph(x)

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


class WithCustomVarStructure(ivy.Module):

    def __init__(self, in_size, out_size, dev_str=None, hidden_size=64):
        self._linear0 = ivy.Linear(in_size, hidden_size, dev_str=dev_str)
        self._linear1 = ivy.Linear(hidden_size, hidden_size, dev_str=dev_str)
        self._linear2 = ivy.Linear(hidden_size, out_size, dev_str=dev_str)
        ivy.Module.__init__(self, dev_str)

    def _create_variables(self, dev_str):
        return ivy.Container(x=self._linear0.v, y=self._linear1.v, z=self._linear2.v)

    def _forward(self, x):
        pass


# with custom var structure
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_with_custom_var_structure(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    module = WithCustomVarStructure(input_channels, output_channels, dev_str=dev_str)
    assert 'x' in module.v
    assert 'y' in module.v
    assert 'z' in module.v


class DoubleLinear(ivy.Module):

    def __init__(self, in_size, out_size, dev_str=None, hidden_size=64):
        self._l0 = ivy.Linear(in_size, hidden_size, dev_str=dev_str)
        self._l1 = ivy.Linear(hidden_size, out_size, dev_str=dev_str)
        ivy.Module.__init__(self, dev_str)

    def _forward(self, x):
        x = self._l0(x)
        x = self._l1(x)
        return x


class WithNestedModules(ivy.Module):

    def __init__(self, in_size, out_size, dev_str=None, hidden_size=64):
        self._dl0 = DoubleLinear(in_size, hidden_size, dev_str=dev_str)
        self._dl1 = DoubleLinear(hidden_size, hidden_size, dev_str=dev_str)
        ivy.Module.__init__(self, dev_str=dev_str)

    def _forward(self, x):
        x = self._dl0(x)
        x = self._dl1(x)
        return x


# top variables
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_top_variables(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    module = WithNestedModules(input_channels, output_channels, dev_str=dev_str)
    for key_chain in ['dl0', 'dl0/l0', 'dl0/l1', 'dl0/l0/b', 'dl0/l0/w', 'dl0/l1/b', 'dl0/l1/w',
                      'dl1', 'dl1/l0', 'dl1/l1', 'dl1/l0/b', 'dl1/l0/w', 'dl1/l1/b', 'dl1/l1/w']:

        # depth 1
        assert key_chain in module._dl0.top_v()
        assert key_chain in module._dl1.top_v()

        # depth 2
        assert key_chain in module._dl0._l0.top_v()
        assert key_chain in module._dl0._l1.top_v()
        assert key_chain in module._dl1._l0.top_v()
        assert key_chain in module._dl1._l1.top_v()


# top module
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_top_module(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    module = WithNestedModules(input_channels, output_channels, dev_str=dev_str)

    # full depth
    assert module._dl0.top_mod() is module
    assert module._dl1.top_mod() is module

    assert module._dl0._l0.top_mod() is module
    assert module._dl0._l1.top_mod() is module
    assert module._dl1._l0.top_mod() is module
    assert module._dl1._l1.top_mod() is module

    # depth 1
    assert module._dl0._l0.top_mod(1) is module._dl0
    assert module._dl0._l1.top_mod(1) is module._dl0
    assert module._dl1._l0.top_mod(1) is module._dl1
    assert module._dl1._l1.top_mod(1) is module._dl1


# module depth
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_module_depth(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    module = WithNestedModules(input_channels, output_channels, dev_str=dev_str)

    # depth 0
    assert module.mod_depth() == 0

    # depth 1
    assert module._dl0.mod_depth() == 1
    assert module._dl1.mod_depth() == 1

    # depth 2
    assert module._dl0._l0.mod_depth() == 2
    assert module._dl0._l1.mod_depth() == 2
    assert module._dl1._l0.mod_depth() == 2
    assert module._dl1._l1.mod_depth() == 2


# module height
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_module_height(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    module = WithNestedModules(input_channels, output_channels, dev_str=dev_str)

    # height 2
    assert module.mod_height() == 2

    # height 1
    assert module._dl0.mod_height() == 1
    assert module._dl1.mod_height() == 1

    # height 0
    assert module._dl0._l0.mod_height() == 0
    assert module._dl0._l1.mod_height() == 0
    assert module._dl1._l0.mod_height() == 0
    assert module._dl1._l1.mod_height() == 0


# sub modules
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_sub_modules(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    module = WithNestedModules(input_channels, output_channels, dev_str=dev_str)

    # depth 0
    sub_mods = module.sub_mods(depth=0)
    assert module.v is sub_mods

    # depth 1
    sub_mods = module.sub_mods(depth=1)
    for v in [module._dl0.v, module._dl1.v]:
        assert v in sub_mods

    # depth 2 (full)
    sub_mods = module.sub_mods()
    for v in [module._dl0._l0.v, module._dl0._l1.v, module._dl1._l0.v, module._dl1._l1.v]:
        assert v in sub_mods


# module intermediate returns
@pytest.mark.parametrize(
    "bs_ic_oc", [([1, 2], 4, 5)])
def test_module_intermediate_rets(bs_ic_oc, dev_str, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')
    module = WithNestedModules(input_channels, output_channels, dev_str=dev_str)

    # depth 1
    ret = module(x, with_intermediate_rets=True, intermediate_ret_depth=1)
    assert ret.shape == tuple(batch_shape + [64])
    int_rets = module.intermediate_rets
    for submod in [module._dl0, module._dl1]:
        ret = int_rets[ivy.Container.format_key(submod.__repr__(False))]
        assert ivy.is_array(ret)
        assert ret.shape == tuple(batch_shape + [64])
    for submod in [module._dl0._l0, module._dl0._l1, module._dl1._l0, module._dl1._l1]:
        assert ivy.Container.format_key(submod.__repr__(False)) not in int_rets

    # depth 2 (full)
    ret = module(x, with_intermediate_rets=True)
    assert ret.shape == tuple(batch_shape + [64])
    int_rets = module.intermediate_rets
    for submod in [module._dl0, module._dl1, module._dl0._l0, module._dl0._l1, module._dl1._l0, module._dl1._l1]:
        ret = int_rets[ivy.Container.format_key(submod.__repr__(False))]
        assert ivy.is_array(ret)
        assert ret.shape == tuple(batch_shape + [64])

    # partial submodules
    ret = module(x, with_intermediate_rets=True, intermediate_ret_submods=[module._dl1, module._dl0._l0])
    assert ret.shape == tuple(batch_shape + [64])
    int_rets = module.intermediate_rets
    for submod in [module._dl1, module._dl0._l0]:
        ret = int_rets[ivy.Container.format_key(submod.__repr__(False))]
        assert ivy.is_array(ret)
        assert ret.shape == tuple(batch_shape + [64])
    for submod in [module._dl0, module._dl0._l1, module._dl1._l0, module._dl1._l1]:
        assert ivy.Container.format_key(submod.__repr__(False)) not in int_rets
