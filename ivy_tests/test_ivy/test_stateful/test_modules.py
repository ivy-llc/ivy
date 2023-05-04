"""Collection of tests for Ivy modules."""

# global
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy
from ivy.functional.ivy.gradients import _variable
import ivy_tests.test_ivy.helpers as helpers


class TrainableModule(ivy.Module):
    def __init__(
        self,
        in_size,
        out_size,
        device=None,
        hidden_size=64,
        v=None,
        with_partial_v=False,
    ):
        self._linear0 = ivy.Linear(in_size, hidden_size, device=device)
        self._linear1 = ivy.Linear(hidden_size, hidden_size, device=device)
        self._linear2 = ivy.Linear(hidden_size, out_size, device=device)
        ivy.Module.__init__(self, device=device, v=v, with_partial_v=with_partial_v)

    def _forward(self, x):
        x = ivy.expand_dims(x, axis=0)
        x = ivy.tanh(self._linear0(x))
        x = ivy.tanh(self._linear1(x))
        return ivy.tanh(self._linear2(x))[0]


# module training
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_training(batch_shape, input_channels, output_channels, on_device):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    module = TrainableModule(input_channels, output_channels, device=on_device)

    def loss_fn(v_):
        out = module(x, v=v_)
        return ivy.mean(out)

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
    assert loss.shape == ()
    # value test
    assert ivy.max(ivy.abs(grads.linear0.b)) > 0
    assert ivy.max(ivy.abs(grads.linear0.w)) > 0
    assert ivy.max(ivy.abs(grads.linear1.b)) > 0
    assert ivy.max(ivy.abs(grads.linear1.w)) > 0
    assert ivy.max(ivy.abs(grads.linear2.b)) > 0
    assert ivy.max(ivy.abs(grads.linear2.w)) > 0
    # compilation test
    if ivy.current_backend_str() == "torch":
        # pytest scripting does not support **kwargs
        return


class TrainableModuleWithList(ivy.Module):
    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        linear0 = ivy.Linear(in_size, hidden_size, device=device)
        linear1 = ivy.Linear(hidden_size, hidden_size, device=device)
        linear2 = ivy.Linear(hidden_size, out_size, device=device)
        self._layers = [linear0, linear1, linear2]
        ivy.Module.__init__(self, device=device)

    def _forward(self, x):
        x = ivy.expand_dims(x, axis=0)
        x = ivy.tanh(self._layers[0](x))
        x = ivy.tanh(self._layers[1](x))
        return ivy.tanh(self._layers[2](x))[0]


# module with list training
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_w_list_training(
    batch_shape, input_channels, output_channels, on_device
):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    module = TrainableModuleWithList(input_channels, output_channels, device=on_device)

    def loss_fn(v_):
        out = module(x, v=v_)
        return ivy.mean(out)

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
    assert loss.shape == ()
    # value test
    assert ivy.max(ivy.abs(grads.layers.v0.b)) > 0
    assert ivy.max(ivy.abs(grads.layers.v0.w)) > 0
    assert ivy.max(ivy.abs(grads.layers.v1.b)) > 0
    assert ivy.max(ivy.abs(grads.layers.v1.w)) > 0
    assert ivy.max(ivy.abs(grads.layers.v2.b)) > 0
    assert ivy.max(ivy.abs(grads.layers.v2.w)) > 0
    # compilation test
    if ivy.current_backend_str() == "torch":
        # pytest scripting does not support **kwargs
        return


# module with partial v
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_w_partial_v(batch_shape, input_channels, output_channels, on_device):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients

        return

    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    v = ivy.Container(
        {
            "linear0": {
                "b": _variable(ivy.random_uniform(shape=[64])),
                "w": _variable(ivy.random_uniform(shape=[64, 4])),
            },
            "linear1": {
                "b": _variable(ivy.random_uniform(shape=[64])),
                "w": _variable(ivy.random_uniform(shape=[64, 64])),
                "extra": _variable(ivy.random_uniform(shape=[64, 64])),
            },
            "linear2": {
                "b": _variable(ivy.random_uniform(shape=[5])),
                "w": _variable(ivy.random_uniform(shape=[5, 64])),
            },
        }
    )
    try:
        TrainableModule(
            input_channels, output_channels, device=on_device, v=v, with_partial_v=True
        )
        raise Exception(
            "TrainableModule did not raise exception desipite being passed "
            "with wrongly shaped variables."
        )
    except ivy.utils.exceptions.IvyException:
        pass
    v = ivy.Container(
        {
            "linear0": {
                "b": _variable(ivy.random_uniform(shape=[64])),
            },
            "linear1": {"w": _variable(ivy.random_uniform(shape=[64, 64]))},
            "linear2": {"b": _variable(ivy.random_uniform(shape=[output_channels]))},
        }
    )
    try:
        TrainableModule(input_channels, output_channels, device=on_device, v=v)
        raise Exception(
            "TrainableModule did not raise exception desipite being passed "
            "with wrongly shaped variables."
        )
    except ivy.utils.exceptions.IvyException:
        pass
    module = TrainableModule(
        input_channels, output_channels, device=on_device, v=v, with_partial_v=True
    )
    module(x)


class ModuleWithNoneAttribute(ivy.Module):
    def __init__(self, device=None, hidden_size=64):
        self.some_attribute = None
        ivy.Module.__init__(self, device=device)

    def _forward(self, x):
        return x


# module with none attribute
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_w_none_attribute(
    batch_shape, input_channels, output_channels, on_device
):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return

    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    module = ModuleWithNoneAttribute(device=on_device)
    module(x)


class TrainableModuleWithDuplicate(ivy.Module):
    def __init__(self, channels, same_layer, device=None):
        if same_layer:
            linear = ivy.Linear(channels, channels, device=device)
            self._linear0 = linear
            self._linear1 = linear
        else:
            w = _variable(ivy.ones((channels, channels)))
            b0 = _variable(ivy.ones((channels,)))
            b1 = _variable(ivy.ones((channels,)))
            v0 = ivy.Container({"w": w, "b": b0})
            v1 = ivy.Container({"w": w, "b": b1})
            self._linear0 = ivy.Linear(channels, channels, device=device, v=v0)
            self._linear1 = ivy.Linear(channels, channels, device=device, v=v1)
        ivy.Module.__init__(self)

    def _forward(self, x):
        x = self._linear0(x)
        return self._linear1(x)


# module training with duplicate
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    channels=st.integers(min_value=1, max_value=64),
    same_layer=st.booleans(),
)
def test_module_training_with_duplicate(batch_shape, channels, same_layer, on_device):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), channels), "float32"
    )
    module = TrainableModuleWithDuplicate(channels, same_layer, device=on_device)

    def loss_fn(v_):
        out = module(x, v=v_)
        return ivy.mean(out)

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
    assert loss.shape == ()
    # value test
    assert ivy.max(ivy.abs(grads.linear0.b)) > 0
    assert ivy.max(ivy.abs(grads.linear0.w)) > 0
    if not same_layer:
        assert ivy.max(ivy.abs(grads.linear1.b)) > 0
    # compilation test
    if ivy.current_backend_str() == "torch":
        # pytest scripting does not support **kwargs
        return


class TrainableModuleWithDict(ivy.Module):
    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        linear0 = ivy.Linear(in_size, hidden_size, device=device)
        linear1 = ivy.Linear(hidden_size, hidden_size, device=device)
        linear2 = ivy.Linear(hidden_size, out_size, device=device)
        self._layers = {"linear0": linear0, "linear1": linear1, "linear2": linear2}
        ivy.Module.__init__(self, device=device)

    def _forward(self, x):
        x = ivy.expand_dims(x, axis=0)
        x = ivy.tanh(self._layers["linear0"](x))
        x = ivy.tanh(self._layers["linear1"](x))
        return ivy.tanh(self._layers["linear2"](x))[0]


# module with dict training
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_w_dict_training(
    batch_shape, input_channels, output_channels, on_device
):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    module = TrainableModuleWithDict(input_channels, output_channels, device=on_device)

    def loss_fn(v_):
        out = module(x, v=v_)
        return ivy.mean(out)

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
    assert loss.shape == ()
    # value test
    assert ivy.max(ivy.abs(grads.layers.linear0.b)) > 0
    assert ivy.max(ivy.abs(grads.layers.linear0.w)) > 0
    assert ivy.max(ivy.abs(grads.layers.linear1.b)) > 0
    assert ivy.max(ivy.abs(grads.layers.linear1.w)) > 0
    assert ivy.max(ivy.abs(grads.layers.linear2.b)) > 0
    assert ivy.max(ivy.abs(grads.layers.linear2.w)) > 0
    # compilation test
    if ivy.current_backend_str() == "torch":
        # pytest scripting does not support **kwargs
        return


class WithCustomVarStructure(ivy.Module):
    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        self._linear0 = ivy.Linear(in_size, hidden_size, device=device)
        self._linear1 = ivy.Linear(hidden_size, hidden_size, device=device)
        self._linear2 = ivy.Linear(hidden_size, out_size, device=device)
        ivy.Module.__init__(self, device=device)

    def _create_variables(self, device, dtype):
        return ivy.Container(x=self._linear0.v, y=self._linear1.v, z=self._linear2.v)

    def _forward(self, x):
        pass


# with custom var structure
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_with_custom_var_structure(
    batch_shape, input_channels, output_channels, on_device
):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return
    module = WithCustomVarStructure(input_channels, output_channels, device=on_device)
    assert "x" in module.v
    assert "y" in module.v
    assert "z" in module.v


class DoubleLinear(ivy.Module):
    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        self._l0 = ivy.Linear(in_size, hidden_size, device=device)
        self._l1 = ivy.Linear(hidden_size, out_size, device=device)
        ivy.Module.__init__(self, device=device)

    def _forward(self, x):
        x = self._l0(x)
        x = self._l1(x)
        return x


class WithNestedModules(ivy.Module):
    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        self._dl0 = DoubleLinear(in_size, hidden_size, device=device)
        self._dl1 = DoubleLinear(hidden_size, hidden_size, device=device)
        ivy.Module.__init__(self, device=device)

    def _forward(self, x):
        x = self._dl0(x)
        x = self._dl1(x)
        x = self._dl1(x)
        return x


# top variables
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_top_variables(batch_shape, input_channels, output_channels, on_device):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return
    module = WithNestedModules(input_channels, output_channels, device=on_device)
    for key_chain in [
        "dl0",
        "dl0/l0",
        "dl0/l1",
        "dl0/l0/b",
        "dl0/l0/w",
        "dl0/l1/b",
        "dl0/l1/w",
        "dl1",
        "dl1/l0",
        "dl1/l1",
        "dl1/l0/b",
        "dl1/l0/w",
        "dl1/l1/b",
        "dl1/l1/w",
    ]:
        # depth 1
        assert key_chain in module._dl0.top_v()
        assert key_chain in module._dl1.top_v()

        # depth 2
        assert key_chain in module._dl0._l0.top_v()
        assert key_chain in module._dl0._l1.top_v()
        assert key_chain in module._dl1._l0.top_v()
        assert key_chain in module._dl1._l1.top_v()


# top module
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_top_module(batch_shape, input_channels, output_channels, on_device):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return

    module = WithNestedModules(input_channels, output_channels, device=on_device)

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


# v with top v key chains
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_v_with_top_v_key_chains(
    batch_shape, input_channels, output_channels, on_device
):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return

    module = WithNestedModules(input_channels, output_channels, device=on_device)

    # full depth
    v = module._dl0.v_with_top_v_key_chains()
    assert "dl0" in v
    assert v.dl0 is module._dl0.v

    v = module._dl1.v_with_top_v_key_chains()
    assert "dl1" in v
    assert v.dl1 is module._dl1.v

    v = module._dl0._l0.v_with_top_v_key_chains()
    assert "dl0" in v
    assert "l0" in v.dl0
    assert v.dl0.l0 is module._dl0._l0.v

    v = module._dl0._l1.v_with_top_v_key_chains()
    assert "dl0" in v
    assert "l1" in v.dl0
    assert v.dl0.l1 is module._dl0._l1.v

    v = module._dl1._l0.v_with_top_v_key_chains()
    assert "dl1" in v
    assert "l0" in v.dl1
    assert v.dl1.l0 is module._dl1._l0.v

    v = module._dl1._l1.v_with_top_v_key_chains()
    assert "dl1" in v
    assert "l1" in v.dl1
    assert v.dl1.l1 is module._dl1._l1.v

    # depth 1

    v = module._dl0._l0.v_with_top_v_key_chains(depth=1)
    assert "l0" in v
    assert v.l0 is module._dl0._l0.v

    v = module._dl0._l1.v_with_top_v_key_chains(depth=1)
    assert "l1" in v
    assert v.l1 is module._dl0._l1.v

    v = module._dl1._l0.v_with_top_v_key_chains(depth=1)
    assert "l0" in v
    assert v.l0 is module._dl1._l0.v

    v = module._dl1._l1.v_with_top_v_key_chains(depth=1)
    assert "l1" in v
    assert v.l1 is module._dl1._l1.v


# module depth
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_depth(batch_shape, input_channels, output_channels, on_device):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return

    module = WithNestedModules(input_channels, output_channels, device=on_device)

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
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_height(batch_shape, input_channels, output_channels, on_device):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return

    module = WithNestedModules(input_channels, output_channels, device=on_device)

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
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_sub_modules(batch_shape, input_channels, output_channels, on_device):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return

    module = WithNestedModules(input_channels, output_channels, device=on_device)

    # depth 0
    sub_mods = module.sub_mods(depth=0)
    assert module.v is sub_mods

    # depth 1
    sub_mods = module.sub_mods(depth=1)
    for v in [module._dl0.v, module._dl1.v]:
        assert v in sub_mods

    # depth 2 (full)
    sub_mods = module.sub_mods()
    for v in [
        module._dl0._l0.v,
        module._dl0._l1.v,
        module._dl1._l0.v,
        module._dl1._l1.v,
    ]:
        assert v in sub_mods


# track submod returns
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_track_submod_rets(
    batch_shape, input_channels, output_channels, on_device
):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return

    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    module = WithNestedModules(input_channels, output_channels, device=on_device)

    # depth 1
    ret = module(x, track_submod_rets=True, submod_depth=1)
    assert ret.shape == tuple(list(batch_shape) + [64])
    sm_rets = module.submod_rets
    for submod in [module._dl0, module._dl1]:
        for ret in sm_rets[submod.get_mod_key()]:
            assert isinstance(ret, np.ndarray)
            assert ret.shape == tuple(list(batch_shape) + [64])
    for submod in [module._dl0._l0, module._dl0._l1, module._dl1._l0, module._dl1._l1]:
        assert (
            ivy.Container.cont_flatten_key_chain(submod.__repr__(), "_") not in sm_rets
        )

    # depth 2 (full)
    ret = module(x, track_submod_rets=True)
    assert ret.shape == tuple(list(batch_shape) + [64])
    sm_rets = module.submod_rets
    for submod in [
        module._dl0,
        module._dl1,
        module._dl0._l0,
        module._dl0._l1,
        module._dl1._l0,
        module._dl1._l1,
    ]:
        for ret in sm_rets[submod.get_mod_key()]:
            assert isinstance(ret, np.ndarray)
            assert ret.shape == tuple(list(batch_shape) + [64])

    # partial submodules
    ret = module(
        x, track_submod_rets=True, submods_to_track=[module._dl1, module._dl0._l0]
    )
    assert ret.shape == tuple(list(batch_shape) + [64])
    sm_rets = module.submod_rets
    for submod in [module._dl1, module._dl0._l0]:
        for ret in sm_rets[submod.get_mod_key()]:
            assert isinstance(ret, np.ndarray)
            assert ret.shape == tuple(list(batch_shape) + [64])
    for submod in [module._dl0, module._dl0._l1, module._dl1._l0, module._dl1._l1]:
        assert (
            ivy.Container.cont_flatten_key_chain(submod.__repr__(), "_") not in sm_rets
        )


# check submod returns
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_check_submod_rets(
    batch_shape, input_channels, output_channels, on_device
):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return

    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    module = WithNestedModules(input_channels, output_channels, device=on_device)

    # depth 1
    ret = module(x, track_submod_rets=True, submod_depth=1)
    assert ret.shape == tuple(list(batch_shape) + [64])
    sm_rets = module.submod_rets
    module(x, expected_submod_rets=sm_rets)
    sm_rets.random_uniform(map_sequences=True)
    try:
        module(x, expected_submod_rets=sm_rets.random_uniform(map_sequences=True))
        raise Exception(
            "forward pass succeeded despite passing random expected_submod_rets, "
            "assertion error expected."
        )
    except ivy.utils.exceptions.IvyException:
        pass

    # depth 2 (full)
    ret = module(x, track_submod_rets=True)
    assert ret.shape == tuple(list(batch_shape) + [64])
    sm_rets = module.submod_rets
    module(x, expected_submod_rets=sm_rets)
    try:
        module(x, expected_submod_rets=sm_rets.random_uniform(map_sequences=True))
        raise Exception(
            "forward pass succeeded despite passing random expected_submod_rets, "
            "assertion error expected."
        )
    except ivy.utils.exceptions.IvyException:
        pass

    # partial submodules
    ret = module(
        x, track_submod_rets=True, submods_to_track=[module._dl1, module._dl0._l0]
    )
    assert ret.shape == tuple(list(batch_shape) + [64])
    sm_rets = module.submod_rets
    module(x, expected_submod_rets=sm_rets)
    try:
        module(x, expected_submod_rets=sm_rets.random_uniform(map_sequences=True))
        raise Exception(
            "forward pass succeeded despite passing random expected_submod_rets, "
            "assertion error expected."
        )
    except ivy.utils.exceptions.IvyException:
        pass

    # with tolerances
    ret = module(x, track_submod_rets=True)
    assert ret.shape == tuple(list(batch_shape) + [64])
    sm_rets_orig = module.submod_rets
    sm_rets = ivy.Container(
        {
            k: {"val": v, "atol": [1e-8] * len(v), "rtol": [1e-5] * len(v)}
            for k, v in sm_rets_orig.items()
        },
        **sm_rets_orig._config
    )
    module(x, expected_submod_rets=sm_rets)
    sm_rets = ivy.Container(
        {k: {"val": v, "atol": 1e-8, "rtol": 1e-5} for k, v in sm_rets_orig.items()},
        **sm_rets_orig._config
    )
    module(x, expected_submod_rets=sm_rets)
    try:
        module(x, expected_submod_rets=sm_rets.random_uniform(map_sequences=True))
        raise Exception(
            "forward pass succeeded despite passing random expected_submod_rets, "
            "assertion error expected."
        )
    except ivy.utils.exceptions.IvyException:
        pass


# track submod call order
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_track_submod_call_order(
    batch_shape, input_channels, output_channels, on_device
):
    # smoke test
    if ivy.current_backend_str() == "numpy":
        # NumPy does not support gradients
        return

    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    module = WithNestedModules(input_channels, output_channels, device=on_device)

    root_key_0 = ivy.Container.cont_flatten_key_chain(module.__repr__(), "_") + "_0"

    dl0_key_0 = ivy.Container.cont_flatten_key_chain(module._dl0.__repr__(), "_") + "_0"
    dl1_key_0 = ivy.Container.cont_flatten_key_chain(module._dl1.__repr__(), "_") + "_0"
    dl1_key_1 = ivy.Container.cont_flatten_key_chain(module._dl1.__repr__(), "_") + "_1"

    dl0_l0_key_0 = (
        ivy.Container.cont_flatten_key_chain(module._dl0._l0.__repr__(), "_") + "_0"
    )
    dl0_l1_key_0 = (
        ivy.Container.cont_flatten_key_chain(module._dl0._l1.__repr__(), "_") + "_0"
    )
    dl1_l0_key_0 = (
        ivy.Container.cont_flatten_key_chain(module._dl1._l0.__repr__(), "_") + "_0"
    )
    dl1_l1_key_0 = (
        ivy.Container.cont_flatten_key_chain(module._dl1._l1.__repr__(), "_") + "_0"
    )

    # depth 1
    ret = module(x, track_submod_call_order=True, submod_depth=1)
    assert ret.shape == tuple(list(batch_shape) + [64])

    sm_co = module.submod_call_order

    assert root_key_0 in sm_co

    assert dl0_key_0 in sm_co[root_key_0]
    assert dl1_key_0 in sm_co[root_key_0]
    assert dl1_key_1 in sm_co[root_key_0]

    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl0_key_0],
            module._dl0.v.cont_flatten_key_chains().to_numpy(),
        ]
    )
    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl1_key_0],
            module._dl1.v.cont_flatten_key_chains().to_numpy(),
        ]
    )
    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl1_key_1],
            module._dl1.v.cont_flatten_key_chains().to_numpy(),
        ]
    )

    # depth 2 (full)
    ret = module(x, track_submod_call_order=True)
    assert ret.shape == tuple(list(batch_shape) + [64])

    sm_co = module.submod_call_order

    assert root_key_0 in sm_co

    assert dl0_key_0 in sm_co[root_key_0]
    assert dl1_key_0 in sm_co[root_key_0]
    assert dl1_key_1 in sm_co[root_key_0]

    assert dl0_l0_key_0 in sm_co[root_key_0][dl0_key_0]
    assert dl0_l1_key_0 in sm_co[root_key_0][dl0_key_0]
    assert dl1_l0_key_0 in sm_co[root_key_0][dl1_key_0]
    assert dl1_l1_key_0 in sm_co[root_key_0][dl1_key_0]
    assert dl1_l0_key_0 in sm_co[root_key_0][dl1_key_1]
    assert dl1_l1_key_0 in sm_co[root_key_0][dl1_key_1]

    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl0_key_0][dl0_l0_key_0],
            module._dl0._l0.v.cont_flatten_key_chains().to_numpy(),
        ]
    )
    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl0_key_0][dl0_l1_key_0],
            module._dl0._l1.v.cont_flatten_key_chains().to_numpy(),
        ]
    )
    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl1_key_0][dl1_l0_key_0],
            module._dl1._l0.v.cont_flatten_key_chains().to_numpy(),
        ]
    )
    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl1_key_0][dl1_l1_key_0],
            module._dl1._l1.v.cont_flatten_key_chains().to_numpy(),
        ]
    )
    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl1_key_1][dl1_l0_key_0],
            module._dl1._l0.v.cont_flatten_key_chains().to_numpy(),
        ]
    )
    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl1_key_1][dl1_l1_key_0],
            module._dl1._l1.v.cont_flatten_key_chains().to_numpy(),
        ]
    )

    # partial submodules
    ret = module(
        x, track_submod_call_order=True, submods_to_track=[module._dl1, module._dl0._l0]
    )
    assert ret.shape == tuple(list(batch_shape) + [64])

    sm_co = module.submod_call_order

    assert root_key_0 in sm_co

    assert dl0_key_0 in sm_co[root_key_0]
    assert dl1_key_0 in sm_co[root_key_0]
    assert dl1_key_1 in sm_co[root_key_0]

    assert dl0_l0_key_0 in sm_co[root_key_0][dl0_key_0]
    assert dl0_l1_key_0 not in sm_co[root_key_0][dl0_key_0]
    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl1_key_0],
            module._dl1.v.cont_flatten_key_chains().to_numpy(),
        ]
    )
    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl1_key_1],
            module._dl1.v.cont_flatten_key_chains().to_numpy(),
        ]
    )

    assert ivy.Container.cont_identical(
        [
            sm_co[root_key_0][dl0_key_0][dl0_l0_key_0],
            module._dl0._l0.v.cont_flatten_key_chains().to_numpy(),
        ]
    )
