"""Collection of tests for Ivy modules."""

# global
import os
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


class ModuleWithNoneAttribute(ivy.Module):
    def __init__(self, device=None, hidden_size=64):
        self.some_attribute = None
        ivy.Module.__init__(self, device=device)

    def _forward(self, x):
        return x


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


class ModuleWithBuffer(ivy.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _forward(self, *args, **kwargs):
        pass


class ModuleWithTrainEval(ivy.Module):
    def __init__(self):
        super().__init__()

    def _forward():
        pass


@given(buffer=st.just({"var1": np.ones((1, 2))}))
def test_get_buffers(buffer, backend_fw):
    with ivy.utils.backend.ContextManager(backend_fw):
        module = ModuleWithBuffer()
        buffers = ivy.Container()
        for name, value in buffer.items():
            value = ivy.array(value)
            buffers[name] = value
            module.register_buffer(name, value)

        assert module.buffers == buffers


@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_save_and_load_as_pickled(
    batch_shape, input_channels, output_channels, on_device, backend_fw
):
    save_filepath = "module.pickled"

    # smoke test
    if backend_fw == "numpy":
        # NumPy does not support gradients
        return

    with ivy.utils.backend.ContextManager(backend_fw):
        x = ivy.astype(
            ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
            "float32",
        )
        module = TrainableModule(input_channels, output_channels, device=on_device)

        def loss_fn(v_):
            out = module(x, v=v_)
            return ivy.mean(out)

        module.save(save_filepath)
        assert os.path.exists(save_filepath)
        loaded_module = ivy.Module.load(save_filepath)

        # train
        loss, grads = ivy.execute_with_gradients(loss_fn, module.v)
        module.v = ivy.gradient_descent_update(module.v, grads, 1e-3)

        loaded_loss, loaded_grads = ivy.execute_with_gradients(loss_fn, loaded_module.v)
        loaded_module.v = ivy.gradient_descent_update(
            loaded_module.v, loaded_grads, 1e-3
        )

        # type test
        assert ivy.is_array(loaded_loss)
        assert isinstance(loaded_grads, ivy.Container)
        # cardinality test
        assert loaded_loss.shape == ()
        # value test
        assert ivy.all_equal(loaded_loss == loss)
        assert ivy.Container.all(loaded_module.v == module.v).cont_all_true()

        os.remove(save_filepath)


@given(dummy=st.booleans())
def test_module_to_device(dummy, on_device, backend_fw):
    with ivy.utils.backend.ContextManager(backend_fw):
        model = TrainableModule(5, 5)
        model.to_device(on_device)

        def assertion(x, on_device):
            if x != on_device:
                print(f"{x} is not equal to {on_device}")
                raise AssertionError

        def model_assert(mod, on_device):
            for obj in mod.v.values():
                if isinstance(obj, ivy.Module):
                    return model_assert(obj, on_device)
                if isinstance(obj, (ivy.Container, dict)):
                    for item2 in obj.values():
                        assertion(item2.device, on_device)

                else:
                    assertion(obj.device, on_device)
            if getattr(mod, "buffers", None):
                for obj in mod.buffers.values():
                    if isinstance(obj, (ivy.Container, dict)):
                        ivy.nested_map(lambda x: assertion(x.device, on_device), obj)
                    else:
                        assertion(obj.device, on_device)

        model_assert(model, on_device)


# module training
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_training(
    batch_shape, input_channels, output_channels, on_device, backend_fw
):
    # smoke test
    if backend_fw == "numpy":
        # NumPy does not support gradients
        return

    with ivy.utils.backend.ContextManager(backend_fw):
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
        # tracing test
        if backend_fw == "torch":
            # pytest scripting does not support **kwargs
            return


# module training with duplicate
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    channels=st.integers(min_value=1, max_value=64),
    same_layer=st.booleans(),
)
def test_module_training_with_duplicate(
    batch_shape, channels, same_layer, on_device, backend_fw
):
    # smoke test
    if backend_fw == "numpy":
        # NumPy does not support gradients
        return

    with ivy.utils.backend.ContextManager(backend_fw):
        x = ivy.astype(
            ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), channels),
            "float32",
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
        # tracing test
        if backend_fw == "torch":
            # pytest scripting does not support **kwargs
            return


# module with dict training
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_w_dict_training(
    batch_shape, input_channels, output_channels, on_device, backend_fw
):
    # smoke test
    if backend_fw == "numpy":
        # NumPy does not support gradients
        return

    with ivy.utils.backend.ContextManager(backend_fw):
        x = ivy.astype(
            ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
            "float32",
        )
        module = TrainableModuleWithDict(
            input_channels, output_channels, device=on_device
        )

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
        # tracing test
        if backend_fw == "torch":
            # pytest scripting does not support **kwargs
            return


# module with list training
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_w_list_training(
    batch_shape, input_channels, output_channels, on_device, backend_fw
):
    # smoke test
    if backend_fw == "numpy":
        # NumPy does not support gradients
        return

    with ivy.utils.backend.ContextManager(backend_fw):
        x = ivy.astype(
            ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
            "float32",
        )
        module = TrainableModuleWithList(
            input_channels, output_channels, device=on_device
        )

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
        # tracing test
        if backend_fw == "torch":
            # pytest scripting does not support **kwargs
            return


# module with none attribute
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_w_none_attribute(
    batch_shape, input_channels, output_channels, on_device, backend_fw
):
    # smoke test
    if backend_fw == "numpy":
        # NumPy does not support gradients
        return

    with ivy.utils.backend.ContextManager(backend_fw):
        x = ivy.astype(
            ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
            "float32",
        )
        module = ModuleWithNoneAttribute(device=on_device)
        module(x)


# module with partial v
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_module_w_partial_v(
    batch_shape, input_channels, output_channels, on_device, backend_fw
):
    # smoke test
    if backend_fw == "numpy":
        # NumPy does not support gradients
        return

    with ivy.utils.backend.ContextManager(backend_fw):
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
                input_channels,
                output_channels,
                device=on_device,
                v=v,
                with_partial_v=True,
            )
            raise Exception(
                "TrainableModule did not raise exception despite being passed "
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
                "linear2": {
                    "b": _variable(ivy.random_uniform(shape=[output_channels]))
                },
            }
        )
        try:
            TrainableModule(input_channels, output_channels, device=on_device, v=v)
            raise Exception(
                "TrainableModule did not raise exception despite being passed "
                "with wrongly shaped variables."
            )
        except ivy.utils.exceptions.IvyException:
            pass
        module = TrainableModule(
            input_channels, output_channels, device=on_device, v=v, with_partial_v=True
        )
        module(x)


@given(mode=st.booleans())
def test_train_eval(mode, backend_fw):
    with ivy.utils.backend.ContextManager(backend_fw):
        cls = ModuleWithTrainEval()
        cls.train(mode)
        assert mode == cls.training
        cls.eval()
        assert not cls.training


# with custom var structure
@given(
    batch_shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=2
    ),
    input_channels=st.integers(min_value=2, max_value=5),
    output_channels=st.integers(min_value=2, max_value=5),
)
def test_with_custom_var_structure(
    batch_shape, input_channels, output_channels, on_device, backend_fw
):
    # smoke test
    if backend_fw == "numpy":
        # NumPy does not support gradients
        return

    with ivy.utils.backend.ContextManager(backend_fw):
        module = WithCustomVarStructure(
            input_channels, output_channels, device=on_device
        )
        assert "x" in module.v
        assert "y" in module.v
        assert "z" in module.v
