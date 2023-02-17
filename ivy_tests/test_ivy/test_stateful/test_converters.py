"""Collection of tests for module converters."""

# global
import pytest
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = SimpleNamespace()
    torch.tanh = SimpleNamespace
    nn = SimpleNamespace()
    nn.Module = SimpleNamespace
    nn.Linear = SimpleNamespace
    torch.optim = SimpleNamespace()
    torch.optim.SGD = SimpleNamespace
    nn.L1Loss = SimpleNamespace

try:
    import jax
    from jax import value_and_grad
    import haiku as hk
    import jax.numpy as jnp
except ImportError:
    jax = SimpleNamespace()
    value_and_grad = SimpleNamespace
    hk = SimpleNamespace()
    hk.Module = SimpleNamespace
    hk.Linear = SimpleNamespace
    hk.transform = SimpleNamespace()
    hk.transform.init = SimpleNamespace
    jnp = SimpleNamespace()
    jnp.expand_dims = SimpleNamespace
    jnp.tanh = SimpleNamespace
    jnp.mean = SimpleNamespace
    jax.random = SimpleNamespace()
    jax.random.PRNGKey = SimpleNamespace
    jax.tree_map = SimpleNamespace

try:
    import tensorflow as tf
except ImportError:
    tf = SimpleNamespace()
    tf.expand_dims = SimpleNamespace
    tf.tanh = SimpleNamespace
    tf.keras = SimpleNamespace()
    tf.keras.Model = SimpleNamespace
    tf.keras.layers = SimpleNamespace()
    tf.keras.layers.Dense = SimpleNamespace
    tf.keras.optimizers = SimpleNamespace()
    tf.keras.optimizers.SGD = SimpleNamespace()
    tf.keras.optimizers.SGD.apply_gradients = SimpleNamespace
    tf.keras.losses = SimpleNamespace()
    tf.keras.losses.MeanAbsoluteError = SimpleNamespace
    tf.GradientTape = SimpleNamespace()
    tf.GradientTape.tape = SimpleNamespace
    tf.GradientTape.watch = SimpleNamespace


# local
import ivy


class IvyModel(ivy.Module):
    def __init__(self, in_size, out_size, hidden_units=64):
        self.linear0 = ivy.Linear(in_size, hidden_units)
        self.linear1 = ivy.Linear(hidden_units, 64)
        self.linear2 = ivy.Linear(hidden_units, out_size)
        ivy.Module.__init__(self)

    def _forward(self, x, *args, **kwargs):
        x = ivy.relu(self.linear0(x))
        x = ivy.relu(self.linear1(x))
        return ivy.sigmoid(self.linear2(x))


class TensorflowLinear(tf.keras.Model):
    def __init__(self, out_size):
        super(TensorflowLinear, self).__init__()
        self._linear = tf.keras.layers.Dense(out_size)

    def build(self, input_shape):
        super(TensorflowLinear, self).build(input_shape)

    def call(self, x):
        return self._linear(x)


class TensorflowModule(tf.keras.Model):
    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        super(TensorflowModule, self).__init__()
        self._linear0 = TensorflowLinear(hidden_size)
        self._linear1 = TensorflowLinear(hidden_size)
        self._linear2 = TensorflowLinear(out_size)

    def call(self, x):
        x = tf.expand_dims(x, 0)
        x = tf.tanh(self._linear0(x))
        x = tf.tanh(self._linear1(x))
        return tf.tanh(self._linear2(x))[0]


class TorchLinearModule(nn.Module):
    def __init__(self, in_size, out_size):
        super(TorchLinearModule, self).__init__()
        self._linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self._linear(x)


class TorchModule(nn.Module):
    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        super(TorchModule, self).__init__()
        self._linear0 = TorchLinearModule(in_size, hidden_size)
        self._linear1 = TorchLinearModule(hidden_size, hidden_size)
        self._linear2 = TorchLinearModule(hidden_size, out_size)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = torch.tanh(self._linear0(x))
        x = torch.tanh(self._linear1(x))
        return torch.tanh(self._linear2(x))[0]


class HaikuLinear(hk.Module):
    def __init__(self, out_size):
        super(HaikuLinear, self).__init__()
        self._linear = hk.Linear(out_size)

    def __call__(self, x):
        return self._linear(x)


class HaikuModule(hk.Module):
    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        super(HaikuModule, self).__init__()
        self._linear0 = HaikuLinear(hidden_size)
        self._linear1 = HaikuLinear(hidden_size)
        self._linear2 = HaikuLinear(out_size)

    def __call__(self, x):
        x = jnp.expand_dims(x, 0)
        x = jnp.tanh(self._linear0(x))
        x = jnp.tanh(self._linear1(x))
        return jnp.tanh(self._linear2(x))[0]


NATIVE_MODULES = {
    "torch": TorchModule,
    "jax": HaikuModule,
    "tensorflow": TensorflowModule,
}

FROM_CONVERTERS = {
    "torch": ivy.Module.from_torch_module,
    "jax": ivy.Module.from_haiku_module,
    "tensorflow": ivy.Module.from_keras_module,
}


@pytest.mark.parametrize("bs_ic_oc", [([1, 2], 4, 5)])
@pytest.mark.parametrize("from_class_and_args", [True, False])
def test_from_backend_module(bs_ic_oc, from_class_and_args):
    # smoke test
    if ivy.current_backend_str() in "numpy":
        # Converters not implemented in numpy
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    native_module_class = NATIVE_MODULES[ivy.current_backend_str()]
    module_converter = FROM_CONVERTERS[ivy.current_backend_str()]

    if from_class_and_args:
        ivy_module = module_converter(
            native_module_class,
            instance_args=[x],
            constructor_kwargs={"in_size": input_channels, "out_size": output_channels},
        )
    else:
        if ivy.current_backend_str() == "jax":

            def forward_fn(*a, **kw):
                model = native_module_class(input_channels, output_channels)
                return model(ivy.to_native(x))

            native_module = hk.transform(forward_fn)
        elif ivy.current_backend_str() == "tensorflow":
            native_module = native_module_class(
                in_size=input_channels, out_size=output_channels
            )
            native_module.build((input_channels,))
        else:
            native_module = native_module_class(
                in_size=input_channels, out_size=output_channels
            )

        fw_kwargs = {}
        if ivy.current_backend_str() == "jax":
            fw_kwargs["params_hk"] = native_module.init(0, x)
        ivy_module = module_converter(native_module, **fw_kwargs)

    def loss_fn(v_=None):
        out = ivy_module(x, v=v_)
        return ivy.mean(out)

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    loss_fn()  # for on-call mode

    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, ivy_module.v)
        if ivy.current_backend_str() == "jax":
            ivy_module.v = ivy.gradient_descent_update(ivy_module.v, grads, 1e-3)
        else:
            w = ivy.gradient_descent_update(ivy_module.v, grads, 1e-3)
            ivy.inplace_update(ivy_module.v, w)
        assert loss < loss_tm1
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).cont_all_true()
