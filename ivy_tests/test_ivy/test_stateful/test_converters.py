"""Collection of tests for module converters."""

# global
import pytest
import torch
import torch.nn as nn
import jax
from jax import value_and_grad
import haiku as hk
import jax.numpy as jnp
import tensorflow as tf
# local
import ivy
import ivy.functional.backends.torch as torch_
import ivy.functional.backends.tensorflow as tf_
import ivy.functional.backends.jax as jax_


class IvyModel(ivy.Module):
    def __init__(self, *, in_size, out_size, hidden_units=64):
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
    def __init__(self, *, in_size, out_size, device=None, hidden_size=64):
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
    def __init__(self, *, in_size, out_size, device=None, hidden_size=64):
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
    def __init__(self, *, in_size, out_size, device=None, hidden_size=64):
        super(HaikuModule, self).__init__()
        self._linear0 = HaikuLinear(hidden_size)
        self._linear1 = HaikuLinear(hidden_size)
        self._linear2 = HaikuLinear(out_size)

    def __call__(self, x):
        x = jnp.expand_dims(x, 0)
        x = jnp.tanh(self._linear0(x))
        x = jnp.tanh(self._linear1(x))
        return jnp.tanh(self._linear2(x))[0]


NATIVE_MODULES = {"torch": TorchModule, "jax": HaikuModule, "tensorflow": TensorflowModule}


# to_ivy_module
@pytest.mark.parametrize("bs_ic_oc", [([1, 2], 4, 5)])
@pytest.mark.parametrize("from_class_and_args", [True, False])
def test_to_ivy_module(bs_ic_oc, from_class_and_args, device):
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
    if from_class_and_args:
        ivy_module = ivy.to_ivy_module(
            native_module_class=native_module_class,
            args=[x],
            kwargs={"in_size": input_channels, "out_size": output_channels},
            device=device,
        )
    else:
        if ivy.current_backend_str() == "jax":

            def forward_fn(*a, **kw):
                model = native_module_class(**kw)
                return model(*a)

            native_module = hk.transform(forward_fn)
        elif ivy.current_backend_str() is "tensorflow":
            native_module = native_module_class(in_size=input_channels, out_size=output_channels)
            native_module.build((input_channels,))
        else:
            native_module = native_module_class(in_size=input_channels, out_size=output_channels)

        ivy_module = ivy.to_ivy_module(
            native_module=native_module,
            args=[x],
            kwargs={"in_size": input_channels, "out_size": output_channels},
            device=device
        )

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
    assert (abs(grads).max() > 0).all_true()


@pytest.mark.parametrize("bs_ic_oc", [([2, 3], 10, 5)])
def test_to_torch_module(bs_ic_oc):
    ivy.set_backend('torch')
    batch_shape, input_channels, output_channels = bs_ic_oc
    torch_model = torch_.to_torch_module(IvyModel, kwargs={"in_size": input_channels, "out_size": output_channels})
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=1e-3)
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    y = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels),
        "float32",
    )
    x_in = ivy.to_native(x)
    target = ivy.to_native(y)
    mae = nn.L1Loss()
    loss_tm1 = 1e12

    def loss_fn():
        preds = torch_model(x_in)
        return mae(target, preds)

    for step in range(10):
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("bs_ic_oc", [([2, 3], 10, 5)])
def test_to_keras_module(bs_ic_oc):
    ivy.set_backend('tensorflow')
    batch_shape, input_channels, output_channels = bs_ic_oc
    tf_model = tf_.to_keras_module(IvyModel, kwargs={"in_size": input_channels, "out_size": output_channels})
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    y = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels),
        "float32",
    )
    x_in = ivy.to_native(x)
    target = ivy.to_native(y)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    mae = tf.keras.losses.MeanAbsoluteError()
    loss_tm1 = 1e12

    def loss_fn():
        preds = tf_model(x_in)
        return mae(target, preds)

    for epoch in range(10):
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(tf_model.trainable_weights)
            loss = loss_fn()
        grads = tape.gradient(loss, tf_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, tf_model.trainable_weights))

        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("bs_ic_oc", [([2, 3], 10, 5)])
def test_to_haiku_module(bs_ic_oc):
    ivy.set_backend('jax')
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    y = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels),
        "float32",
    )
    x_in = ivy.to_native(x)
    target = ivy.to_native(y)
    loss_tm1 = 1e12

    haiku_model = jax_.to_haiku_module(IvyModel, kwargs={"in_size": input_channels, "out_size": output_channels})
    rng = jax.random.PRNGKey(42)
    lr = 0.001

    def forward_fn(*a, **kw):
        model = haiku_model()
        return model(*a, **kw)

    def MAELoss(weights, input_data, target):
        preds = model.apply(weights, rng, input_data)
        return jnp.mean(jnp.abs(target - preds))

    model = hk.transform(forward_fn)

    rng = jax.random.PRNGKey(42)
    params = model.init(rng, x_in)

    def UpdateWeights(weights, gradients):
        return weights - lr * gradients

    for epoch in range(10):
        loss, param_grads = value_and_grad(MAELoss)(params, x_in, target)
        params = jax.tree_map(UpdateWeights, params, param_grads)

        assert loss < loss_tm1
        loss_tm1 = loss
