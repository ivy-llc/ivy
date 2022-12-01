"""Collection of tests for module converters."""

# global
import pytest

try:
    import torch.nn
except ImportError:
    import types

    torch = types.SimpleNamespace()
    torch.nn = types.SimpleNamespace()
    torch.nn.Module = types.SimpleNamespace

try:
    import haiku as hk
except ImportError:
    import types

    hk = types.SimpleNamespace()
    hk.Module = types.SimpleNamespace

try:
    import jax.numpy as jnp
except ImportError:
    import types

    jnp = types.SimpleNamespace()


# local
import ivy


class TorchLinearModule(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(TorchLinearModule, self).__init__()
        self._linear = torch.nn.Linear(in_size, out_size)

    def forward(self, x):
        return self._linear(x)


class TorchModule(torch.nn.Module):
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


NATIVE_MODULES = {"torch": TorchModule, "jax": HaikuModule}


# to_ivy_module
@pytest.mark.parametrize("bs_ic_oc", [([1, 2], 4, 5)])
@pytest.mark.parametrize("from_class_and_args", [True, False])
@pytest.mark.parametrize("inplace_update", [True, False])
def test_to_ivy_module(bs_ic_oc, from_class_and_args, inplace_update, device):
    # smoke test
    if ivy.current_backend_str() not in ("torch", "jax"):
        # Currently only implemented for PyTorch
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    natvie_module_class = NATIVE_MODULES[ivy.current_backend_str()]
    if from_class_and_args:
        ivy_module = ivy.to_ivy_module(
            native_module_class=natvie_module_class,
            args=[input_channels, output_channels],
            device=device,
            inplace_update=inplace_update,
        )
    else:
        if ivy.current_backend_str() == "jax":

            def forward_fn(*a, **kw):
                model = natvie_module_class(input_channels, output_channels)
                return model(*a, **kw)

            native_module = hk.transform(forward_fn)
        else:
            native_module = natvie_module_class(input_channels, output_channels)
        ivy_module = ivy.to_ivy_module(
            native_module, device=device, inplace_update=inplace_update
        )

    def loss_fn(v_=None):
        out = ivy_module(x, v=v_)
        return ivy.mean(out)

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    loss_fn()  # for on-call mode

    if inplace_update:
        # inplace_update mode does not support gradient propagation
        return

    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, ivy_module.v)
        ivy_module.v = ivy.gradient_descent_update(ivy_module.v, grads, 1e-3)
        assert loss < loss_tm1
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()
