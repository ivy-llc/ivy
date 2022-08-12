"""Collection of tests for Ivy sequential."""

import itertools

# global
from hypothesis import given, strategies as st

# local
import ivy


# Helpers #
###########
def _train(module, loss_fn):
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(5):
        loss, grads = ivy.execute_with_gradients(loss_fn, module.v)
        module.v = ivy.gradient_descent_update(module.v, grads, 1e-3)
        loss_tm1 = loss

    assert loss <= loss_tm1


@given(
    input_array=st.lists(
        st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    dims=st.lists(st.integers(1, 10), min_size=1, max_size=5),
    use_activation=st.booleans(),
)
def test_sequential(input_array, dims, use_activation, device, fw):
    dims = [len(input_array)] + dims
    layer_count = len(dims)
    layers = [
        ivy.Linear(dims[i], dims[i + 1], device=device) for i in range(layer_count - 1)
    ]

    if use_activation:
        activations = [ivy.GELU() for _ in range(layer_count - 1)]
        layers = itertools.chain.from_iterable(zip(layers, activations))

    module = ivy.Sequential(*layers)

    input_array = ivy.array(input_array, dtype="float32", device=device)

    if fw != "numpy":
        _train(module, lambda v_: ivy.mean(module(input_array, v=v_)))
