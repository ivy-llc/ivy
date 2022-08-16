"""Collection of tests for Ivy sequential."""

# global
import itertools

from hypothesis import given, strategies as st

# local
import ivy


# Helpers #
###########
def _train(module, input_arr):
    def loss_fn(_v):
        return ivy.abs(ivy.mean(input_arr) - ivy.mean(module(input_arr, v=_v)))

    # initial loss
    loss_tm1, grads = ivy.execute_with_gradients(loss_fn, module.v)
    loss = None
    losses = []
    for i in range(5):
        loss, grads = ivy.execute_with_gradients(loss_fn, module.v)
        module.v = ivy.gradient_descent_update(module.v, grads, 1e-5)
        losses.append(loss)

    # loss is lower or very close to initial loss
    assert loss <= loss_tm1 or ivy.abs(loss - loss_tm1) < 1e-5

    return losses


def _copy_weights(v1, v2):
    # copy weights from layer1 to layer2
    v2.w = ivy.copy_array(v1.w)
    v2.b = ivy.copy_array(v1.b)


@given(
    input_array=st.lists(
        st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    dims=st.lists(st.integers(1, 10), min_size=1, max_size=5),
    use_activation=st.booleans(),
)
def test_sequential_construction_and_value(
    input_array, dims, use_activation, device, fw
):
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
        _train(module, input_array)


class TrainableModule(ivy.Module):
    def __init__(self, in_size, hidden_size, out_size):
        self._linear0 = ivy.Linear(in_size, hidden_size)
        self._linear1 = ivy.Linear(hidden_size, out_size)
        ivy.Module.__init__(self)

    def _forward(self, x):
        x = self._linear0(x)
        return self._linear1(x)


@given(
    input_array=st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    dims=st.lists(st.integers(1, 10), min_size=2, max_size=2),
)
def test_sequential_same_as_class(input_array, dims, fw):
    dims = [len(input_array)] + dims
    layer_count = len(dims)
    layers = [ivy.Linear(dims[i], dims[i + 1]) for i in range(layer_count - 1)]

    m_sequential = ivy.Sequential(*layers)
    m_class = TrainableModule(dims[0], dims[1], dims[2])

    # copy weights
    _copy_weights(m_class.v.linear0, m_sequential.v.submodules.v0)
    _copy_weights(m_class.v.linear1, m_sequential.v.submodules.v1)

    input_array = ivy.array(input_array, dtype="float32")

    if fw != "numpy":
        sequential_loss = _train(m_sequential, input_array)
        class_loss = _train(m_class, input_array)
        assert sequential_loss == class_loss
