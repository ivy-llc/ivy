"""Collection of tests for training normalization layers."""

# global
import pytest

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


# layer norm
@pytest.mark.parametrize(
    "x_n_ns",
    [
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [3]),
    ],
)
@pytest.mark.parametrize("with_v", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_layer_norm_layer_training(x_n_ns, with_v, dtype, tensor_fn, device, call):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        pytest.skip()
    x, normalized_shape = x_n_ns
    x = tensor_fn(x, dtype, device)
    if with_v:
        v = ivy.Container(
            {
                "scale": ivy.variable(ivy.ones(normalized_shape)),
                "offset": ivy.variable(ivy.zeros(normalized_shape)),
            }
        )
    else:
        v = None
    norm_layer = ivy.LayerNorm(normalized_shape, device=device, v=v)

    def loss_fn(v_):
        out = norm_layer(x, v=v_)
        return ivy.mean(out)

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, norm_layer.v)
        norm_layer.v = ivy.gradient_descent_update(norm_layer.v, grads, 1e-3)
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
    assert (abs(grads).max() > 0).all_true()
