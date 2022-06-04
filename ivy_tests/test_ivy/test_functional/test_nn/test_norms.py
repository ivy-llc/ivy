"""Collection of tests for unified neural network layers."""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


# # layer norm
@pytest.mark.parametrize(
    "x_n_ni_n_s_n_o_n_res",
    [
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            -1,
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-0.22473562, 2.0, 6.6742067], [-0.8989425, 5.0, 13.348413]],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array])
def test_layer_norm(x_n_ni_n_s_n_o_n_res, dtype, tensor_fn, device, call):
    # smoke test
    x, norm_idxs, scale, offset, true_res = x_n_ni_n_s_n_o_n_res
    x = tensor_fn(x, dtype, device)
    scale = tensor_fn(scale, dtype, device)
    offset = tensor_fn(offset, dtype, device)
    true_res = tensor_fn(true_res, dtype, device)
    ret = ivy.layer_norm(x, norm_idxs, scale=scale, offset=offset)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(
        call(ivy.layer_norm, x, norm_idxs, scale=scale, offset=offset),
        ivy.to_numpy(true_res),
    )
    # compilation test
    if call in [helpers.torch_call]:
        # this is not a backend implemented function
        return
    helpers.assert_compilable(ivy.layer_norm)
