"""
Collection of tests for templated neural network layers
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


# layer norm
@pytest.mark.parametrize(
    "x_n_ni_n_g_n_b_n_res", [

        ([[1., 2., 3.], [4., 5., 6.]],
         -1,
         [[1., 2., 3.], [4., 5., 6.]],
         [[1., 2., 3.], [4., 5., 6.]],
         [[-0.2247448, 2., 6.6742344],
          [-0.8989792, 5., 13.348469]]),
    ])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_layer_norm(x_n_ni_n_g_n_b_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, norm_idxs, gamma, beta, true_res = x_n_ni_n_g_n_b_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    gamma = tensor_fn(gamma, dtype_str, dev_str)
    beta = tensor_fn(beta, dtype_str, dev_str)
    true_res = tensor_fn(true_res, dtype_str, dev_str)
    ret = ivy.layer_norm(x, norm_idxs, gamma=gamma, beta=beta)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(call(ivy.layer_norm, x, norm_idxs, gamma=gamma, beta=beta), ivy.to_numpy(true_res))
    # compilation test
    if call in [helpers.torch_call]:
        # this is not a backend implemented function
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.layer_norm)
