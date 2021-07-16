"""
Collection of tests for templated meta functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# maml_step
@pytest.mark.parametrize(
    "igs_og", [(1, -0.01), (2, -0.02), (3, -0.03)])
def test_maml_step(dev_str, call, igs_og):

    inner_grad_steps, true_outer_grad = igs_og

    if call in [helpers.np_call, helpers.jnp_call]:
        # Numpy does not support gradients, and jax does not support gradients on custom nested classes
        pytest.skip()

    # config
    batch_size = 1
    inner_learning_rate = 1e-2

    # create variables
    weight = ivy.Container({'weight': ivy.variable(ivy.array([1.]))})
    latent = ivy.Container({'latent': ivy.variable(ivy.array([0.]))})

    # batch
    batch = ivy.Container({'x': ivy.array([[1.], [2.], [3.]])})

    # inner cost function
    def inner_cost_fn(sub_batch, inner_v, outer_v):
        network_pred = inner_v['latent'] * outer_v['weight']
        return -network_pred[0]

    # meta update
    outer_cost, outer_grads = ivy.fomaml_step(
        batch, inner_cost_fn, None, latent, weight, batch_size, inner_grad_steps, inner_learning_rate)
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_outer_grad))
    print('framework: {}, value: {}'.format(call, ivy.to_numpy(outer_grads.weight[0]).item()))
