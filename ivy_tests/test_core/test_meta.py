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


# fomaml_step
@pytest.mark.parametrize(
    "igs_og_wocf_aas", [(1, -0.01, False, False), (2, -0.02, False, False), (3, -0.03, False, False),
                        (1, 0.01, True, False), (2, 0.02, True, False), (3, 0.03, True, False),
                        (1, -0.005, False, True), (2, -0.01, False, True), (3, -0.015, False, True)])
def test_fomaml_step(dev_str, call, igs_og_wocf_aas):

    inner_grad_steps, true_outer_grad, with_outer_cost_fn, average_across_steps = igs_og_wocf_aas

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
    batch = ivy.Container({'x': ivy.array([[0.]])})

    # inner cost function
    def inner_cost_fn(sub_batch, inner_v, outer_v):
        return -(inner_v['latent'] * outer_v['weight'])[0]

    # outer cost function
    def outer_cost_fn(sub_batch, inner_v, outer_v):
        return (inner_v['latent'] * outer_v['weight'])[0]

    # meta update
    outer_cost, outer_grads = ivy.fomaml_step(
        batch, inner_cost_fn, outer_cost_fn if with_outer_cost_fn else None, latent, weight, batch_size,
        inner_grad_steps, inner_learning_rate, average_across_steps)
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_outer_grad))
