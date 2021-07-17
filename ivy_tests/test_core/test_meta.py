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


# First Order #
# ------------#

# fomaml_step
@pytest.mark.parametrize(
    "igs_og_wocf", [(1, -0.01, False), (2, -0.02, False), (3, -0.03, False),
                    (1, 0.01, True), (2, 0.02, True), (3, 0.03, True)])
def test_fomaml_step(dev_str, call, igs_og_wocf):

    inner_grad_steps, true_outer_grad, with_outer_cost_fn = igs_og_wocf

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
        inner_grad_steps, inner_learning_rate)
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_outer_grad))


# reptile_step
@pytest.mark.parametrize(
    "igs_og_wocf", [(1, -0.005, False), (2, -0.01, False), (3, -0.015, False)])
def test_reptile_step(dev_str, call, igs_og_wocf):

    inner_grad_steps, true_outer_grad, with_outer_cost_fn = igs_og_wocf

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
    outer_cost, outer_grads = ivy.reptile_step(
        batch, inner_cost_fn, outer_cost_fn if with_outer_cost_fn else None, latent, weight, batch_size,
        inner_grad_steps, inner_learning_rate)
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_outer_grad))


# Second Order #
# -------------#

# maml_step
@pytest.mark.parametrize(
    "igs_og_wocf", [(1, -0.02, False)])
def test_maml_step(dev_str, call, igs_og_wocf):

    inner_grad_steps, true_outer_grad, with_outer_cost_fn = igs_og_wocf

    if call in [helpers.np_call]:
        # Numpy does not support gradients
        pytest.skip()

    # ToDo: investigate why jax and pytorch are the only frameworks where the second order terms are correct, and fix.
    if call not in [helpers.jnp_call, helpers.torch_call]:
        # Currently only jax and pytorch treat the inner loop optimization as unrolled graph
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
    outer_cost, outer_grads = ivy.maml_step(
        batch, inner_cost_fn, outer_cost_fn if with_outer_cost_fn else None, latent, weight, batch_size,
        inner_grad_steps, inner_learning_rate)
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_outer_grad))
