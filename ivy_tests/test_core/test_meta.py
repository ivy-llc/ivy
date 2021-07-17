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
    "igs_og_wocf_aas_nt", [(1, -0.01, False, False, 1), (2, -0.02, False, False, 1), (3, -0.03, False, False, 1),
                           (1, 0.01, True, False, 1), (2, 0.02, True, False, 1), (3, 0.03, True, False, 1),
                           (1, -0.005, False, True, 1), (2, -0.01, False, True, 1), (3, -0.015, False, True, 1),
                           (1, -0.025, False, False, 2), (2, -0.05, False, False, 2), (3, -0.075, False, False, 2),
                           (1, 0.025, True, False, 2), (2, 0.05, True, False, 2), (3, 0.075, True, False, 2),
                           (1, -0.0125, False, True, 2), (2, -0.025, False, True, 2), (3, -0.0375, False, True, 2)])
def test_fomaml_step(dev_str, call, igs_og_wocf_aas_nt):

    inner_grad_steps, true_outer_grad, with_outer_cost_fn, average_across_steps, num_tasks = igs_og_wocf_aas_nt

    if call in [helpers.np_call, helpers.jnp_call]:
        # Numpy does not support gradients, and jax does not support gradients on custom nested classes
        pytest.skip()

    # config
    inner_learning_rate = 1e-2

    # create variables
    weight = ivy.Container({'weight': ivy.variable(ivy.array([1.]))})
    latent = ivy.Container({'latent': ivy.variable(ivy.array([0.]))})

    # batch
    batch = ivy.Container({'x': ivy.arange(num_tasks+1, 1, dtype_str='float32')})

    # inner cost function
    def inner_cost_fn(sub_batch, inner_v, outer_v):
        return -(sub_batch['x']* inner_v['latent'] * outer_v['weight'])[0]

    # outer cost function
    def outer_cost_fn(sub_batch, inner_v, outer_v):
        return (sub_batch['x'] * inner_v['latent'] * outer_v['weight'])[0]

    # meta update
    outer_cost, outer_grads = ivy.fomaml_step(
        batch, inner_cost_fn, outer_cost_fn if with_outer_cost_fn else None, latent, weight, num_tasks,
        inner_grad_steps, inner_learning_rate, average_across_steps=average_across_steps)
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_outer_grad))


# reptile_step
@pytest.mark.parametrize(
    "igs_og_aas_nt", [(1, -2.0808, False, 1), (2, -2.1649, False, 1), (3, -2.2523, False, 1),
                      (1, -2.0404, True, 1), (2, -2.0819, True, 1), (3, -2.1245, True, 1),
                      (1, -3.2036, False, 2), (2, -3.4221, False, 2), (3, -3.6568, False, 2),
                      (1, -3.1018, True, 2), (2, -3.2086, True, 2), (3, -3.3206, True, 2)])
def test_reptile_step(dev_str, call, igs_og_aas_nt):

    inner_grad_steps, true_outer_grad, average_across_steps, num_tasks = igs_og_aas_nt

    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # Numpy does not support gradients, jax does not support gradients on custom nested classes,
        # and mxnet does not support only_inputs argument to mx.autograd.grad
        pytest.skip()

    # config
    inner_learning_rate = 1e-2

    # create variables
    latent = ivy.Container({'latent': ivy.variable(ivy.array([1.]))})

    # batch
    batch = ivy.Container({'x': ivy.arange(num_tasks+1, 1, dtype_str='float32')})

    # cost function
    def cost_fn(sub_batch, v):
        return -(sub_batch['x'] * v['latent'] ** 2)[0]

    # meta update
    outer_cost, outer_grads = ivy.reptile_step(
        batch, cost_fn, latent, num_tasks, inner_grad_steps, inner_learning_rate,
        average_across_steps=average_across_steps)
    assert np.allclose(ivy.to_numpy(outer_grads.latent[0]), np.array(true_outer_grad), atol=1e-4)


# Second Order #
# -------------#

# maml_step
@pytest.mark.parametrize(
    "igs_og_wocf_aas_nt", [(1, -0.02, False, False, 1), (2, -0.04, False, False, 1), (3, -0.06, False, False, 1),
                           (1, 0.02, True, False, 1), (2, 0.04, True, False, 1), (3, 0.06, True, False, 1),
                           (1, -0.01, False, True, 1), (2, -0.02, False, True, 1), (3, -0.03, False, True, 1),
                           (1, -0.05, False, False, 2), (2, -0.1, False, False, 2), (3, -0.15, False, False, 2),
                           (1, 0.03, True, False, 2), (2, 0.06, True, False, 2), (3, 0.09, True, False, 2),
                           (1, -0.025, False, True, 2), (2, -0.05, False, True, 2), (3, -0.075, False, True, 2)])
def test_maml_step(dev_str, call, igs_og_wocf_aas_nt):

    inner_grad_steps, true_outer_grad, with_outer_cost_fn, average_across_steps, num_tasks = igs_og_wocf_aas_nt

    if call in [helpers.np_call, helpers.mx_call]:
        # Numpy does not support gradients, mxnet.autograd.grad() does not support allow_unused like PyTorch
        pytest.skip()

    # config
    inner_learning_rate = 1e-2

    # create variables
    weight = ivy.Container({'weight': ivy.variable(ivy.array([1.]))})
    latent = ivy.Container({'latent': ivy.variable(ivy.array([0.]))})

    # batch
    batch = ivy.Container({'x': ivy.arange(num_tasks+1, 1, dtype_str='float32')})

    # inner cost function
    def inner_cost_fn(sub_batch, inner_v, outer_v):
        return -(sub_batch['x'] * inner_v['latent'] * outer_v['weight'])[0]

    # outer cost function
    def outer_cost_fn(sub_batch, inner_v, outer_v):
        return (inner_v['latent'] * outer_v['weight'])[0]

    # meta update
    outer_cost, outer_grads = ivy.maml_step(
        batch, inner_cost_fn, outer_cost_fn if with_outer_cost_fn else None, latent, weight, num_tasks,
        inner_grad_steps, inner_learning_rate, average_across_steps=average_across_steps)
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_outer_grad))
