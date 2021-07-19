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

'''
# fomaml step unique vars
@pytest.mark.parametrize(
    "igs_og_wocf_aas_nt", [(1, -0.01, False, False, 1), (2, -0.02, False, False, 1), (3, -0.03, False, False, 1),
                           (1, 0.01, True, False, 1), (2, 0.02, True, False, 1), (3, 0.03, True, False, 1),
                           (1, -0.005, False, True, 1), (2, -0.01, False, True, 1), (3, -0.015, False, True, 1),
                           (1, -0.025, False, False, 2), (2, -0.05, False, False, 2), (3, -0.075, False, False, 2),
                           (1, 0.025, True, False, 2), (2, 0.05, True, False, 2), (3, 0.075, True, False, 2),
                           (1, -0.0125, False, True, 2), (2, -0.025, False, True, 2), (3, -0.0375, False, True, 2)])
def test_fomaml_step_unique_vars(dev_str, call, igs_og_wocf_aas_nt):

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
        return -(sub_batch['x'] * inner_v['latent'] * outer_v['weight'])[0]

    # outer cost function
    def outer_cost_fn(sub_batch, inner_v, outer_v):
        return (sub_batch['x'] * inner_v['latent'] * outer_v['weight'])[0]

    # meta update
    outer_cost, outer_grads = ivy.fomaml_step(
        batch, inner_cost_fn, outer_cost_fn if with_outer_cost_fn else None, latent, weight, num_tasks,
        inner_grad_steps, inner_learning_rate, average_across_steps=average_across_steps)
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_outer_grad))
'''


# fomaml step shared vars
@pytest.mark.parametrize(
    "inner_grad_steps", [1, 2, 3])
@pytest.mark.parametrize(
    "with_outer_cost_fn", [True, False])
@pytest.mark.parametrize(
    "average_across_steps", [True, False])
@pytest.mark.parametrize(
    "num_tasks", [1, 2])
def test_fomaml_step_shared_vars(dev_str, call, inner_grad_steps, with_outer_cost_fn, average_across_steps, num_tasks):

    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # Numpy does not support gradients, jax does not support gradients on custom nested classes,
        # and mxnet does not support only_inputs argument to mx.autograd.grad
        pytest.skip()

    # config
    inner_learning_rate = 1e-2

    # create variable
    latent = ivy.Container({'latent': ivy.variable(ivy.array([1.]))})

    # batch
    batch = ivy.Container({'x': ivy.arange(num_tasks+1, 1, dtype_str='float32')})

    # inner cost function
    def inner_cost_fn(sub_batch_in, v):
        return -(sub_batch_in['x'] * v['latent'] ** 2)[0]

    # outer cost function
    def outer_cost_fn(sub_batch_in, v):
        return (sub_batch_in['x'] * v['latent'] ** 2)[0]

    # numpy
    latent_np = latent.map(lambda x, kc: ivy.to_numpy(x))
    batch_np = batch.map(lambda x, kc: ivy.to_numpy(x))

    # loss grad function
    def loss_grad_fn(sub_batch_in, w_in, outer=False):
        return (1 if (with_outer_cost_fn and outer) else -1) * 2 * sub_batch_in['x'][0] * w_in

    # true gradient
    true_outer_grads = list()
    for sub_batch in batch_np.unstack(0, num_tasks):
        ws = list()
        grads = list()
        ws.append(latent_np)
        for step in range(inner_grad_steps):
            update_grad = loss_grad_fn(sub_batch, ws[-1])
            w = ws[-1] - inner_learning_rate * update_grad
            if with_outer_cost_fn:
                grads.append(loss_grad_fn(sub_batch, ws[-1], outer=True))
            else:
                grads.append(update_grad)
            ws.append(w)
        if with_outer_cost_fn:
            grads.append(loss_grad_fn(sub_batch, ws[-1], outer=True))
        else:
            grads.append(loss_grad_fn(sub_batch, ws[-1]))

        # true outer grad
        if average_across_steps:
            true_outer_grad = sum(grads).latent / len(grads)
        else:
            true_outer_grad = grads[-1].latent
        true_outer_grads.append(true_outer_grad)
    true_outer_grad = sum(true_outer_grads) / len(true_outer_grads)

    # meta update
    outer_cost, outer_grads = ivy.fomaml_step(
        batch, inner_cost_fn, outer_cost_fn if with_outer_cost_fn else None, latent, None, num_tasks,
        inner_grad_steps, inner_learning_rate, average_across_steps=average_across_steps)
    assert np.allclose(ivy.to_numpy(outer_grads.latent[0]), np.array(true_outer_grad))


'''
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

    # create variable
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
'''


# Second Order #
# -------------#

# maml step unique vars
@pytest.mark.parametrize(
    "inner_grad_steps", [1, 2, 3])
@pytest.mark.parametrize(
    "with_outer_cost_fn", [True, False])
@pytest.mark.parametrize(
    "average_across_steps", [True, False])
@pytest.mark.parametrize(
    "num_tasks", [1, 2])
def test_maml_step_unique_vars(dev_str, call, inner_grad_steps, with_outer_cost_fn, average_across_steps, num_tasks):

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
    def inner_cost_fn(sub_batch_in, inner_v, outer_v):
        return -(sub_batch_in['x'] * inner_v['latent'] * outer_v['weight'])[0]

    # outer cost function
    def outer_cost_fn(sub_batch_in, inner_v, outer_v):
        return (sub_batch_in['x'] * inner_v['latent'] * outer_v['weight'])[0]

    # numpy
    weight_np = weight.map(lambda x, kc: ivy.to_numpy(x))
    latent_np = latent.map(lambda x, kc: ivy.to_numpy(x))
    batch_np = batch.map(lambda x, kc: ivy.to_numpy(x))

    # true gradient
    all_outer_grads = list()
    for sub_batch in batch_np.unstack(0, num_tasks):
        all_outer_grads.append(
            [(-2*i*inner_learning_rate*weight_np.weight*sub_batch['x'][0]**2 - sub_batch['x'][0]*latent_np.latent) * \
             (-1 if with_outer_cost_fn else 1) for i in range(inner_grad_steps+1)])
    if average_across_steps:
        true_outer_grad = sum([sum(og) / len(og) for og in all_outer_grads]) / num_tasks
    else:
        true_outer_grad = sum([og[-1] for og in all_outer_grads]) / num_tasks

    # meta update
    outer_cost, outer_grads = ivy.maml_step(
        batch, inner_cost_fn, outer_cost_fn if with_outer_cost_fn else None, latent, weight, num_tasks,
        inner_grad_steps, inner_learning_rate, average_across_steps=average_across_steps)
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_outer_grad))


# maml step shared vars
@pytest.mark.parametrize(
    "inner_grad_steps", [1, 2, 3])
@pytest.mark.parametrize(
    "with_outer_cost_fn", [True, False])
@pytest.mark.parametrize(
    "average_across_steps", [True, False])
@pytest.mark.parametrize(
    "num_tasks", [1, 2])
def test_maml_step_shared_vars(dev_str, call, inner_grad_steps, with_outer_cost_fn, average_across_steps, num_tasks):

    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # Numpy does not support gradients, jax does not support gradients on custom nested classes,
        # and mxnet does not support only_inputs argument to mx.autograd.grad
        pytest.skip()

    # config
    inner_learning_rate = 1e-2

    # create variable
    latent = ivy.Container({'latent': ivy.variable(ivy.array([1.]))})

    # batch
    batch = ivy.Container({'x': ivy.arange(num_tasks+1, 1, dtype_str='float32')})

    # inner cost function
    def inner_cost_fn(sub_batch_in, v):
        return -(sub_batch_in['x'] * v['latent'] ** 2)[0]

    # outer cost function
    def outer_cost_fn(sub_batch_in, v):
        return (sub_batch_in['x'] * v['latent'] ** 2)[0]

    # numpy
    latent_np = latent.map(lambda x, kc: ivy.to_numpy(x))
    batch_np = batch.map(lambda x, kc: ivy.to_numpy(x))

    # loss grad function
    def loss_grad_fn(sub_batch_in, w_in, outer=False):
        return (1 if (with_outer_cost_fn and outer) else -1) * 2*sub_batch_in['x'][0]*w_in
    
    # update grad function
    def update_grad_fn(w_init, sub_batch_in, num_steps, average=False):
        terms = [0]*num_steps + [1]
        collection_of_terms = [terms]
        for s in range(num_steps):
            rhs = [t*2*sub_batch_in['x'][0] for t in terms]
            rhs.pop(0)
            rhs.append(0)
            terms = [t + rh for t, rh in zip(terms, rhs)]
            collection_of_terms.append([t for t in terms])
        if average:
            return [sum([t*inner_learning_rate**(num_steps-i) for i, t in enumerate(tms)]) * w_init.latent
                    for tms in collection_of_terms]
        return sum([t*inner_learning_rate**(num_steps-i) for i, t in enumerate(terms)]) * w_init.latent
    
    # true gradient
    true_outer_grads = list()
    for sub_batch in batch_np.unstack(0, num_tasks):
        ws = list()
        grads = list()
        ws.append(latent_np)
        for step in range(inner_grad_steps):
            update_grad = loss_grad_fn(sub_batch, ws[-1])
            w = ws[-1] - inner_learning_rate * update_grad
            if with_outer_cost_fn:
                grads.append(loss_grad_fn(sub_batch, ws[-1], outer=True))
            else:
                grads.append(update_grad)
            ws.append(w)
        if with_outer_cost_fn:
            grads.append(loss_grad_fn(sub_batch, ws[-1], outer=True))
        else:
            grads.append(loss_grad_fn(sub_batch, ws[-1]))
    
        # true outer grad
        if average_across_steps:
            true_outer_grad =\
                 sum([ig.latent*ug for ig, ug in
                      zip(grads, update_grad_fn(latent_np, sub_batch, inner_grad_steps, average=True))]) / len(grads)
        else:
            true_outer_grad = update_grad_fn(latent_np, sub_batch, inner_grad_steps) * grads[-1].latent
        true_outer_grads.append(true_outer_grad)
    true_outer_grad = sum(true_outer_grads) / len(true_outer_grads)

    # meta update
    outer_cost, outer_grads = ivy.maml_step(
        batch, inner_cost_fn, outer_cost_fn if with_outer_cost_fn else None, latent, None, num_tasks,
        inner_grad_steps, inner_learning_rate, average_across_steps=average_across_steps)
    assert np.allclose(ivy.to_numpy(outer_grads.latent[0]), true_outer_grad[0])
