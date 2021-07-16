# global
import ivy


# Private #
# --------#

def _train_task(sub_batch, inner_cost_fn, inner_v, outer_v, inner_grad_steps, inner_learning_rate, first_order):
    outer_v_in_loop = outer_v.map(lambda x, kc: ivy.stop_gradient(x)) if first_order else outer_v
    for i in range(inner_grad_steps):
        _, inner_grads = ivy.execute_with_gradients(lambda v: inner_cost_fn(sub_batch, v, outer_v_in_loop), inner_v,
                                                    retain_grads=not first_order)
        inner_v = ivy.gradient_descent_update(inner_v, inner_grads, inner_learning_rate)
    return inner_cost_fn(sub_batch, inner_v, outer_v), inner_v


def _train_tasks(batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, batch_size, inner_grad_steps,
                 inner_learning_rate, first_order=True):
    costs = list()
    for sub_batch in batch.unstack(0, batch_size):
        cost, inner_v = _train_task(sub_batch, inner_cost_fn, inner_v, outer_v, inner_grad_steps,
                                    inner_learning_rate, first_order)
        if outer_cost_fn is not None:
            cost = outer_cost_fn(sub_batch, inner_v, outer_v)
        costs.append(cost)
    return sum(costs) / len(costs)


# Public #
# -------#

def fomaml_step(batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, batch_size, inner_grad_steps,
                inner_learning_rate):
    """
    Perform step of first order MAML.

    :param batch: The input batch
    :type batch: ivy.Container
    :param inner_cost_fn: callable for the inner loop cost function, receing sub-batch, inner vars and outer vars
    :type inner_cost_fn: callable
    :param outer_cost_fn: callable for the outer loop cost function, receing sub-batch, inner vars and outer vars
    :type outer_cost_fn: callable
    :param inner_v: Variables to be optimized during the inner loop
    :type inner_v: ivy.Container
    :param outer_v: Variables to be optimized during the outer loop
    :type outer_v: ivy.Container
    :param batch_size: size of the batch
    :type batch_size: int
    :param inner_grad_steps: Number of gradient steps to perform during the inner loop.
    :type inner_grad_steps: int
    :param inner_learning_rate: The learning rate of the inner loop.
    :type inner_learning_rate: float
    """
    return ivy.execute_with_gradients(lambda v: _train_tasks(
        batch, inner_cost_fn, outer_cost_fn, inner_v, v, batch_size, inner_grad_steps, inner_learning_rate,
        first_order=True), outer_v)
