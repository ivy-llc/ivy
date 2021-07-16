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
    return inner_cost_fn(sub_batch, inner_v, outer_v)


def _train_tasks(batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, batch_size, inner_grad_steps,
                 inner_learning_rate, first_order=True):
    costs = list()
    for sub_batch in batch.unstack(0, batch_size):
        costs.append(_train_task(sub_batch, inner_cost_fn, inner_v, outer_v, inner_grad_steps, inner_learning_rate,
                                 first_order))
    if outer_cost_fn is not None:
        return outer_cost_fn(batch, inner_v, outer_v)
    return sum(costs) / len(costs)


# Public #
# -------#

def fomaml_step(batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, batch_size, inner_grad_steps,
                inner_learning_rate):
    return ivy.execute_with_gradients(lambda v: _train_tasks(
        batch, inner_cost_fn, outer_cost_fn, inner_v, v, batch_size, inner_grad_steps, inner_learning_rate,
        first_order=True), outer_v)
