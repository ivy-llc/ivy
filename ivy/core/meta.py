# global
import ivy
from ivy.core.gradients import gradient_descent_update


# Private #
# --------#

def _compute_cost_and_update_grads(cost_fn, order, sub_batch, variables, outer_v, keep_outer_v,
                                   average_across_steps_or_final, all_grads, unique_outer):
    if order == 1:
        cost, inner_grads = ivy.execute_with_gradients(
            lambda v: cost_fn(sub_batch, variables.set_at_key_chains(v) if unique_outer else v),
            variables.at_key_chains(outer_v, ignore_none=True) if keep_outer_v else
            variables.prune_key_chains(outer_v, ignore_none=True), retain_grads=False)
        if average_across_steps_or_final:
            all_grads.append(inner_grads)
    else:
        cost = cost_fn(sub_batch, variables)
    return cost


def _train_task(inner_sub_batch, outer_sub_batch, inner_cost_fn, outer_cost_fn, variables, inner_grad_steps,
                inner_learning_rate, inner_optimization_step, order, average_across_steps, inner_v, keep_innver_v,
                outer_v, keep_outer_v):

    # init
    total_cost = 0
    all_grads = list()

    # inner and outer
    unique_inner = inner_v is not None
    unique_outer = outer_v is not None

    # iterate through inner loop training steps
    for i in range(inner_grad_steps):

        # compute inner gradient for update the inner variables
        cost, inner_update_grads = ivy.execute_with_gradients(
            lambda v: inner_cost_fn(inner_sub_batch, variables.set_at_key_chains(v) if unique_inner else v),
            variables.at_key_chains(inner_v, ignore_none=True) if keep_innver_v else
            variables.prune_key_chains(inner_v, ignore_none=True), retain_grads=order > 1)

        # compute the cost to be optimized, and update all_grads if fist order method
        if outer_cost_fn is None and not unique_inner and not unique_outer:
            all_grads.append(inner_update_grads)
        else:
            cost = _compute_cost_and_update_grads(
                inner_cost_fn if outer_cost_fn is None else outer_cost_fn, order, outer_sub_batch, variables, outer_v,
                keep_outer_v, average_across_steps, all_grads, unique_outer)

        # update cost and update parameters
        total_cost = total_cost + cost
        if unique_inner:
            variables = variables.set_at_key_chains(
                inner_optimization_step(variables.at_key_chains(inner_v) if keep_innver_v else
                                        variables.prune_key_chains(inner_v), inner_update_grads,
                                        inner_learning_rate, inplace=False))
        else:
            variables = inner_optimization_step(variables, inner_update_grads, inner_learning_rate, inplace=False)

    # once training is finished, compute the final cost, and update all_grads if fist order method
    final_cost = _compute_cost_and_update_grads(
        inner_cost_fn if outer_cost_fn is None else outer_cost_fn, order, outer_sub_batch, variables, outer_v,
        keep_outer_v, True, all_grads, unique_outer)

    # average the cost or gradients across all timesteps if this option is chosen
    if average_across_steps:
        total_cost = total_cost + final_cost
        if order == 1:
            all_grads = sum(all_grads) / max(len(all_grads), 1)
        return total_cost / (inner_grad_steps + 1), variables.stop_gradients(), all_grads

    # else return only the final values
    if order == 1:
        all_grads = all_grads[-1]
    return final_cost, variables.stop_gradients(), all_grads


def _train_tasks(batch, inner_sub_batch_fn, outer_sub_batch_fn, inner_cost_fn, outer_cost_fn, variables, num_tasks,
                 inner_grad_steps, inner_learning_rate, inner_optimization_step, order, average_across_steps, inner_v,
                 keep_innver_v, outer_v, keep_outer_v, return_inner_v):
    total_cost = 0
    updated_ivs_to_return = list()
    all_grads = list()
    if isinstance(inner_v, (list, tuple)) and isinstance(inner_v[0], (list, tuple, dict, type(None))):
        inner_v_seq = True
    else:
        inner_v_seq = False
    if isinstance(outer_v, (list, tuple)) and isinstance(outer_v[0], (list, tuple, dict, type(None))):
        outer_v_seq = True
    else:
        outer_v_seq = False
    for i, sub_batch in enumerate(batch.unstack(0, num_tasks)):
        if inner_sub_batch_fn is not None:
            inner_sub_batch = inner_sub_batch_fn(sub_batch)
        else:
            inner_sub_batch = sub_batch
        if outer_sub_batch_fn is not None:
            outer_sub_batch = outer_sub_batch_fn(sub_batch)
        else:
            outer_sub_batch = sub_batch
        iv = inner_v[i] if inner_v_seq else inner_v
        ov = outer_v[i] if outer_v_seq else outer_v
        cost, updated_iv, grads = _train_task(inner_sub_batch, outer_sub_batch, inner_cost_fn, outer_cost_fn, variables,
                                              inner_grad_steps, inner_learning_rate, inner_optimization_step, order,
                                              average_across_steps, iv, keep_innver_v, ov, keep_outer_v)
        if (return_inner_v == 'first' and i == 0) or return_inner_v == 'all':
            updated_ivs_to_return.append(updated_iv)
        total_cost = total_cost + cost
        all_grads.append(grads)
    if order == 1:
        if return_inner_v:
            return total_cost / num_tasks, sum(all_grads) / len(all_grads), updated_ivs_to_return
        return total_cost / num_tasks, sum(all_grads) / len(all_grads)
    if return_inner_v:
        return total_cost / num_tasks, updated_ivs_to_return
    return total_cost / num_tasks


# Public #
# -------#

# First Order

def fomaml_step(batch, inner_cost_fn, outer_cost_fn, variables, num_tasks, inner_grad_steps, inner_learning_rate,
                inner_optimization_step=gradient_descent_update, inner_sub_batch_fn=None, outer_sub_batch_fn=None,
                average_across_steps=False, inner_v=None, keep_inner_v=True, outer_v=None, keep_outer_v=True,
                return_inner_v=False):
    """
    Perform step of first order MAML.

    :param batch: The input batch
    :type batch: ivy.Container
    :param inner_cost_fn: callable for the inner loop cost function, receving task-specific sub-batch,
                            inner vars and outer vars
    :type inner_cost_fn: callable
    :param outer_cost_fn: callable for the outer loop cost function, receving task-specific sub-batch,
                            inner vars and outer vars. If None, the cost from the inner loop will also be
                            optimized in the outer loop.
    :type outer_cost_fn: callable, optional
    :param variables: Variables to be optimized during the meta step
    :type variables: ivy.Container
    :param num_tasks: Number of unique tasks to inner-loop optimize for during the meta step.
                        This must be the leading size of the input batch.
    :type num_tasks: int
    :param inner_grad_steps: Number of gradient steps to perform during the inner loop.
    :type inner_grad_steps: int
    :param inner_learning_rate: The learning rate of the inner loop.
    :type inner_learning_rate: float
    :param inner_optimization_step: The function used for the inner loop optimization.
                                    Default is ivy.gradient_descent_update.
    :type inner_optimization_step: callable, optional
    :param inner_sub_batch_fn: Function to apply to the task sub-batch, before passing to the inner_cost_fn.
                               Default is None.
    :type inner_sub_batch_fn: callable, optional
    :param outer_sub_batch_fn: Function to apply to the task sub-batch, before passing to the outer_cost_fn.
                               Default is None.
    :type outer_sub_batch_fn: callable, optional
    :param average_across_steps: Whether to average the inner loop steps for the outer loop update. Default is False.
    :type average_across_steps: bool, optional
    :param inner_v: Nested variable keys to be optimized during the inner loop, with same keys and boolean values.
    :type inner_v: dict str or list, optional
    :param keep_inner_v: If True, the key chains in inner_v will be kept, otherwise they will be removed.
                            Default is True.
    :type keep_inner_v: bool, optional
    :param outer_v: Nested variable keys to be optimized during the inner loop, with same keys and boolean values.
    :type outer_v: dict str or list, optional
    :param keep_outer_v: If True, the key chains in inner_v will be kept, otherwise they will be removed.
                            Default is True.
    :type keep_outer_v: bool, optional
    :param return_inner_v: Either 'first', 'all', or False. 'first' means the variables for the first task inner loop
                           will also be returned. variables for all tasks will be returned with 'all'. Default is False.
    :type return_inner_v: str, optional
    :return: The cost and the gradients with respect to the outer loop variables.
    """
    return _train_tasks(
        batch, inner_sub_batch_fn, outer_sub_batch_fn, inner_cost_fn, outer_cost_fn, variables, num_tasks,
        inner_grad_steps, inner_learning_rate, inner_optimization_step, 1, average_across_steps, inner_v, keep_inner_v,
        outer_v, keep_outer_v, return_inner_v)


def reptile_step(batch, cost_fn, variables, num_tasks, inner_grad_steps, inner_learning_rate,
                 inner_optimization_step=gradient_descent_update, return_inner_v=False):
    """
    Perform step of Reptile.

    :param batch: The input batch
    :type batch: ivy.Container
    :param cost_fn: callable for the cost function, receivng the task-specific sub-batch and variables
    :type cost_fn: callable
    :param variables: Variables to be optimized
    :type variables: ivy.Container
    :param num_tasks: Number of unique tasks to inner-loop optimize for during the meta step.
                        This must be the leading size of the input batch.
    :type num_tasks: int
    :param inner_grad_steps: Number of gradient steps to perform during the inner loop.
    :type inner_grad_steps: int
    :param inner_learning_rate: The learning rate of the inner loop.
    :type inner_learning_rate: float
    :param inner_optimization_step: The function used for the inner loop optimization.
                                    Default is ivy.gradient_descent_update.
    :type inner_optimization_step: callable, optional
    :param return_inner_v: Either 'first', 'all', or False. 'first' means the variables for the first task inner loop
                           will also be returned. variables for all tasks will be returned with 'all'. Default is False.
    :type return_inner_v: str, optional
    :return: The cost and the gradients with respect to the outer loop variables.
    """
    rets = _train_tasks(
        batch, None, None, cost_fn, None, variables, num_tasks, inner_grad_steps, inner_learning_rate,
        inner_optimization_step, 1, True, None, True, None, True, return_inner_v)
    cost = rets[0]
    grads = rets[1] / inner_learning_rate
    if return_inner_v:
        return cost, grads, rets[2]
    return cost, grads


# Second Order

def maml_step(batch, inner_cost_fn, outer_cost_fn, variables, num_tasks, inner_grad_steps, inner_learning_rate,
              inner_optimization_step=gradient_descent_update, inner_sub_batch_fn=None, outer_sub_batch_fn=None,
              average_across_steps=False, inner_v=None, keep_inner_v=True, outer_v=None, keep_outer_v=True,
              return_inner_v=False):
    """
    Perform step of vanilla second order MAML.

    :param batch: The input batch
    :type batch: ivy.Container
    :param inner_cost_fn: callable for the inner loop cost function, receing sub-batch, inner vars and outer vars
    :type inner_cost_fn: callable
    :param outer_cost_fn: callable for the outer loop cost function, receving task-specific sub-batch,
                            inner vars and outer vars. If None, the cost from the inner loop will also be
                            optimized in the outer loop.
    :type outer_cost_fn: callable, optional
    :param variables: Variables to be optimized during the meta step
    :type variables: ivy.Container
    :param num_tasks: Number of unique tasks to inner-loop optimize for during the meta step.
                        This must be the leading size of the input batch.
    :type num_tasks: int
    :param inner_grad_steps: Number of gradient steps to perform during the inner loop.
    :type inner_grad_steps: int
    :param inner_learning_rate: The learning rate of the inner loop.
    :type inner_learning_rate: float
    :param inner_optimization_step: The function used for the inner loop optimization.
                                    Default is ivy.gradient_descent_update.
    :type inner_optimization_step: callable, optional
    :param inner_sub_batch_fn: Function to apply to the task sub-batch, before passing to the inner_cost_fn.
                               Default is None.
    :type inner_sub_batch_fn: callable, optional
    :param outer_sub_batch_fn: Function to apply to the task sub-batch, before passing to the outer_cost_fn.
                               Default is None.
    :type outer_sub_batch_fn: callable, optional
    :param average_across_steps: Whether to average the inner loop steps for the outer loop update. Default is False.
    :type average_across_steps: bool, optional
    :param inner_v: Nested variable keys to be optimized during the inner loop, with same keys and boolean values.
    :type inner_v: dict str or list, optional
    :param keep_inner_v: If True, the key chains in inner_v will be kept, otherwise they will be removed.
                            Default is True.
    :type keep_inner_v: bool, optional
    :param outer_v: Nested variable keys to be optimized during the inner loop, with same keys and boolean values.
    :type outer_v: dict str or list, optional
    :param keep_outer_v: If True, the key chains in inner_v will be kept, otherwise they will be removed.
                            Default is True.
    :type keep_outer_v: bool, optional
    :param return_inner_v: Either 'first', 'all', or False. 'first' means the variables for the first task inner loop
                           will also be returned. variables for all tasks will be returned with 'all'. Default is False.
    :type return_inner_v: str, optional
    :return: The cost and the gradients with respect to the outer loop variables.
    """
    unique_outer = outer_v is not None
    return ivy.execute_with_gradients(lambda v: _train_tasks(
        batch, inner_sub_batch_fn, outer_sub_batch_fn, inner_cost_fn, outer_cost_fn,
        variables.set_at_key_chains(v) if unique_outer else v, num_tasks, inner_grad_steps, inner_learning_rate,
        inner_optimization_step, 2, average_across_steps, inner_v, keep_inner_v, outer_v, keep_outer_v, return_inner_v),
        variables.at_key_chains(outer_v, ignore_none=True)
        if keep_outer_v else variables.prune_key_chains(outer_v, ignore_none=True))
